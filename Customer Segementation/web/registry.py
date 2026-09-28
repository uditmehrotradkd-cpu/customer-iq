"""Model registry: the live model, trained drafts, publishing and rollback.

Files live under ``store_dir`` so every worker process sees the same state:
  drafts/<id>/            models trained by the agent (auto-expire)
  published/<version>/    published models
  published/CURRENT       name of the live published version (absent = original build-time model)
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path, PurePath

import pandas as pd

from segmentation import CustomerSegmenter, SegmentationConfig

from .generic_service import GENERIC_FILENAME, GenericService
from .services import SegmentationService

log = logging.getLogger(__name__)


def load_service(path: Path) -> SegmentationService | GenericService:
    if (Path(path) / GENERIC_FILENAME).exists():
        return GenericService.load(path)
    return SegmentationService.load(path, auto_train=False)

DRAFT_ID = re.compile(r"^[a-f0-9]{32}$")
DRAFT_TTL_SECONDS = 24 * 60 * 60
RESULT_TTL_SECONDS = 24 * 60 * 60
MAX_DRAFTS = 20
KEEP_PUBLISHED_VERSIONS = 3


class RegistryError(ValueError):
    """User-facing registry problem (unknown draft, busy trainer, bad data)."""


class TrainerBusy(RegistryError):
    pass


class ModelRegistry:
    def __init__(self, base: SegmentationService, store_dir: Path, config: SegmentationConfig | None = None, train_lock: threading.Lock | None = None):
        self.base = base
        self.config = config or SegmentationConfig()
        self.store_dir = Path(store_dir)
        self.drafts_dir = self.store_dir / "drafts"
        self.published_dir = self.store_dir / "published"
        self.pointer = self.published_dir / "CURRENT"
        self.results_dir = self.store_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        self.published_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._train_lock = train_lock or threading.Lock()
        self._live = base
        self._live_key: tuple | None = None
        self._live_version: str | None = None
        self._drafts: OrderedDict[str, SegmentationService] = OrderedDict()

    # ------------------------------------------------------------------ live model
    def _pointer_state(self) -> tuple[str | None, float | None]:
        try:
            stat = self.pointer.stat()
            return self.pointer.read_text(encoding="utf-8").strip(), stat.st_mtime
        except FileNotFoundError:
            return None, None

    def live(self) -> SegmentationService:
        """Current live model; reloads when another worker publishes or resets."""
        version, mtime = self._pointer_state()
        key = (version, mtime)
        if key != self._live_key:
            with self._lock:
                if key != self._live_key:
                    if version and (self.published_dir / version).is_dir():
                        self._live = load_service(self.published_dir / version)
                        self._live_version = version
                    else:
                        self._live = self.base
                        self._live_version = None
                    self._live_key = key
        return self._live

    def status(self) -> dict:
        live = self.live()
        return {
            "live_source": "published" if self._live_version else "original",
            "live_version": self._live_version,
            "mode": live.mode,
            "labels": live.labels(),
            "n_segments": live.segmenter.n_segments,
            "silhouette": round(live.segmenter.silhouette_, 4),
            "trained_at_utc": live.trained_at_utc,
            "segments": [s["name"] for s in live.segments_meta()],
        }

    # ------------------------------------------------------------------ drafts
    def _draft_dir(self, draft_id: str) -> Path:
        if not DRAFT_ID.fullmatch(draft_id or ""):
            raise RegistryError("Invalid draft id.")
        path = self.drafts_dir / draft_id
        if not path.is_dir():
            raise RegistryError("This draft has expired. Train again to create a new one.")
        return path

    def draft(self, draft_id: str) -> SegmentationService:
        if draft_id in self._drafts:
            self._drafts.move_to_end(draft_id)
            return self._drafts[draft_id]
        service = load_service(self._draft_dir(draft_id))
        self._drafts[draft_id] = service
        while len(self._drafts) > 4:
            self._drafts.popitem(last=False)
        return service

    def draft_meta(self, draft_id: str) -> dict:
        return json.loads((self._draft_dir(draft_id) / "agent_meta.json").read_text(encoding="utf-8"))

    def resolve(self, draft_id: str | None) -> SegmentationService:
        return self.draft(draft_id) if draft_id else self.live()

    def _prune_drafts(self) -> None:
        now = time.time()
        drafts = sorted((p for p in self.drafts_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        for i, path in enumerate(drafts):
            if now - path.stat().st_mtime > DRAFT_TTL_SECONDS or i < len(drafts) - MAX_DRAFTS:
                shutil.rmtree(path, ignore_errors=True)
                self._drafts.pop(path.name, None)

    def train_draft(self, raw: pd.DataFrame, n_clusters: int | None, source: str) -> tuple[str, SegmentationService]:
        if not self._train_lock.acquire(blocking=False):
            raise TrainerBusy("Another model is training right now. Please try again in a moment.")
        try:
            self._prune_drafts()
            segmenter = CustomerSegmenter(self.config, n_clusters=n_clusters).fit(raw)
            draft_id = secrets.token_hex(16)
            path = self.drafts_dir / draft_id
            segmenter.save(path)
            meta = {
                "draft_id": draft_id,
                "source": source,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rows_raw": int(segmenter.n_raw_rows_),
                "rows_clean": int(len(segmenter.customers_)),
                "k": segmenter.n_segments,
                "k_requested": n_clusters,
            }
            (path / "agent_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            service = SegmentationService.load(path, auto_train=False)
            self._drafts[draft_id] = service
            log.info("Trained draft %s (k=%d, rows=%d, source=%s)", draft_id, meta["k"], meta["rows_clean"], source)
            return draft_id, service
        finally:
            self._train_lock.release()

    def train_generic(self, table: pd.DataFrame, n_clusters: int | None, source: str) -> tuple[str, GenericService]:
        """Build a workspace from any table (all usable columns) and store it as a draft."""
        if not self._train_lock.acquire(blocking=False):
            raise TrainerBusy("Another model is training right now. Please try again in a moment.")
        try:
            self._prune_drafts()
            service = GenericService.fit(table, n_clusters, source)
            draft_id = secrets.token_hex(16)
            path = service.save(self.drafts_dir / draft_id)
            meta = {
                "draft_id": draft_id,
                "source": source,
                "mode": "generic",
                "created_at_utc": service.trained_at_utc,
                "rows_raw": int(len(table)),
                "rows_clean": int(len(service.customers)),
                "k": service.result.k,
                "k_requested": n_clusters,
            }
            (path / "agent_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._drafts[draft_id] = service
            log.info("Built dataset workspace %s (k=%d, rows=%d, source=%s)", draft_id, meta["k"], meta["rows_clean"], source)
            return draft_id, service
        finally:
            self._train_lock.release()

    # ------------------------------------------------------------------ scored-file results
    def _new_token(self) -> str:
        now = time.time()
        for old in self.results_dir.glob("*.*"):
            if now - old.stat().st_mtime > RESULT_TTL_SECONDS:
                old.unlink(missing_ok=True)
        return secrets.token_hex(16)

    def save_result(self, frame: pd.DataFrame) -> str:
        token = self._new_token()
        frame.to_csv(self.results_dir / f"{token}.csv", index=False)
        return token

    def load_result(self, token: str) -> pd.DataFrame:
        path = self.results_dir / f"{token}.csv"
        if not DRAFT_ID.fullmatch(token or "") or not path.is_file():
            raise RegistryError("This result has expired. Please upload the file again.")
        return pd.read_csv(path)

    def save_report(self, content: bytes) -> str:
        token = self._new_token()
        (self.results_dir / f"{token}.xlsx").write_bytes(content)
        return token

    def load_report(self, token: str) -> bytes:
        path = self.results_dir / f"{token}.xlsx"
        if not DRAFT_ID.fullmatch(token or "") or not path.is_file():
            raise RegistryError("This report has expired. Please upload the file again.")
        return path.read_bytes()

    # ------------------------------------------------------------------ last uploaded file (so chat can act on it later)
    def save_upload(self, content: bytes, filename: str) -> None:
        folder = self.store_dir / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "last.bin").write_bytes(content)
        meta = {"filename": PurePath(filename or "file.csv").name[:120], "saved_at": time.time()}
        (folder / "last.json").write_text(json.dumps(meta), encoding="utf-8")

    def load_upload(self) -> tuple[bytes, str] | None:
        folder = self.store_dir / "uploads"
        try:
            meta = json.loads((folder / "last.json").read_text(encoding="utf-8"))
            if time.time() - float(meta["saved_at"]) > RESULT_TTL_SECONDS:
                return None
            return (folder / "last.bin").read_bytes(), str(meta["filename"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------ publish / reset
    def publish(self, draft_id: str) -> str:
        source = self._draft_dir(draft_id)
        version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{draft_id[:8]}"
        target = self.published_dir / version
        if not target.exists():
            shutil.copytree(source, target)
        tmp = self.pointer.with_suffix(".tmp")
        tmp.write_text(version, encoding="utf-8")
        os.replace(tmp, self.pointer)
        versions = sorted(p for p in self.published_dir.iterdir() if p.is_dir())
        for old in versions[:-KEEP_PUBLISHED_VERSIONS]:
            if old.name != version:
                shutil.rmtree(old, ignore_errors=True)
        log.warning("Published draft %s as live version %s", draft_id, version)
        return version

    def reset(self) -> None:
        try:
            self.pointer.unlink()
            log.warning("Live model reset to the original build-time model")
        except FileNotFoundError:
            pass


class RegistryPool:
    """One registry per signed-in user (own drafts, results and live model) plus a shared anonymous one."""

    MAX_CACHED_USERS = 32

    def __init__(self, shared: ModelRegistry):
        self.shared = shared
        self._users: OrderedDict[str, ModelRegistry] = OrderedDict()
        self._lock = threading.Lock()

    def for_user(self, key: str) -> ModelRegistry:
        if not re.fullmatch(r"[a-f0-9]{24}", key):
            raise ValueError("Invalid workspace key.")
        with self._lock:
            registry = self._users.get(key)
            if registry is None:
                registry = ModelRegistry(
                    self.shared.base,
                    self.shared.store_dir / "users" / key,
                    self.shared.config,
                    train_lock=self.shared._train_lock,
                )
                self._users[key] = registry
                while len(self._users) > self.MAX_CACHED_USERS:
                    self._users.popitem(last=False)
            self._users.move_to_end(key)
            registry.config = self.shared.config
            return registry


def migrate_workspaces(users_dir: Path, accounts: list[dict], key_for) -> int:
    """Rename old numeric workspace folders (users/<id>) to their username-based key."""
    moved = 0
    for account in accounts:
        source = users_dir / str(int(account["id"]))
        target = users_dir / key_for(account["username"])
        if source.is_dir() and not target.exists():
            try:
                source.rename(target)
                moved += 1
            except OSError:
                log.warning("Could not move workspace %s", source)
    return moved
