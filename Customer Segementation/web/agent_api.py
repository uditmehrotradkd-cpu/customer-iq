"""Data-agent endpoints: chat, training, publishing and dataset downloads."""
from __future__ import annotations

import json
import re
from pathlib import PurePath
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from segmentation import load_customers
from segmentation.auto_segment import K_MAX, K_MIN, MIN_SEGMENT_SHARE
from segmentation.config import DATE_COLUMN, DEFAULT_DATA_PATH
from segmentation.data_loader import to_canonical_schema

from .agent import draft_summary, handle_message
from .agent_generic import suggestions_for
from .api import AppSettings, _csv_response, get_registry
from .auth import current_user, require_user
from .datasets import DatasetError, build_dataset
from .file_io import FileFormatError, analyze_table, normalize_columns, read_table, table_profile
from .llm import LLMError, build_messages, plain_text, upload_context
from .registry import ModelRegistry, RegistryError, TrainerBusy
from .report import XLSX_MEDIA_TYPE, scored_report, workspace_report
from .schemas import AgentMessage, DatasetRequest, DraftRef
from .security import RateLimiter
from .services import sanitize_csv

router = APIRouter(prefix="/api/v1", tags=["agent"])
Registry = Annotated[ModelRegistry, Depends(get_registry)]
MAX_TOOL_ROUNDS = 3
# "apply / segment / show my uploaded file on the overview" typed after uploading a file.
APPLY_UPLOAD_RE = re.compile(
    r"^(please\s+|now\s+|can you\s+|could you\s+)*(apply|segment|cluster|switch to|update|rebuild|refresh|change|put|show|use|load)\b.*"
    r"\b(this|that|the|my|uploaded|last)\s+(file|upload|dataset|data|csv|excel|sheet)\b|"
    r"^(please\s+|now\s+)*(update|change|rebuild|refresh)\s+(the\s+|my\s+)?(overview|dashboard|workspace|site|pages)\b",
    re.I,
)


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


AI_TOOLS = {
    "apply_uploaded_file": _tool(
        "apply_uploaded_file",
        "Segment the file the user uploaded most recently and rebuild their whole workspace from it (Overview, Segment "
        "profiles, Explorer, Assign, Model & methodology). Use when the user asks to apply, use, segment or show that file.",
        {"k": {"type": "integer", "minimum": 2, "maximum": 10, "description": "Number of segments; omit to choose automatically."}},
    ),
    "reset_workspace": _tool("reset_workspace", "Switch the user's workspace back to the original demo data. Only when they explicitly ask."),
    "column_stats": _tool(
        "column_stats",
        "Exact statistics of one column of the data behind the current workspace, optionally per segment.",
        {
            "column": {"type": "string", "description": "Column name as listed in the workspace."},
            "by_segment": {"type": "boolean", "description": "Also return the statistics for every segment."},
        },
        ["column"],
    ),
    "download_report": _tool("download_report", "Give the user an Excel report of the current workspace with charts and segment profiles."),
}


def _check_train_quota(request: Request) -> None:
    require_user(request)
    limiter: RateLimiter = request.app.state.train_limiter
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(status_code=429, detail="Training limit reached (a few runs per 10 minutes). Please wait a little.")


def _user_error(exc: Exception) -> HTTPException:
    status = 409 if isinstance(exc, TrainerBusy) else 422
    detail = f"Missing or invalid column: {exc}" if isinstance(exc, KeyError) else str(exc)
    return HTTPException(status_code=status, detail=detail)


def _summary(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() and numeric.notna().sum() >= 0.9 * values.notna().sum():
        d = numeric.describe()
        return {k: round(float(d[k]), 4) for k in ("count", "mean", "std", "min", "50%", "max") if pd.notna(d.get(k))}
    return {"count": int(values.notna().sum()), "top_values": {str(k)[:40]: int(v) for k, v in values.astype(str).value_counts().head(6).items()}}


def column_stats(service, column: str, by_segment: bool = False) -> dict:
    frame = service.customers
    wanted = re.sub(r"[\s_]+", " ", str(column)).strip().lower()
    match = next((c for c in frame.columns if re.sub(r"[\s_]+", " ", str(c)).strip().lower() == wanted), None)
    if match is None:
        return {"error": "Unknown column.", "available_columns": [str(c) for c in frame.columns][:80]}
    result = {"column": str(match), "overall": _summary(frame[match])}
    if by_segment and "Segment_Name" in frame.columns:
        result["by_segment"] = {str(name): _summary(group[match]) for name, group in frame.groupby("Segment_Name")}
    return result


async def _run_tool(name: str, args: dict, request: Request, registry: ModelRegistry, settings) -> tuple[str, dict | None]:
    """Execute one AI tool call; returns (result for the model, response fields for the browser)."""
    try:
        if name == "column_stats":
            return json.dumps(column_stats(registry.live(), str(args.get("column", "")), bool(args.get("by_segment")))), None
        if name == "download_report":
            return "The report download button is shown to the user.", {"dataset": {"kind": "report", "filename": "segment_report.xlsx"}}
        if name == "reset_workspace":
            result = await agent_reset(request, registry)
            return result["reply"], {**result, "applied": True}
        if name == "apply_uploaded_file":
            upload = registry.load_upload()
            if upload is None:
                return "No uploaded file is available. Ask the user to attach one with the paperclip button.", None
            k = args.get("k")
            k = int(k) if isinstance(k, (int, float)) and 2 <= k <= 10 else None
            result = await _apply_upload(request, registry, settings, upload, k)
            return result["reply"], result
    except HTTPException as exc:
        return f"Failed: {exc.detail}", None
    return f"Unknown tool {name}.", None


async def _ask_ai(request: Request, registry: ModelRegistry, body: AgentMessage, settings) -> dict:
    """Answer a free-form question with the configured AI model, grounded in the workspace, able to call tools."""
    user = current_user(request)
    key = f"user:{user['id']}" if user else f"ip:{request.client.host if request.client else 'unknown'}"
    base = {"dataset": None, "preview": None, "draft": None, "links": [], "suggestions": []}
    if not request.app.state.ai_limiter.allow(key):
        return {**base, "reply": "You've asked a lot of AI questions in a short time. Please wait a few minutes and try again.", "intent": "ai_limited"}
    try:
        service = registry.resolve(body.draft_id)
    except RegistryError as exc:
        raise _user_error(exc) from None
    upload = registry.load_upload() if user else None
    upload_text = ""
    if upload:
        try:
            frame, _ = await run_in_threadpool(_parse_upload, upload[0], upload[1], settings)
            upload_text = upload_context(upload[1], await run_in_threadpool(table_profile, frame), len(frame))
        except HTTPException:
            upload = None
    tool_names = ["column_stats", "download_report"] + (["reset_workspace"] + (["apply_uploaded_file"] if upload else []) if user else [])
    tools = [AI_TOOLS[n] for n in tool_names]
    messages = build_messages(body.message, [t.model_dump() for t in body.history], service, user["name"] if user else "a visitor", upload_text, tools=True)
    ai = request.app.state.ai
    action: dict = {}
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            answer = await run_in_threadpool(ai.respond, messages, tools)
            calls = answer.get("tool_calls") or []
            if not calls:
                break
            messages.append({**answer, "role": "assistant"})  # echoed as-is (keeps provider fields such as thought signatures)
            for call in calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = fn.get("name", "")
                if name in tool_names and isinstance(args, dict):
                    output, payload = await _run_tool(name, args, request, registry, settings)
                else:
                    output, payload = f"Unknown tool {name}.", None
                if payload:
                    action.update(payload)
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name, "content": output[:6000]})
            answer = {}
    except LLMError as exc:
        if action:
            return {**base, **action}
        return {**base, "reply": f"Sorry, I couldn't get an answer: {exc}. Please try again in a moment.", "intent": "ai_error", "error": True}
    text = plain_text(str(answer.get("content") or ""))
    result = {**base, **action, "reply": text or action.get("reply") or "I don't have an answer to that.", "ai_model": ai.model}
    result["intent"] = action.get("intent", "ai") if not text else "ai"
    return result


async def _train(registry: ModelRegistry, raw: pd.DataFrame, k: int | None, source: str, settings) -> dict:
    if len(raw) > settings.max_train_rows:
        raise HTTPException(status_code=413, detail=f"Training data is limited to {settings.max_train_rows:,} rows.")
    if len(raw) < settings.min_train_rows:
        raise HTTPException(status_code=422, detail=f"Please provide at least {settings.min_train_rows} customers to train.")
    try:
        draft_id, _ = await run_in_threadpool(registry.train_draft, raw, k, source)
    except (RegistryError, ValueError, KeyError) as exc:
        raise _user_error(exc) from None
    summary = draft_summary(registry, draft_id)
    # Each signed-in user has a private workspace, so a new model is applied straight away.
    try:
        version = await run_in_threadpool(registry.publish, draft_id)
    except RegistryError as exc:
        raise _user_error(exc) from None
    names = ", ".join(f"{s['name']} ({s['customers_pct']:.0f}%)" for s in summary["segments"])
    return {
        "reply": (
            f"Done! Your workspace now runs on a model trained on {source}: {summary['k']} segments from "
            f"{summary['rows_clean']:,} clean customers (silhouette {summary['silhouette']:.3f}; previous model {summary['live']['silhouette']:.3f}). "
            f"Segments: {names}.\nEvery page (Overview, Segment profiles, Explorer, Assign and Model & methodology) now shows your data. "
            "Say \"reset to original\" to switch back to the demo data."
        ),
        "intent": "trained",
        "applied": True,
        "version": version,
        "dataset": {"kind": "report", "filename": f"{_safe_stem(source)}_segment_report.xlsx"},
        "preview": None,
        "draft": summary,
        "links": [["Open Overview", "#/overview"], ["Segment profiles", "#/segments"]],
        "suggestions": ["Download the profiles table", "Generate 500 synthetic customers", "Status"],
    }


@router.post("/agent/message")
async def agent_message(body: AgentMessage, request: Request, registry: Registry, settings: AppSettings) -> dict:
    ai = request.app.state.ai
    if APPLY_UPLOAD_RE.search(body.message.strip()) and current_user(request):
        upload = registry.load_upload()
        if upload is not None:
            return await _apply_upload(request, registry, settings, upload, None)
    try:
        result = await run_in_threadpool(handle_message, body.message, registry, body.draft_id, ai is not None)
    except (RegistryError, DatasetError) as exc:
        raise _user_error(exc) from None

    intent = result["intent"]
    if intent == "general_question":
        return await _ask_ai(request, registry, body, settings)
    if intent == "train":
        _check_train_quota(request)
        raw = await run_in_threadpool(load_customers, DEFAULT_DATA_PATH)
        return await _train(registry, raw, result.get("k"), "built-in dataset", settings)
    if intent == "publish":
        return await agent_publish(DraftRef(draft_id=body.draft_id), request, registry)
    if intent == "reset":
        return await agent_reset(request, registry)
    return result


@router.post("/agent/train")
async def agent_train(
    request: Request,
    registry: Registry,
    settings: AppSettings,
    file: UploadFile | None = File(default=None),
    k: int | None = Form(default=None, ge=2, le=10),
) -> dict:
    if file is not None:
        return await agent_file(request, registry, settings, file=file, action="train", k=k, draft_id=None)
    _check_train_quota(request)
    raw = await run_in_threadpool(load_customers, DEFAULT_DATA_PATH)
    return await _train(registry, raw, k, "built-in dataset", settings)


async def _read_upload(file: UploadFile, settings) -> tuple[pd.DataFrame, dict, bytes]:
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb:g} MB.")
    frame, renames = await run_in_threadpool(_parse_upload, content, file.filename or "", settings)
    return frame, renames, content


def _parse_upload(content: bytes, filename: str, settings) -> tuple[pd.DataFrame, dict]:
    try:
        frame = read_table(content, filename, settings.max_train_rows)
    except FileFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return normalize_columns(frame)


async def _apply_file(request: Request, registry: ModelRegistry, settings, frame: pd.DataFrame, renames: dict, name: str, k: int | None) -> dict:
    """Make an uploaded file the user's workspace: customer files train the customer model, anything else is segmented on its own columns."""
    require_user(request)
    if len(frame) < settings.min_train_rows:
        report = analyze_table(frame, renames)
        result = _analysis_reply(name, report, frame)
        result["reply"] = (
            f"{name} has only {len(frame):,} rows; I need at least {settings.min_train_rows} rows to find reliable segments, "
            "so your pages still show the previous data. Here is what I found in the file:\n" + result["reply"]
        )
        result["file_actions"] = ["score"] if report["compatible"] or registry.live().mode == "generic" else []
        return result
    _check_train_quota(request)
    try:
        canonical = to_canonical_schema(frame)
    except ValueError:
        canonical = None
    if canonical is not None and len(canonical) >= settings.min_train_rows:
        return await _train(registry, canonical, k, f"{name} ({len(canonical):,} rows)", settings)
    return await _cluster(registry, frame, name, k, settings)


async def _apply_upload(request: Request, registry: ModelRegistry, settings, upload: tuple[bytes, str], k: int | None) -> dict:
    content, name = upload
    frame, renames = await run_in_threadpool(_parse_upload, content, name, settings)
    return await _apply_file(request, registry, settings, frame, renames, name, k)


def _analysis_reply(name: str, report: dict, frame: pd.DataFrame) -> dict:
    lines = [f"{name}: {report['rows']:,} rows × {report['columns']} columns."]
    if report["renamed"]:
        lines.append("Mapped columns: " + ", ".join(f"{a} → {b}" for a, b in list(report["renamed"].items())[:8]) + ".")
    if report["compatible"]:
        lines.append("✅ All required customer columns are present, so I can assign segments with the website model or train a new customer model.")
        if report["optional_missing"]:
            lines.append(f"No {', '.join(report['optional_missing'])} column: that's fine, customer tenure will just be left blank.")
    else:
        lines.append(
            f"This isn't the website's customer format (missing {', '.join(report['missing_columns'][:6])}"
            + ("…" if len(report["missing_columns"]) > 6 else "")
            + "), but I can still find segments in it using all of its columns."
        )
    if report["missing_values"]:
        lines.append("Missing values: " + ", ".join(f"{c} ({n})" for c, n in report["missing_values"].items()) + " — they will be imputed.")
    if report["duplicate_rows"]:
        lines.append(f"{report['duplicate_rows']:,} duplicate rows will be removed during training.")
    for stat in report["numeric_summary"]:
        lines.append(f"{stat['column']}: mean {stat['mean']:,}, range {stat['min']:,} – {stat['max']:,}.")
    if not report["compatible"]:
        profile = table_profile(frame)
        numbers = [c for c in profile if c["type"] == "number"]
        texts = [c for c in profile if c["type"] == "text"]
        lines.append(f"{len(numbers)} numeric and {len(texts)} text columns.")
        lines += [f"{c['column']}: mean {c['mean']:,}, range {c['min']:,} – {c['max']:,}." for c in numbers[:5]]
        lines += [f"{c['column']}: {c['distinct']:,} distinct values, most common {c['top']}." for c in texts[:3]]
    lines.append('Choose "Segment & apply" (or say "apply this file") to rebuild Overview, Profiles, Explorer and every other page from this file.')
    head = frame.iloc[:6, :8]
    return {
        "reply": "\n".join(lines),
        "intent": "analyzed",
        "dataset": None,
        "preview": {"columns": [str(c) for c in head.columns], "rows": head.astype(object).where(head.notna(), None).values.tolist(), "total": int(len(frame))},
        "draft": None,
        "links": [],
        "suggestions": [],
        "file_actions": ["auto", "score", "cluster"] if report["compatible"] else ["auto"],
    }


def _safe_stem(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", PurePath(filename).stem).strip("_")[:60] or "data"


async def _score_generic(registry: ModelRegistry, service, frame: pd.DataFrame, name: str) -> dict:
    """Assign rows of a file to the segments of a workspace built from the user's own dataset."""
    try:
        scored = await run_in_threadpool(service.predict_frame, frame)
    except (ValueError, KeyError) as exc:
        raise _user_error(exc) from None
    out = pd.concat([frame.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    token = await run_in_threadpool(registry.save_result, sanitize_csv(out))
    content = await run_in_threadpool(scored_report, service, out, name)
    report_token = await run_in_threadpool(registry.save_report, content)
    stem = _safe_stem(name)
    missing = sorted({s.source for s in service.result.encoder.specs} - set(frame.columns))
    mix = scored["Segment_Name"].value_counts(normalize=True)
    head = out[[c for c in ["Segment_Name", "Assignment_Margin", *list(frame.columns)[:5]] if c in out.columns]].head(6)
    reply = f"Assigned all {len(out):,} rows in {name} to the segments of your {service.source} workspace: " + ", ".join(f"{seg} {share:.0%}" for seg, share in mix.items()) + "."
    if missing:
        reply += f" {len(missing)} column(s) the segments were built on were missing and filled with typical values: {', '.join(missing[:5])}."
    return {
        "reply": reply,
        "intent": "scored",
        "dataset": {"kind": "report", "token": report_token, "filename": f"{stem}_segment_report.xlsx"},
        "downloads": [{"label": "Scored data (CSV)", "dataset": {"kind": "result", "token": token, "filename": f"{stem}_scored.csv"}}],
        "preview": {"columns": list(head.columns), "rows": head.astype(object).where(head.notna(), None).values.tolist(), "total": int(len(out))},
        "draft": None,
        "links": [],
        "suggestions": [],
    }


async def _cluster(registry: ModelRegistry, frame: pd.DataFrame, name: str, k: int | None, settings, note: str = "") -> dict:
    """Segment any table on all of its usable columns and make it the user's workspace (every page switches to it)."""
    if len(frame) < settings.min_train_rows:
        raise HTTPException(status_code=422, detail=f"Please provide at least {settings.min_train_rows} rows to find segments.")
    try:
        draft_id, service = await run_in_threadpool(registry.train_generic, frame, k, name)
        version = await run_in_threadpool(registry.publish, draft_id)
    except (RegistryError, ValueError) as exc:
        raise _user_error(exc) from None
    result = service.result
    stem = _safe_stem(name)
    token = await run_in_threadpool(registry.save_result, service.export_customers(None))
    profile_token = await run_in_threadpool(registry.save_result, sanitize_csv(result.profiles))
    content = await run_in_threadpool(workspace_report, service, name)
    report_token = await run_in_threadpool(registry.save_report, content)
    used = len(result.numeric_columns) + len(result.categorical_columns)
    how_k = (
        f"I tested {K_MIN}–{K_MAX} segments and picked k={result.k} (best separation with at least 3 actionable groups, each ≥ {MIN_SEGMENT_SHARE:.0%} of rows)."
        if result.auto_k
        else f"You asked for {result.k} segments."
    )
    lines = [
        *([note] if note else []),
        f"I found {result.k} segments across all {len(frame):,} rows of {name}, using {used} columns "
        f"({len(result.numeric_columns)} numeric, {len(result.categorical_columns)} categorical). Silhouette {result.silhouette:.3f}.",
        how_k,
    ]
    if result.dropped:
        lines.append("Not used: " + ", ".join(f"{c} ({why})" for c, why in list(result.dropped.items())[:6]) + ".")
    lines.append(
        f"Your whole workspace now runs on {name}: Overview, Segment profiles, Explorer, Assign and Model & methodology all show this dataset. "
        "Download the Excel report for charts and a plain-language summary. Say \"reset to original\" to go back to the demo data."
    )
    preview_cols = ["Segment_Name", "Rows", "Share_%", *[c for c in result.profiles.columns if c.startswith("Avg ")][:5]]
    head = result.profiles[preview_cols]
    return {
        "reply": "\n".join(lines),
        "intent": "clustered",
        "applied": True,
        "version": version,
        "dataset": {"kind": "report", "token": report_token, "filename": f"{stem}_segment_report.xlsx"},
        "downloads": [
            {"label": "Labelled data (CSV)", "dataset": {"kind": "result", "token": token, "filename": f"{stem}_segmented.csv"}},
            {"label": "Segment profiles (CSV)", "dataset": {"kind": "result", "token": profile_token, "filename": f"{stem}_segment_profiles.csv"}},
        ],
        "preview": {"columns": preview_cols, "rows": head.astype(object).where(head.notna(), None).values.tolist(), "total": int(len(head))},
        "segments_view": result.segments,
        "draft": None,
        "links": [["Open Overview", "#/overview"], ["Segment profiles", "#/segments"], ["Explorer", "#/explorer"]],
        "suggestions": ["Why this number of segments?", "Compare all segments", "Download the segment report with charts"],
    }


@router.post("/agent/file")
async def agent_file(
    request: Request,
    registry: Registry,
    settings: AppSettings,
    file: UploadFile = File(...),
    action: Literal["auto", "analyze", "score", "train", "cluster"] = Form(default="analyze"),
    k: int | None = Form(default=None, ge=2, le=10),
    draft_id: str | None = Form(default=None, pattern=r"^[a-f0-9]{32}$"),
) -> dict:
    """Apply (auto), analyse, score, train or auto-segment an uploaded CSV / Excel / JSON file."""
    name = file.filename or "file"
    frame, renames, content = await _read_upload(file, settings)
    if current_user(request):
        await run_in_threadpool(registry.save_upload, content, name)
    if action == "auto":
        return await _apply_file(request, registry, settings, frame, renames, name, k)
    if action == "analyze":
        return _analysis_reply(name, analyze_table(frame, renames), frame)
    if action == "cluster":
        _check_train_quota(request)
        return await _cluster(registry, frame, name, k, settings)
    if action == "score":
        try:
            service = registry.resolve(draft_id)
        except RegistryError as exc:
            raise _user_error(exc) from None
        if service.mode == "generic":
            return await _score_generic(registry, service, frame, name)
    try:
        canonical = to_canonical_schema(frame)
    except ValueError as exc:
        if action == "train":
            _check_train_quota(request)
            note = f"{name} isn't in the customer format the website model needs ({exc}), so I segmented it on all of its own columns instead."
            return await _cluster(registry, frame, name, k, settings, note)
        raise HTTPException(status_code=422, detail=f"{exc}. Choose \"Find segments (all columns)\" to segment this file on its own columns.") from None

    if action == "train":
        _check_train_quota(request)
        return await _train(registry, canonical, k, f"{name} ({len(canonical):,} rows)", settings)

    try:
        service = registry.resolve(draft_id)
        scored = await run_in_threadpool(service.segmenter.predict, canonical)
    except (RegistryError, ValueError, KeyError) as exc:
        raise _user_error(exc) from None
    out = pd.concat([canonical.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    if not any(str(c).startswith(DATE_COLUMN) for c in frame.columns):
        out = out.drop(columns=DATE_COLUMN, errors="ignore")
    token = await run_in_threadpool(registry.save_result, sanitize_csv(out))
    content = await run_in_threadpool(scored_report, service, out, name)
    report_token = await run_in_threadpool(registry.save_report, content)
    stem = _safe_stem(name)
    mix = scored["Segment_Name"].value_counts(normalize=True)
    head = out[[c for c in ("ID", "Segment_Name", "Assignment_Margin", "Income", "MntWines", "Recency") if c in out.columns]].head(6)
    return {
        "reply": f"Assigned all {len(out):,} customers in {name} to segments using {'your draft model' if draft_id else 'the live model'}: "
        + ", ".join(f"{seg} {share:.0%}" for seg, share in mix.items())
        + f". {int((scored['Assignment_Margin'] < 0.1).sum()):,} rows are borderline (they sit between two segments).",
        "intent": "scored",
        "dataset": {"kind": "report", "token": report_token, "filename": f"{stem}_segment_report.xlsx"},
        "downloads": [{"label": "Scored data (CSV)", "dataset": {"kind": "result", "token": token, "filename": f"{stem}_scored.csv"}}],
        "preview": {"columns": list(head.columns), "rows": head.astype(object).where(head.notna(), None).values.tolist(), "total": int(len(out))},
        "draft": None,
        "links": [],
        "suggestions": [],
    }


@router.post("/agent/publish")
async def agent_publish(body: DraftRef, request: Request, registry: Registry) -> dict:
    require_user(request)
    try:
        version = await run_in_threadpool(registry.publish, body.draft_id)
    except RegistryError as exc:
        raise _user_error(exc) from None
    status = registry.status()
    return {
        "reply": f"Applied. Every page in your workspace now uses this model ({status['n_segments']} segments). Say \"reset to original\" to undo.",
        "intent": "published",
        "dataset": None,
        "preview": None,
        "draft": None,
        "suggestions": ["Status", "Reset to original"],
        "version": version,
    }


@router.post("/agent/reset")
async def agent_reset(request: Request, registry: Registry) -> dict:
    require_user(request)
    await run_in_threadpool(registry.reset)
    status = registry.status()
    return {
        "reply": f"Done. Your workspace is back on the original demo model ({status['n_segments']} segments).",
        "intent": "reset_done",
        "dataset": None,
        "preview": None,
        "draft": None,
        "suggestions": ["Status", "Train with 5 segments"],
    }


@router.get("/agent/status")
def agent_status(request: Request, registry: Registry) -> dict:
    status = registry.status()
    live = registry.live()
    if live.mode == "generic":
        status["prompts"] = suggestions_for(live)
    ai = request.app.state.ai
    status["ai"] = {"enabled": ai is not None, "model": ai.model if ai else None}
    return status


@router.post("/datasets")
async def download_dataset(body: DatasetRequest, registry: Registry):
    try:
        if body.kind == "report":
            if body.token:
                content = await run_in_threadpool(registry.load_report, body.token)
            else:
                service = registry.resolve(body.draft_id)
                source = "your data" if registry.status()["live_source"] == "published" else "demo dataset"
                content = await run_in_threadpool(workspace_report, service, source)
            return Response(content, media_type=XLSX_MEDIA_TYPE, headers={"Content-Disposition": 'attachment; filename="segment_report.xlsx"'})
        if body.kind == "result":
            frame = await run_in_threadpool(registry.load_result, body.token or "")
            return _csv_response(frame, "scored_customers.csv")
        service = registry.resolve(body.draft_id)
        frame, filename, _ = await run_in_threadpool(build_dataset, service, body.model_dump())
    except (RegistryError, DatasetError) as exc:
        raise _user_error(exc) from None
    return _csv_response(frame, filename)
