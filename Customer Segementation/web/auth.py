"""User accounts: registration, login sessions (HttpOnly cookie) and per-user chat history.

Accounts live in a cloud PostgreSQL database when ``SEG_DATABASE_URL`` / ``DATABASE_URL`` is set
(so the same login works on every server and survives redeploys), otherwise in a local SQLite file.
Passwords are hashed with scrypt; only a SHA-256 of each session token is stored, so a leaked
database cannot be used to hijack sessions.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

COOKIE_NAME = "ciq_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
MAX_HISTORY_BYTES = 2 * 1024 * 1024
USERNAME_RE = r"^[A-Za-z0-9_.-]{3,32}$"
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
# Signed-in users make several API calls per page; caching token lookups briefly saves cloud round trips.
SESSION_CACHE_SECONDS = 30
log = logging.getLogger(__name__)

SQLITE_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        display_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        history TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        expires_at REAL NOT NULL
    )""",
]
POSTGRES_SCHEMA = [
    "SELECT pg_advisory_xact_lock(724301)",  # several workers may start at once
    """CREATE TABLE IF NOT EXISTS users (
        id BIGSERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        display_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        history TEXT
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS users_username_ci ON users (lower(username))",
    """CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        expires_at DOUBLE PRECISION NOT NULL
    )""",
]


def workspace_key(username: str) -> str:
    """Folder name of a user's workspace; stable across databases because usernames are unique."""
    return hashlib.sha256(username.lower().encode()).hexdigest()[:24]


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


# Verifying against this keeps login time the same for unknown usernames (no user enumeration).
_DUMMY_HASH = hash_password(secrets.token_hex(8))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def clean_display_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f<>]", "", value)).strip()[:40]


class AuthError(ValueError):
    """User-facing account problem."""


class _Db:
    """Runs '?'-style SQL on SQLite or PostgreSQL."""

    def __init__(self, conn, postgres: bool):
        self.conn = conn
        self.postgres = postgres

    def execute(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql.replace("?", "%s") if self.postgres else sql, params)


class UserStore:
    def __init__(self, path: Path | None = None, url: str = ""):
        self.url = url
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._conn = None
        self._sessions: dict[str, tuple[dict, float]] = {}
        if url:
            import psycopg  # only needed for the cloud database

            self._pg = psycopg
            self._integrity_errors: tuple[type[Exception], ...] = (psycopg.IntegrityError,)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._integrity_errors = (sqlite3.IntegrityError,)
        schema = POSTGRES_SCHEMA if url else SQLITE_SCHEMA
        self._run(lambda db: [db.execute(stmt) for stmt in schema])

    @property
    def backend(self) -> str:
        return "postgres" if self.url else "sqlite"

    def _pg_connection(self):
        if self._conn is None or self._conn.closed:
            from psycopg.rows import dict_row

            self._conn = self._pg.connect(self.url, row_factory=dict_row, connect_timeout=10, autocommit=True)
        return self._conn

    def _run(self, fn):
        """Run ``fn(db)`` in one transaction (one retry if a pooled cloud connection went stale)."""
        with self._lock:
            if not self.url:
                conn = sqlite3.connect(self.path, timeout=10)
                conn.row_factory = sqlite3.Row
                try:
                    with conn:
                        return fn(_Db(conn, postgres=False))
                finally:
                    conn.close()
            for attempt in (1, 2):
                conn = self._pg_connection()
                try:
                    with conn.transaction():
                        return fn(_Db(conn, postgres=True))
                except self._pg.OperationalError:
                    try:
                        conn.close()
                    finally:
                        self._conn = None
                    if attempt == 2:
                        raise
                    log.warning("Account database connection dropped; reconnecting")

    @staticmethod
    def _public(row) -> dict:
        return {"id": row["id"], "username": row["username"], "name": row["display_name"], "created_at": row["created_at"]}

    def create_user(self, username: str, name: str, password: str) -> dict:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        password_hash = hash_password(password)

        def insert(db):
            if db.execute("SELECT 1 FROM users WHERE lower(username) = lower(?)", (username,)).fetchone():
                raise AuthError("That username is already taken.")
            return db.execute(
                "INSERT INTO users (username, display_name, password_hash, created_at) VALUES (?, ?, ?, ?) "
                "RETURNING id, username, display_name, created_at",
                (username, clean_display_name(name) or username, password_hash, created),
            ).fetchone()

        try:
            row = self._run(insert)
        except self._integrity_errors:
            raise AuthError("That username is already taken.") from None
        return self._public(row)

    def authenticate(self, username: str, password: str) -> dict | None:
        row = self._run(lambda db: db.execute("SELECT * FROM users WHERE lower(username) = lower(?)", (username.strip(),)).fetchone())
        if row is None:
            verify_password(password, _DUMMY_HASH)
            return None
        return self._public(row) if verify_password(password, row["password_hash"]) else None

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()

        def insert(db):
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            db.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)", (_token_hash(token), user_id, now + SESSION_TTL_SECONDS))

        self._run(insert)
        return token

    def user_for_token(self, token: str | None) -> dict | None:
        if not token or len(token) > 100:
            return None
        key = _token_hash(token)
        cached = self._sessions.get(key)
        if cached and cached[1] > time.time():
            return cached[0]
        row = self._run(
            lambda db: db.execute(
                "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at > ?",
                (key, time.time()),
            ).fetchone()
        )
        user = self._public(row) if row else None
        if user and self.url:
            if len(self._sessions) > 1000:
                self._sessions.clear()
            self._sessions[key] = (user, time.time() + SESSION_CACHE_SECONDS)
        return user

    def delete_session(self, token: str | None) -> None:
        if token:
            key = _token_hash(token)
            self._sessions.pop(key, None)
            self._run(lambda db: db.execute("DELETE FROM sessions WHERE token_hash = ?", (key,)))

    def get_history(self, user_id: int) -> list:
        row = self._run(lambda db: db.execute("SELECT history FROM users WHERE id = ?", (user_id,)).fetchone())
        try:
            data = json.loads(row["history"]) if row and row["history"] else []
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def set_history(self, user_id: int, chats: list) -> None:
        payload = json.dumps(chats, separators=(",", ":"))
        if len(payload.encode()) > MAX_HISTORY_BYTES:
            raise AuthError("Chat history is too large. Delete a few old chats and try again.")
        self._run(lambda db: db.execute("UPDATE users SET history = ? WHERE id = ?", (payload, user_id)))

    def import_users(self, rows: list[dict]) -> int:
        """Copy accounts (with their password hashes and chats) from an older database; existing usernames win."""

        def copy(db) -> int:
            added = 0
            for r in rows:
                if db.execute("SELECT 1 FROM users WHERE lower(username) = lower(?)", (r["username"],)).fetchone():
                    continue
                db.execute(
                    "INSERT INTO users (username, display_name, password_hash, created_at, history) VALUES (?, ?, ?, ?, ?)",
                    (r["username"], r["display_name"], r["password_hash"], r["created_at"], r["history"]),
                )
                added += 1
            return added

        return self._run(copy)


def read_sqlite_users(path: Path) -> list[dict]:
    """All accounts in a SQLite users.db (used to move accounts into the cloud database)."""
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT id, username, display_name, password_hash, created_at, history FROM users ORDER BY id")]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def open_user_store(settings) -> UserStore:
    """Cloud database when configured (importing any accounts from the old local file once), else SQLite."""
    legacy = settings.model_store_dir / "users.db"
    if not settings.database_url:
        return UserStore(legacy)
    store = UserStore(url=settings.database_url)
    marker = settings.model_store_dir / "users.db.imported"
    if legacy.exists() and not marker.exists():
        try:
            added = store.import_users(read_sqlite_users(legacy))
        except store._integrity_errors:
            added = 0  # another worker imported them at the same moment
        marker.write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"), encoding="utf-8")
        log.warning("Imported %d account(s) from %s into the cloud database", added, legacy)
    return store


# ---------------------------------------------------------------------- API
class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(pattern=USERNAME_RE)
    password: str = Field(min_length=8, max_length=128)


class Registration(Credentials):
    name: str = Field(min_length=1, max_length=40)


class HistoryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chats: list[dict] = Field(max_length=50)


def _store(request: Request) -> UserStore:
    return request.app.state.users


def current_user(request: Request) -> dict | None:
    store = getattr(request.app.state, "users", None)
    return store.user_for_token(request.cookies.get(COOKIE_NAME)) if store else None


def require_user(request: Request) -> dict:
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in first.")
    return user


def check_same_origin(request: Request) -> None:
    """Reject cross-site state changes (defence in depth on top of SameSite cookies)."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    origin = request.headers.get("origin")
    if not origin or origin in request.app.state.settings.cors_origins:
        return
    if urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower():
        raise HTTPException(status_code=403, detail="Cross-site request blocked.")


User = Annotated[dict, Depends(require_user)]
router = APIRouter(prefix="/api/v1/auth", tags=["auth"], dependencies=[Depends(check_same_origin)])


def _login_guard(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    if not request.app.state.login_limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many sign-in attempts. Please wait a few minutes.")


def _start_session(request: Request, response: Response, user: dict) -> dict:
    token = _store(request).create_session(user["id"])
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return user


@router.post("/register", dependencies=[Depends(_login_guard)])
def register(body: Registration, request: Request, response: Response) -> dict:
    try:
        user = _store(request).create_user(body.username, body.name, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return _start_session(request, response, user)


@router.post("/login", dependencies=[Depends(_login_guard)])
def login(body: Credentials, request: Request, response: Response) -> dict:
    user = _store(request).authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    return _start_session(request, response, user)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    _store(request).delete_session(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: User) -> dict:
    return user


@router.get("/me/history")
def get_history(user: User, request: Request) -> dict:
    return {"chats": _store(request).get_history(user["id"])}


@router.put("/me/history")
def put_history(body: HistoryBody, user: User, request: Request) -> dict:
    try:
        _store(request).set_history(user["id"], body.chats)
    except AuthError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None
    return {"ok": True}
