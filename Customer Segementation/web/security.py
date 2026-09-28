"""HTTP hardening: security headers, request-size guard and a per-client rate limiter."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
    "media-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; "
    "form-action 'self'; frame-ancestors 'none'"
)
# Swagger UI loads its assets from a CDN, so the docs pages get a relaxed policy.
DOCS_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
    "frame-ancestors 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, enable_hsts: bool = False):
        super().__init__(app)
        self.enable_hsts = enable_hsts

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        is_docs = request.url.path in {"/docs", "/redoc"}
        response.headers["Content-Security-Policy"] = DOCS_CONTENT_SECURITY_POLICY if is_docs else CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith(("/static/vendor/", "/static/media/")):
            response.headers["Cache-Control"] = "public, max-age=604800"
        elif request.url.path.startswith("/static/"):
            # Revalidate app assets via ETag so a redeploy is picked up immediately.
            response.headers["Cache-Control"] = "no-cache"
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared body exceeds the limit before they are read."""

    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
            return JSONResponse({"detail": "Request body too large."}, status_code=413)
        return await call_next(request)


class RateLimiter:
    """Sliding-window limiter keyed by client IP (per process; use a shared store behind a load balancer)."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Throttle state-changing (POST) API calls, which run model inference."""

    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and request.url.path.startswith("/api/"):
            client = request.client.host if request.client else "unknown"
            if not self.limiter.allow(client):
                return JSONResponse(
                    {"detail": "Too many requests. Please retry shortly."},
                    status_code=429,
                    headers={"Retry-After": str(int(self.limiter.window))},
                )
        return await call_next(request)
