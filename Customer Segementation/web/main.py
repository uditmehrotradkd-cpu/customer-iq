"""FastAPI application: REST API + single-page website.

Run locally:  python -m uvicorn web.main:app --reload
Production:   python -m uvicorn web.main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

from .agent_api import router as agent_router  # noqa: E402
from .api import router  # noqa: E402
from .auth import check_same_origin, open_user_store, read_sqlite_users, workspace_key  # noqa: E402
from .auth import router as auth_router  # noqa: E402
from .llm import client_from_settings  # noqa: E402
from .registry import ModelRegistry, RegistryPool, migrate_workspaces  # noqa: E402
from .security import (  # noqa: E402
    BodySizeLimitMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from .services import SegmentationService  # noqa: E402
from .settings import Settings, get_settings  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("web")


def create_app(settings: Settings | None = None, service: SegmentationService | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        base = service or SegmentationService.load(settings.artifacts_dir, settings.auto_train)
        app.state.registry = ModelRegistry(base, settings.model_store_dir)
        app.state.registries = RegistryPool(app.state.registry)
        legacy = settings.model_store_dir / "users.db"
        if legacy.exists():
            migrate_workspaces(settings.model_store_dir / "users", read_sqlite_users(legacy), workspace_key)
        app.state.users = open_user_store(settings)
        log.info("Accounts stored in %s", "the cloud database (PostgreSQL)" if app.state.users.backend == "postgres" else legacy)
        app.state.train_limiter = RateLimiter(settings.train_limit_per_10min, window_seconds=600)
        app.state.login_limiter = RateLimiter(settings.login_limit_per_10min, window_seconds=600)
        app.state.ai = client_from_settings(settings)
        app.state.ai_limiter = RateLimiter(settings.llm_limit_per_10min, window_seconds=600)
        log.info("Model loaded: %d segments (live source: %s)", base.segmenter.n_segments, app.state.registry.status()["live_source"])
        yield

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.version,
        description="Behavioural customer segmentation: profiles, recommendations and real-time assignment.",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )

    # Starlette runs the last-added middleware first.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.enable_hsts)
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(settings.rate_limit_per_minute))
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + 64 * 1024)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "%s %s %s %.1fms id=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
            request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Internal server error."}, status_code=500)

    app.include_router(router, dependencies=[Depends(check_same_origin)])
    app.include_router(agent_router, dependencies=[Depends(check_same_origin)])
    app.include_router(auth_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    return app


app = create_app()
