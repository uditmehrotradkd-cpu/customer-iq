"""Versioned REST API (``/api/v1``)."""
from __future__ import annotations

import io
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool

from .auth import current_user, workspace_key
from .file_io import SUPPORTED_EXTENSIONS
from .registry import ModelRegistry
from .report import XLSX_MEDIA_TYPE, scored_report
from .schemas import AssignmentResult, CustomerInput, GenericRecord, HealthResponse
from .services import SegmentationService
from .settings import Settings

router = APIRouter(prefix="/api/v1")


def get_registry(request: Request) -> ModelRegistry:
    """Signed-in users get their own workspace; anonymous callers share the demo model."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    user = current_user(request)
    return request.app.state.registries.for_user(workspace_key(user["username"])) if user else registry


def get_service(request: Request) -> SegmentationService:
    return get_registry(request).live()


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


Service = Annotated[SegmentationService, Depends(get_service)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]


def _csv_response(frame, filename: str) -> StreamingResponse:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request, service: Service, settings: AppSettings) -> dict:
    return {
        "status": "ok",
        "version": settings.version,
        "model_loaded": True,
        "n_segments": service.segmenter.n_segments,
        "trained_at_utc": service.trained_at_utc,
        "model_source": get_registry(request).status()["live_source"],
        "mode": service.mode,
        "labels": service.labels(),
    }


@router.get("/overview", tags=["segments"])
def overview(service: Service) -> dict:
    return service.overview()


@router.get("/pca", tags=["segments"])
def pca(service: Service) -> dict:
    return service.pca_points()


@router.get("/pca3d", tags=["segments"])
def pca3d(service: Service) -> dict:
    return service.pca3d()


@router.get("/segments", tags=["segments"])
def list_segments(service: Service) -> list[dict]:
    return service.segments_meta()


@router.get("/segments/{segment_id}", tags=["segments"])
def segment_detail(segment_id: int, service: Service) -> dict:
    try:
        return service.segment_detail(segment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Segment not found.") from None


@router.get("/fingerprint", tags=["segments"])
def fingerprint(service: Service) -> dict:
    return service.fingerprint()


@router.get("/explorer/features", tags=["explorer"])
def explorer_features(service: Service) -> dict:
    return service.explorer_features()


@router.get("/explorer/distribution", tags=["explorer"])
def distribution(service: Service, feature: str = Query(max_length=64)) -> dict:
    try:
        return service.distribution(feature)
    except KeyError:
        raise HTTPException(status_code=422, detail="Unknown feature.") from None


@router.get("/explorer/scatter", tags=["explorer"])
def scatter(
    service: Service,
    x: str = Query(max_length=64),
    y: str = Query(max_length=64),
    segments: list[int] | None = Query(default=None),
) -> dict:
    try:
        return service.scatter(x, y, segments)
    except KeyError:
        raise HTTPException(status_code=422, detail="Unknown feature.") from None


@router.get("/explorer/demographics", tags=["explorer"])
def demographics(service: Service, attribute: str = Query(max_length=64)) -> dict:
    try:
        return service.demographics(attribute)
    except KeyError:
        raise HTTPException(status_code=422, detail="Unknown attribute.") from None


@router.get("/diagnostics", tags=["model"])
def diagnostics(service: Service) -> dict:
    return service.diagnostics()


@router.get("/assign/defaults", tags=["scoring"])
def assign_defaults(service: Service) -> dict:
    return service.form_defaults()


@router.post("/assign", response_model=AssignmentResult, tags=["scoring"])
async def assign(customer: CustomerInput, service: Service) -> dict:
    if service.mode != "customer":
        raise HTTPException(status_code=409, detail="This workspace uses your own dataset; use /assign/generic.")
    return await run_in_threadpool(service.assign, customer.model_dump())


@router.post("/assign/generic", tags=["scoring"])
async def assign_generic(body: GenericRecord, service: Service) -> dict:
    if service.mode != "generic":
        raise HTTPException(status_code=409, detail="This workspace uses the customer model; use /assign.")
    return await run_in_threadpool(service.assign, body.values)


@router.post("/score/batch", tags=["scoring"])
async def score_batch(
    service: Service,
    settings: AppSettings,
    file: UploadFile = File(...),
    format: Literal["csv", "xlsx"] = Query(default="csv", description="csv = scored rows; xlsx = report with charts"),
):
    if not (file.filename or "").lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=415, detail="Please upload a CSV, Excel (.xlsx/.xls) or JSON file.")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb:g} MB.")
    try:
        scored = await run_in_threadpool(service.score_file, content, file.filename, settings.max_batch_rows)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not score file: {exc}") from None
    if format == "xlsx":
        content = await run_in_threadpool(scored_report, service, scored, file.filename or "uploaded file")
        return Response(content, media_type=XLSX_MEDIA_TYPE, headers={"Content-Disposition": 'attachment; filename="segment_report.xlsx"'})
    return _csv_response(scored, "scored_customers.csv")


@router.get("/customers/export", tags=["explorer"])
def export_customers(service: Service, segment: int | None = None) -> StreamingResponse:
    try:
        frame = service.export_customers(segment)
    except KeyError:
        raise HTTPException(status_code=404, detail="Segment not found.") from None
    suffix = f"_segment_{segment}" if segment is not None else ""
    return _csv_response(frame, f"customers{suffix}.csv")
