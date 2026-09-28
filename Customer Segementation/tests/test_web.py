"""API and website tests. Run from the project folder: pytest -q"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.config import ARTIFACTS_DIR, DEFAULT_DATA_PATH, MODEL_FILENAME  # noqa: E402
from web.main import create_app  # noqa: E402
from web.security import RateLimiter  # noqa: E402
from web.services import SegmentationService, sanitize_csv  # noqa: E402
from web.settings import Settings  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / MODEL_FILENAME).exists(), reason="Run `python run_pipeline.py train` first"
)

CUSTOMER = {
    "Year_Birth": 1985, "Income": 42000, "Kidhome": 1, "Teenhome": 0, "Recency": 12,
    "MntWines": 120, "MntFruits": 10, "MntMeatProducts": 60, "MntFishProducts": 10,
    "MntSweetProducts": 5, "MntGoldProds": 30, "NumDealsPurchases": 5, "NumWebPurchases": 6,
    "NumCatalogPurchases": 1, "NumStorePurchases": 4, "NumWebVisitsMonth": 8,
    "AcceptedCmp3": True, "Education": "Graduation", "Marital_Status": "Married", "Dt_Customer": "2013-03-15",
}


@pytest.fixture(scope="module")
def service() -> SegmentationService:
    return SegmentationService.load(ARTIFACTS_DIR, auto_train=False)


@pytest.fixture(scope="module")
def client(service):
    app = create_app(Settings(rate_limit_per_minute=1000, allowed_hosts=["testserver"]), service=service)
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["n_segments"] >= 3


def test_index_served_with_security_headers(client):
    res = client.get("/")
    assert res.status_code == 200 and "Customer Segmentation" in res.text
    assert "default-src 'self'" in res.headers["content-security-policy"]
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/static/vendor/chart.umd.min.js").status_code == 200
    assert "media-src 'self'" in res.headers["content-security-policy"]
    for asset in ("/static/vendor/three.module.min.js", "/static/media/hero-data.mp4", "/static/media/persona-premium.jpg", "/static/media/agent-particles.mp4", "/static/js/universe.js", "/static/js/towers.js", "/static/js/theme-init.js", "/static/js/theme.js"):
        assert client.get(asset).status_code == 200, asset


def test_pca3d_payload(client):
    body = client.get("/api/v1/pca3d").json()
    assert len(body["positions"]) == 3 * len(body["segments"])
    assert len(body["centers"]) == len(set(body["segments"]))


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/overview",
        "/api/v1/pca",
        "/api/v1/pca3d",
        "/api/v1/segments",
        "/api/v1/segments/0",
        "/api/v1/fingerprint",
        "/api/v1/explorer/features",
        "/api/v1/explorer/distribution?feature=Total_Spend",
        "/api/v1/explorer/scatter?x=Income&y=Total_Spend&segments=0&segments=1",
        "/api/v1/explorer/demographics?attribute=Age_Band",
        "/api/v1/diagnostics",
        "/api/v1/assign/defaults",
    ],
)
def test_read_endpoints(client, path):
    res = client.get(path)
    assert res.status_code == 200, res.text
    assert res.headers["cache-control"] == "no-store"


def test_unknown_inputs_are_rejected(client):
    assert client.get("/api/v1/segments/99").status_code == 404
    assert client.get("/api/v1/explorer/distribution?feature=__class__").status_code == 422
    assert client.get("/api/v1/explorer/demographics?attribute=ID").status_code == 422


def test_assign_customer(client):
    res = client.post("/api/v1/assign", json=CUSTOMER)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["segment_name"] and body["recommendation"]["actions"]
    assert 0 <= body["assignment_margin"] <= 1
    assert body["fit"] in {"clear", "moderate", "borderline"}


def test_assign_uses_defaults_payload(client):
    defaults = client.get("/api/v1/assign/defaults").json()["defaults"]
    assert client.post("/api/v1/assign", json=defaults).status_code == 200


def test_assign_validation(client):
    bad = {**CUSTOMER, "Year_Birth": 1800}
    assert client.post("/api/v1/assign", json=bad).status_code == 422
    assert client.post("/api/v1/assign", json={**CUSTOMER, "is_admin": True}).status_code == 422
    missing_income = {**CUSTOMER, "Income": None}
    body = client.post("/api/v1/assign", json=missing_income).json()
    assert any("Income" in w for w in body["warnings"])


def test_batch_scoring(client):
    sample = pd.read_csv(DEFAULT_DATA_PATH, nrows=20).to_csv(index=False).encode()
    res = client.post("/api/v1/score/batch", files={"file": ("customers.csv", sample, "text/csv")})
    assert res.status_code == 200, res.text
    scored = pd.read_csv(io.BytesIO(res.content))
    assert len(scored) == 20 and {"Segment", "Segment_Name", "Assignment_Margin"} <= set(scored.columns)


def test_batch_rejects_bad_files(client):
    assert client.post("/api/v1/score/batch", files={"file": ("x.pdf", b"a,b", "application/pdf")}).status_code == 415
    assert client.post("/api/v1/score/batch", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}).status_code == 422


def test_export_customers(client):
    res = client.get("/api/v1/customers/export?segment=1")
    assert res.status_code == 200
    assert "attachment" in res.headers["content-disposition"]
    assert set(pd.read_csv(io.BytesIO(res.content))["Segment"]) == {1}


def test_untrusted_host_rejected(client):
    assert client.get("/api/v1/health", headers={"host": "evil.example"}).status_code == 400


def test_rate_limiter():
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")


def test_rate_limit_middleware(service):
    app = create_app(Settings(rate_limit_per_minute=1, allowed_hosts=["testserver"]), service=service)
    with TestClient(app) as c:
        assert c.post("/api/v1/assign", json=CUSTOMER).status_code == 200
        assert c.post("/api/v1/assign", json=CUSTOMER).status_code == 429


def test_csv_formula_injection_is_neutralised():
    df = pd.DataFrame({"name": ["=HYPERLINK(\"http://x\")", "safe", "+1", "@cmd"], "n": [1, 2, 3, 4]})
    out = sanitize_csv(df)
    assert out["name"].tolist() == ["'=HYPERLINK(\"http://x\")", "safe", "'+1", "'@cmd"]
