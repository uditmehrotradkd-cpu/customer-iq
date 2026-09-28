# Production image for the Customer Segmentation website + API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    SEG_AUTO_TRAIN=false \
    SEG_ENABLE_HSTS=false \
    WEB_CONCURRENCY=2 \
    FORWARDED_ALLOW_IPS=127.0.0.1

WORKDIR /app

COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

COPY segmentation/ segmentation/
COPY web/ web/
COPY data/ data/
COPY run_pipeline.py .

# Train at build time so the image ships with a versioned, reproducible model artifact.
RUN python run_pipeline.py train && rm -rf artifacts/figures

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/api/v1/health' % os.environ.get('PORT', '8000'), timeout=3).status == 200 else 1)"

# Hosting platforms (Render, Railway, Cloud Run) inject PORT; default to 8000 locally.
CMD exec python -m uvicorn web.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
    --workers "$WEB_CONCURRENCY" --proxy-headers --forwarded-allow-ips "$FORWARDED_ALLOW_IPS"
