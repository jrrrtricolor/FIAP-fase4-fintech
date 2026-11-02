# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"

# A mesma lista instala o projeto localmente, nos notebooks e na API.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-compile --requirement /tmp/requirements.txt \
    && pip check

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app:/app/src:/app/ml_prep_kit/src" \

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:////app/data/processed/monitoring/mlflow.db}

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 50000 --gid 0 --home-dir /app app \
    && mkdir -p /app/data/processed \
    && chown -R app:root /app/data

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:root src ./src
COPY --chown=app:root ml_prep_kit ./ml_prep_kit

# Modelos, banco e relatórios são montados em runtime, sem congelá-los na imagem.
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "src.credit_monitoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
