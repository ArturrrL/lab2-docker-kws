# syntax=docker/dockerfile:1
# Multi-stage: навчання в етапі trainer, у runtime потрапляють лише код інференсу і model.pt.

# ================= Stage 1: trainer =================
FROM python:3.10-slim AS trainer
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Залежності окремим шаром — кешуються, поки не зміниться requirements-train.txt
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Датасет окремим шаром ДО копіювання коду: зміна train.py/model.py не перезавантажує 2.3 ГБ
ARG DATASET=speech_commands_v0.02
RUN mkdir -p /build/data && python -c "import torchaudio; \
torchaudio.datasets.SPEECHCOMMANDS('/build/data', url='${DATASET}', download=True)" \
    && rm -f /build/data/*.tar.gz

COPY src/ ./src/

ARG EPOCHS=10
ARG SEED=42
ARG MODEL_VERSION=1.0.0
RUN python -m src.train --classes yes no up down --epochs ${EPOCHS} --seed ${SEED} \
        --data-dir /build/data --out /build/artifacts --model-version ${MODEL_VERSION} \
        --num-workers 0

# ================= Stage 2: runtime =================
FROM python:3.10-slim AS runtime
WORKDIR /app

ARG MODEL_VERSION=1.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="kws-api" \
      org.opencontainers.image.description="Keyword spotting (Speech Commands) inference API" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY --from=trainer /build/artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    LOG_LEVEL=INFO \
    MODEL_VERSION=${MODEL_VERSION} \
    GIT_COMMIT=${GIT_COMMIT} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# exec — щоб uvicorn став PID 1 і коректно отримував SIGTERM від docker stop
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --log-level $(echo ${LOG_LEVEL} | tr A-Z a-z)"]
