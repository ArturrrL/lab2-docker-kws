"""FastAPI-сервіс інференсу аудіокласифікатора.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Конфігурація через змінні середовища:
    MODEL_PATH     шлях до model.pt           (artifacts/model.pt)
    METRICS_PATH   шлях до metrics.json       (поруч із моделлю)
    MODEL_VERSION  перевизначити версію моделі
    LOG_LEVEL      DEBUG / INFO / WARNING     (INFO)
    TORCH_THREADS  кількість потоків torch    (за замовчуванням torch)
    GIT_COMMIT     хеш коміту, з якого зібрано образ
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data import FeatureExtractor, prepare_waveform
from src.model import count_params, load_checkpoint

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
METRICS_PATH = os.getenv("METRICS_PATH", os.path.join(os.path.dirname(MODEL_PATH), "metrics.json"))
MAX_FILE_BYTES = 2 * 1024 * 1024   # 2 МБ ≈ 60 с аудіо 16 кГц/16 біт — з запасом
MAX_DURATION_S = 10.0

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kws-api")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель завантажується ОДИН раз під час старту.
    if os.getenv("TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
    t0 = time.perf_counter()
    model, cfg = load_checkpoint(MODEL_PATH)
    fe = FeatureExtractor().eval()
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    # Розігрів: перший виклик повільніший (виділення пам'яті, ініціалізація ядер).
    with torch.inference_mode():
        model(fe(torch.zeros(1, 1, 16000)))
    state.update(model=model, fe=fe, cfg=cfg, metrics=metrics,
                 version=os.getenv("MODEL_VERSION", cfg.get("model_version", "unknown")))
    log.info("model loaded from %s in %.1f ms, classes=%s",
             MODEL_PATH, (time.perf_counter() - t0) * 1000, cfg["classes"])
    yield
    state.clear()


app = FastAPI(title="KWS API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    m, cfg, metrics = state["model"], state["cfg"], state["metrics"]
    return {
        "architecture": "KWSCNN (4×[Conv3x3-BN-ReLU-MaxPool] → GAP → Linear)",
        "input": f"log-Mel {cfg.get('n_mels', 64)}×101, 16 kHz, 1 s",
        "classes": cfg["classes"],
        "num_params": count_params(m),
        "model_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 1),
        "accuracy": metrics.get("accuracy"),          # % на тестовому наборі
        "model_version": state["version"],
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "torch_version": torch.__version__,
        "trained_at": metrics.get("created_at"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Порожній файл")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail=f"Файл більший за {MAX_FILE_BYTES // 1024} КБ")

    t0 = time.perf_counter()
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)  # [T, C]
    except Exception as e:  # noqa: BLE001 — будь-яка помилка декодування = поганий вхід
        raise HTTPException(status_code=400, detail=f"Не вдалося прочитати аудіо (WAV): {e}") from e
    if audio.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Аудіо не містить відліків")
    if audio.shape[0] / sr > MAX_DURATION_S:
        raise HTTPException(status_code=400, detail=f"Аудіо довше за {MAX_DURATION_S:.0f} с")

    wav = prepare_waveform(torch.from_numpy(audio.T.copy()), sr).unsqueeze(0)  # [1, 1, 16000]
    with torch.inference_mode():
        probs = torch.softmax(state["model"](state["fe"](wav)), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    classes = state["cfg"]["classes"]
    idx = int(probs.argmax())
    log.debug("predict %s → %s (%.3f) in %.2f ms", file.filename, classes[idx], probs[idx], latency_ms)
    return {
        "label": classes[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs, strict=True)},
        "latency_ms": round(latency_ms, 3),
        "model_version": state["version"],
    }
