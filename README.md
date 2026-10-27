# ЛР №2 — Контейнеризація ML-моделі (KWS на Google Speech Commands)

Аудіокласифікатор ключових слів `yes / no / up / down` (CNN на log-Mel спектрограмах),
упакований у Docker з багатоступеневою збіркою, FastAPI-сервісом і CI у GitHub Actions.

## Структура

```
lab2/
├── src/
│   ├── data.py        # Speech Commands, препроцесинг (Mel 64×101), спільний для train і api
│   ├── model.py       # KWSCNN: 4×[Conv-BN-ReLU-MaxPool] → GAP → Linear
│   ├── train.py       # навчання + оцінка → artifacts/{model.pt, labels.json, metrics.json}
│   ├── evaluate.py    # accuracy на тестовому наборі
│   └── api.py         # FastAPI: /health, /model-info, /predict
├── scripts/
│   ├── benchmark.py   # latency сервер/клієнт: mean, p50, p95
│   ├── cold_start.py  # холодний старт + RAM (локально і в Docker)
│   └── make_samples.py# samples/*.wav з тестового набору + bad.txt
├── samples/           # yes.wav, no.wav, up.wav, down.wav, bad.txt (закомітити!)
├── .github/workflows/ci.yml
├── Dockerfile         # multi-stage (trainer → runtime)
├── Dockerfile.single  # single-stage — для порівняння розмірів
├── requirements-{train,api,dev}.txt
├── .dockerignore  Makefile  pyproject.toml  REPORT.md
```

## Покроково

Потрібні Python **3.10** (torch 2.3.1 не має збірок під 3.13) і Docker.

```bash
# 1. Локальне середовище, навчання, семпли
make venv && source .venv/bin/activate
make train            # завантажить датасет у data/ (~2.3 ГБ), ~5–15 хв на CPU
make samples          # samples/yes.wav ... + bad.txt → закомітьте їх
python -m src.evaluate --data-dir data

# 2. Локальний сервіс і вимірювання
make serve            # в окремому терміналі
make bench-local
make cold-local       # (зупиніть make serve перед цим, або змініть порт)
du -sh .venv

# 3. Docker
make build            # multi-stage → kws-api:1.0.0
make build-single     # single-stage → kws-api:1.0.0-single
docker images kws-api
make run && docker ps # чекаємо статус (healthy)
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST -F "file=@samples/yes.wav" http://localhost:8000/predict
curl -X POST -F "file=@samples/bad.txt" http://localhost:8000/predict   # → 400
# Swagger UI: http://localhost:8000/docs

make bench-docker
docker stats --no-stream kws
make eval-docker      # accuracy всередині контейнера (датасет качається в /tmp)
make cold-docker      # використовує порт 8001, kws може працювати паралельно
make history          # docker history для обох образів
```

## Змінні середовища

| Змінна | За замовчуванням | Призначення |
|---|---|---|
| `MODEL_PATH` | `/app/artifacts/model.pt` | шлях до моделі (можна змонтувати інший через `-v`) |
| `PORT` | `8000` | порт uvicorn і HEALTHCHECK |
| `LOG_LEVEL` | `INFO` | рівень логування |
| `MODEL_VERSION` | з `model.pt` | версія у відповідях API |
| `TORCH_THREADS` | авто | потоки PyTorch (для експериментів із `--cpus`) |

Build-args: `EPOCHS` (10; у CI — 3), `SEED` (42), `MODEL_VERSION`, `GIT_COMMIT`, `BUILD_DATE`, `DATASET`.

## CI

`.github/workflows/ci.yml`: lint (ruff) → build (з кешем шарів GHA) → run + чекання `/health` →
smoke-тест `/model-info`, `/predict` на всіх `samples/*.wav`, перевірка, що `bad.txt` дає 400 →
accuracy gate (поріг 80 %). Щоб отримати запуск, що впав, для звіту: Actions → ci → *Run workflow*
з `acc_threshold = 99`.
