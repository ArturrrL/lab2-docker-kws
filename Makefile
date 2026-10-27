IMAGE ?= kws-api
TAG   ?= 1.0.0
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: venv train samples serve build build-single run stop bench-local bench-docker \
        cold-local cold-docker history sizes eval-docker lint

venv:
	python3.10 -m venv .venv && . .venv/bin/activate && \
	pip install -r requirements-train.txt -r requirements-api.txt -r requirements-dev.txt

train:
	python -m src.train --classes yes no up down --epochs 10 --seed 42 --data-dir data --out artifacts

samples:
	python scripts/make_samples.py --data-dir data

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

build:
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE):$(TAG) \
	  --build-arg GIT_COMMIT=$(GIT_COMMIT) --build-arg BUILD_DATE=$(BUILD_DATE) .

build-single:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.single -t $(IMAGE):$(TAG)-single .

run:
	docker run -d --name kws -p 8000:8000 $(IMAGE):$(TAG)

stop:
	docker rm -f kws || true

bench-local:
	python scripts/benchmark.py -n 200 --label local --out results/bench_local.json

bench-docker:
	python scripts/benchmark.py -n 200 --label docker-multi --out results/bench_docker_multi.json

cold-local:
	python scripts/cold_start.py local

cold-docker:
	python scripts/cold_start.py docker --image $(IMAGE):$(TAG)

eval-docker:
	docker exec kws python -m src.evaluate --data-dir /tmp/data

history:
	docker history $(IMAGE):$(TAG) --format "table {{.Size}}\t{{.CreatedBy}}" --no-trunc | cut -c1-150
	docker history $(IMAGE):$(TAG)-single --format "table {{.Size}}\t{{.CreatedBy}}" --no-trunc | cut -c1-150

sizes:
	docker images $(IMAGE)
	du -sh .venv artifacts/model.pt

lint:
	ruff check src/ scripts/
