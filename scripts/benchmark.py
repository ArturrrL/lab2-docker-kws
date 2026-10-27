"""Вимірювання latency /predict: серверна (latency_ms з відповіді) і клієнтська (end-to-end).

    python scripts/benchmark.py --url http://localhost:8000 --file samples/yes.wav -n 200 \
        --out results/bench_docker.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import requests


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary(values: list[float]) -> dict:
    return {"mean": round(statistics.mean(values), 3), "p50": round(pct(values, 50), 3),
            "p95": round(pct(values, 95), 3), "min": round(min(values), 3), "max": round(max(values), 3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--file", default="samples/yes.wav")
    p.add_argument("-n", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--label", default="", help="мітка середовища для звіту (local/docker-multi/...)")
    p.add_argument("--out", default="")
    args = p.parse_args()

    with open(args.file, "rb") as f:
        payload = f.read()
    session = requests.Session()  # keep-alive, щоб не міряти TCP-handshake на кожен запит
    url = args.url.rstrip("/") + "/predict"

    def call() -> tuple[float, float]:
        t0 = time.perf_counter()
        r = session.post(url, files={"file": (os.path.basename(args.file), payload, "audio/wav")})
        client_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return r.json()["latency_ms"], client_ms

    for _ in range(args.warmup):
        call()
    server, client = [], []
    for _ in range(args.n):
        s, c = call()
        server.append(s)
        client.append(c)

    result = {"label": args.label, "n": args.n, "warmup": args.warmup, "file": args.file,
              "server_ms": summary(server), "client_ms": summary(client)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
