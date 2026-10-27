"""Холодний старт і RAM для локального запуску та для Docker.

Локально (venv): запускає uvicorn, чекає /health, міряє RSS процесу через psutil.
    python scripts/cold_start.py local

Docker: docker run → перший успішний /health, RAM з docker stats.
    python scripts/cold_start.py docker --image kws-api:1.0.0 --name kws
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def wait_health(url: str, timeout: float = 120.0) -> float:
    t_end = time.perf_counter() + timeout
    while time.perf_counter() < t_end:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return time.perf_counter()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
    raise TimeoutError(f"{url} не відповів за {timeout} с")


def run_local(port: int) -> dict:
    import psutil

    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app",
                             "--host", "127.0.0.1", "--port", str(port)])
    try:
        t1 = wait_health(f"http://127.0.0.1:{port}/health")
        time.sleep(1)
        # На Windows python.exe з venv — лише запускач, справжній процес є дочірнім.
        parent = psutil.Process(proc.pid)
        procs = [parent, *parent.children(recursive=True)]
        rss = sum(p.memory_info().rss for p in procs) / 2**20
        return {"env": "local", "cold_start_s": round(t1 - t0, 2), "ram_mb": round(rss, 1)}
    finally:
        proc.terminate()
        proc.wait()


def run_docker(image: str, name: str, port: int, keep: bool) -> dict:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    t0 = time.perf_counter()
    subprocess.run(["docker", "run", "-d", "--name", name, "-p", f"{port}:8000", image],
                   check=True, capture_output=True)
    t1 = wait_health(f"http://127.0.0.1:{port}/health")
    time.sleep(1)
    mem = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                         check=True, capture_output=True, text=True).stdout.strip()
    if not keep:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return {"env": f"docker:{image}", "cold_start_s": round(t1 - t0, 2), "ram": mem}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["local", "docker"])
    p.add_argument("--image", default="kws-api:1.0.0")
    p.add_argument("--name", default="kws-cold")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--runs", type=int, default=3, help="кількість повторів, береться медіана")
    p.add_argument("--keep", action="store_true", help="не видаляти контейнер після вимірювання")
    args = p.parse_args()

    results = []
    for _ in range(args.runs):
        if args.mode == "local":
            r = run_local(args.port)
        else:
            r = run_docker(args.image, args.name, args.port, args.keep)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
    times = sorted(r["cold_start_s"] for r in results)
    print(json.dumps({"median_cold_start_s": times[len(times) // 2]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
