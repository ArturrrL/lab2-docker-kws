"""Accuracy моделі на тестовому наборі Speech Commands.

Приклади:
    python -m src.evaluate --model artifacts/model.pt --data-dir data
    docker exec kws python -m src.evaluate --data-dir /tmp/data
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from src.data import FeatureExtractor, make_loader
from src.model import load_checkpoint


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, fe: FeatureExtractor) -> float:
    """Повертає accuracy у відсотках."""
    model.eval()
    correct = total = 0
    for wav, y in loader:
        pred = model(fe(wav)).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("MODEL_PATH", "artifacts/model.pt"))
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    model, cfg = load_checkpoint(args.model)
    loader = make_loader(args.data_dir, "testing", cfg["classes"], args.batch_size,
                         shuffle=False, seed=0)
    t0 = time.perf_counter()
    acc = evaluate(model, loader, FeatureExtractor())
    print(json.dumps({
        "test_accuracy": round(acc, 2),
        "n_test": len(loader.dataset),
        "model_size_kb": round(os.path.getsize(args.model) / 1024, 1),
        "eval_time_s": round(time.perf_counter() - t0, 1),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
