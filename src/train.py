"""Навчання CNN на Google Speech Commands → artifacts/{model.pt, labels.json, metrics.json}.

    python -m src.train --classes yes no up down --epochs 10 --seed 42 --out artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from tqdm import tqdm

from src.data import N_MELS, SAMPLE_RATE, FeatureExtractor, make_loader
from src.evaluate import evaluate
from src.model import KWSCNN, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def predict_all(model, loader, fe) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for wav, y in loader:
        y_pred += model(fe(wav)).argmax(dim=1).tolist()
        y_true += y.tolist()
    return y_true, y_pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    p.add_argument("--out", default="artifacts")
    p.add_argument("--model-version", default=os.getenv("MODEL_VERSION", "1.0.0"))
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out, exist_ok=True)

    kw = dict(root=args.data_dir, classes=args.classes, batch_size=args.batch_size,
              seed=args.seed, num_workers=args.num_workers)
    train_dl = make_loader(subset="training", shuffle=True, **kw)
    val_dl = make_loader(subset="validation", shuffle=False, **kw)
    test_dl = make_loader(subset="testing", shuffle=False, **kw)
    print(f"train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    fe = FeatureExtractor()
    model = KWSCNN(n_classes=len(args.classes), width=args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for wav, y in tqdm(train_dl, desc=f"epoch {epoch}", mininterval=10, leave=False):
            x = fe(wav)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        val_acc = evaluate(model, val_dl, fe)
        history.append({"epoch": epoch, "loss": round(running / n, 4), "val_acc": round(val_acc, 2)})
        print(f"epoch {epoch:02d}  loss={running / n:.4f}  val_acc={val_acc:.2f}%", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_dl, fe)
    y_true, y_pred = predict_all(model, test_dl, fe)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=args.classes, output_dict=True)
    print(f"test_acc={test_acc:.2f}%")

    config = {"classes": args.classes, "width": args.width, "n_mels": N_MELS,
              "sample_rate": SAMPLE_RATE, "model_version": args.model_version}
    model_path = os.path.join(args.out, "model.pt")
    torch.save({"state_dict": model.state_dict(), "config": config}, model_path)

    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(args.classes, f, ensure_ascii=False)

    metrics = {
        "accuracy": round(test_acc, 2),          # % на тестовому наборі
        "val_accuracy": round(best_val, 2),
        "n_train": len(train_dl.dataset), "n_test": len(test_dl.dataset),
        "epochs": args.epochs, "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
        "num_params": count_params(model),
        "model_size_kb": round(os.path.getsize(model_path) / 1024, 1),
        "train_time_s": round(train_time, 1),
        "model_version": args.model_version,
        "torch_version": torch.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "confusion_matrix": cm,
        "per_class_f1": {c: round(report[c]["f1-score"], 4) for c in args.classes},
        "history": history,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    short = {k: v for k, v in metrics.items() if k not in ("history", "confusion_matrix")}
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
