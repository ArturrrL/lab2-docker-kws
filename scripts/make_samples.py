"""Копіює кілька .wav із тестового набору в samples/ (yes.wav, no.wav, ...) + bad.txt для тесту 400.

    python scripts/make_samples.py --data-dir data --classes yes no up down
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from torchaudio.datasets import SPEECHCOMMANDS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--classes", nargs="+", default=["yes", "no", "up", "down"])
    p.add_argument("--out", default="samples")
    p.add_argument("--per-class", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = SPEECHCOMMANDS(args.data_dir, download=True, subset="testing")
    for cls in args.classes:
        files = [w for w in ds._walker if Path(w).parent.name == cls][: args.per_class]
        for i, src in enumerate(files):
            name = f"{cls}.wav" if i == 0 else f"{cls}_{i}.wav"
            shutil.copy(src, os.path.join(args.out, name))
            print(f"{src} → {args.out}/{name}")
    with open(os.path.join(args.out, "bad.txt"), "w") as f:
        f.write("це не аудіо\n")


if __name__ == "__main__":
    main()
