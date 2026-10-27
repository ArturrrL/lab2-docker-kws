"""Завантаження Google Speech Commands і препроцесинг (Mel-спектрограми).

Препроцесинг винесено в окремий модуль, щоб train.py, evaluate.py і api.py
використовували ОДНАКОВЕ перетворення аудіо → ознаки.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16_000
NUM_SAMPLES = SAMPLE_RATE  # 1 секунда
N_MELS = 64
N_FFT = 400       # 25 мс
HOP_LENGTH = 160  # 10 мс  → 101 кадр на 1 с


class FeatureExtractor(torch.nn.Module):
    """waveform [B, 1, 16000] → log-Mel [B, 1, 64, 101], нормалізована per-sample."""

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.to_db(self.mel(wav))
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (x - mean) / std


def fix_length(wav: torch.Tensor, n: int = NUM_SAMPLES) -> torch.Tensor:
    """Обрізає або доповнює нулями до n відліків. wav: [C, T]."""
    t = wav.shape[-1]
    if t > n:
        return wav[..., :n]
    if t < n:
        return F.pad(wav, (0, n - t))
    return wav


def prepare_waveform(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Довільний сигнал [C, T] або [T] → моно 16 кГц, 1 с, форма [1, 16000]."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return fix_length(wav)


class SpeechCommandsSubset(Dataset):
    """Підмножина Speech Commands тільки з потрібними класами."""

    def __init__(self, root: str, subset: str, classes: list[str],
                 url: str = "speech_commands_v0.02", download: bool = True) -> None:
        os.makedirs(root, exist_ok=True)
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root, url=url, download=download, subset=subset)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # Фільтруємо список файлів за назвою теки (= мітка), не читаючи аудіо.
        self.ds._walker = [p for p in self.ds._walker if Path(p).parent.name in self.class_to_idx]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        wav, sr, label, *_ = self.ds[i]
        return prepare_waveform(wav, sr), self.class_to_idx[label]


def make_loader(root: str, subset: str, classes: list[str], batch_size: int,
                shuffle: bool, seed: int, num_workers: int = 2) -> DataLoader:
    ds = SpeechCommandsSubset(root, subset, classes)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers, pin_memory=False)
