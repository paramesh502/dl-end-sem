import time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import spectrogram

# Real stride-1 timing: scores SMAP P-1 / MSL C-1 test streams with the
# TRAINED week-1 model (model_week1.pt, written by train.py).
# Run from the week1 folder:  python time_stride.py

torch.set_num_threads(6)          # same as train.py

L, SUBSET, BATCH = 256, 1000, 256
STREAMS = {"SMAP P-1": "../SMAP_MSL_test_P-1.npy",
           "MSL C-1": "../SMAP_MSL_test_C-1.npy"}


class SpectrogramAutoencoder(nn.Module):      # identical to train.py
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def to_spectrogram_batch(x):                  # same settings as train.py, batched
    _, _, S = spectrogram(x, nperseg=64, noverlap=32, nfft=64,
                          scaling="density", mode="magnitude", axis=-1)
    return np.log1p(S).astype(np.float32)     # (n, 33, 7)


def make_score_fn():
    ck = torch.load("model_week1.pt", map_location="cpu")
    model = SpectrogramAutoencoder()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean, std = ck["train_mean"], ck["train_std"]

    def score(x):                             # x: (n, 256) float32 windows
        x = (x - mean) / std
        img = torch.from_numpy(to_spectrogram_batch(x))[:, None]
        with torch.no_grad():
            rec = model(img)
            return ((rec - img) ** 2).mean(dim=(1, 2, 3)).numpy()
    return score


def windows(x, start, n):
    idx = np.arange(start, start + n)[:, None] + np.arange(L)[None, :]
    return x[idx]


score, total_s = make_score_fn(), 0.0
score(windows(np.zeros(L + BATCH, np.float32), 0, BATCH))   # untimed warm-up

for name, f in STREAMS.items():
    x = np.load(f)[:, 0].astype(np.float32)
    n_all = len(x) - L + 1
    n = min(SUBSET, n_all)
    t0 = time.perf_counter()
    for s in range(0, n, BATCH):
        score(windows(x, s, min(BATCH, n - s)))
    dt = time.perf_counter() - t0
    proj = dt / n * n_all
    total_s += proj
    print(f"{name}: {n} windows in {dt:.3f}s ({dt/n*1000:.3f}s per 1,000); "
          f"full stream {n_all} windows -> projected {proj:.1f}s")
print(f"Projected total: {total_s:.1f}s = {total_s/60:.2f} min")
