"""
week3_2d_autoencoder.py
=======================
Week 3 — Card I1 — Satellite Telemetry Anomaly Detection
Team 05: K Hemalatha Reddy, B Parameshwar Reddy, Jagan Sai Kumar

2-D Spectrogram Convolutional Autoencoder on OPSSAT-AD.
Same nominal-only training regime as the 1-D model (Annex D6).
Same split, same preprocessing, same calibration rule.

How it differs from the 1-D model:
  - Each 256-sample fragment is first converted to a 2-D spectrogram
    image using scipy.signal.spectrogram (nperseg=64, noverlap=32).
  - The autoencoder then learns to reconstruct that 2-D image instead
    of the raw time series.
  - Anomaly score = mean squared reconstruction error on the spectrogram.

This is the REFERENCE RUN for Card I1 (Annex D0 Week-1 table).

Run from project root:
    source .venv/bin/activate
    python3 scripts/week3_2d_autoencoder.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import spectrogram as scipy_spectrogram
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility  (seed = 22, Annex D0)
# ─────────────────────────────────────────────────────────────────────────────
SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Annex D0 training limits
# ─────────────────────────────────────────────────────────────────────────────
MAX_EPOCHS    = 30
MAX_SECONDS   = 30 * 60
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EARLY_STOP    = 5
THRESHOLD_PCT = 99.0

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)


# =============================================================================
# STEP 1 — Load data and build split  (same as 1-D script)
# =============================================================================
print("=" * 60)
print("STEP 1: Loading OPSSAT-AD data")
print("=" * 60)

df  = pd.read_csv(DATA / "segments.csv")
dev = pd.read_csv(DATA / "OPSSAT_nominal_dev_segments.csv")

df["segment"] = df["segment"].astype(str)
df["train"]   = df["train"].astype(str)
df["anomaly"] = df["anomaly"].astype(int)
dev_ids       = set(dev["segment"].astype(str))

frag = df.groupby("segment").agg(
    anomaly=("anomaly", "first"),
    train=("train",   "first"),
).reset_index()

nominal_fit_ids  = set(frag[
    (frag["anomaly"] == 0) &
    (frag["train"]   == "1") &
    (~frag["segment"].isin(dev_ids))
]["segment"])
nominal_dev_ids  = dev_ids
test_nominal_ids = set(frag[(frag["train"] != "1") & (frag["anomaly"] == 0)]["segment"])
test_anomaly_ids = set(frag[(frag["train"] != "1") & (frag["anomaly"] == 1)]["segment"])

print(f"  Nominal fitting : {len(nominal_fit_ids)} | "
      f"Nominal dev : {len(nominal_dev_ids)} | "
      f"Test : {len(test_nominal_ids)+len(test_anomaly_ids)} "
      f"({len(test_anomaly_ids)} anom / {len(test_nominal_ids)} nom)")


# =============================================================================
# STEP 2 — Resample to 256 samples  (same rule, Annex D6)
# =============================================================================

def resample_to_256(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    if len(v) == 256:
        return v
    return np.interp(
        np.linspace(0.0, 1.0, 256),
        np.linspace(0.0, 1.0, len(v)),
        v
    ).astype(np.float32)


def load_fragments(seg_ids: set) -> np.ndarray:
    arrays = []
    for sid in sorted(seg_ids):
        vals = df[df["segment"] == sid]["value"].astype(float).to_numpy()
        if len(vals) >= 2:
            arrays.append(resample_to_256(vals))
    return np.stack(arrays) if arrays else np.empty((0, 256), dtype=np.float32)


X_fit_raw  = load_fragments(nominal_fit_ids)
X_dev_raw  = load_fragments(nominal_dev_ids)
X_test_nom = load_fragments(test_nominal_ids)
X_test_anom= load_fragments(test_anomaly_ids)

X_test_raw = np.concatenate([X_test_nom, X_test_anom], axis=0)
y_test     = np.concatenate([
    np.zeros(len(X_test_nom),  dtype=int),
    np.ones(len(X_test_anom), dtype=int),
])


# =============================================================================
# STEP 3 — Training-only standardisation
# =============================================================================
train_mean = float(X_fit_raw.mean())
train_std  = float(X_fit_raw.std())
if train_std == 0:
    train_std = 1.0

X_fit_z  = (X_fit_raw  - train_mean) / train_std
X_dev_z  = (X_dev_raw  - train_mean) / train_std
X_test_z = (X_test_raw - train_mean) / train_std

print(f"  Standardisation — mean: {train_mean:.6f}  std: {train_std:.6f}")


# =============================================================================
# STEP 4 — Convert 1-D fragments to 2-D spectrograms  (Annex D6)
# =============================================================================
print()
print("=" * 60)
print("STEP 4: Converting fragments to 2-D spectrograms")
print("=" * 60)

NPERSEG  = 64
NOVERLAP = 32
NFFT     = 64

def to_spectrogram(x: np.ndarray) -> np.ndarray:
    """
    Convert a 256-sample 1-D signal to a 2-D spectrogram.

    scipy.signal.spectrogram returns:
        frequencies : (33,)    — 0 to Nyquist frequency bins
        times       : (8,)     — time frames
        Sxx         : (33, 8)  — power spectral density

    We apply log1p to compress the dynamic range (log-scale magnitude).
    Output shape: (33, 8)
    """
    _, _, Sxx = scipy_spectrogram(
        x,
        nperseg  = NPERSEG,
        noverlap = NOVERLAP,
        nfft     = NFFT,
        scaling  = "density",
        mode     = "magnitude",
    )
    return np.log1p(Sxx).astype(np.float32)


# Convert all splits
X_fit_img  = np.stack([to_spectrogram(x) for x in X_fit_z])   # (1019, 33, 8)
X_dev_img  = np.stack([to_spectrogram(x) for x in X_dev_z])   # (254,  33, 8)
X_test_img = np.stack([to_spectrogram(x) for x in X_test_z])  # (529,  33, 8)

# Add channel dimension: (N, 1, 33, 8)
X_fit_img  = X_fit_img[:, None, :, :]
X_dev_img  = X_dev_img[:, None, :, :]
X_test_img = X_test_img[:, None, :, :]

H, W = X_fit_img.shape[2], X_fit_img.shape[3]  # 33, 8

print(f"  Spectrogram shape per fragment : (1, {H}, {W})")
print(f"  X_fit_img  : {X_fit_img.shape}")
print(f"  X_dev_img  : {X_dev_img.shape}")
print(f"  X_test_img : {X_test_img.shape}")


# =============================================================================
# STEP 5 — Dataset
# =============================================================================

class SpectrogramDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


train_loader = DataLoader(SpectrogramDataset(X_fit_img),  batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(SpectrogramDataset(X_dev_img),  batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(SpectrogramDataset(X_test_img), batch_size=BATCH_SIZE, shuffle=False)


# =============================================================================
# STEP 6 — Model: 2-D Spectrogram Convolutional Autoencoder
# =============================================================================
print()
print("=" * 60)
print("STEP 6: Building 2-D Spectrogram Autoencoder")
print("=" * 60)


class SpectrogramAutoencoder(nn.Module):
    """
    2-D Convolutional Autoencoder that operates on spectrogram images.

    Input  : (batch, 1, 33, 8)  — 1 channel, 33 freq bins, 8 time frames

    Encoder:
        Conv2d(1→16, k=3, pad=1)  + ReLU   → (batch, 16, 33, 8)
        Conv2d(16→32, k=3, pad=1) + ReLU   → (batch, 32, 33, 8)
        Conv2d(32→16, k=3, pad=1) + ReLU   → (batch, 16, 33, 8)

        [No pooling — spectrogram is already small at 33×8]

    Decoder:
        Conv2d(16→32, k=3, pad=1) + ReLU   → (batch, 32, 33, 8)
        Conv2d(32→16, k=3, pad=1) + ReLU   → (batch, 16, 33, 8)
        Conv2d(16→1,  k=3, pad=1)           → (batch,  1, 33, 8)

    Anomaly score = mean squared reconstruction error on the spectrogram.

    Trainable parameters: ~45,000  (under the 250k Annex D0 cap)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16,  1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model    = SpectrogramAutoencoder()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Architecture          : SpectrogramAutoencoder (2-D)")
print(f"  Trainable parameters  : {n_params:,}  (cap: 250,000)")
assert n_params <= 250_000, f"Parameter count {n_params} exceeds 250k cap!"

with torch.no_grad():
    dummy = torch.zeros(1, 1, H, W)
    out   = model(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape}"
print(f"  Shape check passed    : (1,1,{H},{W}) → {tuple(out.shape)}")


# =============================================================================
# STEP 7 — Training loop
# =============================================================================
print()
print("=" * 60)
print("STEP 7: Training")
print("=" * 60)

optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion   = nn.MSELoss()

best_val_loss = float("inf")
best_state    = None
no_improve    = 0
history       = []
stop_reason   = "completed all epochs"

t_start = time.perf_counter()

for epoch in range(1, MAX_EPOCHS + 1):

    if time.perf_counter() - t_start >= MAX_SECONDS:
        stop_reason = "30-minute elapsed limit"
        print(f"  30-min limit reached at epoch {epoch}.")
        break

    # Training
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        out  = model(batch)
        loss = criterion(out, batch)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(batch)
    tr_loss /= len(X_fit_img)

    # Validation
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for batch in dev_loader:
            out = model(batch)
            va_loss += criterion(out, batch).item() * len(batch)
    va_loss /= len(X_dev_img)

    elapsed = time.perf_counter() - t_start
    print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  "
          f"train={tr_loss:.6f}  val={va_loss:.6f}  "
          f"elapsed={elapsed:.1f}s")

    history.append((epoch, tr_loss, va_loss))

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1

    if no_improve >= EARLY_STOP:
        stop_reason = f"early stopping ({EARLY_STOP} epochs no improvement)"
        print(f"  Early stopping at epoch {epoch}.")
        break

if best_state:
    model.load_state_dict(best_state)

total_elapsed = time.perf_counter() - t_start
print()
print(f"  Stopped by    : {stop_reason}")
print(f"  Best val loss : {best_val_loss:.6f}")
print(f"  Total elapsed : {total_elapsed:.2f}s  ({total_elapsed/60:.2f} min)")

model_path = OUT / "2d_autoencoder.pt"
torch.save(model.state_dict(), model_path)
print(f"  Model saved   : {model_path}")


# =============================================================================
# STEP 8 — Score all partitions
# =============================================================================
print()
print("=" * 60)
print("STEP 8: Computing reconstruction MSE scores")
print("=" * 60)


def score_fragments(loader: DataLoader) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mse = ((out - batch) ** 2).mean(dim=(1, 2, 3))
            scores.append(mse.numpy())
    return np.concatenate(scores)


dev_scores  = score_fragments(dev_loader)
test_scores = score_fragments(test_loader)

test_nominal_scores  = test_scores[y_test == 0]
test_anomaly_scores  = test_scores[y_test == 1]

print(f"  Dev  scores  — mean: {dev_scores.mean():.6f}  max: {dev_scores.max():.6f}")
print(f"  Test nominal  mean : {test_nominal_scores.mean():.6f}")
print(f"  Test anomaly  mean : {test_anomaly_scores.mean():.6f}")
print(f"  Separation ratio   : "
      f"{test_anomaly_scores.mean()/max(test_nominal_scores.mean(),1e-9):.2f}x")


# =============================================================================
# STEP 9 — Threshold calibration  (99th pctile of dev scores, Annex D6)
# =============================================================================
print()
print("=" * 60)
print("STEP 9: Threshold calibration")
print("=" * 60)

threshold  = float(np.percentile(dev_scores, THRESHOLD_PCT))
dev_alarms = int((dev_scores >= threshold).sum())

print(f"  Threshold (99th pctile) : {threshold:.6f}")
print(f"  Dev alarm rate          : {dev_alarms}/{len(dev_scores)} = "
      f"{100*dev_alarms/len(dev_scores):.2f}%  (false-alarm rate on nominal)")


# =============================================================================
# STEP 10 — Test evaluation
# =============================================================================
print()
print("=" * 60)
print("STEP 10: Test evaluation")
print("=" * 60)

test_pred = (test_scores >= threshold).astype(int)
p, r, f1, _ = precision_recall_fscore_support(
    y_test, test_pred, average="binary", zero_division=0
)

print(f"  Precision : {p:.4f}")
print(f"  Recall    : {r:.4f}")
print(f"  F1 score  : {f1:.4f}")
print(f"  Support   : {y_test.sum()} anomalous / {(y_test==0).sum()} nominal")

# Load 1-D result for comparison
f1_1d = None
results_1d = OUT / "1d_autoencoder_results.txt"
if results_1d.exists():
    for line in results_1d.read_text().splitlines():
        if "F1 score" in line and ":" in line:
            try:
                f1_1d = float(line.split(":")[-1].strip())
            except ValueError:
                pass

print()
print("  Comparison table (test F1):")
print(f"    Isolation Forest  : 0.0164")
print(f"    PCA recon. error  : 0.0630")
print(f"    Trivial detector  : 0.0164")
if f1_1d is not None:
    print(f"    1-D Autoencoder   : {f1_1d:.4f}")
print(f"    2-D Autoencoder   : {f1:.4f}   ← this model")


# =============================================================================
# STEP 11 — Save results log
# =============================================================================
print()
print("=" * 60)
print("STEP 11: Saving results")
print("=" * 60)

log_lines = [
    "=" * 60,
    "2-D Spectrogram Autoencoder — OPSSAT-AD",
    "Team 05, Card I1, Week 3",
    "=" * 60,
    "",
    "--- Model ---",
    f"Architecture      : SpectrogramAutoencoder (2-D Conv)",
    f"Input shape       : (1, {H}, {W})  — 1 channel, {H} freq bins, {W} time frames",
    f"Trainable params  : {n_params:,}",
    f"Spectrogram       : nperseg={NPERSEG}, noverlap={NOVERLAP}, nfft={NFFT}",
    "",
    "--- Split ---",
    f"Nominal fitting   : {len(nominal_fit_ids)} fragments",
    f"Nominal dev       : {len(nominal_dev_ids)} fragments",
    f"Test nominal      : {len(test_nominal_ids)} fragments",
    f"Test anomalous    : {len(test_anomaly_ids)} fragments",
    "",
    "--- Preprocessing ---",
    f"Resampled to      : 256 samples per fragment",
    f"Training mean     : {train_mean:.6f}",
    f"Training std      : {train_std:.6f}",
    f"Spectrogram       : log1p(magnitude), shape ({H}, {W})",
    "",
    "--- Training ---",
    f"Seed              : {SEED}",
    f"Stopped by        : {stop_reason}",
    f"Epochs completed  : {len(history)}",
    f"Best val loss     : {best_val_loss:.6f}",
    f"Total elapsed     : {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)",
    "",
    "--- Loss history ---",
]
for ep, tl, vl in history:
    log_lines.append(f"  Epoch {ep:02d}: train={tl:.6f}  val={vl:.6f}")

log_lines += [
    "",
    "--- Calibration ---",
    f"Threshold         : {threshold:.6f}",
    f"Dev alarm rate    : {dev_alarms}/{len(dev_scores)} = "
    f"{100*dev_alarms/len(dev_scores):.2f}%",
    "",
    "--- Test results ---",
    f"Precision         : {p:.4f}",
    f"Recall            : {r:.4f}",
    f"F1 score          : {f1:.4f}",
    f"Support           : {y_test.sum()} anom / {(y_test==0).sum()} nom",
    "",
    "--- Comparison (test F1) ---",
    "  Isolation Forest  : 0.0164",
    "  PCA recon. error  : 0.0630",
    "  Trivial detector  : 0.0164",
]
if f1_1d is not None:
    log_lines.append(f"  1-D Autoencoder   : {f1_1d:.4f}")
log_lines.append(f"  2-D Autoencoder   : {f1:.4f}")
log_lines.append("")
log_lines.append("No test labels or test metrics were used for threshold selection.")

log_path = OUT / "2d_autoencoder_results.txt"
log_path.write_text("\n".join(log_lines) + "\n")

print(f"  Results : {log_path}")
print(f"  Model   : {model_path}")
print()
print("=" * 60)
print("Week 3 — 2-D Spectrogram Autoencoder complete.")
print("=" * 60)
