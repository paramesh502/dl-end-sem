"""
week4_fusion.py
===============
Week 4 — Card I1 — Satellite Telemetry Anomaly Detection
Team 05: K Hemalatha Reddy, B Parameshwar Reddy, Jagan Sai Kumar

Score-level fusion of 1-D and 2-D autoencoders (Annex D6 / Card I1).

Procedure:
  1. Load saved 1-D and 2-D autoencoder weights.
  2. Run inference on dev and test sets for both models → raw MSE scores.
  3. Standardise each model's scores using median/IQR fitted on dev scores only.
  4. Fuse: fused_score = 0.5 * std_1d + 0.5 * std_2d  (Annex D0 fixed weight).
  5. Threshold at 99th percentile of fused DEV scores (never test).
  6. Evaluate on test set.

Run from project root:
    source .venv/bin/activate
    python3 scripts/week4_fusion.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import spectrogram as scipy_spectrogram
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility  (Annex D0: seed = 22)
# ─────────────────────────────────────────────────────────────────────────────
SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram parameters (must match week3_2d_autoencoder.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
NPERSEG  = 64
NOVERLAP = 32
NFFT     = 64

THRESHOLD_PCT = 99.0
BATCH_SIZE    = 32


# =============================================================================
# MODEL DEFINITIONS
# (Copied verbatim from the original scripts to keep this file self-contained.)
# =============================================================================

class Conv1dAutoencoder(nn.Module):
    """
    1-D Convolutional Autoencoder (same architecture as week3_1d_autoencoder.py).

    Input  : (batch, 1, 256)

    Encoder:
        Conv1d(1→16, k=3, pad=1) + ReLU
        MaxPool1d(2)                         256→128
        Conv1d(16→32, k=3, pad=1) + ReLU
        MaxPool1d(2)                         128→64
        Conv1d(32→16, k=3, pad=1) + ReLU
                                    bottleneck: (batch, 16, 64)

    Decoder:
        Conv1d(16→32, k=3, pad=1) + ReLU
        Upsample(×2)                         64→128
        Conv1d(32→16, k=3, pad=1) + ReLU
        Upsample(×2)                         128→256
        Conv1d(16→1,  k=3, pad=1)

    Output : (batch, 1, 256)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1,  out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SpectrogramAutoencoder(nn.Module):
    """
    2-D Spectrogram Autoencoder (same architecture as week3_2d_autoencoder.py).

    Input  : (batch, 1, 33, 8)

    Encoder:
        Conv2d(1→16,  k=3, pad=1) + ReLU
        Conv2d(16→32, k=3, pad=1) + ReLU
        Conv2d(32→16, k=3, pad=1) + ReLU

    Decoder:
        Conv2d(16→32, k=3, pad=1) + ReLU
        Conv2d(32→16, k=3, pad=1) + ReLU
        Conv2d(16→1,  k=3, pad=1)

    Output : (batch, 1, 33, 8)
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


# =============================================================================
# STEP 1 — Load data and build split  (same as original scripts)
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

print(f"  Nominal fitting : {len(nominal_fit_ids)}")
print(f"  Nominal dev     : {len(nominal_dev_ids)}")
print(f"  Test nominal    : {len(test_nominal_ids)}")
print(f"  Test anomalous  : {len(test_anomaly_ids)}")


# =============================================================================
# STEP 2 — Resample to 256 samples  (same rule, Annex D6)
# =============================================================================
print()
print("=" * 60)
print("STEP 2: Resampling fragments to 256 samples")
print("=" * 60)


def resample_to_256(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    if len(v) == 256:
        return v
    return np.interp(
        np.linspace(0.0, 1.0, 256),
        np.linspace(0.0, 1.0, len(v)),
        v,
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
X_test_anom = load_fragments(test_anomaly_ids)

X_test_raw = np.concatenate([X_test_nom, X_test_anom], axis=0)
y_test     = np.concatenate([
    np.zeros(len(X_test_nom),  dtype=int),
    np.ones(len(X_test_anom), dtype=int),
])

print(f"  X_fit_raw  : {X_fit_raw.shape}")
print(f"  X_dev_raw  : {X_dev_raw.shape}")
print(f"  X_test_raw : {X_test_raw.shape}  ({y_test.sum()} pos / {(y_test==0).sum()} neg)")


# =============================================================================
# STEP 3 — Training-only standardisation  (Annex D6)
# =============================================================================
print()
print("=" * 60)
print("STEP 3: Standardising (training data only)")
print("=" * 60)

train_mean = float(X_fit_raw.mean())
train_std  = float(X_fit_raw.std())
if train_std == 0:
    train_std = 1.0

X_fit_z  = (X_fit_raw  - train_mean) / train_std
X_dev_z  = (X_dev_raw  - train_mean) / train_std
X_test_z = (X_test_raw - train_mean) / train_std

print(f"  Training mean : {train_mean:.6f}")
print(f"  Training std  : {train_std:.6f}")


# =============================================================================
# STEP 4 — Build 2-D spectrogram arrays  (same as week3_2d_autoencoder.py)
# =============================================================================
print()
print("=" * 60)
print("STEP 4: Computing spectrograms for 2-D model")
print("=" * 60)


def to_spectrogram(x: np.ndarray) -> np.ndarray:
    """
    Convert a 256-sample standardised signal to a log-magnitude spectrogram.
    Returns shape (33, W) where W depends on nperseg/noverlap.
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


X_dev_img  = np.stack([to_spectrogram(x) for x in X_dev_z])[:, None, :, :]
X_test_img = np.stack([to_spectrogram(x) for x in X_test_z])[:, None, :, :]

H, W = X_dev_img.shape[2], X_dev_img.shape[3]
print(f"  Spectrogram shape : (1, {H}, {W})")
print(f"  X_dev_img  : {X_dev_img.shape}")
print(f"  X_test_img : {X_test_img.shape}")


# =============================================================================
# STEP 5 — Build Datasets and DataLoaders
# =============================================================================

class FragmentDataset1D(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x[:, None, :], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


class SpectrogramDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


dev_loader_1d   = DataLoader(FragmentDataset1D(X_dev_z),   batch_size=BATCH_SIZE, shuffle=False)
test_loader_1d  = DataLoader(FragmentDataset1D(X_test_z),  batch_size=BATCH_SIZE, shuffle=False)

dev_loader_2d   = DataLoader(SpectrogramDataset(X_dev_img),  batch_size=BATCH_SIZE, shuffle=False)
test_loader_2d  = DataLoader(SpectrogramDataset(X_test_img), batch_size=BATCH_SIZE, shuffle=False)


# =============================================================================
# STEP 6 — Load saved models
# =============================================================================
print()
print("=" * 60)
print("STEP 6: Loading saved model weights")
print("=" * 60)

model_1d = Conv1dAutoencoder()
model_1d.load_state_dict(torch.load(OUT / "1d_autoencoder.pt", map_location="cpu"))
model_1d.eval()

model_2d = SpectrogramAutoencoder()
model_2d.load_state_dict(torch.load(OUT / "2d_autoencoder.pt", map_location="cpu"))
model_2d.eval()

n_params_1d = sum(p.numel() for p in model_1d.parameters() if p.requires_grad)
n_params_2d = sum(p.numel() for p in model_2d.parameters() if p.requires_grad)

print(f"  Conv1dAutoencoder       : {n_params_1d:,} params  ✓ loaded")
print(f"  SpectrogramAutoencoder  : {n_params_2d:,} params  ✓ loaded")


# =============================================================================
# STEP 7 — Run inference: raw MSE scores for both models
# =============================================================================
print()
print("=" * 60)
print("STEP 7: Running inference to get raw MSE scores")
print("=" * 60)


def score_1d(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mse = ((out - batch) ** 2).mean(dim=(1, 2))
            scores.append(mse.numpy())
    return np.concatenate(scores)


def score_2d(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mse = ((out - batch) ** 2).mean(dim=(1, 2, 3))
            scores.append(mse.numpy())
    return np.concatenate(scores)


dev_scores_1d  = score_1d(model_1d, dev_loader_1d)
test_scores_1d = score_1d(model_1d, test_loader_1d)

dev_scores_2d  = score_2d(model_2d, dev_loader_2d)
test_scores_2d = score_2d(model_2d, test_loader_2d)

print(f"  1-D dev  scores : min={dev_scores_1d.min():.6f}  max={dev_scores_1d.max():.6f}  "
      f"mean={dev_scores_1d.mean():.6f}")
print(f"  1-D test scores : min={test_scores_1d.min():.6f}  max={test_scores_1d.max():.6f}  "
      f"mean={test_scores_1d.mean():.6f}")
print(f"  2-D dev  scores : min={dev_scores_2d.min():.6f}  max={dev_scores_2d.max():.6f}  "
      f"mean={dev_scores_2d.mean():.6f}")
print(f"  2-D test scores : min={test_scores_2d.min():.6f}  max={test_scores_2d.max():.6f}  "
      f"mean={test_scores_2d.mean():.6f}")


# =============================================================================
# STEP 8 — Standardise scores with median / IQR  (fitted on DEV only)
# =============================================================================
print()
print("=" * 60)
print("STEP 8: Standardising scores with median/IQR (dev-only calibration)")
print("=" * 60)


def standardise(scores: np.ndarray, median_val: float, iqr_val: float) -> np.ndarray:
    return (scores - median_val) / iqr_val


# Fit on DEV — never on test
median_1d = float(np.median(dev_scores_1d))
iqr_1d    = float(np.percentile(dev_scores_1d, 75) - np.percentile(dev_scores_1d, 25))
iqr_1d    = max(iqr_1d, 1e-9)

median_2d = float(np.median(dev_scores_2d))
iqr_2d    = float(np.percentile(dev_scores_2d, 75) - np.percentile(dev_scores_2d, 25))
iqr_2d    = max(iqr_2d, 1e-9)

print(f"  1-D : median={median_1d:.6f}  IQR={iqr_1d:.6f}")
print(f"  2-D : median={median_2d:.6f}  IQR={iqr_2d:.6f}")

std_dev_1d  = standardise(dev_scores_1d,  median_1d, iqr_1d)
std_test_1d = standardise(test_scores_1d, median_1d, iqr_1d)

std_dev_2d  = standardise(dev_scores_2d,  median_2d, iqr_2d)
std_test_2d = standardise(test_scores_2d, median_2d, iqr_2d)

print(f"  1-D std dev  : min={std_dev_1d.min():.3f}  max={std_dev_1d.max():.3f}")
print(f"  2-D std dev  : min={std_dev_2d.min():.3f}  max={std_dev_2d.max():.3f}")


# =============================================================================
# STEP 9 — Fuse: equal-weight average  (Annex D0)
# =============================================================================
print()
print("=" * 60)
print("STEP 9: Fusing scores (0.5 × 1D + 0.5 × 2D)")
print("=" * 60)

fused_dev_scores  = 0.5 * std_dev_1d  + 0.5 * std_dev_2d
fused_test_scores = 0.5 * std_test_1d + 0.5 * std_test_2d

print(f"  Fused dev  scores : min={fused_dev_scores.min():.4f}  "
      f"max={fused_dev_scores.max():.4f}  mean={fused_dev_scores.mean():.4f}")
print(f"  Fused test scores : min={fused_test_scores.min():.4f}  "
      f"max={fused_test_scores.max():.4f}  mean={fused_test_scores.mean():.4f}")

# Score separation check
fused_test_nom  = fused_test_scores[y_test == 0]
fused_test_anom = fused_test_scores[y_test == 1]
print(f"  Fused test nominal mean  : {fused_test_nom.mean():.4f}")
print(f"  Fused test anomaly mean  : {fused_test_anom.mean():.4f}")


# =============================================================================
# STEP 10 — Threshold calibration  (99th pctile of FUSED DEV scores)
# =============================================================================
print()
print("=" * 60)
print("STEP 10: Threshold calibration (99th pctile of fused dev scores)")
print("=" * 60)

threshold  = float(np.percentile(fused_dev_scores, THRESHOLD_PCT))
dev_alarms = int((fused_dev_scores >= threshold).sum())

print(f"  Fused threshold (99th pctile) : {threshold:.6f}")
print(f"  Dev alarm rate               : {dev_alarms}/{len(fused_dev_scores)} = "
      f"{100*dev_alarms/len(fused_dev_scores):.2f}%  (false-alarm rate on nominal)")
print()
print("  NOTE: Dev is nominal-only. F1 is undefined on dev.")
print("        No test information used for this threshold.")


# =============================================================================
# STEP 11 — Test evaluation  (ONE time, all choices locked)
# =============================================================================
print()
print("=" * 60)
print("STEP 11: Test evaluation")
print("=" * 60)
print("  (Threshold locked. No changes after seeing these numbers.)")
print()

test_pred = (fused_test_scores >= threshold).astype(int)
p, r, f1, _ = precision_recall_fscore_support(
    y_test, test_pred, average="binary", zero_division=0
)

print(f"  Precision : {p:.4f}")
print(f"  Recall    : {r:.4f}")
print(f"  F1 score  : {f1:.4f}")
print(f"  Support   : {y_test.sum()} anomalous / {(y_test==0).sum()} nominal")


# =============================================================================
# STEP 12 — Read Attention AE F1 from file
# =============================================================================
f1_attn = None
attn_results = OUT / "attention_autoencoder_results.txt"
if attn_results.exists():
    for line in attn_results.read_text().splitlines():
        if "F1 score" in line and ":" in line:
            try:
                f1_attn = float(line.split(":")[-1].strip())
                break
            except ValueError:
                pass

attn_str = f"{f1_attn:.4f}" if f1_attn is not None else "N/A"


# =============================================================================
# STEP 13 — Print full comparison table
# =============================================================================
print()
print("=" * 60)
print("  Comparison table (test F1):")
print("=" * 60)
print(f"    Isolation Forest  : 0.0164")
print(f"    PCA recon. error  : 0.0630")
print(f"    Trivial detector  : 0.0164")
print(f"    1-D Autoencoder   : 0.1417  (no-attention ablation)")
print(f"    2-D Autoencoder   : 0.1311")
print(f"    Attention AE      : {attn_str}")
print(f"    Fusion (1D+2D)    : {f1:.4f}   ← this model")
print()

if f1 > 0.1417:
    print("  → Fusion BEATS the 1-D autoencoder (previous best).")
elif f1 > 0.0630:
    print("  → Fusion beats classical baselines, but not 1-D autoencoder.")
else:
    print("  → Fusion does not beat baselines yet.")
    print("    (This is a valid result — report it honestly.)")


# =============================================================================
# STEP 14 — Save results
# =============================================================================
print()
print("=" * 60)
print("STEP 14: Saving results")
print("=" * 60)

log_lines = [
    "=" * 60,
    "Fusion (1-D + 2-D Autoencoder) — OPSSAT-AD",
    "Team 05, Card I1, Week 4",
    "=" * 60,
    "",
    "--- Fusion method ---",
    "Strategy          : Score-level equal-weight average (Annex D0)",
    "Formula           : fused_score = 0.5 * std_1d + 0.5 * std_2d",
    "Score norm        : Median / IQR, fitted on nominal dev scores only",
    "",
    "--- Models ---",
    f"1-D model         : {OUT / '1d_autoencoder.pt'}",
    f"  Architecture    : Conv1dAutoencoder",
    f"  Params          : {n_params_1d:,}",
    f"2-D model         : {OUT / '2d_autoencoder.pt'}",
    f"  Architecture    : SpectrogramAutoencoder",
    f"  Params          : {n_params_2d:,}",
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
    f"2-D spectrogram   : nperseg={NPERSEG}, noverlap={NOVERLAP}, nfft={NFFT}, log1p(magnitude)",
    "",
    "--- Score normalisation (dev only) ---",
    f"1-D median        : {median_1d:.6f}",
    f"1-D IQR           : {iqr_1d:.6f}",
    f"2-D median        : {median_2d:.6f}",
    f"2-D IQR           : {iqr_2d:.6f}",
    "",
    "--- Calibration (99th pctile of fused dev scores) ---",
    f"Threshold         : {threshold:.6f}",
    f"Dev alarm rate    : {dev_alarms}/{len(fused_dev_scores)} = "
    f"{100*dev_alarms/len(fused_dev_scores):.2f}% (false-alarm rate on nominal)",
    "NOTE: Dev is nominal-only. F1 undefined on dev.",
    "",
    "--- Test results ---",
    f"Precision         : {p:.4f}",
    f"Recall            : {r:.4f}",
    f"F1 score          : {f1:.4f}",
    f"Support           : {y_test.sum()} anomalous / {(y_test==0).sum()} nominal",
    "",
    "--- Comparison table (test F1) ---",
    "  Isolation Forest  : 0.0164",
    "  PCA recon. error  : 0.0630",
    "  Trivial detector  : 0.0164",
    "  1-D Autoencoder   : 0.1417  (no-attention ablation)",
    "  2-D Autoencoder   : 0.1311",
    f"  Attention AE      : {attn_str}",
    f"  Fusion (1D+2D)    : {f1:.4f}   <- this model",
    "",
    "No test labels or test metrics were used for threshold selection.",
]

log_path = OUT / "fusion_results.txt"
log_path.write_text("\n".join(log_lines) + "\n")

print(f"  Results saved to : {log_path}")
print()
print("=" * 60)
print("Week 4 — Fusion (1D+2D) complete.")
print("=" * 60)
