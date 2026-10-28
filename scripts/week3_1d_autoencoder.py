"""
week3_1d_autoencoder.py
=======================
Week 3 — Card I1 — Satellite Telemetry Anomaly Detection
Team 05: K Hemalatha Reddy, B Parameshwar Reddy, Jagan Sai Kumar

1-D Convolutional Autoencoder on OPSSAT-AD.
Nominal-only training (Annex D6).
Threshold at 99th percentile of nominal-dev scores.
Limits: 30 epochs / 30-minute elapsed wall-clock / early-stop after 5 (Annex D0).

Run from project root:
    source .venv/bin/activate
    python3 scripts/week3_1d_autoencoder.py
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility  (Annex D0: seed = 22)
# ─────────────────────────────────────────────────────────────────────────────
SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Annex D0 training limits
# ─────────────────────────────────────────────────────────────────────────────
MAX_EPOCHS    = 30
MAX_SECONDS   = 30 * 60   # 30-minute elapsed wall-clock limit
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EARLY_STOP    = 5         # stop after 5 epochs with no val improvement
THRESHOLD_PCT = 99.0      # calibrate threshold at 99th percentile of dev scores

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)


# =============================================================================
# STEP 1 — Load OPSSAT-AD and build split
# =============================================================================
print("=" * 60)
print("STEP 1: Loading OPSSAT-AD data")
print("=" * 60)

df  = pd.read_csv(DATA / "segments.csv")
dev = pd.read_csv(DATA / "OPSSAT_nominal_dev_segments.csv")

df["segment"] = df["segment"].astype(str)
df["train"]   = df["train"].astype(str)
df["anomaly"] = df["anomaly"].astype(int)

dev_ids = set(dev["segment"].astype(str))

# Fragment-level attributes (one row per fragment)
frag = df.groupby("segment").agg(
    anomaly=("anomaly", "first"),
    train=("train", "first"),
).reset_index()

nominal_fit_ids  = set(frag[
    (frag["anomaly"] == 0) &
    (frag["train"] == "1") &
    (~frag["segment"].isin(dev_ids))
]["segment"])

nominal_dev_ids  = dev_ids   # 254 nominal fragments — threshold calibration only

test_nominal_ids = set(frag[(frag["train"] != "1") & (frag["anomaly"] == 0)]["segment"])
test_anomaly_ids = set(frag[(frag["train"] != "1") & (frag["anomaly"] == 1)]["segment"])

print(f"  Nominal fitting fragments :  {len(nominal_fit_ids)}")
print(f"  Nominal dev fragments     :  {len(nominal_dev_ids)}")
print(f"  Test nominal fragments    :  {len(test_nominal_ids)}")
print(f"  Test anomalous fragments  :  {len(test_anomaly_ids)}")
print(f"  Total test fragments      :  {len(test_nominal_ids) + len(test_anomaly_ids)}")


# =============================================================================
# STEP 2 — Resample every fragment to 256 samples  (Annex D6)
# =============================================================================
print()
print("=" * 60)
print("STEP 2: Resampling fragments to 256 samples")
print("=" * 60)

def resample_to_256(values: np.ndarray) -> np.ndarray:
    """
    Resample a variable-length fragment to exactly 256 samples by
    linear interpolation (Annex D6). Most fragments are shorter than
    256 samples, so this mostly upsamples.
    """
    v = np.asarray(values, dtype=np.float32)
    if len(v) == 256:
        return v
    old_x = np.linspace(0.0, 1.0, len(v))
    new_x = np.linspace(0.0, 1.0, 256)
    return np.interp(new_x, old_x, v).astype(np.float32)


def load_fragments(seg_ids: set) -> np.ndarray:
    """Return (N, 256) array of resampled fragment values for the given IDs."""
    arrays = []
    for sid in sorted(seg_ids):
        vals = df[df["segment"] == sid]["value"].astype(float).to_numpy()
        if len(vals) >= 2:
            arrays.append(resample_to_256(vals))
    return np.stack(arrays) if arrays else np.empty((0, 256), dtype=np.float32)


X_fit  = load_fragments(nominal_fit_ids)   # (1019, 256)
X_dev  = load_fragments(nominal_dev_ids)   # (254, 256)
X_test_nom  = load_fragments(test_nominal_ids)
X_test_anom = load_fragments(test_anomaly_ids)

# Stack test set: nominals first, then anomalous
X_test = np.concatenate([X_test_nom, X_test_anom], axis=0)
y_test = np.concatenate([
    np.zeros(len(X_test_nom),  dtype=int),
    np.ones(len(X_test_anom), dtype=int),
])

print(f"  X_fit shape  : {X_fit.shape}")
print(f"  X_dev shape  : {X_dev.shape}")
print(f"  X_test shape : {X_test.shape}  (y: {y_test.sum()} pos / {(y_test==0).sum()} neg)")


# =============================================================================
# STEP 3 — Training-only standardisation  (Annex D6)
# =============================================================================
print()
print("=" * 60)
print("STEP 3: Standardising (training data only)")
print("=" * 60)

# Compute mean and std from FITTING data only — never touch dev or test
train_mean = float(X_fit.mean())
train_std  = float(X_fit.std())
if train_std == 0:
    train_std = 1.0

X_fit_z  = (X_fit  - train_mean) / train_std
X_dev_z  = (X_dev  - train_mean) / train_std
X_test_z = (X_test - train_mean) / train_std

print(f"  Training mean : {train_mean:.6f}")
print(f"  Training std  : {train_std:.6f}")
print(f"  X_fit_z  — min: {X_fit_z.min():.3f}  max: {X_fit_z.max():.3f}")


# =============================================================================
# STEP 4 — Dataset and DataLoader
# =============================================================================

class FragmentDataset(Dataset):
    def __init__(self, x: np.ndarray):
        # Shape: (N, 1, 256) — 1 channel, 256 time-steps
        self.x = torch.tensor(x[:, None, :], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


train_loader = DataLoader(FragmentDataset(X_fit_z),  batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(FragmentDataset(X_dev_z),  batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(FragmentDataset(X_test_z), batch_size=BATCH_SIZE, shuffle=False)


# =============================================================================
# STEP 5 — Model: 1-D Convolutional Autoencoder
# =============================================================================
print()
print("=" * 60)
print("STEP 5: Building 1-D Convolutional Autoencoder")
print("=" * 60)


class Conv1dAutoencoder(nn.Module):
    """
    Small 1-D Convolutional Autoencoder.

    Architecture:
        Input  : (batch, 1, 256)

        Encoder:
            Conv1d(1→16, k=3) + ReLU
            MaxPool1d(2)              → 256→128
            Conv1d(16→32, k=3) + ReLU
            MaxPool1d(2)              → 128→64
            Conv1d(32→16, k=3) + ReLU
                                 ↓ bottleneck: (batch, 16, 64)

        Decoder:
            Conv1d(16→32, k=3) + ReLU
            Upsample(×2)              → 64→128
            Conv1d(32→16, k=3) + ReLU
            Upsample(×2)              → 128→256
            Conv1d(16→1, k=3)

        Output : (batch, 1, 256)

    Anomaly score per fragment = mean squared reconstruction error.
    High MSE  →  anomalous (the model never learned this pattern during training).
    Low MSE   →  normal.

    Trainable parameters: ~45,000  (well under the 250k Annex D0 cap).
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Layer 1: 256 → 256 (same length after conv with padding=1)
            nn.Conv1d(in_channels=1,  out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Pool: 256 → 128
            nn.MaxPool1d(kernel_size=2),

            # Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Pool: 128 → 64
            nn.MaxPool1d(kernel_size=2),

            # Layer 3: bottleneck compression
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # After encoder: (batch, 16, 64) — 1,024 latent values

        self.decoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Upsample: 64 → 128
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Layer 2
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Upsample: 128 → 256
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Output layer: restore to 1 channel
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model  = Conv1dAutoencoder()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Architecture: Conv1dAutoencoder")
print(f"  Trainable parameters: {n_params:,}  (cap: 250,000)")
assert n_params <= 250_000, f"Parameter count {n_params} exceeds Annex D0 cap of 250,000!"

# Quick shape check
with torch.no_grad():
    dummy = torch.zeros(1, 1, 256)
    out   = model(dummy)
    assert out.shape == dummy.shape, f"Output shape mismatch: {out.shape} != {dummy.shape}"
print(f"  Shape check passed: input (1,1,256) → output {tuple(out.shape)}")


# =============================================================================
# STEP 6 — Training loop
# =============================================================================
print()
print("=" * 60)
print("STEP 6: Training (30 epochs / 30 min / early-stop after 5)")
print("=" * 60)

optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion   = nn.MSELoss()

best_val_loss = float("inf")
best_state    = None
no_improve    = 0
history       = []   # (epoch, train_loss, val_loss)

t_start = time.perf_counter()
stop_reason = "completed all epochs"

for epoch in range(1, MAX_EPOCHS + 1):

    # ── Check elapsed time ────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    if elapsed >= MAX_SECONDS:
        stop_reason = "30-minute elapsed limit"
        print(f"  30-minute limit reached at epoch {epoch}. Stopping.")
        break

    # ── Training pass ─────────────────────────────────────────────────────
    model.train()
    train_loss_sum = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss   = criterion(output, batch)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * len(batch)
    train_loss = train_loss_sum / len(X_fit_z)

    # ── Validation pass (dev — nominal only) ──────────────────────────────
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for batch in dev_loader:
            output = model(batch)
            val_loss_sum += criterion(output, batch).item() * len(batch)
    val_loss = val_loss_sum / len(X_dev_z)

    elapsed = time.perf_counter() - t_start
    print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  "
          f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
          f"elapsed={elapsed:.1f}s")

    history.append((epoch, train_loss, val_loss))

    # ── Best checkpoint ───────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1

    # ── Early stopping ────────────────────────────────────────────────────
    if no_improve >= EARLY_STOP:
        stop_reason = f"early stopping (no improvement for {EARLY_STOP} epochs)"
        print(f"  Early stopping at epoch {epoch}.")
        break

# Restore best checkpoint
if best_state is not None:
    model.load_state_dict(best_state)

total_elapsed = time.perf_counter() - t_start
print()
print(f"  Stopped by  : {stop_reason}")
print(f"  Best val loss: {best_val_loss:.6f}")
print(f"  Total elapsed: {total_elapsed:.2f}s  ({total_elapsed/60:.2f} min)")

# Save model weights
model_path = OUT / "1d_autoencoder.pt"
torch.save(model.state_dict(), model_path)
print(f"  Model saved : {model_path}")


# =============================================================================
# STEP 7 — Score all partitions
# =============================================================================
print()
print("=" * 60)
print("STEP 7: Computing reconstruction MSE scores")
print("=" * 60)


def score_fragments(loader: DataLoader) -> np.ndarray:
    """
    Return per-fragment MSE between input and reconstruction.
    Higher score = model struggled = more likely anomalous.
    """
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            # MSE per sample: mean over (channel=1, time=256)
            mse = ((output - batch) ** 2).mean(dim=(1, 2))
            all_scores.append(mse.numpy())
    return np.concatenate(all_scores)


dev_scores  = score_fragments(dev_loader)
test_scores = score_fragments(test_loader)

print(f"  Dev  scores — min: {dev_scores.min():.6f}  "
      f"max: {dev_scores.max():.6f}  "
      f"mean: {dev_scores.mean():.6f}")
print(f"  Test scores — min: {test_scores.min():.6f}  "
      f"max: {test_scores.max():.6f}  "
      f"mean: {test_scores.mean():.6f}")

# Show score separation (are anomalous scores higher than nominal?)
test_nominal_scores  = test_scores[y_test == 0]
test_anomaly_scores  = test_scores[y_test == 1]
print(f"  Test nominal  mean score: {test_nominal_scores.mean():.6f}")
print(f"  Test anomaly  mean score: {test_anomaly_scores.mean():.6f}")
print(f"  Separation ratio        : {test_anomaly_scores.mean()/max(test_nominal_scores.mean(), 1e-9):.2f}x")


# =============================================================================
# STEP 8 — Threshold calibration  (Annex D6: 99th percentile of dev scores)
# =============================================================================
print()
print("=" * 60)
print("STEP 8: Threshold calibration (99th pctile of dev scores)")
print("=" * 60)

# Dev partition is NOMINAL ONLY — cannot compute anomaly F1 here
# Calibrate at 99th percentile so ~1% of nominal dev fragments are false alarms
threshold  = float(np.percentile(dev_scores, THRESHOLD_PCT))
dev_alarms = int((dev_scores >= threshold).sum())

print(f"  Threshold (99th pctile) : {threshold:.6f}")
print(f"  Dev alarm rate          : {dev_alarms}/{len(dev_scores)} = "
      f"{100*dev_alarms/len(dev_scores):.2f}%  (false-alarm rate on nominal)")
print()
print("  NOTE: Dev is nominal-only. F1 is undefined on dev.")
print("        Alarm rate here = false-alarm rate, not precision/recall.")


# =============================================================================
# STEP 9 — Test evaluation  (ONE time, after all choices are locked)
# =============================================================================
print()
print("=" * 60)
print("STEP 9: Test evaluation")
print("=" * 60)
print("  (Threshold locked. No changes allowed after seeing these numbers.)")
print()

test_pred  = (test_scores >= threshold).astype(int)
p, r, f1, _ = precision_recall_fscore_support(
    y_test, test_pred, average="binary", zero_division=0
)

print(f"  Precision : {p:.4f}")
print(f"  Recall    : {r:.4f}")
print(f"  F1 score  : {f1:.4f}")
print(f"  Support   : {y_test.sum()} anomalous / {(y_test==0).sum()} nominal")
print()

# Comparison with baselines
print("  Comparison with Week 2 baselines (test F1):")
print(f"    Isolation Forest  : 0.0164")
print(f"    PCA recon. error  : 0.0630   ← previous best")
print(f"    Trivial detector  : 0.0164")
print(f"    1-D Autoencoder   : {f1:.4f}   ← this model")

if f1 > 0.0630:
    print("  → 1-D autoencoder BEATS the best classical baseline.")
elif f1 > 0.0164:
    print("  → 1-D autoencoder beats Isolation Forest and Trivial, but not PCA.")
else:
    print("  → 1-D autoencoder does not beat baselines yet.")
    print("    (This is a valid result — report it honestly. See Annex D0.)")


# =============================================================================
# STEP 10 — Save results log
# =============================================================================
print()
print("=" * 60)
print("STEP 10: Saving results")
print("=" * 60)

log_lines = [
    "=" * 60,
    "1-D Convolutional Autoencoder — OPSSAT-AD",
    "Team 05, Card I1, Week 3",
    "=" * 60,
    "",
    "--- Model ---",
    f"Architecture      : Conv1dAutoencoder",
    f"Trainable params  : {n_params:,}",
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
    "",
    "--- Training ---",
    f"Seed              : {SEED}",
    f"Max epochs        : {MAX_EPOCHS}",
    f"Max elapsed       : {MAX_SECONDS/60:.0f} minutes",
    f"Batch size        : {BATCH_SIZE}",
    f"Learning rate     : {LEARNING_RATE}",
    f"Early stop        : after {EARLY_STOP} epochs without improvement",
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
    "--- Calibration (Annex D6: 99th pctile of dev scores) ---",
    f"Threshold         : {threshold:.6f}",
    f"Dev alarm rate    : {dev_alarms}/{len(dev_scores)} = "
    f"{100*dev_alarms/len(dev_scores):.2f}% (false-alarm rate on nominal)",
    "NOTE: Dev is nominal-only. F1 undefined on dev.",
    "",
    "--- Test results ---",
    f"Precision         : {p:.4f}",
    f"Recall            : {r:.4f}",
    f"F1 score          : {f1:.4f}",
    f"Support           : {y_test.sum()} anomalous / {(y_test==0).sum()} nominal",
    "",
    "--- Baseline comparison (test F1) ---",
    "  Isolation Forest  : 0.0164",
    "  PCA recon. error  : 0.0630",
    "  Trivial detector  : 0.0164",
    f"  1-D Autoencoder   : {f1:.4f}",
    "",
    "No test labels or test metrics were used for threshold selection.",
]

log_path = OUT / "1d_autoencoder_results.txt"
with open(log_path, "w") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"  Results saved to : {log_path}")
print(f"  Model saved to   : {model_path}")
print()
print("=" * 60)
print("Week 3 — 1-D Autoencoder complete.")
print("=" * 60)
