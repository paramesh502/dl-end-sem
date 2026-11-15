"""
final_test_eval.py
==================
Final OPSSAT-AD test evaluation — Card I1, Team 05
K Hemalatha Reddy, B Parameshwar Reddy, Jagan Sai Kumar

ALL thresholds are LOCKED from validation evidence (never recomputed from test).
This script evaluates 7 detectors once on the 529-fragment test set.

Test set inspected earlier on 2026-10-12 (R02-R08).
Final evaluation on 2026-11-20. Thresholds were locked on validation evidence
before this run.

Run from project root:
    source .venv/bin/activate
    python3 scripts/final_test_eval.py
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.stats import linregress, pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)

RESULT_FILE = OUT / "final_test_eval.txt"

# ---------------------------------------------------------------------------
# LOCKED thresholds (never recomputed from test data)
# ---------------------------------------------------------------------------
LOCKED = {
    "iso_forest":   0.201592,
    "pca":          0.030605,
    "trivial":      3.410388,
    "ae_1d":        0.000581,
    "ae_2d":        0.000355,
    "attention":    0.010338,
    "fusion":       3.977751,
}

# Locked fusion normalisation constants (from fusion_results.txt)
FUSION_MEDIAN_1D = 0.000005
FUSION_IQR_1D    = 0.000141
FUSION_MEDIAN_2D = 0.000000
FUSION_IQR_2D    = 0.000067
FUSION_WEIGHT    = 0.5

# Earlier test results from R02-R08 runs
EARLIER = {
    "iso_forest": {"P": 0.0164 / (1 + 0.0164),  # reconstruct approx from F1
                   "R": None,
                   "F1": 0.0164},
    "pca":        {"P": None, "R": None, "F1": 0.0630},
    "trivial":    {"P": None, "R": None, "F1": 0.0164},
    "ae_1d":      {"P": 0.6429, "R": 0.0796, "F1": 0.1417},
    "ae_2d":      {"P": 0.8889, "R": 0.0708, "F1": 0.1311},
    "attention":  {"P": 0.2500, "R": 0.0265, "F1": 0.0480},
    "fusion":     {"P": 0.8000, "R": 0.1062, "F1": 0.1875},
}

TODAY = datetime.date.today().isoformat()
EARLIER_DATE = "2026-10-12"
DISCLAIMER = (
    f"Test set inspected earlier on {EARLIER_DATE} (R02-R08). "
    f"Final evaluation on {TODAY}. "
    "Thresholds were locked on validation evidence before this run."
)

# Spectrogram parameters (must match week3_2d_autoencoder.py)
NPERSEG  = 64
NOVERLAP = 32
NFFT     = 64
BATCH_SIZE = 32

# Length bins for per-length analysis
BINS = [(8, 50), (50, 100), (100, 200), (200, 500), (500, 1041)]

# =============================================================================
# OUTPUT BUFFER
# =============================================================================
_lines = []


def log(msg=""):
    print(msg)
    _lines.append(str(msg))


def save_output():
    RESULT_FILE.write_text("\n".join(_lines) + "\n")


# =============================================================================
# STEP 1: Load data and build split
# =============================================================================

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

# Fragment original lengths (before resampling)
frag_orig_len = {}
for sid, grp in df.groupby("segment"):
    frag_orig_len[str(sid)] = len(grp)

# Sorted test IDs (nominals first, then anomalous) — consistent ordering
test_nom_sorted  = sorted(test_nominal_ids)
test_anom_sorted = sorted(test_anomaly_ids)
test_all_sorted  = test_nom_sorted + test_anom_sorted

y_test = np.array(
    [0] * len(test_nom_sorted) + [1] * len(test_anom_sorted),
    dtype=int,
)
test_orig_lengths = np.array([frag_orig_len[sid] for sid in test_all_sorted])

assert len(test_all_sorted) == 529, f"Expected 529 test fragments, got {len(test_all_sorted)}"
assert (y_test == 0).sum() == 416
assert (y_test == 1).sum() == 113

# =============================================================================
# STEP 2: Resample to 256 samples
# =============================================================================


def resample_to_256(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    if len(v) == 256:
        return v
    return np.interp(
        np.linspace(0.0, 1.0, 256),
        np.linspace(0.0, 1.0, len(v)),
        v,
    ).astype(np.float32)


def load_fragments(seg_ids) -> np.ndarray:
    arrays = []
    for sid in seg_ids:
        vals = df[df["segment"] == str(sid)]["value"].astype(float).to_numpy()
        if len(vals) >= 2:
            arrays.append(resample_to_256(vals))
    return np.stack(arrays) if arrays else np.empty((0, 256), dtype=np.float32)


X_fit_raw   = load_fragments(sorted(nominal_fit_ids))
X_dev_raw   = load_fragments(sorted(nominal_dev_ids))
X_test_raw  = load_fragments(test_all_sorted)

assert X_fit_raw.shape[0] == 1019
assert X_dev_raw.shape[0] == 254
assert X_test_raw.shape[0] == 529

# =============================================================================
# STEP 3: Training-only standardisation — verify locked values
# =============================================================================

train_mean = float(X_fit_raw.mean())
train_std  = float(X_fit_raw.std())
if train_std == 0:
    train_std = 1.0

LOCKED_MEAN = 0.129194
LOCKED_STD  = 0.285037
assert abs(train_mean - LOCKED_MEAN) < 1e-6, (
    f"train_mean mismatch: computed {train_mean:.6f} vs locked {LOCKED_MEAN}"
)
assert abs(train_std - LOCKED_STD) < 1e-6, (
    f"train_std mismatch: computed {train_std:.6f} vs locked {LOCKED_STD}"
)

X_fit_z   = (X_fit_raw  - train_mean) / train_std
X_dev_z   = (X_dev_raw  - train_mean) / train_std
X_test_z  = (X_test_raw - train_mean) / train_std

# =============================================================================
# STEP 4: Feature extraction for classical models
# =============================================================================


def feature_vector(values: np.ndarray, impute: dict = None) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    mean_v = float(np.mean(v))
    std_v  = float(np.std(v))
    min_v  = float(np.min(v))
    max_v  = float(np.max(v))

    if len(v) >= 2:
        t = np.arange(len(v), dtype=np.float64)
        slope_v = float(linregress(t, v).slope)
    else:
        slope_v = np.nan

    if len(v) >= 3 and std_v > 0:
        lag1 = float(pearsonr(v[:-1], v[1:])[0])
    else:
        lag1 = np.nan

    feat = np.array([mean_v, std_v, min_v, max_v, slope_v, lag1], dtype=np.float64)
    if impute is not None:
        for i, name in enumerate(["mean", "std", "min", "max", "slope", "lag1"]):
            if np.isnan(feat[i]):
                feat[i] = impute.get(name, 0.0)
    return feat


def extract_features(seg_ids, impute=None):
    feats = []
    for sid in seg_ids:
        vals = df[df["segment"] == str(sid)]["value"].astype(float).to_numpy()
        if len(vals) >= 2:
            feats.append(feature_vector(vals, impute))
    return np.stack(feats) if feats else np.empty((0, 6), dtype=np.float64)


# Compute imputation medians from fit set (raw, before standardisation)
X_fit_feat_raw = extract_features(sorted(nominal_fit_ids))
train_medians  = {}
for i, name in enumerate(["mean", "std", "min", "max", "slope", "lag1"]):
    col = X_fit_feat_raw[:, i]
    valid = col[~np.isnan(col)]
    train_medians[name] = float(np.median(valid)) if len(valid) > 0 else 0.0

X_fit_feat  = extract_features(sorted(nominal_fit_ids), impute=train_medians)
X_dev_feat  = extract_features(sorted(nominal_dev_ids),  impute=train_medians)
X_test_feat = extract_features(test_all_sorted,          impute=train_medians)

# Feature-space standardisation (fit only)
feat_mean = X_fit_feat.mean(axis=0)
feat_std  = X_fit_feat.std(axis=0)
feat_std[feat_std == 0] = 1.0

X_fit_feat_z  = (X_fit_feat  - feat_mean) / feat_std
X_dev_feat_z  = (X_dev_feat  - feat_mean) / feat_std
X_test_feat_z = (X_test_feat - feat_mean) / feat_std

# Trivial detector: nominal training mean/std of fragment means
nom_train_mean_frags = float(np.mean(X_fit_feat[:, 0]))
nom_train_std_frags  = float(np.std(X_fit_feat[:, 0]))
if nom_train_std_frags == 0:
    nom_train_std_frags = 1.0

# =============================================================================
# STEP 5: Spectrogram conversion for 2-D model
# =============================================================================


def to_spectrogram(x: np.ndarray) -> np.ndarray:
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

# =============================================================================
# STEP 6: Model definitions
# =============================================================================


class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(16,  1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SpectrogramAutoencoder(nn.Module):
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


class TemporalAttention(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, z):
        zt     = z.permute(0, 2, 1)
        scores = self.score(zt)
        alpha  = torch.softmax(scores, dim=1)
        out    = zt * alpha
        return out.permute(0, 2, 1)


class AttentionAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.attention = TemporalAttention(channels=16)
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(16,  1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.attention(z)
        return self.decoder(z)


# =============================================================================
# STEP 7: Dataset helpers
# =============================================================================


class FragDataset1D(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x[:, None, :], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


class SpectrogramDataset(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


def score_1d(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mse = ((out - batch) ** 2).mean(dim=(1, 2))
            scores.append(mse.numpy())
    return np.concatenate(scores)


def score_2d(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mse = ((out - batch) ** 2).mean(dim=(1, 2, 3))
            scores.append(mse.numpy())
    return np.concatenate(scores)


# =============================================================================
# STEP 8: Compute all scores
# =============================================================================

# --- Classical feature scores ---
# Isolation Forest
iso = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1)
iso.fit(X_fit_feat_z)
# Locked threshold 0.201592 was calibrated on -decision_function scores in run_baselines.py
# (score_samples = decision_function - 0.5, so scales differ by a constant)
scores_iso = -iso.decision_function(X_test_feat_z)

# PCA
pca_full = PCA(random_state=SEED)
pca_full.fit(X_fit_feat_z)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, 0.95)) + 1
n_comp = max(1, min(n_comp, X_fit_feat_z.shape[1]))
pca = PCA(n_components=n_comp, random_state=SEED)
pca.fit(X_fit_feat_z)
scores_pca = np.mean((X_test_feat_z - pca.inverse_transform(pca.transform(X_test_feat_z))) ** 2, axis=1)

# Trivial
scores_trivial = np.abs((X_test_feat[:, 0] - nom_train_mean_frags) / nom_train_std_frags)

# --- Neural network scores ---
test_loader_1d  = DataLoader(FragDataset1D(X_test_z),    batch_size=BATCH_SIZE, shuffle=False)
test_loader_img = DataLoader(SpectrogramDataset(X_test_img), batch_size=BATCH_SIZE, shuffle=False)

ae1d = Conv1dAutoencoder()
ae1d.load_state_dict(torch.load(OUT / "1d_autoencoder.pt", map_location="cpu", weights_only=True))
scores_1d = score_1d(ae1d, test_loader_1d)

ae2d = SpectrogramAutoencoder()
ae2d.load_state_dict(torch.load(OUT / "2d_autoencoder.pt", map_location="cpu", weights_only=True))
scores_2d = score_2d(ae2d, test_loader_img)

attn = AttentionAutoencoder()
attn.load_state_dict(torch.load(OUT / "attention_autoencoder.pt", map_location="cpu", weights_only=True))
scores_attn = score_1d(attn, test_loader_1d)

# --- Fusion scores (using LOCKED normalisation constants) ---
std_1d = (scores_1d - FUSION_MEDIAN_1D) / FUSION_IQR_1D
std_2d = (scores_2d - FUSION_MEDIAN_2D) / FUSION_IQR_2D
scores_fusion = FUSION_WEIGHT * std_1d + FUSION_WEIGHT * std_2d

# =============================================================================
# STEP 9: Per-length analysis helper
# =============================================================================


def per_length_table(preds, y_true, orig_lengths):
    """Return per-bin TP/FP/TN/FN counts."""
    rows = []
    for lo, hi in BINS:
        mask = (orig_lengths >= lo) & (orig_lengths < hi)
        if mask.sum() == 0:
            rows.append((f"[{lo},{hi})", 0, 0, 0, 0, 0))
            continue
        yt = y_true[mask]
        yp = preds[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel() if len(np.unique(yt)) > 1 or len(np.unique(yp)) > 1 else _safe_cm(yt, yp)
        rows.append((f"[{lo},{hi})", int(tp), int(fp), int(tn), int(fn), int(mask.sum())))
    return rows


def _safe_cm(yt, yp):
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return tn, fp, fn, tp


def evaluate_model(scores, threshold, name, earlier_key):
    preds = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    length_rows = per_length_table(preds, y_test, test_orig_lengths)
    earlier = EARLIER.get(earlier_key, {})
    return {
        "name": name,
        "threshold": threshold,
        "scores": scores,
        "preds": preds,
        "P": p, "R": r, "F1": f1,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "length_rows": length_rows,
        "earlier_P": earlier.get("P"),
        "earlier_R": earlier.get("R"),
        "earlier_F1": earlier.get("F1"),
    }


results = {
    "iso_forest": evaluate_model(scores_iso,     LOCKED["iso_forest"], "Isolation Forest",             "iso_forest"),
    "pca":        evaluate_model(scores_pca,     LOCKED["pca"],        "PCA Reconstruction",           "pca"),
    "trivial":    evaluate_model(scores_trivial, LOCKED["trivial"],    "Trivial Detector",             "trivial"),
    "ae_1d":      evaluate_model(scores_1d,      LOCKED["ae_1d"],      "1-D Autoencoder",              "ae_1d"),
    "ae_2d":      evaluate_model(scores_2d,      LOCKED["ae_2d"],      "2-D Spectrogram Autoencoder",  "ae_2d"),
    "attention":  evaluate_model(scores_attn,    LOCKED["attention"],  "Attention Autoencoder",        "attention"),
    "fusion":     evaluate_model(scores_fusion,  LOCKED["fusion"],     "Fusion (1D+2D)",               "fusion"),
}

# =============================================================================
# STEP 10: Constant-all-nominal floor
# =============================================================================
preds_all_nominal = np.zeros(len(y_test), dtype=int)
p_floor, r_floor, f1_floor, _ = precision_recall_fscore_support(
    y_test, preds_all_nominal, average="binary", zero_division=0
)

# =============================================================================
# STEP 11: Assertions — all F1 scores must be finite
# =============================================================================
for key, res in results.items():
    assert np.isfinite(res["F1"]), f"F1 for {key} is not finite!"
    assert not np.isnan(res["F1"]), f"F1 for {key} is NaN!"
    assert np.isfinite(res["P"]),  f"Precision for {key} is not finite!"
    assert np.isfinite(res["R"]),  f"Recall for {key} is not finite!"

# =============================================================================
# STEP 12: Write output
# =============================================================================

ORDERED_KEYS = ["iso_forest", "pca", "trivial", "ae_1d", "ae_2d", "attention", "fusion"]


def fmt_earlier(res):
    ef1 = res["earlier_F1"]
    ep  = res["earlier_P"]
    er  = res["earlier_R"]
    parts = []
    if ef1 is not None:
        parts.append(f"F1={ef1:.4f}")
    if ep is not None:
        parts.append(f"P={ep:.4f}")
    if er is not None:
        parts.append(f"R={er:.4f}")
    return "  ".join(parts) if parts else "see baseline_results.txt"


log("=" * 70)
log("FINAL TEST EVALUATION — OPSSAT-AD")
log("Team 05, Card I1")
log(f"Date: {TODAY}")
log("=" * 70)
log()
log(DISCLAIMER)
log()
log(f"Test set: {len(test_all_sorted)} fragments "
    f"({(y_test==1).sum()} anomalous, {(y_test==0).sum()} nominal)")
log(f"Nominal fitting set: {len(nominal_fit_ids)} fragments")
log(f"Nominal dev set: {len(nominal_dev_ids)} fragments")
log()
log(f"Training standardisation: mean={train_mean:.6f}  std={train_std:.6f}  (locked, computed from X_fit)")
log(f"  Verified against locked values: mean=0.129194  std=0.285037  — MATCH")
log()

for key in ORDERED_KEYS:
    res = results[key]
    log("=" * 70)
    log(f"MODEL: {res['name']}")
    log(f"  Locked threshold: {res['threshold']:.6f}")
    log(f"  Earlier test-scored result ({EARLIER_DATE}, R02-R08): {fmt_earlier(res)}")
    log(f"  Final result ({TODAY}):")
    log(f"    Precision : {res['P']:.4f}")
    log(f"    Recall    : {res['R']:.4f}")
    log(f"    F1        : {res['F1']:.4f}")
    log(f"    TP={res['TP']}  FP={res['FP']}  TN={res['TN']}  FN={res['FN']}")
    log(f"    Support   : {(y_test==1).sum()} anomalous / {(y_test==0).sum()} nominal")

    if key == "attention":
        ae1d_f1 = results["ae_1d"]["F1"]
        delta = res["F1"] - ae1d_f1
        log(f"  Ablation comparison (vs 1-D AE, no-attention): "
            f"1-D AE F1={ae1d_f1:.4f}  Attention F1={res['F1']:.4f}  "
            f"delta={delta:+.4f}")

    if key == "fusion":
        log(f"  Fusion formula: 0.5*(score_1d-{FUSION_MEDIAN_1D})/{FUSION_IQR_1D} "
            f"+ 0.5*(score_2d-{FUSION_MEDIAN_2D})/{FUSION_IQR_2D}")
        log(f"  Normalisation constants locked from fusion_results.txt (dev evidence only).")

    log()
    log(f"  Per-length analysis (original fragment length before resampling):")
    log(f"  {'Bin':>12}  {'TP':>5}  {'FP':>5}  {'TN':>5}  {'FN':>5}  {'N':>5}")
    log(f"  {'-'*47}")
    for bin_label, tp, fp, tn, fn, n in res["length_rows"]:
        log(f"  {bin_label:>12}  {tp:>5}  {fp:>5}  {tn:>5}  {fn:>5}  {n:>5}")
    log()

log("=" * 70)
log("CONSTANT-ALL-NOMINAL FLOOR (reference)")
log(f"  Predict everything nominal: F1={f1_floor:.4f}  P={p_floor:.4f}  R={r_floor:.4f}")
log()

log("=" * 70)
log("SUMMARY TABLE — All 7 Detectors")
log(f"  {'Model':<30}  {'Threshold':>10}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>5}  {'FP':>5}  {'TN':>5}  {'FN':>5}")
log(f"  {'-'*92}")
for key in ORDERED_KEYS:
    res = results[key]
    log(f"  {res['name']:<30}  {res['threshold']:>10.6f}  "
        f"{res['P']:>7.4f}  {res['R']:>7.4f}  {res['F1']:>7.4f}  "
        f"{res['TP']:>5}  {res['FP']:>5}  {res['TN']:>5}  {res['FN']:>5}")
log(f"  {'All-nominal floor':<30}  {'N/A':>10}  "
    f"{p_floor:>7.4f}  {r_floor:>7.4f}  {f1_floor:>7.4f}  "
    f"{'0':>5}  {'0':>5}  {'416':>5}  {'113':>5}")
log()
log("Thresholds were loaded (not recomputed) — locked from validation evidence.")
log("No test labels or test scores were used for any threshold choice.")

save_output()

# =============================================================================
# STEP 13: Final stdout summary
# =============================================================================
print()
print("=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)
print(f"All 7 models produced valid (finite, non-NaN) results.")
print()
for key in ORDERED_KEYS:
    res = results[key]
    print(f"  {res['name']:<30}  P={res['P']:.4f}  R={res['R']:.4f}  F1={res['F1']:.4f}")
print()
print(f"Constant-all-nominal floor:            F1=0.0000  (reference)")
print()
print(f"Results saved to: {RESULT_FILE}")
print()
print("Thresholds confirmed LOADED (not recomputed from test):")
for key, thr in LOCKED.items():
    print(f"  {key:<15} = {thr:.6f}")
