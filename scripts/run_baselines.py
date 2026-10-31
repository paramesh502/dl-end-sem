"""
run_baselines.py
================
Implements and calibrates the three classical baselines required by Card I1,
Annex D6, on OPSSAT-AD and SMAP/MSL (P-1 and C-1).

Baselines:
  1. Isolation Forest
  2. PCA reconstruction error
  3. Trivial constant/threshold detector

OPSSAT-AD (validation only):
  - Nominal development data only (no anomalous-class labels).
  - Reports calibrated threshold and nominal-development alarm rate.
  - Does NOT report precision/recall/F1 (those need both classes and, per the
    revised Handout §14, no test partition is scored before the results freeze).

SMAP/MSL (calibration only):
  - Per-stream calibration via guard-context windows (revised Annex D6): a
    calibration window may take earlier context from the guard, never from fit
    rows, and only calibration points are scored.
  - Reports calibrated threshold, calibration alarm count and alarm rate.
  - Does NOT score the test stream (revised Handout §14).

Run from project root:
    python3 scripts/run_baselines.py

Results saved to: results/baseline_results.txt
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

LOG  = OUT / "baseline_results.txt"

lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

SEED = 22
np.random.seed(SEED)


def feature_vector(values: np.ndarray, train_median_impute: dict = None) -> np.ndarray:
    """
    Compute the 6-element summary feature vector required by D6:
      mean, std, min, max, slope, lag-1 correlation.
    Undefined features are imputed by training median.
    """
    v = np.asarray(values, dtype=np.float64)
    mean_v  = float(np.mean(v))
    std_v   = float(np.std(v))
    min_v   = float(np.min(v))
    max_v   = float(np.max(v))

    # Slope via linear regression
    if len(v) >= 2:
        t = np.arange(len(v), dtype=np.float64)
        slope_v = float(linregress(t, v).slope)
    else:
        slope_v = np.nan

    # Lag-1 correlation
    if len(v) >= 3 and std_v > 0:
        lag1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    else:
        lag1 = np.nan

    feat = np.array([mean_v, std_v, min_v, max_v, slope_v, lag1], dtype=np.float64)

    if train_median_impute is not None:
        for i, name in enumerate(["mean", "std", "min", "max", "slope", "lag1"]):
            if np.isnan(feat[i]):
                feat[i] = train_median_impute.get(name, 0.0)

    return feat


def resample_to_256(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 256:
        return values
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, 256)
    return np.interp(new_x, old_x, values).astype(np.float32)


def calibrate_threshold(scores: np.ndarray, percentile: float = 99.0) -> float:
    """Return the p-th percentile of scores (99th by default, per D6)."""
    return float(np.percentile(scores, percentile))


# ===========================================================================
# PART 1 — OPSSAT-AD
# ===========================================================================

log("=" * 70)
log("PART 1: OPSSAT-AD Classical Baselines  (validation: threshold + dev alarm rate)")
log("=" * 70)
log()

segs = pd.read_csv(DATA / "segments.csv")
dev_df = pd.read_csv(DATA / "OPSSAT_nominal_dev_segments.csv")
dev_ids = set(dev_df["segment"].astype(str))

segs["segment"] = segs["segment"].astype(str)
segs["train"]   = segs["train"].astype(str)
segs["anomaly"] = segs["anomaly"].astype(int)

# Fragment-level group
frag = segs.groupby("segment").agg(
    anomaly=("anomaly", "first"),
    train=("train", "first"),
    channel=("channel", "first"),
).reset_index()
frag["segment"] = frag["segment"].astype(str)

# Build feature matrices (nominal fitting / development only; test NOT scored)
nominal_fit_ids = set(frag[(frag["anomaly"]==0) & (frag["train"]=="1") &
                           (~frag["segment"].isin(dev_ids))]["segment"])
nominal_dev_ids = dev_ids  # all nominal
test_ids        = set(frag[frag["train"]!="1"]["segment"])  # partition size only

log(f"OPSSAT-AD split counts:")
log(f"  Nominal fitting fragments :  {len(nominal_fit_ids)}")
log(f"  Nominal development frags :  {len(nominal_dev_ids)}")
log(f"  Test fragments (not scored):  {len(test_ids)}")
log()

def extract_opssat_features(seg_ids):
    """Return feature matrix (N, 6) and list of segment IDs in order."""
    feats, ids = [], []
    for sid in sorted(seg_ids):
        vals = segs[segs["segment"]==sid]["value"].astype(float).to_numpy()
        if len(vals) < 2:
            continue
        feats.append(feature_vector(vals))
        ids.append(sid)
    return np.stack(feats) if feats else np.empty((0, 6)), ids

# Get raw feature arrays
X_fit_raw, fit_ids = extract_opssat_features(nominal_fit_ids)

# Compute training-only median imputation values (for undefined features)
train_medians = {}
for i, name in enumerate(["mean", "std", "min", "max", "slope", "lag1"]):
    col = X_fit_raw[:, i]
    valid = col[~np.isnan(col)]
    train_medians[name] = float(np.median(valid)) if len(valid) > 0 else 0.0

# Re-extract with imputation (fit + dev only; test not extracted)
def extract_imputed(seg_ids):
    feats, ids = [], []
    for sid in sorted(seg_ids):
        vals = segs[segs["segment"]==sid]["value"].astype(float).to_numpy()
        if len(vals) < 2:
            continue
        feats.append(feature_vector(vals, train_medians))
        ids.append(sid)
    return np.stack(feats) if feats else np.empty((0, 6)), ids

X_fit, fit_ids       = extract_imputed(nominal_fit_ids)
X_dev, dev_ids_list  = extract_imputed(nominal_dev_ids)

# Training-only standardisation (mean/std from fitting data)
train_mean_feat = X_fit.mean(axis=0)
train_std_feat  = X_fit.std(axis=0)
train_std_feat[train_std_feat == 0] = 1.0

X_fit_z = (X_fit - train_mean_feat) / train_std_feat
X_dev_z = (X_dev - train_mean_feat) / train_std_feat

log(f"Dev fragments scored: {len(dev_ids_list)} (nominal-only; F1 undefined on dev)")
log()


# -----------------------------------------------------------------------
# Baseline 1: Isolation Forest
# -----------------------------------------------------------------------
log("-" * 50)
log("Baseline 1: Isolation Forest")
log("  100 trees, seed=22, training-only standardisation")
log("-" * 50)

t0 = time.perf_counter()
iso = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1)
iso.fit(X_fit_z)
fit_time_iso = time.perf_counter() - t0

# Scores: decision_function returns higher = more normal; negate for anomaly score
iso_dev_scores = -iso.decision_function(X_dev_z)

iso_threshold = calibrate_threshold(iso_dev_scores, 99.0)
iso_dev_alarms = (iso_dev_scores >= iso_threshold).sum()

log(f"  Calibrated threshold (99th pctile of dev scores): {iso_threshold:.6f}")
log(f"  Dev alarm rate: {iso_dev_alarms}/{len(iso_dev_scores)} "
    f"= {100*iso_dev_alarms/len(iso_dev_scores):.2f}%")
log(f"  Fit time: {fit_time_iso:.2f}s")
log()


# -----------------------------------------------------------------------
# Baseline 2: PCA reconstruction error
# -----------------------------------------------------------------------
log("-" * 50)
log("Baseline 2: PCA reconstruction error")
log("  Retains smallest dimension explaining >=95% training variance")
log("-" * 50)

t0 = time.perf_counter()
pca_full = PCA(random_state=SEED)
pca_full.fit(X_fit_z)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.searchsorted(cumvar, 0.95)) + 1
n_components = max(1, min(n_components, X_fit_z.shape[1]))

pca = PCA(n_components=n_components, random_state=SEED)
pca.fit(X_fit_z)
fit_time_pca = time.perf_counter() - t0

def pca_score(X: np.ndarray) -> np.ndarray:
    X_proj = pca.inverse_transform(pca.transform(X))
    return np.mean((X - X_proj) ** 2, axis=1)

pca_dev_scores = pca_score(X_dev_z)

pca_threshold = calibrate_threshold(pca_dev_scores, 99.0)
pca_dev_alarms = (pca_dev_scores >= pca_threshold).sum()

log(f"  PCA components retained: {n_components} (explaining >= 95% variance)")
log(f"  Calibrated threshold (99th pctile of dev scores): {pca_threshold:.6f}")
log(f"  Dev alarm rate: {pca_dev_alarms}/{len(pca_dev_scores)} "
    f"= {100*pca_dev_alarms/len(pca_dev_scores):.2f}%")
log(f"  Fit time: {fit_time_pca:.2f}s")
log()


# -----------------------------------------------------------------------
# Baseline 3: Trivial detector
# -----------------------------------------------------------------------
log("-" * 50)
log("Baseline 3: Trivial constant/threshold detector")
log("  Distance of fragment mean from nominal-training mean (in std units)")
log("-" * 50)

# D6: distance of fragment mean from nominal-training mean in training-std units
# Feature 0 is 'mean'; training-only standardisation means we use raw means.
nom_train_grand_mean = float(np.mean(X_fit[:, 0]))  # mean of fragment means
nom_train_grand_std  = float(np.std(X_fit[:, 0]))
if nom_train_grand_std == 0:
    nom_train_grand_std = 1.0

def trivial_score(X_raw: np.ndarray) -> np.ndarray:
    """Absolute z-score of the fragment mean."""
    return np.abs((X_raw[:, 0] - nom_train_grand_mean) / nom_train_grand_std)

triv_dev_scores = trivial_score(X_dev)

triv_threshold = calibrate_threshold(triv_dev_scores, 99.0)
triv_dev_alarms = (triv_dev_scores >= triv_threshold).sum()

log(f"  Nominal training grand mean (fragment means): {nom_train_grand_mean:.6f}")
log(f"  Nominal training grand std  (fragment means): {nom_train_grand_std:.6f}")
log(f"  Calibrated threshold (99th pctile of dev scores): {triv_threshold:.6f}")
log(f"  Dev alarm rate: {triv_dev_alarms}/{len(triv_dev_scores)} "
    f"= {100*triv_dev_alarms/len(triv_dev_scores):.2f}%")
log()


log("=" * 70)
log("OPSSAT-AD SUMMARY TABLE  (threshold + dev alarm rate; no test scored)")
log("=" * 70)
log(f"{'Method':<26} {'Threshold':>12}   {'Dev alarm rate':>18}")
log("-" * 60)
log(f"{'Isolation Forest':<26} {iso_threshold:>12.6f}   "
    f"{iso_dev_alarms}/{len(iso_dev_scores)} ({100*iso_dev_alarms/len(iso_dev_scores):.2f}%)")
log(f"{'PCA recon. error':<26} {pca_threshold:>12.6f}   "
    f"{pca_dev_alarms}/{len(pca_dev_scores)} ({100*pca_dev_alarms/len(pca_dev_scores):.2f}%)")
log(f"{'Trivial (mean dist.)':<26} {triv_threshold:>12.6f}   "
    f"{triv_dev_alarms}/{len(triv_dev_scores)} ({100*triv_dev_alarms/len(triv_dev_scores):.2f}%)")
log()
log("NOTE: Dev partition is NOMINAL ONLY. Precision/recall/F1 are undefined on dev.")
log("      Reported dev alarm rate = false-alarm rate on nominal fragments.")
log("      Per revised Handout §14, no test partition is scored before the results freeze.")
log()


# ===========================================================================
# PART 2 — SMAP/MSL  (calibration only; guard-context windows, revised Annex D6)
# ===========================================================================

log("=" * 70)
log("PART 2: SMAP/MSL Classical Baselines  (P-1 and C-1) — calibration only")
log("=" * 70)
log()

GUARD_STEPS = 256
WINDOW       = 256
TRAIN_STRIDE = 64


def window_features(stream: np.ndarray, col: int = 0,
                    stride: int = TRAIN_STRIDE) -> tuple:
    """
    Extract the 6-element feature vector from windows over a long stream.
    Used for FITTING windows (stride 64). Returns (N_windows, 6) features and
    window-center indices.
    """
    vals = stream[:, col]
    N = len(vals)
    feats, centers = [], []
    i = 0
    while i + WINDOW <= N:
        w = vals[i:i+WINDOW]
        feats.append(feature_vector(w))
        centers.append(i + WINDOW // 2)
        i += stride
    # Cover the tail when stride > 1
    if i < N and stride > 1 and N >= WINDOW:
        w = vals[N-WINDOW:N]
        feats.append(feature_vector(w))
        centers.append(N - WINDOW // 2)
    return np.array(feats), np.array(centers)


def cal_point_windows(train_arr: np.ndarray, col: int,
                      fit_end: int, cal_start: int, cal_end: int) -> np.ndarray:
    """
    Guard-context calibration windows (revised Annex D6):
      - One window per calibration point, right-aligned (the window ends at the
        point, which it scores at its right edge).
      - The window may take earlier context from the GUARD, never from fit rows
        (the left edge is always >= fit_end).
      - No padding is required: the guard supplies left context for the earliest
        calibration points, and every window ends at a calibration point inside
        the training stream.
    Returns (N_cal, 6) feature matrix, one row per calibration point
    (319 for P-1, 176 for C-1).
    """
    vals = train_arr[:, col]
    feats = []
    for p in range(cal_start, cal_end):
        start = p - WINDOW + 1                 # left edge of the window
        assert start >= fit_end, "calibration window must not enter fit rows"
        feats.append(feature_vector(vals[start:p+1]))
    return np.array(feats)


for stream_id, benchmark in [("P-1", "SMAP"), ("C-1", "MSL")]:
    log("-" * 60)
    log(f"Stream: {benchmark} {stream_id}")
    log("-" * 60)

    train_arr = np.load(DATA / f"SMAP_MSL_train_{stream_id}.npy", allow_pickle=False)

    N_train = len(train_arr)
    fit_end   = int(N_train * 0.8)
    cal_start = fit_end + GUARD_STEPS          # guard = [fit_end, cal_start)
    cal_end   = N_train                         # cal  = [cal_start, cal_end)

    fit_stream = train_arr[:fit_end]
    n_cal = cal_end - cal_start

    log(f"  Fit steps : {len(fit_stream)}  [0-{fit_end})")
    log(f"  Guard     : {GUARD_STEPS} steps  [{fit_end}-{cal_start})")
    log(f"  Cal steps : {n_cal}  [{cal_start}-{cal_end})  -> {n_cal} calibration-point scores")
    log(f"  Calibration rule (revised Annex D6): windows take earlier context from the")
    log(f"  guard, never from fit rows; only calibration points are scored.")
    log()

    col = 0

    # Fit windows (stride 64) and fit-only standardisation
    X_fit_s, _ = window_features(fit_stream, col, stride=TRAIN_STRIDE)
    train_mean_s = X_fit_s.mean(axis=0)
    train_std_s  = X_fit_s.std(axis=0)
    train_std_s[train_std_s == 0] = 1.0
    X_fit_sz = (X_fit_s - train_mean_s) / train_std_s

    # Guard-context calibration windows: one per cal point (319 / 176)
    X_cal_s = cal_point_windows(train_arr, col, fit_end, cal_start, cal_end)
    X_cal_sz = (X_cal_s - train_mean_s) / train_std_s
    assert X_cal_s.shape[0] == n_cal, (X_cal_s.shape[0], n_cal)

    log(f"  Fit windows : {X_fit_s.shape[0]} (stride {TRAIN_STRIDE})")
    log(f"  Cal windows : {X_cal_s.shape[0]} (one per cal point, guard-context)")

    # ------------------------------------------------------------------
    # IsolationForest on stream
    # ------------------------------------------------------------------
    log("  [IsolationForest]")
    iso_s = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1)
    iso_s.fit(X_fit_sz)

    cal_iso = -iso_s.decision_function(X_cal_sz)
    iso_thr_s = calibrate_threshold(cal_iso, 99.0)
    iso_alarms = int((cal_iso >= iso_thr_s).sum())
    log(f"    Threshold (99th pctile of cal-point scores): {iso_thr_s:.6f}")
    log(f"    Cal alarm count / rate: {iso_alarms}/{n_cal} = {100*iso_alarms/n_cal:.2f}%")

    # ------------------------------------------------------------------
    # PCA reconstruction on stream
    # ------------------------------------------------------------------
    log("  [PCA reconstruction]")
    pca_s = PCA(random_state=SEED)
    pca_s.fit(X_fit_sz)
    cumvar_s = np.cumsum(pca_s.explained_variance_ratio_)
    nc_s = int(np.searchsorted(cumvar_s, 0.95)) + 1
    nc_s = max(1, min(nc_s, X_fit_sz.shape[1]))
    pca_s = PCA(n_components=nc_s, random_state=SEED)
    pca_s.fit(X_fit_sz)

    def pca_stream_score(X_z):
        X_proj = pca_s.inverse_transform(pca_s.transform(X_z))
        return np.mean((X_z - X_proj)**2, axis=1)

    cal_pca = pca_stream_score(X_cal_sz)
    pca_thr_s = calibrate_threshold(cal_pca, 99.0)
    pca_alarms = int((cal_pca >= pca_thr_s).sum())
    log(f"    Components (>=95% var): {nc_s}")
    log(f"    Threshold (99th pctile of cal-point scores): {pca_thr_s:.6f}")
    log(f"    Cal alarm count / rate: {pca_alarms}/{n_cal} = {100*pca_alarms/n_cal:.2f}%")

    # ------------------------------------------------------------------
    # Trivial detector on stream
    # ------------------------------------------------------------------
    log("  [Trivial detector]")
    nom_grand_mean_s = float(np.mean(X_fit_s[:, 0]))
    nom_grand_std_s  = float(np.std(X_fit_s[:, 0]))
    if nom_grand_std_s == 0:
        nom_grand_std_s = 1.0

    def trivial_stream_score(X_raw):
        return np.abs((X_raw[:, 0] - nom_grand_mean_s) / nom_grand_std_s)

    cal_triv = trivial_stream_score(X_cal_s)
    triv_thr_s = calibrate_threshold(cal_triv, 99.0)
    triv_alarms = int((cal_triv >= triv_thr_s).sum())
    log(f"    Threshold (99th pctile of cal-point scores): {triv_thr_s:.6f}")
    log(f"    Cal alarm count / rate: {triv_alarms}/{n_cal} = {100*triv_alarms/n_cal:.2f}%")
    log()


log("=" * 70)
log("NOTES")
log("=" * 70)
log("1. OPSSAT-AD dev partition is NOMINAL ONLY — F1 is undefined there.")
log("   Dev alarm rates above represent FALSE ALARM RATES on nominal data.")
log("2. SMAP/MSL calibration uses guard-context windows (revised Annex D6): a")
log("   calibration window may take earlier context from the guard, never from")
log("   fit rows, and only calibration points are scored. P-1 yields 319")
log("   calibration-point scores; C-1 yields 176. No boundary padding is needed")
log("   (the guard supplies left context; every window ends at a cal point).")
log("3. Thresholds are the 99th percentile of nominal dev (OPSSAT) / cal-point")
log("   (SMAP/MSL) scores, computed after window scores are reduced to point")
log("   scores (one window per point here, so the reduction is 1:1).")
log("4. Per revised Handout §14, NO test partition is scored before the results")
log("   freeze. Earlier test-scored runs are recorded (test_scored=yes) in")
log("   T05_I1_runlog.csv; test metrics are NOT recomputed here.")
log("5. SMAP/MSL data is pre-scaled using test-set min/max (inherited limitation).")
log()
log("Baselines complete.")

# Write log
with open(LOG, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults saved to: {LOG}")
