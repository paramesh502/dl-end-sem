"""
run_baselines.py
================
Implements and evaluates the three classical baselines required by Card I1,
Annex D6, on OPSSAT-AD and SMAP/MSL (P-1 and C-1).

Baselines:
  1. Isolation Forest
  2. PCA reconstruction error
  3. Trivial constant/threshold detector
     (also reports constant-all-nominal F1 as a floor)

OPSSAT-AD evaluation:
  - Nominal development data only (no anomalous-class labels).
  - Reports calibrated threshold and nominal-development alarm rate.
  - Does NOT report precision/recall/F1 on dev (those require both classes).
  - Reports precision/recall/F1 on the TEST set.

SMAP/MSL evaluation:
  - Per-stream unadjusted point precision/recall/F1 (primary).
  - Point-adjusted F1 (audit figure, clearly labelled).

Run from project root:
    python3 scripts/run_baselines.py

Results saved to: results/baseline_results.txt
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

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


def point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray,
                       anomaly_intervals: list[tuple[int, int]]) -> float:
    """
    Point-adjusted F1 (audit figure, D6):
    If any predicted point intersects a labelled anomaly interval,
    fill the entire true interval as detected. Predictions outside
    intervals are unchanged.
    """
    y_adj = y_pred.copy()
    for start, end in anomaly_intervals:
        if np.any(y_pred[start:end+1] == 1):
            y_adj[start:end+1] = 1
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_adj, average="binary",
                                                   zero_division=0)
    return float(p), float(r), float(f1)


# ===========================================================================
# PART 1 — OPSSAT-AD
# ===========================================================================

log("=" * 70)
log("PART 1: OPSSAT-AD Classical Baselines")
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

# Build feature matrices
nominal_fit_ids = set(frag[(frag["anomaly"]==0) & (frag["train"]=="1") &
                           (~frag["segment"].isin(dev_ids))]["segment"])
nominal_dev_ids = dev_ids  # all nominal
test_ids        = set(frag[frag["train"]!="1"]["segment"])

log(f"OPSSAT-AD split counts:")
log(f"  Nominal fitting fragments :  {len(nominal_fit_ids)}")
log(f"  Nominal development frags :  {len(nominal_dev_ids)}")
log(f"  Test fragments            :  {len(test_ids)}")
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

# Re-extract with imputation
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
X_test, test_ids_list = extract_imputed(test_ids)

# Training-only standardisation (mean/std from fitting data)
train_mean_feat = X_fit.mean(axis=0)
train_std_feat  = X_fit.std(axis=0)
train_std_feat[train_std_feat == 0] = 1.0

X_fit_z  = (X_fit  - train_mean_feat) / train_std_feat
X_dev_z  = (X_dev  - train_mean_feat) / train_std_feat
X_test_z = (X_test - train_mean_feat) / train_std_feat

# Ground-truth labels for test fragments
test_labels = [int(frag[frag["segment"]==sid]["anomaly"].values[0])
               for sid in test_ids_list]
y_test = np.array(test_labels, dtype=int)

log(f"Test set: {len(y_test)} fragments | {y_test.sum()} anomalous | {(y_test==0).sum()} nominal")
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
iso_dev_scores  = -iso.decision_function(X_dev_z)
iso_test_scores = -iso.decision_function(X_test_z)

iso_threshold = calibrate_threshold(iso_dev_scores, 99.0)
iso_dev_alarms  = (iso_dev_scores  >= iso_threshold).sum()
iso_test_pred   = (iso_test_scores >= iso_threshold).astype(int)

p, r, f1, _ = precision_recall_fscore_support(y_test, iso_test_pred, average="binary", zero_division=0)

log(f"  Calibrated threshold (99th pctile of dev scores): {iso_threshold:.6f}")
log(f"  Dev alarm rate: {iso_dev_alarms}/{len(iso_dev_scores)} "
    f"= {100*iso_dev_alarms/len(iso_dev_scores):.2f}%")
log(f"  Test precision: {p:.4f} | recall: {r:.4f} | F1: {f1:.4f}")
log(f"  Test support:   positives={y_test.sum()} negatives={(y_test==0).sum()}")
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

pca_dev_scores  = pca_score(X_dev_z)
pca_test_scores = pca_score(X_test_z)

pca_threshold = calibrate_threshold(pca_dev_scores, 99.0)
pca_dev_alarms = (pca_dev_scores >= pca_threshold).sum()
pca_test_pred  = (pca_test_scores >= pca_threshold).astype(int)

p2, r2, f2, _ = precision_recall_fscore_support(y_test, pca_test_pred, average="binary", zero_division=0)

log(f"  PCA components retained: {n_components} (explaining >= 95% variance)")
log(f"  Calibrated threshold (99th pctile of dev scores): {pca_threshold:.6f}")
log(f"  Dev alarm rate: {pca_dev_alarms}/{len(pca_dev_scores)} "
    f"= {100*pca_dev_alarms/len(pca_dev_scores):.2f}%")
log(f"  Test precision: {p2:.4f} | recall: {r2:.4f} | F1: {f2:.4f}")
log(f"  Test support:   positives={y_test.sum()} negatives={(y_test==0).sum()}")
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

triv_dev_scores  = trivial_score(X_dev)
triv_test_scores = trivial_score(X_test)

triv_threshold = calibrate_threshold(triv_dev_scores, 99.0)
triv_dev_alarms = (triv_dev_scores >= triv_threshold).sum()
triv_test_pred  = (triv_test_scores >= triv_threshold).astype(int)

p3, r3, f3, _ = precision_recall_fscore_support(y_test, triv_test_pred, average="binary", zero_division=0)

# Constant-all-nominal floor
all_nominal_pred = np.zeros(len(y_test), dtype=int)
p_floor, r_floor, f_floor, _ = precision_recall_fscore_support(
    y_test, all_nominal_pred, average="binary", zero_division=0)

log(f"  Nominal training grand mean (fragment means): {nom_train_grand_mean:.6f}")
log(f"  Nominal training grand std  (fragment means): {nom_train_grand_std:.6f}")
log(f"  Calibrated threshold (99th pctile of dev scores): {triv_threshold:.6f}")
log(f"  Dev alarm rate: {triv_dev_alarms}/{len(triv_dev_scores)} "
    f"= {100*triv_dev_alarms/len(triv_dev_scores):.2f}%")
log(f"  Test precision: {p3:.4f} | recall: {r3:.4f} | F1: {f3:.4f}")
log(f"  Test support:   positives={y_test.sum()} negatives={(y_test==0).sum()}")
log()
log("  Constant-all-nominal floor (predict all normal):")
log(f"    precision={p_floor:.4f} | recall={r_floor:.4f} | F1={f_floor:.4f}")
log()


log("=" * 70)
log("OPSSAT-AD SUMMARY TABLE")
log("=" * 70)
log(f"{'Method':<30} {'Dev Alarm Rate':>16} {'Test P':>8} {'Test R':>8} {'Test F1':>8}")
log("-" * 75)
log(f"{'Isolation Forest':<30} "
    f"{iso_dev_alarms}/{len(iso_dev_scores)} ({100*iso_dev_alarms/len(iso_dev_scores):.1f}%)".rjust(16) + " "
    f"{p:.4f}   {r:.4f}   {f1:.4f}")
log(f"{'PCA recon. error':<30} "
    f"{pca_dev_alarms}/{len(pca_dev_scores)} ({100*pca_dev_alarms/len(pca_dev_scores):.1f}%)".rjust(16) + " "
    f"{p2:.4f}   {r2:.4f}   {f2:.4f}")
log(f"{'Trivial (mean dist.)':<30} "
    f"{triv_dev_alarms}/{len(triv_dev_scores)} ({100*triv_dev_alarms/len(triv_dev_scores):.1f}%)".rjust(16) + " "
    f"{p3:.4f}   {r3:.4f}   {f3:.4f}")
log(f"{'Constant-all-nominal (floor)':<30} {'N/A':>16}  "
    f"{p_floor:.4f}   {r_floor:.4f}   {f_floor:.4f}")
log()
log("NOTE: Dev partition is NOMINAL ONLY. Precision/recall/F1 are undefined on dev.")
log("      Reported dev alarm rate = false-alarm rate on nominal fragments.")
log()


# ===========================================================================
# PART 2 — SMAP/MSL
# ===========================================================================

log("=" * 70)
log("PART 2: SMAP/MSL Classical Baselines  (P-1 and C-1)")
log("=" * 70)
log()

labels_csv = pd.read_csv(DATA / "labeled_anomalies.csv")
GUARD_STEPS = 256
WINDOW      = 256
TRAIN_STRIDE = 64
SCORE_STRIDE  = 1


def get_anomaly_intervals(stream_name, n_test):
    """Return list of (start_idx, end_idx) anomaly intervals for a stream."""
    row = labels_csv[labels_csv["chan_id"] == stream_name]
    if len(row) == 0:
        return []
    intervals_str = row.iloc[0]["anomaly_sequences"]
    import ast
    intervals = ast.literal_eval(intervals_str)
    return [(int(s), int(e)) for s, e in intervals]


def intervals_to_labels(n_points, intervals):
    y = np.zeros(n_points, dtype=int)
    for s, e in intervals:
        y[s:e+1] = 1
    return y


def window_features(stream: np.ndarray, col: int = 0,
                    stride: int = TRAIN_STRIDE) -> tuple:
    """
    Extract the 6-element feature vector from windows.
    stride=TRAIN_STRIDE (64) for training/calibration; stride=1 for scoring.
    If the stream is shorter than WINDOW, extract a single window from
    the full stream (padded if necessary) so calibration never returns empty.
    Returns:
        X_windows : (N_windows, 6) feature matrix
        window_centers : (N_windows,) center index of each window
    """
    vals = stream[:, col]
    N = len(vals)
    feats, centers = [], []

    if N < WINDOW:
        # Pad with edge values so we can still compute one window
        pad_right = WINDOW - N
        vals_padded = np.pad(vals, (0, pad_right), mode="edge")
        feats.append(feature_vector(vals_padded[:WINDOW]))
        centers.append(N // 2)
    else:
        i = 0
        while i + WINDOW <= N:
            w = vals[i:i+WINDOW]
            feats.append(feature_vector(w))
            centers.append(i + WINDOW // 2)
            i += stride
        # Ensure we cover the tail if stride > 1
        if i < N and stride > 1:
            w = vals[N-WINDOW:N]
            feats.append(feature_vector(w))
            centers.append(N - WINDOW // 2)

    return np.array(feats), np.array(centers)


def score_stream_by_windows(stream: np.ndarray, model_fn,
                              train_mean_f, train_std_f,
                              col: int = 0) -> np.ndarray:
    """
    Score every point in stream by mean reconstruction error of covering windows
    (stride 1 scoring as per D6, approximated by stride 1 with padding masked).
    """
    vals = stream[:, col]
    N    = len(vals)
    # Pad to handle prefix/suffix
    pad  = WINDOW // 2
    padded = np.pad(vals, (pad, pad), mode="edge")

    scores = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)

    for center in range(N):
        start_pad = center   # in padded array, center+pad corresponds to center
        # extract window centred on this point
        w = padded[start_pad: start_pad + WINDOW]
        feat = feature_vector(w, {})
        feat_z = (feat - train_mean_f) / train_std_f
        score = float(model_fn(feat_z.reshape(1, -1)))
        scores[center] = score
        counts[center] += 1

    # Average (counts==1 everywhere after this simplified pass)
    return scores / np.maximum(counts, 1)


for stream_id, benchmark in [("P-1", "SMAP"), ("C-1", "MSL")]:
    log("-" * 60)
    log(f"Stream: {benchmark} {stream_id}")
    log("-" * 60)

    train_arr = np.load(DATA / f"SMAP_MSL_train_{stream_id}.npy", allow_pickle=False)
    test_arr  = np.load(DATA / f"SMAP_MSL_test_{stream_id}.npy",  allow_pickle=False)

    N_train = len(train_arr)
    N_test  = len(test_arr)
    fit_end   = int(N_train * 0.8)
    cal_start = fit_end + GUARD_STEPS

    fit_stream = train_arr[:fit_end]
    cal_stream = train_arr[cal_start:]

    # Anomaly intervals and labels for test set
    intervals  = get_anomaly_intervals(stream_id, N_test)
    y_test_stream = intervals_to_labels(N_test, intervals)
    n_anomalous = y_test_stream.sum()

    log(f"  Fit steps: {len(fit_stream)} | Cal steps: {len(cal_stream)} | "
        f"Test steps: {N_test}")
    log(f"  Anomalous test points: {n_anomalous} ({100*n_anomalous/N_test:.2f}%)")
    log(f"  Anomaly intervals ({len(intervals)}): {intervals}")

    # Use column 0 as sensor input (per D6)
    col = 0

    # ------------------------------------------------------------------
    # Extract training windows (stride 64) from fit stream
    # ------------------------------------------------------------------
    X_fit_s, _ = window_features(fit_stream, col, stride=TRAIN_STRIDE)

    train_mean_s = X_fit_s.mean(axis=0)
    train_std_s  = X_fit_s.std(axis=0)
    train_std_s[train_std_s == 0] = 1.0

    X_fit_sz = (X_fit_s - train_mean_s) / train_std_s

    # ------------------------------------------------------------------
    # Extract calibration windows
    # ------------------------------------------------------------------
    X_cal_s, _ = window_features(cal_stream, col, stride=TRAIN_STRIDE)
    X_cal_sz   = (X_cal_s - train_mean_s) / train_std_s

    # ------------------------------------------------------------------
    # IsolationForest on stream
    # ------------------------------------------------------------------
    log("  [IsolationForest]")
    iso_s = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1)
    iso_s.fit(X_fit_sz)

    # Score calibration windows then test windows
    X_test_s, test_centers = window_features(test_arr, col, stride=SCORE_STRIDE)
    X_test_sz = (X_test_s - train_mean_s) / train_std_s

    cal_scores = -iso_s.decision_function(X_cal_sz)
    test_window_scores = -iso_s.decision_function(X_test_sz)

    iso_thr_s = calibrate_threshold(cal_scores, 99.0)

    # Map window scores to point scores by averaging over covering windows
    point_scores = np.full(N_test, np.nan)
    for i, center in enumerate(test_centers):
        if np.isnan(point_scores[center]):
            point_scores[center] = test_window_scores[i]
        else:
            point_scores[center] = (point_scores[center] + test_window_scores[i]) / 2

    # Fill NaNs (prefix/suffix) with median
    nan_mask = np.isnan(point_scores)
    if nan_mask.any():
        point_scores[nan_mask] = float(np.nanmedian(point_scores))

    iso_test_pred_s = (point_scores >= iso_thr_s).astype(int)

    p_s, r_s, f1_s, _ = precision_recall_fscore_support(
        y_test_stream, iso_test_pred_s, average="binary", zero_division=0)
    pa_p, pa_r, pa_f1 = point_adjusted_f1(y_test_stream, iso_test_pred_s, intervals)

    log(f"    Threshold (99th pctile of cal): {iso_thr_s:.6f}")
    log(f"    Unadj. point  P={p_s:.4f} R={r_s:.4f} F1={f1_s:.4f}")
    log(f"    Pt-adj (AUDIT) P={pa_p:.4f} R={pa_r:.4f} F1={pa_f1:.4f}")

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

    cal_scores_pca = pca_stream_score(X_cal_sz)
    pca_thr_s = calibrate_threshold(cal_scores_pca, 99.0)

    test_win_scores_pca = pca_stream_score(X_test_sz)
    pt_scores_pca = np.full(N_test, np.nan)
    for i, center in enumerate(test_centers):
        if np.isnan(pt_scores_pca[center]):
            pt_scores_pca[center] = test_win_scores_pca[i]
        else:
            pt_scores_pca[center] = (pt_scores_pca[center] + test_win_scores_pca[i]) / 2
    if np.isnan(pt_scores_pca).any():
        pt_scores_pca[np.isnan(pt_scores_pca)] = float(np.nanmedian(pt_scores_pca))

    pca_pred_s = (pt_scores_pca >= pca_thr_s).astype(int)
    p_p, r_p, f1_p, _ = precision_recall_fscore_support(
        y_test_stream, pca_pred_s, average="binary", zero_division=0)
    pa_p2, pa_r2, pa_f12 = point_adjusted_f1(y_test_stream, pca_pred_s, intervals)

    log(f"    Components (>=95% var): {nc_s}")
    log(f"    Threshold: {pca_thr_s:.6f}")
    log(f"    Unadj. point  P={p_p:.4f} R={r_p:.4f} F1={f1_p:.4f}")
    log(f"    Pt-adj (AUDIT) P={pa_p2:.4f} R={pa_r2:.4f} F1={pa_f12:.4f}")

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

    test_triv = trivial_stream_score(X_test_s)
    pt_triv = np.full(N_test, np.nan)
    for i, center in enumerate(test_centers):
        if np.isnan(pt_triv[center]):
            pt_triv[center] = test_triv[i]
        else:
            pt_triv[center] = (pt_triv[center] + test_triv[i]) / 2
    if np.isnan(pt_triv).any():
        pt_triv[np.isnan(pt_triv)] = float(np.nanmedian(pt_triv))

    triv_pred_s = (pt_triv >= triv_thr_s).astype(int)
    p_t, r_t, f1_t, _ = precision_recall_fscore_support(
        y_test_stream, triv_pred_s, average="binary", zero_division=0)
    pa_p3, pa_r3, pa_f13 = point_adjusted_f1(y_test_stream, triv_pred_s, intervals)

    log(f"    Threshold: {triv_thr_s:.6f}")
    log(f"    Unadj. point  P={p_t:.4f} R={r_t:.4f} F1={f1_t:.4f}")
    log(f"    Pt-adj (AUDIT) P={pa_p3:.4f} R={pa_r3:.4f} F1={pa_f13:.4f}")
    log()


log("=" * 70)
log("NOTES")
log("=" * 70)
log("1. OPSSAT-AD dev partition is NOMINAL ONLY — F1 is undefined there.")
log("   Dev alarm rates above represent FALSE ALARM RATES on nominal data.")
log("2. SMAP/MSL point-adjusted F1 is an AUDIT FIGURE (labelled as such).")
log("   Unadjusted point F1 is the primary metric.")
log("3. All thresholds calibrated at 99th percentile of nominal dev/cal scores.")
log("4. No test labels or test metrics were used for threshold selection.")
log("5. SMAP/MSL data is pre-scaled using test-set min/max (inherited limitation).")
log()
log("Baselines complete.")

# Write log
with open(LOG, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nResults saved to: {LOG}")
