"""
smap_msl_final_eval.py
======================
Final SMAP/MSL test evaluation with LOCKED thresholds.
Team 05, Card I1

Re-fits the Conv1dAutoencoder on SMAP P-1 and MSL C-1 from scratch using
the EXACT corrected covering-window scoring protocol (seed=22, deterministic).
Scores each test stream once, applies locked thresholds, reports unadjusted
and point-adjusted F1 for neural + 3 classical baselines.

Locked thresholds (from smap_msl_results.txt — NOT recomputed):
  P-1: 1.995969
  C-1: 0.513500

Run from project root:
    source .venv/bin/activate
    python3 scripts/smap_msl_final_eval.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Constants (must match week4_smap_msl.py exactly)
# ---------------------------------------------------------------------------
SEED          = 22
MAX_EPOCHS    = 30
MAX_SECONDS   = 1800
BATCH_SIZE    = 32
LR            = 1e-3
EARLY_STOP    = 5
THRESHOLD_PCT = 99.0
WINDOW        = 256
TRAIN_STRIDE  = 64
GUARD         = 256

# Locked thresholds — do NOT recompute
LOCKED_THRESHOLDS = {
    "P-1": 1.995969,
    "C-1": 0.513500,
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Anomaly ground truth (from labeled_anomalies.csv)
# intervals: inclusive start, exclusive end
# ---------------------------------------------------------------------------
ANOMALY_INTERVALS = {
    "P-1": [(2149, 2349), (4536, 4844), (3539, 3779)],
    "C-1": [(550, 750), (2100, 2210)],
}


def build_labels(T: int, intervals):
    """Build binary label array of length T from (start, end) intervals."""
    labels = np.zeros(T, dtype=np.int32)
    for start, end in intervals:
        labels[start:end] = 1
    return labels


# ===========================================================================
# Model — identical to week4_smap_msl.py
# ===========================================================================
class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_windows(series: np.ndarray, window: int, stride: int) -> np.ndarray:
    T = len(series)
    starts = np.arange(0, T - window + 1, stride)
    return np.stack([series[s:s + window] for s in starts], axis=0)


def cal_point_mse_neural(model: nn.Module, series_z: np.ndarray,
                          fit_end: int, cal_start: int, cal_end: int,
                          window: int = WINDOW) -> np.ndarray:
    """
    Score each point in [cal_start, cal_end) by mean squared reconstruction
    error AT THAT POINT, averaged over all covering windows.
    Windows start in [fit_end, cal_end); guard context allowed, fit rows never.
    """
    model.eval()
    cal_scores = np.zeros(cal_end - cal_start, dtype=np.float64)
    cal_counts = np.zeros(cal_end - cal_start, dtype=np.int64)

    with torch.no_grad():
        for s in range(fit_end, cal_end):
            w_end = s + window
            if w_end > len(series_z):
                break
            win   = torch.tensor(series_z[s:w_end][None, None, :], dtype=torch.float32)
            recon = model(win)
            err_per_pos = ((recon - win) ** 2).squeeze().numpy()  # (256,)
            cov_start = max(cal_start, s)
            cov_end   = min(cal_end, s + window)
            # Vectorised accumulation over covered cal points
            rel_positions = np.arange(cov_start - s, cov_end - s)
            cal_indices   = np.arange(cov_start - cal_start, cov_end - cal_start)
            cal_scores[cal_indices] += err_per_pos[rel_positions]
            cal_counts[cal_indices] += 1

    assert (cal_counts > 0).all(), "Some cal point was not covered"
    return (cal_scores / cal_counts).astype(np.float32)


def test_point_mse_neural(model: nn.Module, series_z: np.ndarray,
                           window: int = WINDOW) -> np.ndarray:
    """
    Score each test point by the mean squared reconstruction error AT THAT POINT
    averaged over all covering windows [s, s+window) where s in [0, T-window].

    Points with no covering window (t < window-1 and stream too short) fall back
    to window [0, window) at position t.

    Vectorised: accumulate per-position errors using numpy slice assignment.
    """
    T = len(series_z)
    scores = np.zeros(T, dtype=np.float64)
    counts = np.zeros(T, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for s in range(0, T - window + 1):
            win   = torch.tensor(series_z[s:s + window][None, None, :], dtype=torch.float32)
            recon = model(win)
            err_per_pos = ((recon - win) ** 2).squeeze().numpy()  # (256,)
            scores[s:s + window] += err_per_pos
            counts[s:s + window] += 1

    no_cover = counts == 0
    if no_cover.any():
        win   = torch.tensor(series_z[:window][None, None, :], dtype=torch.float32)
        with torch.no_grad():
            recon = model(win)
        err_per_pos = ((recon - win) ** 2).squeeze().numpy()
        for t in np.where(no_cover)[0]:
            scores[t] = float(err_per_pos[min(t, window - 1)])
            counts[t] = 1

    return (scores / np.maximum(counts, 1)).astype(np.float32)


def test_point_scores_classical(detector_score_fn,
                                 series_z: np.ndarray,
                                 window: int = WINDOW) -> np.ndarray:
    """
    Covering-window mean for classical detectors on the test stream.
    For each window [s, s+window), compute detector score, accumulate
    into every point that window covers.
    Each point's score is the mean of all window scores that cover it.
    """
    T = len(series_z)
    scores = np.zeros(T, dtype=np.float64)
    counts = np.zeros(T, dtype=np.int64)

    for s in range(0, T - window + 1):
        win_score = detector_score_fn(series_z[s:s + window])
        scores[s:s + window] += win_score
        counts[s:s + window] += 1

    no_cover = counts == 0
    if no_cover.any():
        win_score = detector_score_fn(series_z[:window])
        for t in np.where(no_cover)[0]:
            scores[t] = win_score
            counts[t] = 1

    return (scores / np.maximum(counts, 1)).astype(np.float64)


def cal_point_scores_classical(detector_score_fn,
                                series_z: np.ndarray,
                                fit_end: int, cal_start: int, cal_end: int,
                                window: int = WINDOW) -> np.ndarray:
    """Covering-window mean for classical detectors on calibration portion."""
    cal_scores = np.zeros(cal_end - cal_start, dtype=np.float64)
    cal_counts = np.zeros(cal_end - cal_start, dtype=np.int64)

    for s in range(fit_end, cal_end):
        w_end = s + window
        if w_end > len(series_z):
            break
        win_score = detector_score_fn(series_z[s:w_end])
        cov_start = max(cal_start, s)
        cov_end   = min(cal_end, s + window)
        cal_start_idx = cov_start - cal_start
        cal_end_idx   = cov_end   - cal_start
        cal_scores[cal_start_idx:cal_end_idx] += win_score
        cal_counts[cal_start_idx:cal_end_idx] += 1

    assert (cal_counts > 0).all(), "Some cal point uncovered"
    return (cal_scores / cal_counts).astype(np.float64)


def point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray, intervals):
    """
    Point-adjusted prediction: for each contiguous anomalous interval,
    if ANY predicted point is 1, set all predictions in that interval to 1.
    Returns adjusted precision, recall, f1.
    """
    y_adj = y_pred.copy()
    for start, end in intervals:
        if y_pred[start:end].any():
            y_adj[start:end] = 1
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_adj, average="binary", zero_division=0)
    return p, r, f


# ===========================================================================
# Fit one stream: returns model, mu, sig, train_z, fit_end, cal_start, cal_end, win_fit
# ===========================================================================
def fit_stream(name: str, train_npy: Path, log_lines: list):
    print(f"\n{'='*65}")
    print(f"  FIT: {name}")
    print(f"{'='*65}")
    log_lines.append(f"\n{'='*65}")
    log_lines.append(f"  FIT: {name}")
    log_lines.append(f"{'='*65}")

    train_raw = np.load(train_npy)[:, 0].astype(np.float32)
    T_train   = len(train_raw)

    fit_end   = int(0.8 * T_train)
    cal_start = fit_end + GUARD
    cal_end   = T_train
    n_cal     = cal_end - cal_start

    mu  = float(train_raw[:fit_end].mean())
    sig = float(train_raw[:fit_end].std())
    if sig == 0.0:
        sig = 1.0
    train_z = (train_raw - mu) / sig

    win_fit = extract_windows(train_z[:fit_end], WINDOW, TRAIN_STRIDE)

    msg = (f"  T_train={T_train}, fit=[0,{fit_end}), guard=[{fit_end},{cal_start}), "
           f"cal=[{cal_start},{cal_end}) n_cal={n_cal}")
    print(msg)
    log_lines.append(msg)
    log_lines.append(f"  mu={mu:.6f}  sig={sig:.6f}  n_fit_win={len(win_fit)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = Conv1dAutoencoder()

    class WindowDataset(Dataset):
        def __init__(self, w):
            self.w = torch.tensor(w[:, None, :], dtype=torch.float32)
        def __len__(self):  return len(self.w)
        def __getitem__(self, i): return self.w[i]

    train_loader = DataLoader(WindowDataset(win_fit), batch_size=BATCH_SIZE, shuffle=True)
    optimizer    = torch.optim.Adam(model.parameters(), lr=LR)
    criterion    = nn.MSELoss()
    best_loss    = float("inf")
    best_state   = None
    no_improve   = 0
    t0           = time.perf_counter()

    print("  Training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        if time.perf_counter() - t0 >= MAX_SECONDS:
            print(f"  Time limit at epoch {epoch}")
            break
        model.train()
        tr_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * len(batch)
        tr_loss = tr_sum / len(win_fit)

        val_scores = cal_point_mse_neural(model, train_z, fit_end, cal_start, cal_end)
        val_loss   = float(val_scores.mean())
        print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  train={tr_loss:.6f}  "
              f"val_cal_mean={val_loss:.6f}  elapsed={time.perf_counter()-t0:.1f}s")

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.perf_counter() - t0
    print(f"  Fit complete in {elapsed:.1f}s, best_val_cal_mean={best_loss:.6f}")
    log_lines.append(f"  Fit elapsed: {elapsed:.1f}s, best_val_cal_mean={best_loss:.6f}")

    return model, mu, sig, train_z, fit_end, cal_start, cal_end, win_fit


# ===========================================================================
# Evaluate one stream (neural + 3 classical)
# ===========================================================================
def evaluate_stream(stream_id: str, train_npy: Path, test_npy: Path,
                    log_lines: list, out_lines: list):

    locked_thr = LOCKED_THRESHOLDS[stream_id]
    intervals  = ANOMALY_INTERVALS[stream_id]

    # Fit neural model
    model, mu, sig, train_z, fit_end, cal_start, cal_end, win_fit = \
        fit_stream(stream_id, train_npy, log_lines)

    # Load & standardise test stream with SAME mu/sig
    test_raw = np.load(test_npy)[:, 0].astype(np.float32)
    T_test   = len(test_raw)
    test_z   = (test_raw - mu) / sig

    y_true = build_labels(T_test, intervals)
    n_anom = int(y_true.sum())
    print(f"\n  Test length: {T_test}, anomalous points: {n_anom}")
    log_lines.append(f"  Test length: {T_test}, anomalous points: {n_anom}")

    out_lines.append(f"\n{'='*65}")
    out_lines.append(f"  STREAM: {stream_id}")
    out_lines.append(f"{'='*65}")
    out_lines.append(f"  Test length          : {T_test}")
    out_lines.append(f"  Anomalous points     : {n_anom}")
    out_lines.append(f"  Anomaly intervals    : {intervals}")

    # -----------------------------------------------------------------------
    # 1. Neural model
    # -----------------------------------------------------------------------
    print(f"\n  Scoring test stream with neural model (covering-window mean)...")
    t0 = time.perf_counter()
    test_scores_neural = test_point_mse_neural(model, test_z)
    elapsed_score = time.perf_counter() - t0
    assert np.all(np.isfinite(test_scores_neural)), \
        "Neural test scores contain non-finite values"

    y_pred_neural = (test_scores_neural >= locked_thr).astype(np.int32)
    n_alarms = int(y_pred_neural.sum())
    print(f"  Neural scoring done in {elapsed_score:.1f}s. "
          f"Alarms: {n_alarms}/{T_test} = {100*y_pred_neural.mean():.2f}%")

    p_adj_n, r_adj_n, f1_adj_n = point_adjusted_f1(y_true, y_pred_neural, intervals)
    report_neural = classification_report(y_true, y_pred_neural, digits=4, zero_division=0)
    _, _, f1_neural, _ = precision_recall_fscore_support(
        y_true, y_pred_neural, average="binary", zero_division=0)

    out_lines.append(f"\n  --- Neural Autoencoder (R11/R12) ---")
    out_lines.append(f"  Locked threshold (P99 cal, NOT recomputed): {locked_thr:.6f}")
    out_lines.append(f"  Test alarms: {n_alarms}/{T_test} = {100*y_pred_neural.mean():.2f}%")
    out_lines.append(f"\n  [PRIMARY] Unadjusted point precision/recall/F1:")
    for ln in report_neural.splitlines():
        out_lines.append("    " + ln)
    out_lines.append(f"\n  [AUDIT ONLY — not used for tuning] Point-adjusted F1:")
    out_lines.append(
        f"    Precision={p_adj_n:.4f}  Recall={r_adj_n:.4f}  F1={f1_adj_n:.4f}")

    # -----------------------------------------------------------------------
    # 2. Isolation Forest
    # -----------------------------------------------------------------------
    print("  Fitting IsolationForest...")
    iso_clf = IsolationForest(n_estimators=100, random_state=SEED)
    iso_clf.fit(win_fit)

    def iso_score_fn(w):
        return float(-iso_clf.score_samples(w.reshape(1, -1))[0])

    print("  Calibrating IsolationForest threshold on cal portion...")
    iso_cal = cal_point_scores_classical(iso_score_fn, train_z, fit_end, cal_start, cal_end)
    iso_thr = float(np.percentile(iso_cal, THRESHOLD_PCT))
    print(f"  IsoForest cal threshold (P99): {iso_thr:.6f}")

    print("  Scoring test stream with IsolationForest...")
    iso_test = test_point_scores_classical(iso_score_fn, test_z)
    assert np.all(np.isfinite(iso_test)), "IsoForest scores contain non-finite values"
    y_pred_iso = (iso_test >= iso_thr).astype(np.int32)

    p_adj_i, r_adj_i, f1_adj_i = point_adjusted_f1(y_true, y_pred_iso, intervals)
    report_iso = classification_report(y_true, y_pred_iso, digits=4, zero_division=0)
    _, _, f1_iso, _ = precision_recall_fscore_support(
        y_true, y_pred_iso, average="binary", zero_division=0)

    out_lines.append(f"\n  --- Isolation Forest (n_estimators=100, random_state=22) ---")
    out_lines.append(
        f"  Calibrated threshold (P99 cal covering-window mean): {iso_thr:.6f}")
    out_lines.append(
        f"  Test alarms: {int(y_pred_iso.sum())}/{T_test} = {100*y_pred_iso.mean():.2f}%")
    out_lines.append(f"\n  [PRIMARY] Unadjusted point precision/recall/F1:")
    for ln in report_iso.splitlines():
        out_lines.append("    " + ln)
    out_lines.append(f"\n  [AUDIT ONLY — not used for tuning] Point-adjusted F1:")
    out_lines.append(
        f"    Precision={p_adj_i:.4f}  Recall={r_adj_i:.4f}  F1={f1_adj_i:.4f}")

    # -----------------------------------------------------------------------
    # 3. PCA reconstruction
    # -----------------------------------------------------------------------
    print("  Fitting PCA (n_components=4)...")
    pca = PCA(n_components=4)
    pca.fit(win_fit)

    def pca_score_fn(w):
        w2    = w.reshape(1, -1)
        recon = pca.inverse_transform(pca.transform(w2))
        return float(np.mean((w2 - recon) ** 2))

    print("  Calibrating PCA threshold on cal portion...")
    pca_cal = cal_point_scores_classical(pca_score_fn, train_z, fit_end, cal_start, cal_end)
    pca_thr = float(np.percentile(pca_cal, THRESHOLD_PCT))
    print(f"  PCA cal threshold (P99): {pca_thr:.6f}")

    print("  Scoring test stream with PCA...")
    pca_test = test_point_scores_classical(pca_score_fn, test_z)
    assert np.all(np.isfinite(pca_test)), "PCA scores contain non-finite values"
    y_pred_pca = (pca_test >= pca_thr).astype(np.int32)

    p_adj_p, r_adj_p, f1_adj_p = point_adjusted_f1(y_true, y_pred_pca, intervals)
    report_pca = classification_report(y_true, y_pred_pca, digits=4, zero_division=0)
    _, _, f1_pca, _ = precision_recall_fscore_support(
        y_true, y_pred_pca, average="binary", zero_division=0)

    out_lines.append(f"\n  --- PCA Reconstruction (n_components=4) ---")
    out_lines.append(
        f"  Calibrated threshold (P99 cal covering-window mean): {pca_thr:.6f}")
    out_lines.append(
        f"  Test alarms: {int(y_pred_pca.sum())}/{T_test} = {100*y_pred_pca.mean():.2f}%")
    out_lines.append(f"\n  [PRIMARY] Unadjusted point precision/recall/F1:")
    for ln in report_pca.splitlines():
        out_lines.append("    " + ln)
    out_lines.append(f"\n  [AUDIT ONLY — not used for tuning] Point-adjusted F1:")
    out_lines.append(
        f"    Precision={p_adj_p:.4f}  Recall={r_adj_p:.4f}  F1={f1_adj_p:.4f}")

    # -----------------------------------------------------------------------
    # 4. Trivial detector: |z-score of window mean|
    # -----------------------------------------------------------------------
    fit_mean = float(train_z[:fit_end].mean())
    fit_std  = float(train_z[:fit_end].std())
    if fit_std == 0.0:
        fit_std = 1.0

    def trivial_score_fn(w):
        return float(abs((w.mean() - fit_mean) / fit_std))

    print("  Calibrating Trivial threshold on cal portion...")
    triv_cal = cal_point_scores_classical(trivial_score_fn, train_z, fit_end, cal_start, cal_end)
    triv_thr = float(np.percentile(triv_cal, THRESHOLD_PCT))
    print(f"  Trivial cal threshold (P99): {triv_thr:.6f}")

    print("  Scoring test stream with Trivial detector...")
    triv_test = test_point_scores_classical(trivial_score_fn, test_z)
    assert np.all(np.isfinite(triv_test)), "Trivial scores contain non-finite values"
    y_pred_triv = (triv_test >= triv_thr).astype(np.int32)

    p_adj_t, r_adj_t, f1_adj_t = point_adjusted_f1(y_true, y_pred_triv, intervals)
    report_triv = classification_report(y_true, y_pred_triv, digits=4, zero_division=0)
    _, _, f1_triv, _ = precision_recall_fscore_support(
        y_true, y_pred_triv, average="binary", zero_division=0)

    out_lines.append(f"\n  --- Trivial Detector (|z-score of window mean|) ---")
    out_lines.append(
        f"  Calibrated threshold (P99 cal covering-window mean): {triv_thr:.6f}")
    out_lines.append(
        f"  Test alarms: {int(y_pred_triv.sum())}/{T_test} = {100*y_pred_triv.mean():.2f}%")
    out_lines.append(f"\n  [PRIMARY] Unadjusted point precision/recall/F1:")
    for ln in report_triv.splitlines():
        out_lines.append("    " + ln)
    out_lines.append(f"\n  [AUDIT ONLY — not used for tuning] Point-adjusted F1:")
    out_lines.append(
        f"    Precision={p_adj_t:.4f}  Recall={r_adj_t:.4f}  F1={f1_adj_t:.4f}")

    return {
        "stream":  stream_id,
        "neural":  f1_neural,
        "iso":     f1_iso,
        "pca":     f1_pca,
        "trivial": f1_triv,
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log_lines = [
        "=" * 65,
        "SMAP/MSL Final Test Evaluation — LOCKED THRESHOLDS",
        "Team 05, Card I1",
        f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "Neural thresholds: LOCKED (from smap_msl_results.txt, NOT recomputed)",
        f"  P-1 locked threshold: {LOCKED_THRESHOLDS['P-1']:.6f}",
        f"  C-1 locked threshold: {LOCKED_THRESHOLDS['C-1']:.6f}",
        "",
        "Classical thresholds: calibrated at P99 of cal-portion covering-window scores.",
        "",
        "Scoring: covering-window mean (corrected Annex D6 protocol, 2026-11-14).",
        "PRIMARY metric: unadjusted point F1.",
        "AUDIT ONLY: point-adjusted F1 (not used for tuning or selection).",
        "",
    ]

    out_lines = [
        "=" * 65,
        "SMAP/MSL Final Test Evaluation — LOCKED THRESHOLDS",
        "Team 05, Card I1",
        f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "Neural thresholds LOCKED (from smap_msl_results.txt — NOT recomputed):",
        f"  P-1: {LOCKED_THRESHOLDS['P-1']:.6f}",
        f"  C-1: {LOCKED_THRESHOLDS['C-1']:.6f}",
        "",
        "Scoring protocol: covering-window mean (corrected Annex D6, 2026-11-14).",
        "  Each test point scored by mean squared reconstruction error at that point",
        "  averaged over all windows covering it.",
        "",
        "[PRIMARY] = unadjusted point F1 (standard sklearn classification_report)",
        "[AUDIT ONLY] = point-adjusted F1 (not used for tuning or selection)",
        "",
    ]

    all_f1s = []

    for stream_id, train_name, test_name in [
        ("P-1", "SMAP_MSL_train_P-1.npy", "SMAP_MSL_test_P-1.npy"),
        ("C-1", "SMAP_MSL_train_C-1.npy", "SMAP_MSL_test_C-1.npy"),
    ]:
        f1s = evaluate_stream(
            stream_id=stream_id,
            train_npy=DATA / train_name,
            test_npy=DATA  / test_name,
            log_lines=log_lines,
            out_lines=out_lines,
        )
        all_f1s.append(f1s)

    # Summary table
    out_lines.append(f"\n{'='*65}")
    out_lines.append("SUMMARY — Unadjusted F1 (PRIMARY)")
    out_lines.append(f"{'='*65}")
    out_lines.append(
        f"  {'Stream':<8}  {'Neural':>8}  {'IsoForest':>10}  {'PCA':>8}  {'Trivial':>8}")
    out_lines.append(f"  {'-'*48}")
    for f1s in all_f1s:
        out_lines.append(
            f"  {f1s['stream']:<8}  {f1s['neural']:>8.4f}  {f1s['iso']:>10.4f}  "
            f"{f1s['pca']:>8.4f}  {f1s['trivial']:>8.4f}"
        )

    # Assertion: all F1 scores finite
    for f1s in all_f1s:
        for key in ("neural", "iso", "pca", "trivial"):
            assert np.isfinite(f1s[key]), \
                f"{f1s['stream']} {key} F1 is not finite: {f1s[key]}"
    out_lines.append("\nAll F1 scores are finite. Assertion PASSED.")

    out_path = OUT / "smap_msl_final_eval.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nResults saved to: {out_path}")

    # Print summary to stdout
    print()
    for ln in out_lines[-12:]:
        print(ln)

    return all_f1s


if __name__ == "__main__":
    main()
