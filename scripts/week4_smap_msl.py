"""
week4_smap_msl.py
=================
Week 4 — Card I1 — Satellite Telemetry Anomaly Detection
Team 05: K Hemalatha Reddy, B Parameshwar Reddy, Jagan Sai Kumar

Fit the NO-ATTENTION 1-D convolutional autoencoder INDEPENDENTLY on:
  • SMAP  P-1  (train 2,872 steps, column 0, 25 features total)
  • MSL   C-1  (train 2,158 steps, column 0, 55 features total)

Split protocol (Annex D6, revised): first 80% fit, a 256-step guard gap, the
remaining 20% calibration. Scoring uses 256-step windows. Calibration uses
GUARD-CONTEXT windows per the revised Annex D6 rule: a calibration window may
take earlier context from the guard, never from fit rows, and only calibration
points are scored (one window per cal point, right-aligned).

Threshold    : 99th percentile of CALIBRATION-point reconstruction MSE scores.
Calibration  : per-stream; reports threshold, cal alarm count and cal alarm rate.

Per revised Handout §14, NO test partition is scored before the results freeze.
This script therefore reports calibration figures only and does not load or
score the test stream.

Run from project root:
    source .venv/bin/activate
    python3 scripts/week4_smap_msl.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (Annex D0)
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 22
MAX_EPOCHS    = 30
MAX_SECONDS   = 30 * 60
BATCH_SIZE    = 32
LR            = 1e-3
EARLY_STOP    = 5
THRESHOLD_PCT = 99.0
WINDOW        = 256
TRAIN_STRIDE  = 64
GUARD         = 256

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)


# =============================================================================
# Model — identical to week3_1d_autoencoder.py  (no attention)
# =============================================================================
class Conv1dAutoencoder(nn.Module):
    """
    Small 1-D Convolutional Autoencoder — NO ATTENTION.

    Architecture:
        Input  : (batch, 1, 256)

        Encoder:
            Conv1d(1→16, k=3, pad=1) + ReLU
            MaxPool1d(2)              → 256→128
            Conv1d(16→32, k=3, pad=1) + ReLU
            MaxPool1d(2)              → 128→64
            Conv1d(32→16, k=3, pad=1) + ReLU
                                 ↓ bottleneck: (batch, 16, 64)

        Decoder:
            Conv1d(16→32, k=3, pad=1) + ReLU
            Upsample(×2)              → 64→128
            Conv1d(32→16, k=3, pad=1) + ReLU
            Upsample(×2)              → 128→256
            Conv1d(16→1, k=3, pad=1)

        Output : (batch, 1, 256)
    """

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


# =============================================================================
# Helper: extract sliding windows from a 1-D time series
# =============================================================================
def extract_windows(series: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Return (N, window) array of sliding windows.
    series : 1-D float32 array of length T
    """
    T = len(series)
    starts = np.arange(0, T - window + 1, stride)
    return np.stack([series[s:s + window] for s in starts], axis=0)


# =============================================================================
# Helper: guard-context calibration windows (one per cal point, right-aligned)
# =============================================================================
def cal_point_mse_neural(model: nn.Module, series_z: np.ndarray,
                          fit_end: int, cal_start: int, cal_end: int,
                          window: int = WINDOW) -> np.ndarray:
    """
    Annex D6 (corrected per 14-Nov-2026 email): score each calibration point
    by the MEAN SQUARED RECONSTRUCTION ERROR AT THAT POINT, averaged over all
    windows that cover it.

    For cal point p, the covering windows are those whose start index s satisfies:
        max(fit_end, p - window + 1)  <=  s  <=  p
    (s >= fit_end: guard context allowed, fit rows never entered)
    (s <= p: the window [s, s+window) covers position p)

    Every cal point is covered by exactly (p - max(fit_end, p-window+1) + 1) windows,
    which equals min(window, p - fit_end) windows.  For the first cal point
    (p = cal_start = fit_end + guard = fit_end + 256), this is exactly 256 windows.

    Returns: (N_cal,) float32 — one mean-over-covering-windows score per cal point.
    """
    model.eval()
    cal_scores = np.zeros(cal_end - cal_start, dtype=np.float64)
    cal_counts = np.zeros(cal_end - cal_start, dtype=np.int64)

    # Slide every window that starts in [fit_end, cal_end) and covers at
    # least one cal point.  stride=1 for scoring.
    with torch.no_grad():
        for s in range(fit_end, cal_end):
            w_end = s + window
            if w_end > len(series_z):
                break  # window runs off the end of the training stream
            win  = torch.tensor(series_z[s:w_end][None, None, :], dtype=torch.float32)
            recon = model(win)
            # Reconstruction error at each position
            err_per_pos = ((recon - win) ** 2).squeeze().numpy()  # (256,)
            # Which cal points does this window cover?
            cov_start = max(cal_start, s)
            cov_end   = min(cal_end, s + window)
            for p in range(cov_start, cov_end):
                pos_in_win = p - s
                cal_idx    = p - cal_start
                cal_scores[cal_idx] += err_per_pos[pos_in_win]
                cal_counts[cal_idx] += 1

    assert (cal_counts > 0).all(), "Some cal point was not covered by any window"
    return (cal_scores / cal_counts).astype(np.float32)


def cal_point_scores_classical(detector_score_fn,
                                series_z: np.ndarray,
                                fit_end: int, cal_start: int, cal_end: int,
                                window: int = WINDOW) -> np.ndarray:
    """
    Same covering-window average for classical detectors.

    detector_score_fn: takes a (1, N_features) array, returns a scalar score
                       for one window (after standardisation applied outside).

    For each window [s, s+window), compute the detector's window score, then
    accumulate that score into every cal point the window covers.

    Returns: (N_cal,) float64 — mean detector window score per cal point.
    """
    cal_scores = np.zeros(cal_end - cal_start, dtype=np.float64)
    cal_counts = np.zeros(cal_end - cal_start, dtype=np.int64)

    for s in range(fit_end, cal_end):
        w_end = s + window
        if w_end > len(series_z):
            break
        win_score = detector_score_fn(series_z[s:w_end])
        cov_start = max(cal_start, s)
        cov_end   = min(cal_end, s + window)
        for p in range(cov_start, cov_end):
            cal_scores[p - cal_start] += win_score
            cal_counts[p - cal_start] += 1

    assert (cal_counts > 0).all()
    return (cal_scores / cal_counts).astype(np.float64)


# =============================================================================
# Helper: score each window independently → per-window MSE array
# =============================================================================
def score_windows(model: nn.Module, windows: np.ndarray) -> np.ndarray:
    """
    windows: (N, 256) float32
    Returns: (N,) float32  — per-window mean MSE
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(windows), BATCH_SIZE):
            batch = torch.tensor(
                windows[i:i + BATCH_SIZE, None, :], dtype=torch.float32
            )  # (B, 1, 256)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            scores.append(mse.numpy())
    return np.concatenate(scores)


# =============================================================================
# Core routine: fit + calibrate one stream (NO test scoring)
# =============================================================================
def run_stream(name: str, train_npy: Path):
    print()
    print("=" * 65)
    print(f"  STREAM: {name}")
    print("=" * 65)

    # ── 1. Load training data (column 0 only) ────────────────────────────
    train_raw = np.load(train_npy)[:, 0].astype(np.float32)
    T_train   = len(train_raw)
    print(f"  Train length : {T_train}")

    # ── 2. Split (Annex D6, revised: guard is always 256; no capping) ─────
    fit_end  = int(0.8 * T_train)              # last index (exclusive) of fit
    cal_start = fit_end + GUARD                 # guard = [fit_end, cal_start)
    cal_end   = T_train                         # cal   = [cal_start, cal_end)
    n_cal     = cal_end - cal_start

    fit_series = train_raw[:fit_end]
    print(f"  Fit  : [0, {fit_end})  ({len(fit_series)} steps)")
    print(f"  Guard: [{fit_end}, {cal_start})  ({GUARD} steps)")
    print(f"  Cal  : [{cal_start}, {cal_end})  ({n_cal} steps) -> {n_cal} calibration-point scores")

    # ── 3. Standardise (fit statistics only) ─────────────────────────────
    mu  = float(fit_series.mean())
    sig = float(fit_series.std())
    if sig == 0.0:
        sig = 1.0
    train_z = (train_raw - mu) / sig            # full series, fit-standardised
    print(f"  Fit mean: {mu:.6f}   std: {sig:.6f}")

    # ── 4. Window extraction ─────────────────────────────────────────────
    win_fit = extract_windows(train_z[:fit_end], WINDOW, TRAIN_STRIDE)  # stride-64

    print(f"  Fit windows  : {win_fit.shape}  (stride {TRAIN_STRIDE})")
    print(f"  Cal scoring  : mean-over-covering-windows per point")
    print(f"                 guard context allowed; fit rows never entered")

    # ── 5. Reproducibility seed ──────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 6. Build model ───────────────────────────────────────────────────
    model = Conv1dAutoencoder()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_par <= 250_000, f"Param count {n_par} exceeds cap"
    print(f"  Model params : {n_par:,}")

    with torch.no_grad():
        dummy = torch.zeros(1, 1, WINDOW)
        assert model(dummy).shape == dummy.shape

    # ── 7. DataLoader for training ───────────────────────────────────────
    class WindowDataset(Dataset):
        def __init__(self, w: np.ndarray):
            self.w = torch.tensor(w[:, None, :], dtype=torch.float32)
        def __len__(self):  return len(self.w)
        def __getitem__(self, i): return self.w[i]

    train_loader = DataLoader(
        WindowDataset(win_fit), batch_size=BATCH_SIZE, shuffle=True
    )

    # ── 8. Training loop (validation proxy = mean cal-point score) ───────
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    criterion  = nn.MSELoss()
    best_loss  = float("inf")
    best_state = None
    no_improve = 0
    history    = []
    stop_reason = "completed all epochs"
    t0 = time.perf_counter()

    print(f"\n  Training …")
    for epoch in range(1, MAX_EPOCHS + 1):
        if time.perf_counter() - t0 >= MAX_SECONDS:
            stop_reason = "30-minute elapsed limit"
            print(f"  30-min limit reached at epoch {epoch}. Stopping.")
            break

        model.train()
        tr_loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * len(batch)
        tr_loss = tr_loss_sum / len(win_fit)

        # Validation proxy: mean cal-point score using covering-window mean.
        val_scores = cal_point_mse_neural(model, train_z, fit_end, cal_start, cal_end)
        val_loss   = float(val_scores.mean())

        elapsed = time.perf_counter() - t0
        print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  "
              f"train={tr_loss:.6f}  val_cal_mean={val_loss:.6f}  "
              f"elapsed={elapsed:.1f}s")
        history.append((epoch, tr_loss, val_loss))

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP:
            stop_reason = f"early stopping (no improvement for {EARLY_STOP} epochs)"
            print(f"  Early stop at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_total = time.perf_counter() - t0
    print(f"\n  Stopped by    : {stop_reason}")
    print(f"  Best val loss : {best_loss:.6f}")
    print(f"  Total elapsed : {elapsed_total:.2f}s ({elapsed_total/60:.2f} min)")

    # ── 9. Calibration threshold (Annex D6, corrected 14-Nov-2026) ───────
    # Score each cal point by the mean reconstruction error AT THAT POINT
    # averaged over all windows covering it (guard context allowed, fit rows never).
    print(f"\n  Computing calibration point scores (mean-over-covering-windows) …")
    t_cal = time.perf_counter()
    cal_point_scores = cal_point_mse_neural(
        model, train_z, fit_end, cal_start, cal_end)   # (N_cal,)
    t_cal = time.perf_counter() - t_cal
    assert len(cal_point_scores) == n_cal
    threshold  = float(np.percentile(cal_point_scores, THRESHOLD_PCT))
    cal_alarms = int((cal_point_scores >= threshold).sum())
    cal_rate   = cal_alarms / n_cal
    print(f"\n  Cal point scores        : {n_cal}")
    print(f"  Cal threshold (99th pct) : {threshold:.6f}")
    print(f"  Cal alarm count / rate  : {cal_alarms}/{n_cal} = {100*cal_rate:.2f}%")
    print(f"  Cal scoring time        : {t_cal:.2f}s")
    print(f"  (Per §14, the test stream is NOT scored here.)")

    return {
        "name":          name,
        "threshold":     threshold,
        "cal_alarms":    cal_alarms,
        "cal_rate":      cal_rate,
        "n_cal":         n_cal,
        "history":       history,
        "stop_reason":   stop_reason,
        "n_params":      n_par,
        "elapsed":       elapsed_total,
        "mu":            mu,
        "sig":           sig,
        "T_train":       T_train,
        "fit_end":       fit_end,
        "cal_start":     cal_start,
        "n_train_win":   len(win_fit),
        "best_loss":     best_loss,
        "cal_scoring_time": t_cal,
    }


# =============================================================================
# MAIN
# =============================================================================
print("=" * 65)
print("Week 4 — 1-D Conv Autoencoder on SMAP P-1 and MSL C-1 (calibration only)")
print("Team 05, Card I1")
print("=" * 65)

# ── P-1 ──────────────────────────────────────────────────────────────────────
res_p1 = run_stream(
    name="SMAP P-1",
    train_npy=DATA / "SMAP_MSL_train_P-1.npy",
)

# ── C-1 ──────────────────────────────────────────────────────────────────────
res_c1 = run_stream(
    name="MSL C-1",
    train_npy=DATA / "SMAP_MSL_train_C-1.npy",
)

# =============================================================================
# Print final summary (calibration only; no test metrics)
# =============================================================================
print()
print("=" * 65)
print("FINAL SUMMARY  (calibration only; no test scored before the freeze)")
print("=" * 65)
for res in [res_p1, res_c1]:
    print(f"\n{res['name']}:")
    print(f"  Cal point scores : {res['n_cal']}")
    print(f"  Cal threshold    : {res['threshold']:.6f}  (99th pctile of cal-point MSE)")
    print(f"  Cal alarm count  : {res['cal_alarms']}")
    print(f"  Cal alarm rate   : {100*res['cal_rate']:.2f}%")

# =============================================================================
# Save results file
# =============================================================================
lines = []
lines += [
    "=" * 65,
    "Week 4 — 1-D Conv Autoencoder (No Attention) — SMAP / MSL",
    "Team 05, Card I1",
    "=" * 65,
    "",
    "Calibration only (revised Handout §14): no test partition is scored before",
    "the results freeze. Earlier test-scored runs (R09, R10) are recorded as",
    "test_scored=yes in T05_I1_runlog.csv; their test metrics are NOT recomputed.",
    "",
    "Calibration uses covering-window mean per revised Annex D6 (corrected 2026-11-14):",
    "each cal point is scored by the mean reconstruction error AT THAT POINT",
    "averaged over all windows that cover it, using guard context but never fit rows.",
    "P-1: 319 cal-point scores; C-1: 176 cal-point scores.",
    "",
]

for res in [res_p1, res_c1]:
    lines += [
        "-" * 65,
        f"Stream : {res['name']}",
        "-" * 65,
        f"  Model params     : {res['n_params']:,}",
        f"  Train length     : {res['T_train']}",
        f"  Fit  portion      : [0, {res['fit_end']})",
        f"  Guard gap         : {GUARD} steps  [{res['fit_end']}, {res['cal_start']})",
        f"  Cal  portion      : [{res['cal_start']}, {res['T_train']})  "
        f"({res['n_cal']} cal steps)",
        f"  Fit mean / std    : {res['mu']:.6f} / {res['sig']:.6f}",
        f"  Train windows      : {res['n_train_win']} (stride 64)",
        f"  Cal scoring        : mean-over-covering-windows per point (corrected 2026-11-14)",
        f"                       guard context allowed; fit rows never entered",
        f"  Stopped by        : {res['stop_reason']}",
        f"  Elapsed           : {res['elapsed']:.2f}s ({res['elapsed']/60:.2f} min)",
        f"  Cal scoring time  : {res['cal_scoring_time']:.2f}s",
        f"  Cal point scores  : {res['n_cal']}",
        f"  Cal threshold     : {res['threshold']:.6f}  (99th pctile of cal-point MSE)",
        f"  Cal alarm count   : {res['cal_alarms']}/{res['n_cal']}",
        f"  Cal alarm rate    : {100*res['cal_rate']:.2f}%",
        "",
        "  Training history (train MSE / val_cal mean MSE per epoch):",
    ]
    for ep, tl, vl in res["history"]:
        lines.append(f"    Epoch {ep:02d}: train={tl:.6f}  val_cal={vl:.6f}")
    lines += [""]

out_path = OUT / "smap_msl_results.txt"
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nResults saved to: {out_path}")
print()
print("=" * 65)
print("Week 4 complete (calibration only).")
print("=" * 65)
