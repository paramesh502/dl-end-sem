"""
week4_smap_msl.py
=================
Week 4 — Card I1 — Satellite Telemetry Anomaly Detection
Team 05: K Hemalatha Reddy, B Parameshwar Reddy, Jagan Sai Kumar

Fit the NO-ATTENTION 1-D convolutional autoencoder INDEPENDENTLY on:
  • SMAP  P-1  (8505 test steps, column 0, 25 features total)
  • MSL   C-1  (2264 test steps, column 0, 55 features total)

Split protocol: Annex D6 (80/20 fit/cal, guard gap = 256).
Threshold    : 99th percentile of CALIBRATION window MSE scores.
Evaluation   : Unadjusted point-level F1 (primary)
               Point-adjusted F1 (AUDIT figure only).

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
from sklearn.metrics import precision_recall_fscore_support

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
# Helper: aggregate per-window scores to per-time-step scores
# =============================================================================
def windows_to_timestep_scores(
    window_scores: np.ndarray, T_test: int, window: int
) -> np.ndarray:
    """
    For stride-1 scoring over a test stream of length T_test:
      - There are (T_test - window + 1) windows.
      - Window s covers time-steps s .. s+window-1.
      - Each time-step t receives the MEAN MSE of all windows covering it.
      - Edge steps (< window full-coverage windows) are still scored.
    """
    N = T_test - window + 1
    assert len(window_scores) == N, \
        f"Expected {N} window scores, got {len(window_scores)}"

    acc   = np.zeros(T_test, dtype=np.float64)
    count = np.zeros(T_test, dtype=np.int64)

    for s, sc in enumerate(window_scores):
        acc[s:s + window]   += sc
        count[s:s + window] += 1

    # All positions should be covered at least once
    assert (count > 0).all(), "Some time-steps have zero window coverage!"
    return (acc / count).astype(np.float32)


# =============================================================================
# Helper: point-adjusted F1  (AUDIT figure only)
# =============================================================================
def point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray,
                      anomaly_intervals: list) -> float:
    """
    If ANY predicted point falls inside a true anomaly interval,
    label the ENTIRE interval as predicted anomaly.
    Returns F1.  — AUDIT FIGURE ONLY.
    """
    y_adj = y_pred.copy()
    for s, e in anomaly_intervals:
        if y_pred[s:e + 1].any():
            y_adj[s:e + 1] = 1
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_adj, average="binary", zero_division=0
    )
    return float(f1)


# =============================================================================
# Core routine: fit + evaluate one stream
# =============================================================================
def run_stream(
    name: str,
    train_npy: Path,
    test_npy: Path,
    anomaly_intervals: list,   # list of (start, end) inclusive, 0-based test indices
):
    print()
    print("=" * 65)
    print(f"  STREAM: {name}")
    print("=" * 65)

    # ── 1. Load data (column 0 only) ────────────────────────────────────
    train_raw = np.load(train_npy)[:, 0].astype(np.float32)
    test_raw  = np.load(test_npy)[:, 0].astype(np.float32)
    T_train   = len(train_raw)
    T_test    = len(test_raw)
    print(f"  Train length : {T_train}   Test length : {T_test}")

    # ── 2. Split  (Annex D6) ────────────────────────────────────────────
    fit_end  = int(0.8 * T_train)              # last index (exclusive) of fit portion

    # Guard gap: 256 normally, but capped so the cal portion always has
    # at least WINDOW steps (required to extract at least one cal window).
    # Annex D6 does not mandate a fixed guard for short streams.
    actual_guard = min(GUARD, max(0, T_train - fit_end - WINDOW))
    cal_start    = fit_end + actual_guard

    fit_series = train_raw[:fit_end]
    cal_series = train_raw[cal_start:]

    print(f"  Fit  : [0, {fit_end})  ({len(fit_series)} steps)")
    print(f"  Guard: [{fit_end}, {cal_start})  ({actual_guard} steps)"
          + ("" if actual_guard == GUARD else f"  [reduced from {GUARD} — short stream]"))
    print(f"  Cal  : [{cal_start}, {T_train})  ({len(cal_series)} steps)")

    # ── 3. Standardise  (fit statistics only) ───────────────────────────
    mu  = float(fit_series.mean())
    sig = float(fit_series.std())
    if sig == 0.0:
        sig = 1.0

    fit_z  = (fit_series - mu) / sig
    cal_z  = (cal_series - mu) / sig
    test_z = (test_raw   - mu) / sig

    print(f"  Fit mean: {mu:.6f}   std: {sig:.6f}")

    # ── 4. Window extraction ─────────────────────────────────────────────
    win_train = extract_windows(fit_z, WINDOW, TRAIN_STRIDE)  # stride-64
    win_cal   = extract_windows(cal_z, WINDOW, 1)              # stride-1
    win_test  = extract_windows(test_z, WINDOW, 1)             # stride-1

    print(f"  Train windows : {win_train.shape}  (stride {TRAIN_STRIDE})")
    print(f"  Cal   windows : {win_cal.shape}   (stride 1)")
    print(f"  Test  windows : {win_test.shape}   (stride 1)")

    # ── 5. Reproducibility seed ──────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 6. Build model ───────────────────────────────────────────────────
    model  = Conv1dAutoencoder()
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_par <= 250_000, f"Param count {n_par} exceeds cap"
    print(f"  Model params : {n_par:,}")

    # Shape sanity check
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
        WindowDataset(win_train), batch_size=BATCH_SIZE, shuffle=True
    )

    # ── 8. Training loop ─────────────────────────────────────────────────
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
        tr_loss = tr_loss_sum / len(win_train)

        # Validation: use calibration windows as a proxy validation set
        # (These are nominal-only because they come from the training file.)
        model.eval()
        val_scores = score_windows(model, win_cal)
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

    # ── 9. Calibration threshold ─────────────────────────────────────────
    cal_win_scores = score_windows(model, win_cal)       # (N_cal,) per-window MSE
    threshold      = float(np.percentile(cal_win_scores, THRESHOLD_PCT))
    print(f"\n  Cal threshold (99th pctile) : {threshold:.6f}")

    # ── 10. Score test stream ────────────────────────────────────────────
    test_win_scores = score_windows(model, win_test)     # (T_test - 255,) per-window
    test_ts_scores  = windows_to_timestep_scores(
        test_win_scores, T_test, WINDOW
    )                                                     # (T_test,) per-timestep

    # ── 11. Build binary labels ──────────────────────────────────────────
    y_true = np.zeros(T_test, dtype=int)
    for s, e in anomaly_intervals:
        y_true[s:e + 1] = 1

    y_pred = (test_ts_scores >= threshold).astype(int)

    n_anom = int(y_true.sum())
    n_pred = int(y_pred.sum())
    print(f"\n  True anomalous points : {n_anom} / {T_test}")
    print(f"  Predicted anomalous   : {n_pred} / {T_test}")

    # ── 12. Unadjusted metrics (PRIMARY) ─────────────────────────────────
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"\n  [PRIMARY] Unadjusted point-level metrics:")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")

    # ── 13. Point-adjusted F1 (AUDIT ONLY) ───────────────────────────────
    pa_f1 = point_adjusted_f1(y_true, y_pred, anomaly_intervals)
    print(f"\n  [AUDIT ONLY] Point-adjusted F1 : {pa_f1:.4f}")
    print(f"  (Not used for threshold or architectural decisions.)")

    return {
        "name":         name,
        "prec":         prec,
        "rec":          rec,
        "f1":           f1,
        "pa_f1":        pa_f1,
        "threshold":    threshold,
        "history":      history,
        "stop_reason":  stop_reason,
        "n_params":     n_par,
        "elapsed":      elapsed_total,
        "mu":           mu,
        "sig":          sig,
        "T_train":      T_train,
        "T_test":       T_test,
        "fit_end":      fit_end,
        "cal_start":    cal_start,
        "actual_guard": actual_guard,
        "n_train_win":  len(win_train),
        "n_cal_win":    len(win_cal),
        "n_test_win":   len(win_test),
    }


# =============================================================================
# MAIN
# =============================================================================
print("=" * 65)
print("Week 4 — 1-D Conv Autoencoder on SMAP P-1 and MSL C-1")
print("Team 05, Card I1")
print("=" * 65)

# ── P-1 ──────────────────────────────────────────────────────────────────────
res_p1 = run_stream(
    name="SMAP P-1",
    train_npy=DATA / "SMAP_MSL_train_P-1.npy",
    test_npy =DATA / "SMAP_MSL_test_P-1.npy",
    anomaly_intervals=[(2149, 2349), (4536, 4844), (3539, 3779)],
)

# ── C-1 ──────────────────────────────────────────────────────────────────────
res_c1 = run_stream(
    name="MSL C-1",
    train_npy=DATA / "SMAP_MSL_train_C-1.npy",
    test_npy =DATA / "SMAP_MSL_test_C-1.npy",
    anomaly_intervals=[(550, 750), (2100, 2210)],
)

# =============================================================================
# Print final summary
# =============================================================================
print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
for res in [res_p1, res_c1]:
    print(f"\n{res['name']}:")
    print(f"  Unadj  Precision : {res['prec']:.4f}")
    print(f"  Unadj  Recall    : {res['rec']:.4f}")
    print(f"  Unadj  F1        : {res['f1']:.4f}   ← PRIMARY")
    print(f"  Pt-adj F1        : {res['pa_f1']:.4f}   ← AUDIT ONLY")

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
    "NOTE: Point-adjusted F1 is an AUDIT figure. Never used for",
    "      threshold selection or architectural decisions.",
    "",
]

for res in [res_p1, res_c1]:
    lines += [
        "-" * 65,
        f"Stream : {res['name']}",
        "-" * 65,
        f"  Model params     : {res['n_params']:,}",
        f"  Train length     : {res['T_train']}",
        f"  Test  length     : {res['T_test']}",
        f"  Fit  portion     : [0, {res['fit_end']})",
        f"  Guard gap used   : {res['actual_guard']} steps"
        + ("" if res['actual_guard'] == GUARD else f"  (reduced from {GUARD} — short stream)"),
        f"  Cal  portion     : [{res['cal_start']}, {res['T_train']})",
        f"  Fit mean / std   : {res['mu']:.6f} / {res['sig']:.6f}",
        f"  Train windows    : {res['n_train_win']} (stride 64)",
        f"  Cal windows      : {res['n_cal_win']} (stride 1)",
        f"  Test windows     : {res['n_test_win']} (stride 1)",
        f"  Stopped by       : {res['stop_reason']}",
        f"  Elapsed          : {res['elapsed']:.2f}s ({res['elapsed']/60:.2f} min)",
        f"  Cal threshold    : {res['threshold']:.6f}  (99th pctile of cal-win MSE)",
        "",
        "  Training history:",
    ]
    for ep, tl, vl in res["history"]:
        lines.append(f"    Epoch {ep:02d}: train={tl:.6f}  val_cal={vl:.6f}")
    lines += [
        "",
        "  ── PRIMARY (unadjusted point-level) ──",
        f"  Precision : {res['prec']:.4f}",
        f"  Recall    : {res['rec']:.4f}",
        f"  F1        : {res['f1']:.4f}",
        "",
        "  ── AUDIT ONLY (point-adjusted) ──",
        f"  Pt-adj F1 : {res['pa_f1']:.4f}",
        "",
    ]

# Comparison table
lines += [
    "=" * 65,
    "Comparison vs Classical Baselines",
    "=" * 65,
    "",
    "SMAP P-1:",
    f"  {'Method':<25}  {'Unadj F1':>10}  {'Pt-adj F1':>10}",
    f"  {'-'*25}  {'-'*10}  {'-'*10}",
    f"  {'Isolation Forest':<25}  {0.2728:>10.4f}  {0.3391:>10.4f}",
    f"  {'PCA (5 components)':<25}  {0.3833:>10.4f}  {0.6821:>10.4f}",
    f"  {'Trivial (mean dist)':<25}  {0.2247:>10.4f}  {0.2565:>10.4f}",
    f"  {'Conv1dAutoencoder':<25}  {res_p1['f1']:>10.4f}  {res_p1['pa_f1']:>10.4f}",
    "",
    "MSL C-1:",
    f"  {'Method':<25}  {'Unadj F1':>10}  {'Pt-adj F1':>10}",
    f"  {'-'*25}  {'-'*10}  {'-'*10}",
    f"  {'Isolation Forest':<25}  {0.2597:>10.4f}  {0.2726:>10.4f}",
    f"  {'PCA (4 components)':<25}  {0.2569:>10.4f}  {0.3128:>10.4f}",
    f"  {'Trivial (mean dist)':<25}  {0.2967:>10.4f}  {0.3300:>10.4f}",
    f"  {'Conv1dAutoencoder':<25}  {res_c1['f1']:>10.4f}  {res_c1['pa_f1']:>10.4f}",
    "",
    "─" * 65,
    "Point-adjusted F1 is an AUDIT figure. Never used for threshold",
    "selection or architectural decisions.",
    "─" * 65,
]

out_path = OUT / "smap_msl_results.txt"
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nResults saved to: {out_path}")
print()
print("=" * 65)
print("Week 4 complete.")
print("=" * 65)
