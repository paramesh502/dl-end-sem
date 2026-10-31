"""
stride1_timing.py
==================
Card I1 — stride-1 scoring timing (Annex D0 stage to project for I1).

Times the full stride-1 scoring pipeline on P-1 and C-1, with the trained
1-D autoencoder architecture and the classical Isolation-Forest control, using
test-shaped inputs (the test sensor stream). NO test labels are loaded and NO
test metric is computed — this is a timing only.

Pipeline timed (per stream, per model):
  1. Window construction  — stride-1, 256-step windows, with right-edge
     replication padding so there is one window per test point (full workload,
     including padding). Window count = T_test.
  2. Inference            — forward pass (neural) / feature extraction +
     IsolationForest scoring (classical).
  3. Reduction            — window scores reduced to point scores by the mean
     over covering windows.

Output: results/stride1_timing.txt  (and a 24-hour programme projection).

Run from project root:
    source .venv/bin/activate
    python3 scripts/stride1_timing.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest

# ── Constants ────────────────────────────────────────────────────────────────
SEED   = 22
WINDOW = 256
TRAIN_STRIDE = 64
BATCH  = 32

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 1-D autoencoder architecture (proxy; matches results/1d_autoencoder.pt) ──
class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── feature vector for the classical control (matches run_baselines.py) ──────
def feature_vector(values):
    v = np.asarray(values, dtype=np.float64)
    mean_v = float(np.mean(v)); std_v = float(np.std(v))
    min_v = float(np.min(v)); max_v = float(np.max(v))
    t = np.arange(len(v), dtype=np.float64)
    slope_v = float(linregress(t, v).slope) if len(v) >= 2 else 0.0
    lag1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if (len(v) >= 3 and std_v > 0) else 0.0
    return np.array([mean_v, std_v, min_v, max_v, slope_v, lag1])


# ── window construction: one window per test point, right-edge padded ─────────
def build_windows_padded(series, window=WINDOW):
    """Return (T, window) array: window i = padded[i:i+window], covering point i
    at its start; right-edge replication padding supplies the last `window-1`
    windows. Every test point is covered; window count = T (includes padding).
    """
    T = len(series)
    pad = window - 1
    padded = np.concatenate([series, np.full(pad, series[-1])])  # right-edge replication
    idx = np.arange(T)[:, None] + np.arange(window)[None, :]       # (T, window)
    return padded[idx]


# ── reduction: window scores -> point scores (mean over covering windows) ────
def reduce_to_points(window_scores, T, window=WINDOW):
    acc = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.int64)
    for s, sc in enumerate(window_scores):
        acc[s:s + window] += sc
        cnt[s:s + window] += 1
    return acc / np.maximum(cnt, 1)


def time_neural(series_z, model):
    """Time neural stride-1 scoring on the standardised stream."""
    T = len(series_z)
    t0 = time.perf_counter()
    windows = build_windows_padded(series_z)                      # (T, 256)
    t_construct = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(windows), BATCH):
            batch = torch.tensor(windows[i:i + BATCH, None, :], dtype=torch.float32)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            scores.append(mse.numpy())
    window_scores = np.concatenate(scores)
    t_infer = time.perf_counter() - t0

    t0 = time.perf_counter()
    point_scores = reduce_to_points(window_scores, T)             # not used for metrics
    t_reduce = time.perf_counter() - t0
    _ = point_scores  # timing only; no metric computed

    return {
        "n_windows": len(windows),
        "n_points": T,
        "construct_s": t_construct,
        "infer_s": t_infer,
        "reduce_s": t_reduce,
        "total_s": t_construct + t_infer + t_reduce,
    }


def time_classical(series, fit_arr):
    """Time classical (IsolationForest) stride-1 scoring. Fits on fit windows
    (stride 64), then times the test scoring pipeline (construction+features,
    IsoForest scoring, reduction)."""
    T = len(series)

    # Fit on fit windows (stride 64) — fit cost reported separately, not in the
    # scoring pipeline total.
    t0 = time.perf_counter()
    fit_vals = fit_arr
    fw_feats = []
    i = 0
    while i + WINDOW <= len(fit_vals):
        fw_feats.append(feature_vector(fit_vals[i:i + WINDOW]))
        i += TRAIN_STRIDE
    X_fit = np.array(fw_feats)
    fm, fs = X_fit.mean(axis=0), X_fit.std(axis=0); fs[fs == 0] = 1.0
    X_fit_z = (X_fit - fm) / fs
    iso = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1)
    iso.fit(X_fit_z)
    fit_s = time.perf_counter() - t0

    # Scoring pipeline (timed): construction + feature extraction
    t0 = time.perf_counter()
    windows = build_windows_padded(series)                        # (T, 256)
    feats = np.array([feature_vector(w) for w in windows])        # (T, 6)
    X_z = (feats - fm) / fs
    t_construct = time.perf_counter() - t0

    # IsoForest scoring (inference)
    t0 = time.perf_counter()
    window_scores = -iso.decision_function(X_z)
    t_infer = time.perf_counter() - t0

    # Reduction to point scores
    t0 = time.perf_counter()
    point_scores = reduce_to_points(window_scores, T)             # not used for metrics
    t_reduce = time.perf_counter() - t0
    _ = point_scores

    return {
        "n_windows": len(windows),
        "n_points": T,
        "construct_s": t_construct,
        "infer_s": t_infer,
        "reduce_s": t_reduce,
        "total_s": t_construct + t_infer + t_reduce,
        "fit_s": fit_s,
    }


# ── load proxy model ─────────────────────────────────────────────────────────
proxy_note = ""
model = Conv1dAutoencoder()
n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
pt_path = OUT / "1d_autoencoder.pt"
if pt_path.exists():
    try:
        sd = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(sd)
        proxy_note = (f"Loaded trained 1-D autoencoder weights from {pt_path} "
                      f"({n_par:,} params). These weights were trained on OPSSAT-AD "
                      f"nominal data and are used here only as a proxy architecture "
                      f"for timing the stride-1 pipeline on P-1/C-1; cross-domain "
                      f"reconstructions are not meaningful — only wall-clock timing "
                      f"is reported.")
    except Exception as e:
        model = Conv1dAutoencoder()
        proxy_note = (f"Could not load {pt_path} ({e}); instantiated Conv1dAutoencoder "
                      f"with random weights as a proxy for timing only.")
else:
    proxy_note = (f"{pt_path} not found; instantiated Conv1dAutoencoder with random "
                  f"weights as a proxy for timing only.")

results = {}
fit_times_log = {"P-1": None, "C-1": None}
for stream in ["P-1", "C-1"]:
    train_raw = np.load(DATA / f"SMAP_MSL_train_{stream}.npy")[:, 0].astype(np.float32)
    test_raw  = np.load(DATA / f"SMAP_MSL_test_{stream}.npy")[:, 0].astype(np.float32)  # test-shaped input only
    T_train = len(train_raw); T_test = len(test_raw)
    fit_end = int(0.8 * T_train)
    fit_series = train_raw[:fit_end]
    mu, sig = float(fit_series.mean()), float(fit_series.std()); sig = sig or 1.0
    fit_z = (fit_series - mu) / sig
    test_z = (test_raw - mu) / sig                                       # fit-standardised test input

    # stream 1-D AE fit time (measured here for the projection)
    t0 = time.perf_counter()
    fw = []
    i = 0
    while i + WINDOW <= len(fit_z):
        fw.append(fit_z[i:i + WINDOW]); i += TRAIN_STRIDE
    fw = np.array(fw)
    torch.manual_seed(SEED)
    m = Conv1dAutoencoder()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(5):  # 5 epochs — enough to measure a per-epoch rate; not a real fit
        m.train()
        for b in range(0, len(fw), BATCH):
            xb = torch.tensor(fw[b:b + BATCH, None, :], dtype=torch.float32)
            opt.zero_grad(); out = m(xb); crit(out, xb).backward(); opt.step()
    fit_s_5ep = time.perf_counter() - t0
    fit_times_log[stream] = (fit_s_5ep, len(fw))

    neural = time_neural(test_z, model)
    classical = time_classical(test_raw, fit_series)
    results[stream] = {"T_test": T_test, "neural": neural, "classical": classical}

# ── assemble report ──────────────────────────────────────────────────────────
lines = []
lines.append("=" * 72)
lines.append("Card I1 — Stride-1 Scoring Timing (Annex D0 stage to project for I1)")
lines.append("Team 05  ·  seed = 22")
lines.append("=" * 72)
lines.append("")
lines.append("Proxy model:")
lines.append(f"  {proxy_note}")
lines.append("")
lines.append("Pipeline per stream (test-shaped input only; no test labels loaded;")
lines.append("no test metric computed):")
lines.append("  - Window construction: stride-1, 256-step windows, right-edge")
lines.append("    replication padding so there is one window per test point")
lines.append("    (full workload, including padding). Window count = T_test.")
lines.append("  - Reduction: window scores -> point scores by the mean over the")
lines.append("    windows covering each point.")
lines.append("")
for stream in ["P-1", "C-1"]:
    r = results[stream]
    lines.append("-" * 72)
    lines.append(f"Stream {stream}  (test steps = {r['T_test']})")
    lines.append("-" * 72)
    for kind, label in [("neural", "NEURAL — 1-D autoencoder (proxy weights)"),
                        ("classical", "CLASSICAL — IsolationForest")]:
        d = r[kind]
        lines.append(f"  [{label}]")
        lines.append(f"    Window count          : {d['n_windows']}")
        lines.append(f"    Point scores         : {d['n_points']}")
        lines.append(f"    Construction time (s) : {d['construct_s']:.4f}")
        lines.append(f"    Inference time (s)    : {d['infer_s']:.4f}")
        lines.append(f"    Reduction time (s)   : {d['reduce_s']:.4f}")
        lines.append(f"    TOTAL (s)            : {d['total_s']:.4f}")
        if kind == "classical":
            lines.append(f"    [IsoForest fit on fit windows: {d['fit_s']:.4f} s, not in scoring total]")

# Projection
lines.append("")
lines.append("=" * 72)
lines.append("Projection against the 24-hour budget")
lines.append("=" * 72)
neural_total = results["P-1"]["neural"]["total_s"] + results["C-1"]["neural"]["total_s"]
classical_total = results["P-1"]["classical"]["total_s"] + results["C-1"]["classical"]["total_s"]
stride1_total = neural_total + classical_total
lines.append(f"Measured stride-1 scoring (P-1 + C-1, neural + classical controls):")
lines.append(f"  P-1  neural  : {results['P-1']['neural']['total_s']:.4f} s")
lines.append(f"  C-1  neural  : {results['C-1']['neural']['total_s']:.4f} s")
lines.append(f"  P-1  classical: {results['P-1']['classical']['total_s']:.4f} s")
lines.append(f"  C-1  classical: {results['C-1']['classical']['total_s']:.4f} s")
lines.append(f"  stride-1 subtotal : {stride1_total:.4f} s")

# Fit times: OPSSAT neural fits (from runlog R05-R07) + stream 1-D AE fits.
# Stream fit times below are 5-epoch probes (per-epoch rate); the actual recorded
# full 30-epoch stream fits are R09 (~3.1 s P-1) and R10 (~1.8 s C-1).
opssat_fits = {"1-D AE (R05)": 14.22, "2-D AE (R06)": 20.79,
               "Attention AE (R07)": 15.32}
stream_fit_full = {"P-1 1-D AE (R09, calib. re-run)": 3.07,
                   "C-1 1-D AE (R10, calib. re-run)": 1.83}
fit_total = sum(opssat_fits.values()) + sum(stream_fit_full.values())
lines.append("")
lines.append("Required neural fits (actual recorded elapsed, T05_I1_runlog.csv):")
for k, v in opssat_fits.items():
    lines.append(f"  {k}: {v:.2f} s")
for k, v in stream_fit_full.items():
    lines.append(f"  {k}: {v:.2f} s")
lines.append(f"  (runlog R09/R10 record the original test-scored runs at 1.07/0.36 s;")
lines.append(f"   the corrected-calibration re-runs are 3.07/1.83 s — both << 29 s.)")
lines.append(f"  fusion (R08): score-level, no training (< 0.5 s)")
lines.append(f"  classical fits (R02-R04 + streams): < 1 s each (negligible)")
lines.append(f"  fit subtotal : {fit_total:.2f} s")
lines.append("")
lines.append("Stream-model timings establish the per-fit cost: stream 1-D AE fits are")
lines.append(f"~1-3 s (R09/R10 and their calibration re-runs), NOT the 29.18 s of the")
lines.append("Week-1 spectrogram reference fit (R01). The 29.18 s reference is the largest")
lines.append("single fit; the OPSSAT fits (R05-R07) are 14-21 s. None approach a budget concern.")

prog_elapsed = fit_total + stride1_total
hours = prog_elapsed / 3600.0
lines.append("")
lines.append(f"Programme projection (fits + stride-1 scoring):")
lines.append(f"  fits + stride-1 scoring = {fit_total:.2f} + {stride1_total:.4f} = {prog_elapsed:.2f} s")
lines.append(f"                         = {hours:.4f} h  ({100*prog_elapsed/86400:.4f}% of 24 h)")
lines.append(f"  24 h budget = 86400 s; margin = {86400 - prog_elapsed:.0f} s")
lines.append("")
lines.append("Conclusion: the full programme (all required fits + stride-1 scoring of P-1")
lines.append(f"and C-1 with the trained models and the classical controls on the same")
lines.append(f"windows) completes in ~{prog_elapsed:.0f} s, well within the 24-hour budget")
lines.append(f"(uses {100*prog_elapsed/86400:.3f}% of it). If one required fit took as much")
lines.append(f"as 29.18 s (the spectrogram reference upper bound), the {len(opssat_fits)+len(stream_fit_full)}")
lines.append(f"required neural fits would still take only")
lines.append(f"  {len(opssat_fits)+len(stream_fit_full)} x 29.18 = {29.18*(len(opssat_fits)+len(stream_fit_full)):.1f} s,")
lines.append(f"plus {stride1_total:.2f} s of stride-1 scoring = "
             f"{29.18*(len(opssat_fits)+len(stream_fit_full))+stride1_total:.1f} s, still "
             f"~{100*(29.18*(len(opssat_fits)+len(stream_fit_full))+stride1_total)/86400:.3f}% of 24 h.")
lines.append("")
lines.append("No test labels were loaded and no test metric was computed for this timing.")
lines.append("=" * 72)

out_path = OUT / "stride1_timing.txt"
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nSaved to: {out_path}")
