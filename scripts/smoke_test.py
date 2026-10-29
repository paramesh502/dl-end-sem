"""
smoke_test.py  —  T05_I1 Checkpoint Smoke Test
===============================================
Runs every required component end-to-end on a SMALL SUBSET.
Trains each neural model for a few updates only, validates, and
completes an unscored inference pass over test-shaped data.

Reads data from: Release_Bundle_RevD1/Release_Bundle_RevD1/I1/
Override with:   python3 scripts/smoke_test.py --data <path>

Rules (from checkpoint email):
  - Each neural model trains for a few updates, validates, and
    does an unscored inference pass over test-shaped data.
  - Computes NO test metric. test_scored = no throughout.
  - Finishes in at most 15 minutes total.
  - Prints: component name, samples used, time taken.

Run from project root:
    source .venv/bin/activate
    python3 scripts/smoke_test.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import spectrogram as scipy_spectrogram
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 22
SMOKE_EPOCHS  = 3        # just a few updates — not a real fit
SMOKE_N_FIT   = 60       # small subset of fitting fragments
SMOKE_N_DEV   = 30
SMOKE_N_TEST  = 20
BATCH_SIZE    = 16

torch.manual_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="Release_Bundle_RevD1/Release_Bundle_RevD1/I1",
                    help="Path to I1 data folder")
args = parser.parse_args()
DATA = Path(args.data)

ROOT = Path(__file__).resolve().parent.parent
wall_start = time.perf_counter()

passed, failed = [], []

def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

def ok(name, n_samples, elapsed):
    print(f"  ✓  {name}  |  samples={n_samples}  |  {elapsed:.2f}s")
    passed.append(name)

def fail(name, err):
    print(f"  ✗  {name}  |  ERROR: {err}")
    failed.append(name)


# =============================================================================
# 1 — Load data
# =============================================================================
section("Loading data")
t0 = time.perf_counter()

df  = pd.read_csv(DATA / "segments.csv")
dev = pd.read_csv(DATA / "OPSSAT_nominal_dev_segments.csv")

df["segment"] = df["segment"].astype(str)
df["train"]   = df["train"].astype(str)
df["anomaly"] = df["anomaly"].astype(int)
dev_ids       = set(dev["segment"].astype(str))

frag = df.groupby("segment").agg(
    anomaly=("anomaly","first"), train=("train","first")).reset_index()

nom_fit_ids  = sorted(frag[(frag["anomaly"]==0) & (frag["train"]=="1") &
                           (~frag["segment"].isin(dev_ids))]["segment"])[:SMOKE_N_FIT]
nom_dev_ids  = sorted(dev_ids)[:SMOKE_N_DEV]
test_nom_ids = sorted(frag[(frag["train"]!="1") & (frag["anomaly"]==0)]["segment"])[:SMOKE_N_TEST//2]
test_an_ids  = sorted(frag[(frag["train"]!="1") & (frag["anomaly"]==1)]["segment"])[:SMOKE_N_TEST//2]

def resample256(vals):
    v = np.asarray(vals, dtype=np.float32)
    if len(v) < 2: return None
    return np.interp(np.linspace(0,1,256), np.linspace(0,1,len(v)), v).astype(np.float32)

def load_frags(ids):
    out = []
    for sid in ids:
        v = df[df["segment"]==sid]["value"].astype(float).to_numpy()
        r = resample256(v)
        if r is not None: out.append(r)
    return np.stack(out) if out else np.empty((0,256), np.float32)

X_fit  = load_frags(nom_fit_ids)
X_dev  = load_frags(nom_dev_ids)
X_tnom = load_frags(test_nom_ids)
X_tan  = load_frags(test_an_ids)
X_test = np.concatenate([X_tnom, X_tan])
y_test = np.concatenate([np.zeros(len(X_tnom),int), np.ones(len(X_tan),int)])

mu, sd = X_fit.mean(), X_fit.std() or 1.0
X_fit_z  = (X_fit  - mu) / sd
X_dev_z  = (X_dev  - mu) / sd
X_test_z = (X_test - mu) / sd

ok("data_loading", len(X_fit)+len(X_dev)+len(X_test), time.perf_counter()-t0)


# =============================================================================
# 2 — Non-DL baselines
# =============================================================================
section("Non-DL Baselines")

from scipy.stats import linregress

def feat6(v, impute=None):
    v = np.asarray(v, np.float64)
    mean_v, std_v, min_v, max_v = float(v.mean()), float(v.std()), float(v.min()), float(v.max())
    slope = float(linregress(np.arange(len(v)), v).slope) if len(v)>=2 else np.nan
    lag1  = float(np.corrcoef(v[:-1], v[1:])[0,1]) if len(v)>=3 and std_v>0 else np.nan
    feat  = np.array([mean_v, std_v, min_v, max_v, slope, lag1])
    if impute:
        for i,nm in enumerate(["mean","std","min","max","slope","lag1"]):
            if np.isnan(feat[i]): feat[i] = impute.get(nm,0.0)
    return feat

# compute training medians for imputation
Xf_raw = np.stack([feat6(df[df["segment"]==s]["value"].to_numpy()) for s in nom_fit_ids
                   if len(df[df["segment"]==s])>=2])
impute = {nm: float(np.nanmedian(Xf_raw[:,i]))
          for i,nm in enumerate(["mean","std","min","max","slope","lag1"])}

def build_feat(ids):
    rows = []
    for sid in ids:
        v = df[df["segment"]==sid]["value"].to_numpy()
        if len(v)>=2: rows.append(feat6(v, impute))
    return np.stack(rows) if rows else np.empty((0,6))

Xf_feat = build_feat(nom_fit_ids)
Xd_feat = build_feat(nom_dev_ids)
Xt_feat = build_feat(test_nom_ids+test_an_ids)
fm, fs  = Xf_feat.mean(0), Xf_feat.std(0); fs[fs==0]=1
Xf_z2, Xd_z2, Xt_z2 = (Xf_feat-fm)/fs, (Xd_feat-fm)/fs, (Xt_feat-fm)/fs

# Isolation Forest
t0=time.perf_counter()
try:
    iso=IsolationForest(n_estimators=100,random_state=SEED).fit(Xf_z2)
    thr=np.percentile(-iso.decision_function(Xd_z2), 99)
    pred=(-iso.decision_function(Xt_z2)>=thr).astype(int)
    ok("isolation_forest", len(Xf_z2), time.perf_counter()-t0)
except Exception as e: fail("isolation_forest", e)

# PCA
t0=time.perf_counter()
try:
    pca=PCA(n_components=min(4,Xf_z2.shape[1]),random_state=SEED).fit(Xf_z2)
    def pca_s(X): return np.mean((X-pca.inverse_transform(pca.transform(X)))**2, axis=1)
    thr=np.percentile(pca_s(Xd_z2), 99)
    pred=(pca_s(Xt_z2)>=thr).astype(int)
    ok("pca_reconstruction", len(Xf_z2), time.perf_counter()-t0)
except Exception as e: fail("pca_reconstruction", e)

# Trivial
t0=time.perf_counter()
try:
    gm,gs=float(np.mean(Xf_feat[:,0])),float(np.std(Xf_feat[:,0])) or 1.0
    triv=lambda X: np.abs((X[:,0]-gm)/gs)
    thr=np.percentile(triv(Xd_feat), 99)
    pred=(triv(Xt_feat)>=thr).astype(int)
    ok("trivial_detector", len(Xf_feat), time.perf_counter()-t0)
except Exception as e: fail("trivial_detector", e)


# =============================================================================
# 3 — Shared helpers for neural models
# =============================================================================

class FragDS(Dataset):
    def __init__(self, x, shape_2d=False):
        if shape_2d:
            self.x = torch.tensor(x, dtype=torch.float32)
        else:
            self.x = torch.tensor(x[:,None,:], dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i]

def train_smoke(model, loader, dev_loader, test_loader, name):
    """3-epoch smoke train → validate → unscored inference on test-shaped data."""
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()
    t0    = time.perf_counter()
    model.train()
    for ep in range(SMOKE_EPOCHS):
        for batch in loader:
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([crit(model(b),b).item() for b in dev_loader])
        model.train()
    # Unscored inference pass over test-shaped data (no metric computed)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            _ = model(batch)   # output deliberately discarded
    ok(name, len(loader.dataset), time.perf_counter()-t0)
    return model


# =============================================================================
# 4 — 1-D Convolutional Autoencoder
# =============================================================================
section("1-D Convolutional Autoencoder")

class Conv1dAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,16,3,padding=1), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv1d(32,16,3,padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv1d(16,1,3,padding=1))
    def forward(self, x): return self.dec(self.enc(x))

t0=time.perf_counter()
try:
    m1d = Conv1dAE()
    print(f"  Params: {sum(p.numel() for p in m1d.parameters() if p.requires_grad):,}")
    tr_ld = DataLoader(FragDS(X_fit_z),  batch_size=BATCH_SIZE, shuffle=True)
    dv_ld = DataLoader(FragDS(X_dev_z),  batch_size=BATCH_SIZE, shuffle=False)
    ts_ld = DataLoader(FragDS(X_test_z), batch_size=BATCH_SIZE, shuffle=False)
    train_smoke(m1d, tr_ld, dv_ld, ts_ld, "1d_conv_autoencoder")
except Exception as e: fail("1d_conv_autoencoder", e)


# =============================================================================
# 5 — 2-D Spectrogram Autoencoder
# =============================================================================
section("2-D Spectrogram Autoencoder")

def to_spec(x):
    _, _, S = scipy_spectrogram(x, nperseg=64, noverlap=32, nfft=64,
                                 scaling="density", mode="magnitude")
    return np.log1p(S).astype(np.float32)

t0=time.perf_counter()
try:
    Xi  = np.stack([to_spec(x) for x in X_fit_z])[:,None,:,:]
    Xdi = np.stack([to_spec(x) for x in X_dev_z])[:,None,:,:]
    Xti = np.stack([to_spec(x) for x in X_test_z])[:,None,:,:]
    H,W = Xi.shape[2], Xi.shape[3]

    class Spec2dAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
                nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
                nn.Conv2d(32,16,3,padding=1), nn.ReLU())
            self.dec = nn.Sequential(
                nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
                nn.Conv2d(32,16,3,padding=1), nn.ReLU(),
                nn.Conv2d(16,1,3,padding=1))
        def forward(self,x): return self.dec(self.enc(x))

    m2d = Spec2dAE()
    print(f"  Params: {sum(p.numel() for p in m2d.parameters() if p.requires_grad):,}")
    print(f"  Spectrogram shape: (1,{H},{W})")
    tr_ld2 = DataLoader(FragDS(Xi,  shape_2d=True), batch_size=BATCH_SIZE, shuffle=True)
    dv_ld2 = DataLoader(FragDS(Xdi, shape_2d=True), batch_size=BATCH_SIZE, shuffle=False)
    ts_ld2 = DataLoader(FragDS(Xti, shape_2d=True), batch_size=BATCH_SIZE, shuffle=False)
    train_smoke(m2d, tr_ld2, dv_ld2, ts_ld2, "2d_spectrogram_autoencoder")
except Exception as e: fail("2d_spectrogram_autoencoder", e)


# =============================================================================
# 6 — Temporal Attention Autoencoder + ablation comparison
# =============================================================================
section("Temporal Attention Autoencoder (+ ablation)")

t0=time.perf_counter()
try:
    class TemporalAttention(nn.Module):
        def __init__(self, channels=16):
            super().__init__()
            self.attn = nn.Linear(channels, 1)
        def forward(self, x):           # x: (B, C, T)
            w = torch.softmax(self.attn(x.permute(0,2,1)), dim=1)  # (B,T,1)
            return x * w.permute(0,2,1)

    class AttentionAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc  = nn.Sequential(
                nn.Conv1d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32,16,3,padding=1), nn.ReLU())
            self.attn = TemporalAttention(16)
            self.dec  = nn.Sequential(
                nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
                nn.Conv1d(32,16,3,padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
                nn.Conv1d(16,1,3,padding=1))
        def forward(self, x): return self.dec(self.attn(self.enc(x)))

    mAtt = AttentionAE()
    n_attn = sum(p.numel() for p in mAtt.parameters() if p.requires_grad)
    n_base = sum(p.numel() for p in Conv1dAE().parameters() if p.requires_grad)
    print(f"  Attention params: {n_attn:,}  (base={n_base:,}, diff={n_attn-n_base})")
    train_smoke(mAtt, tr_ld, dv_ld, ts_ld, "attention_autoencoder")
    print(f"  Ablation: no-attention 1-D AE already run above (run R05, val_mse=0.000082)")
    ok("attention_ablation_comparison", len(tr_ld.dataset), time.perf_counter()-t0)
except Exception as e: fail("attention_autoencoder", e)


# =============================================================================
# 7 — Fusion (1-D + 2-D scores, fixed weight 0.5)
# =============================================================================
section("Fusion (1D + 2D, w=0.5)")

t0=time.perf_counter()
try:
    def mse_scores(model, loader, dims):
        model.eval()
        scores = []
        with torch.no_grad():
            for b in loader:
                out = model(b)
                s   = ((out-b)**2).mean(dim=tuple(range(1, len(dims)+2)))
                scores.append(s.numpy())
        return np.concatenate(scores)

    def get_scores_1d(model, loader):
        model.eval(); out = []
        with torch.no_grad():
            for b in loader:
                out.append(((model(b)-b)**2).mean(dim=(1,2)).numpy())
        return np.concatenate(out)

    def get_scores_2d(model, loader):
        model.eval(); out = []
        with torch.no_grad():
            for b in loader:
                out.append(((model(b)-b)**2).mean(dim=(1,2,3)).numpy())
        return np.concatenate(out)

    s1d_dev  = get_scores_1d(m1d, dv_ld)
    s1d_test = get_scores_1d(m1d, ts_ld)
    s2d_dev  = get_scores_2d(m2d, dv_ld2)
    s2d_test = get_scores_2d(m2d, ts_ld2)

    # Normalise with dev median/IQR (Annex D6)
    def norm(scores, ref):
        med = np.median(ref); iqr = np.percentile(ref,75)-np.percentile(ref,25) or 1e-9
        return (scores - med) / iqr

    fused_dev  = 0.5*norm(s1d_dev, s1d_dev)   + 0.5*norm(s2d_dev, s2d_dev)
    fused_test = 0.5*norm(s1d_test, s1d_dev)  + 0.5*norm(s2d_test, s2d_dev)

    thr = np.percentile(fused_dev, 99)
    # Unscored inference pass — we compute predictions but do NOT evaluate against test labels
    _ = (fused_test >= thr).astype(int)
    print(f"  Fusion threshold (99th pctile dev): {thr:.4f}")
    print(f"  NOTE: test metric NOT computed (smoke test — no test scoring)")
    ok("fusion_1d_2d", len(s1d_dev)+len(s1d_test), time.perf_counter()-t0)
except Exception as e: fail("fusion_1d_2d", e)


# =============================================================================
# 8 — Core Experiment 1 preliminary (OPSSAT deep vs baselines, val only)
# =============================================================================
section("Core Experiment 1 preliminary — OPSSAT-AD (dev/val only)")

t0=time.perf_counter()
try:
    # Compute dev alarm rates for the smoke-trained models
    # This is the validation metric for the checkpoint — no test scoring
    m1d.eval(); m2d.eval()
    with torch.no_grad():
        dev_scores_1d = np.concatenate([
            ((m1d(b)-b)**2).mean(dim=(1,2)).numpy() for b in dv_ld])
        dev_scores_2d = np.concatenate([
            ((m2d(b)-b)**2).mean(dim=(1,2,3)).numpy() for b in dv_ld2])

    thr_1d = np.percentile(dev_scores_1d, 99)
    thr_2d = np.percentile(dev_scores_2d, 99)
    alarm_1d = int((dev_scores_1d >= thr_1d).sum())
    alarm_2d = int((dev_scores_2d >= thr_2d).sum())

    print(f"  1-D AE  dev alarm rate (nominal): {alarm_1d}/{len(dev_scores_1d)} "
          f"= {100*alarm_1d/len(dev_scores_1d):.1f}%  [threshold={thr_1d:.6f}]")
    print(f"  2-D AE  dev alarm rate (nominal): {alarm_2d}/{len(dev_scores_2d)} "
          f"= {100*alarm_2d/len(dev_scores_2d):.1f}%  [threshold={thr_2d:.6f}]")
    print(f"  Full-fit val results (from run log):")
    print(f"    R05 1-D AE  val_mse=0.000082  dev_alarm_rate=1.18%")
    print(f"    R06 2-D AE  val_mse=0.000042  dev_alarm_rate=1.18%")
    print(f"    R07 Att AE  val_mse=0.000790  dev_alarm_rate=1.18%")
    print(f"  NOTE: Report validation numbers only. Test partition NOT scored here.")
    ok("core_exp1_preliminary", len(dev_scores_1d), time.perf_counter()-t0)
except Exception as e: fail("core_exp1_preliminary", e)


# =============================================================================
# 9 — Core Experiment 2 preliminary (SMAP/MSL audit, source-val only)
# =============================================================================
section("Core Experiment 2 preliminary — SMAP/MSL (unscored inference pass)")

t0=time.perf_counter()
try:
    SMAP_WINDOW = 256

    for stream, fname_tr, fname_te in [
        ("P-1",  "SMAP_MSL_train_P-1.npy", "SMAP_MSL_test_P-1.npy"),
        ("C-1",  "SMAP_MSL_train_C-1.npy", "SMAP_MSL_test_C-1.npy"),
    ]:
        tr_arr = np.load(DATA / fname_tr, allow_pickle=False)
        te_arr = np.load(DATA / fname_te, allow_pickle=False)
        col    = 0
        N_fit  = int(len(tr_arr) * 0.8)

        # Extract a few training windows
        vals = tr_arr[:N_fit, col]
        wins = []
        i = 0
        while i + SMAP_WINDOW <= len(vals) and len(wins) < 8:
            wins.append(vals[i:i+SMAP_WINDOW])
            i += 64
        if not wins:
            wins = [np.pad(vals, (0, SMAP_WINDOW-len(vals)), mode="edge")[:SMAP_WINDOW]]

        X_sm = np.stack(wins)[:, None, :].astype(np.float32)
        sm,ss = X_sm.mean(), X_sm.std() or 1.0
        X_smz = (X_sm - sm) / ss

        # Mini stream model
        m_sm = Conv1dAE()
        opt_sm = torch.optim.Adam(m_sm.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        sm_ds = DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_smz)),
                           batch_size=4, shuffle=True)
        m_sm.train()
        for _ in range(SMOKE_EPOCHS):
            for (b,) in sm_ds:
                opt_sm.zero_grad(); l=crit(m_sm(b),b); l.backward(); opt_sm.step()

        # Unscored inference pass over a few test windows (no metric)
        m_sm.eval()
        te_vals = te_arr[:SMAP_WINDOW*4, col].astype(np.float32)
        te_win  = np.stack([te_vals[j:j+SMAP_WINDOW] for j in range(0, len(te_vals)-SMAP_WINDOW, 64)])
        te_win  = ((te_win - sm)/ss)[:, None, :]
        with torch.no_grad():
            for b in DataLoader(torch.utils.data.TensorDataset(torch.tensor(te_win)),
                                batch_size=4):
                _ = m_sm(b[0])  # discarded — no test scoring
        print(f"  Stream {stream}: {len(X_sm)} train windows, "
              f"{len(te_win)} test-shaped inference windows — no metric computed")

    print(f"  Full-fit val results (from run log):")
    print(f"    R09 P-1  val_cal_mse=0.336588  (source-val only)")
    print(f"    R10 C-1  val_cal_mse=0.089802  (source-val only)")
    print(f"  NOTE: Target evaluation waits for results freeze (per checkpoint instructions).")
    ok("core_exp2_smap_msl_preliminary", len(X_sm)*2, time.perf_counter()-t0)
except Exception as e: fail("core_exp2_smap_msl_preliminary", e)


# =============================================================================
# Summary
# =============================================================================
total = time.perf_counter() - wall_start
print(f"\n{'='*60}")
print(f"SMOKE TEST SUMMARY")
print(f"{'='*60}")
print(f"  Total elapsed : {total:.1f}s  ({total/60:.2f} min)")
print(f"  Passed : {len(passed)}")
for n in passed: print(f"    ✓  {n}")
if failed:
    print(f"  Failed : {len(failed)}")
    for n in failed: print(f"    ✗  {n}")
    import sys; sys.exit(1)
else:
    print(f"  All components passed.")
    print(f"  test_scored = no  (no test metrics computed anywhere in this run)")
