import time, numpy as np

L, SUBSET, BATCH = 256, 1000, 256
STREAMS = {"SMAP P-1": "../SMAP_MSL_test_P-1.npy", "MSL C-1": "../SMAP_MSL_test_C-1.npy"}

def make_score_fn():
    def score(x):   # dummy: replace with your model later
        return ((x - x.mean(1, keepdims=True)) ** 2).mean(1)
    return score

def windows(x, start, n):
    idx = np.arange(start, start + n)[:, None] + np.arange(L)[None, :]
    return x[idx]

score, total_s = make_score_fn(), 0.0
for name, f in STREAMS.items():
    x = np.load(f)[:, 0].astype(np.float32)
    n_all = len(x) - L + 1
    n = min(SUBSET, n_all)
    t0 = time.perf_counter()
    for s in range(0, n, BATCH):
        score(windows(x, s, min(BATCH, n - s)))
    dt = time.perf_counter() - t0
    proj = dt / n * n_all
    total_s += proj
    print(f"{name}: {n} windows in {dt:.3f}s; full stream {n_all} windows -> projected {proj:.1f}s")
print(f"Projected total: {total_s:.1f}s = {total_s/60:.2f} min")
