# 22AIE304 Deep Learning — End-Semester Project
## Team 05 · Card I1 · Satellite Telemetry Anomaly Detection

**Members:** K Hemalatha Reddy · B Parameshwar Reddy · Jagan Sai Kumar  
**Repository:** https://github.com/paramesh502/dl-end-sem  
**Checkpoint tag:** `checkpoint`

---

## One-command smoke test (checkpoint requirement)

```bash
source .venv/bin/activate
python3 scripts/smoke_test.py
```

Runs every required component end-to-end on a small subset.  
Takes **< 5 minutes**. Computes **no test metric** (`test_scored = no` throughout).  
Override the data path with `--data <path>` if your I1 data folder is elsewhere.

---

## Setup

```bash
cd /path/to/22AIE304_Team05
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Data files expected at `Release_Bundle_RevD1/Release_Bundle_RevD1/I1/`:
- `segments.csv`, `dataset.csv`, `labeled_anomalies.csv`
- `OPSSAT_nominal_dev_segments.csv`
- `SMAP_MSL_train_P-1.npy`, `SMAP_MSL_test_P-1.npy`
- `SMAP_MSL_train_C-1.npy`, `SMAP_MSL_test_C-1.npy`

The four `.npy` files are not in the repository (excluded by `.gitignore`).  
They are issued with the course bundle.

---

## Reproduce individual fits

Every script runs from the project root with the venv active:

```bash
# Week 2 — splits and baselines
python3 scripts/generate_splits.py
python3 scripts/run_baselines.py

# Week 3 — neural models
python3 scripts/week3_1d_autoencoder.py
python3 scripts/week3_2d_autoencoder.py

# Week 4 — attention, fusion, SMAP/MSL
python3 scripts/week4_attention_autoencoder.py
python3 scripts/week4_fusion.py
python3 scripts/week4_smap_msl.py
```

All scripts use `seed=22`, at most 30 epochs / 30-minute elapsed wall-clock.  
No test metrics are computed during threshold calibration.

---

## Project structure

```
scripts/           All pipeline scripts (splits → baselines → models)
results/           CSV split files, result logs, trained model weights (.pt)
proposal/          T05_I1_Proposal.docx, T05_I1_ReadinessNote.docx
Release_Bundle_RevD1/  Assignment docs and OPSSAT-AD CSV data files
STUDY_GUIDE.md     Complete project study guide with all results
T05_I1_runlog.csv  Run log (one row per fit, Annex D0 format)
requirements.txt   Pinned Python dependencies
```

---

## Key results (OPSSAT-AD test set — 113 anomalous / 416 nominal)

| Method | F1 |
|---|---|
| Best classical baseline (PCA) | 0.0630 |
| 1-D Convolutional Autoencoder | 0.1417 |
| 2-D Spectrogram Autoencoder | 0.1311 |
| Attention Autoencoder | 0.0480 |
| **Fusion (1D + 2D, w=0.5)** | **0.1875** |

All thresholds at 99th percentile of nominal-dev scores. No test labels used for calibration.
