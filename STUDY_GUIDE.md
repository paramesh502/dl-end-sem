# T05\_I1 — Satellite Telemetry Anomaly Detection
## 22AIE304 Deep Learning — Week 2 Proposal
**Team 05 · Card I1 · Revision D1**
K Hemalatha Reddy · B Parameshwar Reddy · Jagan Sai Kumar

---

## 1. Question and Claim

**Research question.** Can a small nominal-only deep autoencoder reliably detect anomalies in
satellite telemetry fragments, and does adding a spectrogram representation improve detection
compared to classical baselines?

**Core experiment 1.** One small 1-D convolutional autoencoder versus Isolation Forest, PCA
reconstruction error, and a trivial threshold detector on OPSSAT-AD under the nominal-only
regime. All methods are calibrated at the 99th percentile of nominal-development scores.

**Core experiment 2.** Audit of SMAP P-1 and MSL C-1: a no-attention 1-D autoencoder fitted
independently per stream versus the same three classical controls, using the published train/test
files and the point-adjusted F1 as a clearly labelled secondary audit figure.

**Scope and limits.** Both experiments use nominal-only training. Any supervised formulation is a
separately labelled cross-regime demonstration and does not support architectural conclusions.
The point-adjusted F1 for SMAP/MSL is an audit figure and is never used for threshold selection.
Conclusions are limited to the tested algorithms, the declared label-access regime, and the native
protocol of each benchmark.

---

## 2. Data and EDA

**OPSSAT-AD** (Zenodo record 12588359): 2,123 single-channel telemetry fragments across 9 channels
(CADC0872–CADC0894). Fragment-level anomaly labels: 434 anomalous (20.44%), 1,689 nominal. The
provided train/test split gives 1,594 training fragments and 529 test fragments. Fragment lengths
range from 8 to 1,040 samples; 1,748 of 2,123 (82.3%) are shorter than 256 samples, so the
256-sample resampling predominantly upsamples. Two channels are thin: CADC0886 (5 nominal
training fragments, 0 in development) and CADC0890 (3 nominal training fragments, 0 nominal
test fragments).

**SMAP/MSL** (issued with cards, from TranAD commit 7ffb98d): pre-scaled continuous streams in
[−1, 1] (test-set scaling — inherited limitation, not fixable). P-1: train 2,872 × 25, test
8,505 × 25, 3 anomaly intervals covering 751 test points (8.83%). C-1: train 2,158 × 55, test
2,264 × 55, 2 anomaly intervals covering 312 test points (13.78%). Column 0 is the primary
sensor input per Annex D6. C-1's training stream reaches 2.19 while its test stream spans
exactly [−1, 1], consistent with test-set scaling.

**Split ID files:** `T05_I1_opssat.csv` and `T05_I1_smap_msl.csv` (see §3).

---

## 3. Split Protocol

**OPSSAT-AD.**
Split unit: fragment (one contiguous telemetry segment per channel). The provided train/test
split is used directly. Within the 1,273 nominal training fragments, segment IDs are ordered by
SHA-256 of `22|<segment_id>` and the first floor(20%) = 254 are reserved as nominal development;
the remaining 1,019 are fitting fragments. Anomalous training fragments (321) are excluded from
the nominal-only core. Leakage check: no fragment appears in both fitting and development (verified
by reproduced split matching the issued `OPSSAT_nominal_dev_segments.csv` exactly — 0 differing
IDs). Test fragments are kept strictly unseen for all threshold decisions.

**SMAP/MSL.**
Split unit: contiguous time-step range within a stream. The published train/test files are used
as issued. The nominal training stream is divided: first 80% as fitting (P-1: steps 0–2296,
C-1: steps 0–1725), a 256-step guard gap (P-1: 2297–2552, C-1: 1726–1981), and the last 20%
as calibration (P-1: 2553–2871, C-1: 1982–2157). Anomaly detection uses 256-step windows,
stride 64 for training and stride 1 for scoring. Padded values are masked out of losses.

**Split files.** `T05_I1_opssat.csv` · `T05_I1_smap_msl.csv` · `T05_I1_split_summary.csv`

---

## 4. Non-DL Baseline

All baselines use training-only standardisation and the same feature representation:
mean, std, min, max, slope, lag-1 correlation (6 features). Undefined features are imputed
by training median. All thresholds are calibrated at the 99th percentile of nominal
development / calibration scores. No test labels were used for any threshold decision.

**OPSSAT-AD results** (validation metrics on nominal dev only; test metrics on 529 test fragments):

| Method | Dev alarm rate | Test Precision | Test Recall | Test F1 | Support (pos/neg) |
|---|---|---|---|---|---|
| Isolation Forest (100 trees, seed=22) | 3/254 (1.2%) | 0.1111 | 0.0088 | 0.0164 | 113 / 416 |
| PCA recon. error (4 components, ≥95% var) | 3/254 (1.2%) | 0.2857 | 0.0354 | 0.0630 | 113 / 416 |
| Trivial (mean distance) | 3/254 (1.2%) | 0.1111 | 0.0088 | 0.0164 | 113 / 416 |
| Constant-all-nominal (floor) | — | 0.0000 | 0.0000 | 0.0000 | 113 / 416 |

**Note:** Dev partition contains nominal fragments only. Precision/recall/F1 are undefined on dev.
Dev alarm rate = false-alarm rate on nominal data.

**SMAP P-1 results** (unadjusted point; point-adjusted is audit-only):

| Method | Threshold | Unadj. P | Unadj. R | Unadj. F1 | Pt-adj F1 (AUDIT) |
|---|---|---|---|---|---|
| Isolation Forest | 0.0294 | 0.1656 | 0.7736 | 0.2728 | 0.3391 |
| PCA (5 components) | 0.0908 | 0.3295 | 0.4581 | 0.3833 | 0.6821 |
| Trivial (mean dist.) | 0.6668 | 0.1292 | 0.8602 | 0.2247 | 0.2565 |

**MSL C-1 results:**

| Method | Threshold | Unadj. P | Unadj. R | Unadj. F1 | Pt-adj F1 (AUDIT) |
|---|---|---|---|---|---|
| Isolation Forest | −0.0183 | 0.1505 | 0.9455 | 0.2597 | 0.2726 |
| PCA (4 components) | 0.0027 | 0.1532 | 0.7949 | 0.2569 | 0.3128 |
| Trivial (mean dist.) | 1.0178 | 0.1783 | 0.8814 | 0.2967 | 0.3300 |

SMAP/MSL test-set scaling is an inherited limitation (noted per D6).
Point-adjusted F1 intentionally uses test labels in the scoring adjustment; never used for tuning.

---

## 5. Literature

Anomaly detection in satellite telemetry is a well-studied problem.
Hundman et al. [1] introduced the SMAP and MSL datasets with the LSTM-based telemanom detector and
the point-adjusted F1 metric, which motivated the "published criticism of SMAP/MSL" this project
is testing. Xu et al. [2] introduced TFAD and reviewed the limitations of point-adjusted evaluation.
Schmidl et al. [3] benchmarked 71 time-series anomaly detectors and found classical methods
competitive with deep learning under strict evaluation. Blazquez-García et al. [4] surveyed
anomaly detection for time series, cataloguing reconstruction-based autoencoders, isolation-based
and distance-based methods. Audibert et al. [5] proposed USAD (Unsupervised Anomaly Detection on
multivariate time series) using dual-autoencoder adversarial training. Ruff et al. [6] proposed
Deep SVDD for one-class classification using deep networks, closely related to the nominal-only
regime in this project. Park et al. [7] proposed LSTM-VAE, combining recurrent and variational
autoencoders for anomaly detection. He et al. [8] proposed Isolation Forest, the primary classical
baseline here. Jolliffe & Cadima [9] review PCA and its use as a reconstruction baseline.
Leblanc & Bland [10] discuss threshold calibration and the effect of base rate on precision-recall
tradeoffs, directly relevant to the low-anomaly-fraction OPSSAT-AD fragments.

References:
[1] Hundman et al. "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." KDD 2018.
[2] Xu et al. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." ICLR 2022.
[3] Schmidl et al. "Anomaly Detection in Time Series: A Comprehensive Evaluation." VLDB 2022.
[4] Blazquez-García et al. "A Review on Outlier/Anomaly Detection in Time Series Data." ACM CSUR 2021.
[5] Audibert et al. "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." KDD 2020.
[6] Ruff et al. "Deep One-Class Classification." ICML 2018.
[7] Park et al. "A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder." RA-L 2018.
[8] Liu et al. "Isolation Forest." ICDM 2008.
[9] Jolliffe & Cadima. "Principal Component Analysis: A Review and Recent Developments." Phil. Trans. R. Soc. A 2016.
[10] Leblanc & Bland. "How to Use the False Positive Rate and the Precision–Recall Trade-Off." BJCP 2020.

---

## 6. Plan to Week 5 Demo

| Component | Owner | Reviewer | Target Week |
|---|---|---|---|
| 1-D convolutional autoencoder (OPSSAT-AD) | Parameshwar | Jagan | Week 3 |
| 2-D spectrogram autoencoder (OPSSAT-AD) | Hemalatha | Parameshwar | Week 3 |
| Temporal attention + ablation | Jagan | Hemalatha | Week 4 |
| Fusion (recon score + image score, weight 0.5) | Parameshwar | Jagan | Week 4 |
| SMAP/MSL stream autoencoders (P-1, C-1) | Hemalatha | Jagan | Week 4 |
| Core Experiment 1 (OPSSAT deep vs. baselines) | Jagan | Parameshwar | Week 5 |
| Core Experiment 2 (SMAP/MSL audit) | Parameshwar | Hemalatha | Week 5 |
| Preliminary demo run | All | All | Week 5 (30 Oct 2026) |

All components use the card's reference setting: 30-epoch / 30-minute limit, seed 22.
Attention is temporal attention within the 1-D autoencoder, with the no-attention fit as
its ablation. Fusion weight is fixed at 0.5 (nominal-only development cannot optimise F1).
No-attention models are used for the named fusion (D0 rule).

---

## 7. Risks and Requests

**Compute budget.**
The reference machine is 6 cores, 16 GB RAM (D0). Week-1 fit (2-D spectrogram autoencoder on
OPSSAT-AD, 30 epochs/30 min) completed successfully on our macOS machine (Python 3.14.7, Apple
M-series or Intel CPU). SMAP/MSL stride-1 scoring over 8,505-step test stream is the most
expensive inference step. No GPU is required; MPS acceleration may help if available.

Required neural fits (Card I1 D0 table): 5 total — OPSSAT sequential + image + attention +
two stream fits (P-1, C-1). This is well within the 24-hour elapsed budget.

**Data issues.**
SMAP/MSL files are issued with the cards (original S3 source returns 403). Pre-scaled using
test-set min/max (inherited limitation, reported per D6). Both streams verified against the
labeled_anomalies.csv interval file.

**OPSSAT thin channels.** CADC0886 and CADC0890 have very few nominal training fragments
(5 and 3 respectively, with none in development for both). Channel-level statistics should
account for this; per-channel evaluation will show these channels separately.

**No approval requests at this stage.** No dataset substitution, scope change, or protocol
deviation is requested.

---

## 8. What We've Run So Far — Actual Results

All numbers are taken directly from the result files in `results/`. Thresholds are calibrated at
the 99th percentile of nominal-only dev/cal scores. No test labels were used for any threshold
decision.

**OPSSAT-AD (fragment-level, test set — 113 anomalous / 416 nominal):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest | 0.1111 | 0.0088 | 0.0164 |
| PCA recon. error | 0.2857 | 0.0354 | 0.0630 |
| Trivial detector | 0.1111 | 0.0088 | 0.0164 |
| 1-D Autoencoder (no attention) | 0.6429 | 0.0796 | 0.1417 |
| 2-D Spectrogram AE | 0.8889 | 0.0708 | 0.1311 |
| Attention Autoencoder | 0.2500 | 0.0265 | 0.0480 |
| Fusion (1D AE + 2D AE, w=0.5) | 0.8000 | 0.1062 | 0.1875 |

**SMAP P-1 (point-level, 8505 test steps):**

| Method | Unadj. P | Unadj. R | Unadj. F1 | Pt-adj F1 (AUDIT) |
|---|---|---|---|---|
| Isolation Forest | 0.1656 | 0.7736 | 0.2728 | 0.3391 |
| PCA (5 components) | 0.3295 | 0.4581 | 0.3833 | 0.6821 |
| Trivial (mean dist.) | 0.1292 | 0.8602 | 0.2247 | 0.2565 |
| 1-D Autoencoder (no attention) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**MSL C-1 (point-level, 2264 test steps):**

| Method | Unadj. P | Unadj. R | Unadj. F1 | Pt-adj F1 (AUDIT) |
|---|---|---|---|---|
| Isolation Forest | 0.1505 | 0.9455 | 0.2597 | 0.2726 |
| PCA (4 components) | 0.1532 | 0.7949 | 0.2569 | 0.3128 |
| Trivial (mean dist.) | 0.1783 | 0.8814 | 0.2967 | 0.3300 |
| 1-D Autoencoder (no attention) | 0.1865 | 0.9840 | 0.3136 | 0.3179 |

Note: Pt-adj F1 is an AUDIT figure, never used for threshold selection.

---

## 9. Week 4 — What Was Done

### 9.1 Temporal Attention

Temporal attention is a lightweight mechanism that learns to weight the 64 time positions in the
bottleneck representation — shaped (batch, 16, 64) — rather than treating all positions equally.
For each of the 64 time steps, a `Linear(16→1)` layer produces a scalar score; softmax normalises
these into attention weights α that sum to 1 across the time dimension. The bottleneck tensor is
then multiplied element-wise by α, so the model can amplify the most informative time windows and
suppress less relevant ones when reconstructing the fragment.

The entire module adds only **17 parameters** to the base autoencoder: the `Linear(16, 1)` layer
has 16 weights and 1 bias. The base 1-D autoencoder has 6,353 trainable parameters; the attention
variant has 6,370.

### 9.2 Ablation Comparison

The no-attention 1-D autoencoder from Week 3 (F1 = 0.1417) is the ablation baseline per Card I1.
The attention autoencoder achieved F1 = **0.0480** on the same OPSSAT-AD test set (precision
0.2500, recall 0.0265).

The attention mechanism did **not** improve over the base architecture in this case. The attention
AE F1 (0.0480) is lower than the no-attention ablation baseline (0.1417), meaning the base
architecture already captures enough temporal structure for reconstruction-based anomaly detection
on this dataset. One likely reason: the attention module learns to down-weight parts of the
bottleneck during training on nominal data, which can reduce reconstruction error for anomalous
fragments as well — hurting discrimination at threshold time.

### 9.3 Fusion

The fusion score combines two complementary views of each fragment:

- **1-D reconstruction error** — time-domain signal; the Conv1dAutoencoder's MSE on the raw
  resampled waveform.
- **2-D spectrogram reconstruction error** — frequency-domain signal; the SpectrogramAutoencoder's
  MSE on the log-magnitude spectrogram.

Per Annex D0, the fusion weight is fixed at **0.5** (equal weight). The reason is that nominal-only
development cannot optimise the anomalous-class F1 directly — any weight search using dev scores
would implicitly optimise for nominal reconstruction quality, not anomaly detection. Searching
over weights on dev data would therefore be methodologically unsound.

Before combining, both scores are standardised to a common scale using the **median and IQR** of
dev scores fitted on nominal data only. Median/IQR normalisation is used (rather than mean/std)
because it is robust to outliers on nominal data. The fused score is:

```
fused = 0.5 × (score_1d − median_1d) / IQR_1d  +  0.5 × (score_2d − median_2d) / IQR_2d
```

The threshold is calibrated at the **99th percentile of fused dev scores** (threshold = 3.9778,
dev alarm rate 3/254 = 1.18%). On the test set, fusion achieved precision 0.8000, recall 0.1062,
**F1 = 0.1875** — the best single F1 score across all OPSSAT-AD methods run so far.

### 9.4 SMAP/MSL Windowed Evaluation

Unlike OPSSAT-AD's fragment-level evaluation (one label per fragment), SMAP and MSL require
**point-level** predictions at each test time step.

The approach uses a **256-step sliding window** scored by its mean MSE. During training, windows
are extracted with stride 64 (P-1: 32 training windows, C-1: 23 training windows). During
test-time scoring, stride 1 is used so every possible window is scored. Each test time-step's
anomaly score is then the **mean of all window scores that cover it** — giving a single score per
test point across the full stream (8,505 points for P-1, 2,264 points for C-1). The threshold is
the 99th percentile of calibration-window scores.

The **unadjusted F1** is the primary metric: point-level predictions are compared directly to
point-level labels without any interval manipulation.

The **point-adjusted F1** (AUDIT ONLY) fills entire true anomaly intervals whenever any predicted
point intersects them. This inflates the metric significantly (P-1 classical baselines go from
unadj. F1 ≈ 0.27–0.38 up to pt-adj F1 ≈ 0.26–0.68) and is clearly labelled AUDIT ONLY per
Annex D6. It is never used for threshold selection or architectural decisions.

Results on P-1: the stream autoencoder produced unadj. F1 = **0.0000** (all predictions were
below threshold — the model did not flag any test points as anomalous). On C-1 it achieved
unadj. F1 = **0.3136**, which exceeds all three classical baselines on that stream.

---

## 10. Week 5 Plan

The Week 5 demo is on **30 Oct 2026**. The following tasks remain.

**Core Experiment 1 — OPSSAT deep vs. baselines (mostly done)**

- All 5 required neural fits are logged: 1-D AE (Week 3), 2-D AE (Week 3), Attention AE
  (Week 4), SMAP P-1 stream AE (Week 4), MSL C-1 stream AE (Week 4). Verify each is present
  in the Card I1 D0 results table.
- Prepare the final OPSSAT-AD comparison table (baselines + 1-D AE + 2-D AE + Attention AE +
  Fusion) for the report. All rows are already available from the result files.

**Core Experiment 2 — SMAP/MSL audit (done)**

- Stream autoencoder results for P-1 and C-1 are logged in `results/smap_msl_results.txt`.
- Final audit table comparing the no-attention 1-D AE vs classical baselines on both streams
  is ready (see §8 above).
- Ensure the point-adjusted F1 is clearly labelled AUDIT ONLY everywhere it appears in the
  report and presentation slides.

**Demo preparation**

- Run all scripts in sequence on a clean run from project root to confirm reproducibility:
  Week 2 baselines → Week 3 1-D AE → Week 3 2-D AE → Week 4 Attention AE →
  Week 4 Fusion → Week 4 SMAP/MSL stream AEs.
- Verify `seed=22` is set consistently in every script.
- Confirm that no test labels are used for any threshold decision (already verified per D0).
- Prepare a 5-minute verbal walkthrough covering: data split → model architecture →
  training regime → calibration → results table → ablation interpretation → fusion rationale.
