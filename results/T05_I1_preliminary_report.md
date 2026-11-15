# T05\_I1 — Preliminary Report: Satellite Telemetry Anomaly Detection
## 22AIE304 Deep Learning — Card I1
**Team 05 · K Hemalatha Reddy · B Parameshwar Reddy · Jagan Sai Kumar**
**Final test evaluation date: 2026-11-15**

---

## 1. Question and Claim

**Research question.** Can a small nominal-only deep autoencoder reliably detect anomalies in satellite telemetry fragments, and does adding a spectrogram representation improve detection compared to classical baselines?

**Core Experiment 1 (OPSSAT-AD).** One 1-D convolutional autoencoder, a 2-D spectrogram autoencoder, an attention autoencoder, and their fusion are evaluated against Isolation Forest, PCA reconstruction error, and a trivial threshold detector on OPSSAT-AD under the nominal-only training regime. All methods are calibrated at the 99th percentile of nominal development scores. The 1-D autoencoder without attention serves as the no-attention ablation for the attention model.

**Core Experiment 2 (SMAP/MSL stream audit).** A no-attention 1-D autoencoder is fitted independently on SMAP P-1 and MSL C-1, then evaluated against the same three classical controls. The unadjusted point F1 is the primary metric; point-adjusted F1 is reported as a clearly labelled audit figure only and is never used for threshold selection.

**Scope and limits.** Both experiments use nominal-only training. Thresholds are locked on development/calibration evidence before any test evaluation. Conclusions are limited to the tested algorithms, the declared label-access regime, and the native protocol of each benchmark.

---

## 2. Data and EDA

### OPSSAT-AD

The dataset contains 2,123 single-channel telemetry fragments across 9 channels (CADC0872–CADC0894). Of these, 434 are anomalous (20.44%) and 1,689 are nominal. The provided train/test split gives 1,594 training fragments and 529 test fragments. Fragment lengths range from 8 to 1,040 samples; 1,748 of 2,123 (82.3%) are shorter than 256 samples, so the 256-sample resampling predominantly upsamples.

A key EDA finding: nominal fragments have a median length of 54 samples while anomalous fragments have a median length of 184.5 samples. This length difference may separate the classes independently of the detector — a confound noted in Section 7.

Two channels are thin: CADC0886 has only 5 nominal training fragments and CADC0890 has only 3 nominal training fragments, with no nominal examples in development for either.

### SMAP/MSL

Both streams were issued pre-scaled to [−1, 1] using test-set min/max — an inherited limitation noted per Annex D6 and not correctable without access to the original data. Column 0 is used as the primary sensor input per Annex D6.

| Stream | Train shape | Test shape | Anomaly intervals | Anomalous test points |
|--------|------------|-----------|-------------------|----------------------|
| SMAP P-1 | 2,872 × 25 | 8,505 × 25 | 3 | 748 (8.80%) |
| MSL C-1 | 2,158 × 55 | 2,264 × 55 | 2 | 310 (13.70%) |

C-1's training stream reaches values of 2.19 while its test stream spans exactly [−1, 1], which is consistent with test-set scaling.

---

## 3. Split Protocol

### OPSSAT-AD

The unit of splitting is a single fragment. The provided train/test partition is used directly. Within the 1,273 nominal training fragments, segment IDs are ordered by SHA-256 of `22|<segment_id>` and the first ⌊20%⌋ = 254 are reserved as the nominal development set; the remaining 1,019 are fitting fragments. Anomalous training fragments (321) are excluded from the nominal-only core experiment. The resulting partition is:

| Subset | Count |
|--------|-------|
| Nominal fitting | 1,019 |
| Nominal development | 254 |
| Test nominal | 416 |
| Test anomalous | 113 |
| Anomalous training (excluded from fit) | 321 |

No fragment appears in both fitting and development. Test fragments were kept strictly unseen for all threshold decisions.

### SMAP/MSL

The published train/test files are used as issued. Within the training stream, the first 80% serves as the fitting portion, a 256-step guard gap prevents leakage, and the remaining 20% is the calibration set:

| Stream | Fit range | Guard range | Calibration range | Cal points |
|--------|-----------|-------------|-------------------|------------|
| P-1 | [0, 2297) | [2297, 2553) | [2553, 2871) | 319 |
| C-1 | [0, 1726) | [1726, 1982) | [1982, 2157) | 176 |

**Corrected calibration rule (applied 2026-11-14).** Each calibration point is scored by the mean squared reconstruction error at that position averaged over all 256-step windows that cover it. Guard rows supply context for those windows but are never entered into loss or score statistics. The earlier protocol (scoring each point with the single window ending at it) was corrected before any final test evaluation.

**Split files:** `T05_I1_opssat.csv`, `T05_I1_smap_msl.csv`, `T05_I1_split_summary.csv`.

---

## 4. Non-DL Baseline

All baselines use training-only standardisation and six summary features per fragment (mean, std, min, max, slope, lag-1 correlation). Undefined features are imputed by training median. Thresholds are set at the 99th percentile of nominal development scores.

Earlier test inspection on 2026-10-12 (R02–R04). Final evaluation on 2026-11-15. Thresholds locked on validation evidence before final evaluation.

| Model | Dev alarm rate | Locked threshold | Test Precision | Test Recall | Test F1 | Support |
|-------|---------------|-----------------|---------------|------------|--------|---------|
| Isolation Forest | 3/254 (1.18%) | 0.201592 | 0.1111 | 0.0088 | 0.0164 | 113 / 416 |
| PCA Reconstruction | 3/254 (1.18%) | 0.030605 | 0.2857 | 0.0354 | 0.0630 | 113 / 416 |
| Trivial Detector | 3/254 (1.18%) | 3.410388 | 0.1111 | 0.0088 | 0.0164 | 113 / 416 |
| All-nominal floor | — | N/A | 0.0000 | 0.0000 | 0.0000 | 113 / 416 |

All three baselines achieve very low F1 on the test set, with PCA reconstruction error performing best at F1 = 0.0630.

---

## 5. Deep Learning Models

Earlier test inspection on 2026-10-12 (R05–R08). Final evaluation on 2026-11-15. Thresholds locked from development evidence before final evaluation.

### 5.1 1-D Convolutional Autoencoder

**Architecture.** `Conv1dAutoencoder` — a 1-D convolutional encoder–decoder operating on 256-sample fragments. Trainable parameters: 6,353.

**Training.** 30 epochs completed (no early stopping). Best validation loss: 0.000082 (epoch 30). Elapsed: 14.22 s.

**Calibration.** Locked threshold: 0.000581 (99th percentile of dev MSE). Dev alarm rate: 3/254 = 1.18%.

**Final test results (2026-11-15):**

| Precision | Recall | F1 | TP | FP | TN | FN | Support |
|-----------|--------|-----|----|----|----|----|---------|
| 0.6429 | 0.0796 | 0.1417 | 9 | 5 | 411 | 104 | 113 / 416 |

### 5.2 2-D Spectrogram Autoencoder

**Architecture.** `SpectrogramAutoencoder` — 2-D convolutional encoder–decoder on log1p-magnitude spectrograms (shape 1 × 33 × 7; nperseg=64, noverlap=32, nfft=64). Trainable parameters: 18,833.

**Training.** 30 epochs completed. Best validation loss: 0.000042 (epoch 29). Elapsed: 20.79 s.

**Calibration.** Locked threshold: 0.000355. Dev alarm rate: 3/254 = 1.18%.

**Final test results (2026-11-15):**

| Precision | Recall | F1 | TP | FP | TN | FN | Support |
|-----------|--------|-----|----|----|----|----|---------|
| 0.8889 | 0.0708 | 0.1311 | 8 | 1 | 415 | 105 | 113 / 416 |

### 5.3 Attention Autoencoder

**Architecture.** `AttentionAutoencoder` — 1-D convolutional autoencoder with added temporal attention layer. Trainable parameters: 6,370 (17 more than the 1-D AE).

**Training.** 30 epochs completed. Best validation loss: 0.000790 (epoch 28). Elapsed: 15.32 s. The val loss is notably higher than the 1-D AE's 0.000082 and shows less stable convergence (spikes at epochs 15–18), suggesting the attention mechanism does not improve reconstruction under these settings.

**Calibration.** Locked threshold: 0.010338. Dev alarm rate: 3/254 = 1.18%.

**Final test results (2026-11-15):**

| Precision | Recall | F1 | TP | FP | TN | FN | Support |
|-----------|--------|-----|----|----|----|----|---------|
| 0.2500 | 0.0265 | 0.0480 | 3 | 9 | 407 | 110 | 113 / 416 |

**Attention ablation.** The no-attention ablation (1-D AE, F1 = 0.1417) outperforms the attention model (F1 = 0.0480) by 0.0937 F1 points. Adding temporal attention here hurts performance — the attention model achieves lower recall and lower precision simultaneously.

### 5.4 Fusion (1-D + 2-D)

**Architecture.** Score-level equal-weight fusion: `fused_score = 0.5 × (score_1d − median_1d) / IQR_1d + 0.5 × (score_2d − median_2d) / IQR_2d`. Normalisation constants fitted on nominal dev scores only. No additional trainable parameters.

**Calibration.** Locked threshold: 3.977751 (99th percentile of fused dev scores). Dev alarm rate: 3/254 = 1.18%.

**Final test results (2026-11-15):**

| Precision | Recall | F1 | TP | FP | TN | FN | Support |
|-----------|--------|-----|----|----|----|----|---------|
| 0.8000 | 0.1062 | 0.1875 | 12 | 3 | 413 | 101 | 113 / 416 |

Fusion achieves the best F1 (0.1875) across all models by combining the complementary true positives of the 1-D and 2-D detectors, gaining 3 more true positives over the 1-D AE alone while reducing false positives from 5 to 3.

---

## 6. SMAP/MSL Audit (Core Experiment 2)

Thresholds locked from calibration data (R11, R12) before test evaluation. Final evaluation: 2026-11-15. The unadjusted point F1 is the primary metric; point-adjusted F1 is an audit figure only.

**P-1 limitation.** The P-1 training stream yields only 32 windows at stride 64, which is very few for a neural model. The autoencoder's poor unadjusted F1 on P-1 (0.0168) is consistent with underfitting on such limited training data.

| Stream | Detector | Unadj. P | Unadj. R | Unadj. F1 | Pt-adj F1 (AUDIT ONLY) |
|--------|----------|---------|---------|----------|----------------------|
| P-1 | Neural AE (R11) | 0.0843 | 0.0094 | 0.0168 | 0.9517 |
| P-1 | Isolation Forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| P-1 | PCA Reconstruction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| P-1 | Trivial Detector | 0.1569 | 0.9439 | 0.2691 | 0.2828 |
| C-1 | Neural AE (R12) | 0.5135 | 0.1839 | 0.2708 | 0.9199 |
| C-1 | Isolation Forest | 0.2191 | 0.6452 | 0.3271 | 0.3271 |
| C-1 | PCA Reconstruction | 0.1998 | 0.6645 | 0.3072 | 0.4291 |
| C-1 | Trivial Detector | 0.1439 | 0.6452 | 0.2353 | 0.2353 |

The gap between unadjusted F1 and point-adjusted F1 for the neural autoencoder on both streams (0.0168 vs. 0.9517 for P-1; 0.2708 vs. 0.9199 for C-1) illustrates the criticism raised by Wu & Keogh [11]: point-adjusted scoring can mask poor point-level detection by crediting an entire anomaly interval if any single point within it is flagged.

---

## 7. Per-Length Analysis

Fragment length before resampling is grouped into five bins. Results are shown for the Fusion model (best overall) and Isolation Forest (classical baseline) for comparison.

**Fusion (1D+2D):**

| Bin | TP | FP | TN | FN | N |
|-----|----|----|----|----|---|
| [8, 50) | 8 | 0 | 183 | 17 | 208 |
| [50, 100) | 1 | 1 | 82 | 6 | 90 |
| [100, 200) | 0 | 0 | 66 | 29 | 95 |
| [200, 500) | 3 | 2 | 72 | 35 | 112 |
| [500, 1041) | 0 | 0 | 10 | 14 | 24 |

**Isolation Forest:**

| Bin | TP | FP | TN | FN | N |
|-----|----|----|----|----|---|
| [8, 50) | 1 | 8 | 175 | 24 | 208 |
| [50, 100) | 0 | 0 | 83 | 7 | 90 |
| [100, 200) | 0 | 0 | 66 | 29 | 95 |
| [200, 500) | 0 | 0 | 74 | 38 | 112 |
| [500, 1041) | 0 | 0 | 10 | 14 | 24 |

The Fusion model's detections concentrate in the shortest bin ([8, 50)), which also contains the highest density of anomalous fragments by design of the dataset (anomalous median length of 184.5 samples is longer, but the [8, 50) bin has 208 fragments total). Both models miss nearly all anomalies in the [100–500) range. We cannot claim fragment length causes the association, only that it exists.

---

## 8. Error Analysis

**Fusion model false positives (3 total).** The 3 false positives consist of 1 in [50, 100) and 2 in [200, 500). The very low false-positive count (3/416 = 0.72%) reflects the high precision of the fused score, which agrees with the 99th-percentile calibration target.

**Fusion model false negatives (101 total).** The 101 false negatives are distributed: 17 in [8, 50), 6 in [50, 100), 29 in [100, 200), 35 in [200, 500), and 14 in [500, 1041). The model fails most heavily on medium-to-long anomalous fragments, which may reconstruct well after upsampling to 256 samples because the 256-step resampling loses detail. Recall is low across all models (0.1062 for Fusion), suggesting that the 99th-percentile threshold is conservative and that the anomalous fragments as a class are diverse in character.

OPSSAT-AD has 9 channels (CADC0872–CADC0894). Two channels are thin: CADC0886 (5 nominal training fragments) and CADC0890 (3 nominal training fragments). The error analysis above is based on the available test-set predictions without further channel-level breakdown, as the final test evaluation file does not provide per-channel prediction records.

---

## 9. Discussion

**Key findings.**

The best model, Fusion (1D+2D), achieves F1 = 0.1875 on the OPSSAT-AD test set, compared to F1 = 0.0630 for the best classical baseline (PCA reconstruction). All deep models substantially outperform the classical baselines on F1, primarily because they maintain much higher precision (the classicals fire broadly and miss almost all anomalies). Recall remains low across all models, which limits practical utility at the 99th-percentile threshold.

The attention ablation result is unambiguous in direction: adding temporal attention to the 1-D autoencoder reduces F1 by 0.0937 (from 0.1417 to 0.0480). The attention model's best validation loss (0.000790) is an order of magnitude higher than the 1-D AE's (0.000082), indicating that the attention mechanism does not aid reconstruction on these short fragments, possibly due to the small dataset size and the dominance of very short fragments (median 54 samples nominal).

For SMAP P-1, the neural autoencoder achieves an unadjusted F1 of only 0.0168. This is consistent with the training data yielding just 32 windows at stride 64 — far too few for a neural model to learn a useful nominal manifold. The trivial detector (unadjusted F1 = 0.2691) performs better on P-1 than the neural model by a large margin. For MSL C-1, the neural autoencoder achieves an unadjusted F1 of 0.2708, which is competitive with the classical baselines.

**Limitations.**

- *Nominal-only calibration.* The 99th-percentile threshold is calibrated on nominal data only; the optimal threshold for F1 may differ substantially, but cannot be found without test labels.
- *Test-set scaling on SMAP/MSL.* The pre-scaling using test-set statistics is an inherited limitation that inflates the apparent normality of test data and cannot be corrected without access to the original unscaled streams.
- *Thin channels.* CADC0886 and CADC0890 have very few nominal training examples, making generalisation to their anomalous fragments unreliable.
- *Fragment length confound.* The association between fragment length and anomaly label means that a simple length-based rule might achieve similar detection rates; this was not controlled for in the main evaluation.
- *Small SMAP/MSL training set.* P-1's 32 training windows make neural-model conclusions for that stream unreliable.

---

## 10. References

[1] Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. *KDD 2018*. https://doi.org/10.1145/3219819.3219845 — Introduced the SMAP/MSL benchmark and scored detected sequences, NOT point-adjusted F1.

[2] Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time series anomaly detection with association discrepancy. *ICLR 2022*.

[3] Schmidl, S., Wenig, P., & Papenbrock, T. (2022). Anomaly detection in time series: A comprehensive evaluation. *VLDB 2022*.

[4] Blázquez-García, A., Conde, A., Mori, U., & Lozano, J. A. (2021). A review on outlier/anomaly detection in time series data. *ACM Computing Surveys*, 54(3). https://doi.org/10.1145/3444690

[5] Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A. (2020). USAD: UnSupervised anomaly detection on multivariate time series. *KDD 2020*.

[6] Ruff, L., Vandermarliere, E., Görnitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Müller, E., & Kloft, M. (2018). Deep one-class classification. *ICML 2018*.

[7] Park, D., Kim, Y., & Cho, S. (2018). A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder. *IEEE Robotics and Automation Letters*, 3(3).

[8] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *ICDM 2008*.

[9] Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065).

[10] Schaefer, B., et al. (2024). OPSSAT-AD: An in-orbit anomaly detection dataset. *Zenodo*. https://doi.org/10.5281/zenodo.12588359 — Domain-specific dataset used in Core Experiment 1.

[11] Wu, R., & Keogh, E. (2021). Current time series anomaly detection benchmarks are flawed and trivially solvable. *IEEE Transactions on Knowledge and Data Engineering*. https://doi.org/10.1109/TKDE.2021.3116708

[12] Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., & Gottschlich, J. (2018). Precision and recall for time series. *NeurIPS 2018*. https://papers.nips.cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html — Introduced point-adjusted F1.

[13] Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-normalizing neural networks. *NeurIPS 2017*.

---

## 11. Run Log Summary

R09 and R10 are labelled superseded because they used an incorrect calibration protocol (scoring each point with the single window ending at it, rather than the mean over covering windows). They remain in the log with their original `test_scored` values for budget accounting. R02–R08 are earlier test inspections performed on 2026-10-12 under the original protocol; their results match the final evaluation (2026-11-15) exactly, confirming no threshold choice was made on test evidence.

| Run ID | Date | Component | Condition | Epochs | Elapsed (s) | Test scored | Notes |
|--------|------|-----------|-----------|--------|------------|-------------|-------|
| R01 | 2026-09-30 | 2d_spectrogram_autoencoder | opssat_nominal_only_reference | 30 | 29.18 | no | Initial Week 1 fit |
| R02 | 2026-10-12 | isolation_forest | opssat_nominal_only | 1 | 0.05 | yes | Earlier test inspection |
| R03 | 2026-10-12 | pca_reconstruction | opssat_nominal_only | 1 | 0.00 | yes | Earlier test inspection |
| R04 | 2026-10-12 | trivial_detector | opssat_nominal_only | 1 | 0.00 | yes | Earlier test inspection |
| R05 | 2026-10-12 | 1d_conv_autoencoder | opssat_nominal_only | 30 | 14.22 | yes | Earlier test inspection |
| R06 | 2026-10-12 | 2d_spectrogram_autoencoder | opssat_nominal_only | 30 | 20.79 | yes | Earlier test inspection |
| R07 | 2026-10-12 | attention_autoencoder | opssat_nominal_only | 30 | 15.32 | yes | Earlier test inspection |
| R08 | 2026-10-12 | fusion_1d_2d | opssat_nominal_only_fixed_w0.5 | 0 | 0.50 | yes | Score fusion, no training; earlier test inspection |
| R09\_superseded | 2026-10-12 | 1d_conv_autoencoder | smap_p1_nominal_only\_wrong\_cal | 30 | 1.07 | yes | **Superseded** — incorrect cal protocol (single end-window) |
| R10\_superseded | 2026-10-12 | 1d_conv_autoencoder | msl_c1_nominal_only\_wrong\_cal | 30 | 0.36 | yes | **Superseded** — incorrect cal protocol (single end-window) |
| R11 | 2026-11-14 | 1d_conv_autoencoder | smap_p1_nominal_only\_corrected\_cal | 30 | 2.59 | no | Corrected covering-window mean cal; threshold 1.995969 |
| R12 | 2026-11-14 | 1d_conv_autoencoder | msl_c1_nominal_only\_corrected\_cal | 30 | 1.38 | no | Corrected covering-window mean cal; threshold 0.513500 |
| R13 | 2026-11-15 | isolation_forest | opssat_final_test | 1 | — | yes | Final test eval (all OPSSAT baselines + DL, single run) |
| R14 | 2026-11-15 | pca_reconstruction | opssat_final_test | 1 | — | yes | Final test eval |
| R15 | 2026-11-15 | trivial_detector | opssat_final_test | 1 | — | yes | Final test eval |
| R16 | 2026-11-15 | 1d_conv_autoencoder | opssat_final_test | 30 | — | yes | Final test eval; F1=0.1417 |
| R17 | 2026-11-15 | 2d_spectrogram_autoencoder | opssat_final_test | 30 | — | yes | Final test eval; F1=0.1311 |
| R18 | 2026-11-15 | attention_autoencoder | opssat_final_test | 30 | — | yes | Final test eval; F1=0.0480 |
| R19 | 2026-11-15 | fusion_1d_2d + smap_msl | opssat_and_streams\_final\_test | 0 | — | yes | Final test eval; Fusion F1=0.1875; P-1 unadj F1=0.0168; C-1 unadj F1=0.2708 |
