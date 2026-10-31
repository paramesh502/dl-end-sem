# Mid-Project Checkpoint Note — T05 I1

22AIE304 Deep Learning · Due 23:59 IST, Wednesday 4 November 2026

## Cover

| Field | Entry |
| --- | --- |
| Team number | 05 |
| Card | I1 — Satellite Telemetry Anomaly Detection |
| Members | K Hemalatha Reddy (BL.SC.U4AIE24018) · B Parameshwar Reddy (BL.SC.U4AIE24008) · Jagan Sai Kumar (BL.SC.U4AIE24023) |
| Repository | https://github.com/paramesh502/dl-end-sem |
| Commit (this checkpoint) | 027e1899494567ef67a193853835fc0acaba9cd3 (HEAD at this checkpoint; final tag added by the release step) |
| Bundle | Revision D2 (verified — see §7) |

## 1. Required Components

11/11 required components pass the smoke test (scripts/smoke_test.py, 4.3 s total). Run IDs refer to T05_I1_runlog.csv.

| Component | Smoke test | Run ID | Owner | Reviewer |
| --- | --- | --- | --- | --- |
| Isolation Forest | ✓ | R02 | Parameshwar | Jagan |
| PCA reconstruction | ✓ | R03 | Parameshwar | Jagan |
| Trivial detector | ✓ | R04 | Parameshwar | Jagan |
| 1-D Conv Autoencoder | ✓ | R05 | Parameshwar | Jagan |
| 2-D Spectrogram Autoencoder | ✓ | R06 | Hemalatha | Parameshwar |
| Temporal Attention AE + ablation | ✓ | R07 | Jagan | Hemalatha |
| Fusion (1D+2D, w=0.5) | ✓ | R08 | Parameshwar | Jagan |
| SMAP P-1 stream AE | ✓ | R09 | Hemalatha | Jagan |
| MSL C-1 stream AE | ✓ | R10 | Hemalatha | Jagan |

Per revised Handout §14, the earlier runs R02–R10 that scored the test set are recorded as test_scored=yes in T05_I1_runlog.csv (with their dates); R01 (the Week-1 reference) is test_scored=no (unscored inference only). No test partition is scored again before the results freeze, and a detector is changed only on development evidence, with the reason and the development figures recorded before and after. A forward-policy note in the run log states that, from this checkpoint onwards, test_scored=no for all future runs. At the freeze every final detector is evaluated once more and the new result reported beside the earlier one, with a plain statement that the test sets had already been inspected.

## 2. Preliminary Core-Experiment Results (validation only — no test scoring)

Thresholds are calibrated on nominal development (OPSSAT-AD) or calibration (SMAP/MSL) data only, before any test scoring. The development partition is nominal-only, so precision/recall/F1 are undefined on dev; the reported validation figure is the calibrated threshold and the alarm (false-alarm) rate it gives on nominal data. Per revised Handout §14, no test partition is scored in this checkpoint; the figures below are thresholds and alarm rates only.

OPSSAT-AD — threshold (99th percentile of nominal-development scores) and development alarm rate (false-alarm rate on the 254 nominal dev fragments):

| System | Threshold | Dev alarm rate |
| --- | --- | --- |
| 1-D autoencoder | 0.000581 | 3/254 (1.18%) |
| 2-D spectrogram autoencoder | 0.000355 | 3/254 (1.18%) |
| Attention autoencoder | 0.010338 | 3/254 (1.18%) |
| Fusion (1-D + 2-D, w=0.5) | 3.977751 | 3/254 (1.18%) |
| Isolation Forest | 0.201592 | 3/254 (1.18%) |
| PCA reconstruction | 0.030605 | 3/254 (1.18%) |
| Trivial (mean distance) | 3.410388 | 3/254 (1.18%) |

SMAP/MSL — per-stream threshold (99th percentile of calibration-point scores), calibration alarm count and calibration alarm rate, from the new guard-context calibration runs (results/baseline_results.txt and results/smap_msl_results.txt):

| Stream | Detector | Threshold | Cal alarm count | Cal alarm rate |
| --- | --- | --- | --- | --- |
| P-1 | 1-D autoencoder | 0.450251 | 4/319 | 1.25% |
| P-1 | Isolation Forest | 0.062456 | 4/319 | 1.25% |
| P-1 | PCA reconstruction | 0.156650 | 4/319 | 1.25% |
| P-1 | Trivial (mean distance) | 1.720557 | 4/319 | 1.25% |
| C-1 | 1-D autoencoder | 0.129134 | 2/176 | 1.14% |
| C-1 | Isolation Forest | 0.040320 | 2/176 | 1.14% |
| C-1 | PCA reconstruction | 0.040891 | 2/176 | 1.14% |
| C-1 | Trivial (mean distance) | 1.765357 | 2/176 | 1.14% |

Calibration rule (revised Annex D6): a calibration window may take earlier context from the guard, never from fit rows, and only calibration points are scored. One window per calibration point (right-aligned, ending at the point) gives exactly 319 calibration-point scores for P-1 and 176 for C-1. No boundary padding is required — the 256-step guard supplies left context for the earliest calibration points (P-1 guard [2297,2553), cal [2553,2872); C-1 guard [1726,1982), cal [1982,2158)), and every window ends at a calibration point inside the training stream. Model fitting and preprocessing statistics use fit rows only, never guard or calibration rows. Window scores are reduced to point scores one-to-one (the window ending at a point scores that point), so each calibration-point score is the reduced point score; thresholds are the 99th percentile of these calibration-point scores, and the alarm count and rate are computed on the calibration points. The earlier C-1 implementation capped the guard at 176 and started calibration at step 1902 — that is removed; the guard is now always 256 and calibration starts at fit_end+256.

Preliminary core-experiment result: the nominal-development calibration above (with its support — 254 OPSSAT-AD development fragments; 319 P-1 and 176 C-1 calibration points) and an unscored inference pass (scripts/smoke_test.py, 11/11 components pass; and the Week-1 reference run R01). No test partition is scored and no test metric is computed; anomalous-class validation F1 is not required. P-1 convergence is slow — validation (cal-point) reconstruction MSE falls from ~1.38 to ~0.36 over 30 epochs with only 32 training windows; C-1 converges from ~0.81 to ~0.09 (early stopped after no improvement for 5 epochs). Target point-F1 evaluation waits for the results freeze (20 November 2026).

## 3. Elapsed Time and 24-Hour Budget

Annex D0's stage to project for I1 is stride-1 scoring of P-1 and C-1. It was timed with the trained 1-D autoencoder (proxy weights from results/1d_autoencoder.pt, 6,353 params) and the classical Isolation-Forest control on the same windows, using test-shaped inputs only — no test labels loaded, no test metric computed. The window count includes padding (one 256-step window per test point, right-edge replication padding). Full results in results/stride1_timing.txt.

| Stream · model | Windows | Construction (s) | Inference (s) | Reduction (s) | Total (s) |
| --- | --- | --- | --- | --- | --- |
| P-1 neural (1-D AE) | 8505 | 0.0055 | 2.2395 | 0.0105 | 2.2555 |
| P-1 classical (IsoForest) | 8505 | 1.5158 | 0.0081 | 0.0093 | 1.5332 |
| C-1 neural (1-D AE) | 2264 | 0.0006 | 0.5939 | 0.0028 | 0.5972 |
| C-1 classical (IsoForest) | 2264 | 0.4030 | 0.0042 | 0.0024 | 0.4096 |

Required neural fits (actual recorded elapsed, T05_I1_runlog.csv): 1-D AE 14.22 s, 2-D AE 20.79 s, Attention AE 15.32 s, P-1 1-D AE 3.07 s (corrected-calibration re-run; runlog R09 records the original test-scored run at 1.07 s), C-1 1-D AE 1.83 s (re-run; runlog R10 original 0.36 s). Fusion is score-level (no training); classical fits are < 1 s each. Fit subtotal ≈ 55.23 s; stride-1 scoring subtotal ≈ 4.80 s.

Programme projection = 55.23 + 4.80 = 60.03 s = 0.0167 h = 0.0695% of the 24-hour budget (margin ≈ 86340 s). The stream-model timings establish the per-fit cost: stream 1-D AE fits are ~1–3 s, NOT the 29.18 s of the Week-1 spectrogram reference fit (R01). Even if every required neural fit cost 29.18 s, the five fits plus stride-1 scoring would total ≈ 150.7 s (~0.174% of 24 h). The earlier dummy-scorer projection is withdrawn; this is a trained-model measurement.

## 4. Plan

Checkpoint due Wednesday 4 November 2026 (this note). The earlier Week-5 demo target of 30 October is replaced by this checkpoint. Results freeze and the preliminary 4–6 page report are due 20 November 2026. Remaining steps before the freeze: final test evaluation of each detector (once, reported beside the earlier test-scored figures), error analysis for both core experiments, and the CO4 spectrogram fragment-level analysis (Handout §9 of the revised pack).

## 5. Risks and Requests

1. Compute against 24 hours. The stride-1 scoring of P-1 and C-1 was timed with the trained 1-D autoencoder and the classical controls on the same windows (results/stride1_timing.txt): stride-1 subtotal ≈ 4.80 s; with all required fits (≈ 55.23 s) the programme is ≈ 60 s, i.e. 0.0695% of the 24-hour budget. Stream fits are ~1–3 s, not the 29.18 s reference. No compute risk; no extension requested.

2. Length difference. Nominal fragments are much shorter than anomalous ones (median 54 vs 184.5 samples); length may separate the classes independently of the detector, even after resampling to 256 samples. This is recorded as a risk to interpretation (not to compute). A per-length analysis (detection against original fragment length, from the same predictions) will be reported in the final report; it can show association, not causation.

3. C-1 calibration rule. The calibration segment of C-1 (176 steps) is shorter than one 256-step window — a gap in Annex D6 as originally written. The revised Annex D6 rule is now implemented: a calibration window may take earlier context from the guard, never from fit rows, and only calibration points are scored. This yields exactly 176 calibration-point scores for C-1 and 319 for P-1, with no boundary padding. Model fitting and preprocessing statistics use fit rows only.

Requests: None.

## 6. Changes, Open Items, and Requests

Changes made for this checkpoint (each on development evidence only; no test partition re-scored):

• Split-ID format (Handout §13): regenerated T05_I1_opssat.csv (sample_id = segment only; reason = anomalous_train_unused on excluded rows), T05_I1_smap_msl.csv (sample_id = half-open range <stream>:<file>:<start>-<end>; unit_id = P-1/C-1; label empty; reason = guard on guard rows) and recomputed the summary hashes. Membership unchanged.

• Run log (revised §14): R02–R10 marked test_scored=yes (they scored test); R01 remains no. Forward-policy note added: test_scored=no for all future runs; no test partition scored again before the freeze.

• Calibration (revised Annex D6): guard-context calibration implemented in scripts/run_baselines.py and scripts/week4_smap_msl.py; the C-1 guard cap (176) removed. Test scoring removed from both scripts (no test partition re-scored). Calibration thresholds changed (before → after): P-1 IsoForest 0.029441 → 0.062456, PCA 0.090753 → 0.156650, Trivial 0.666753 → 1.720557, 1-D AE 0.382350 → 0.450251; C-1 IsoForest -0.018319 → 0.040320, PCA 0.002729 → 0.040891, Trivial 1.017836 → 1.765357, 1-D AE 0.089802 → 0.129134.

• Roll numbers corrected: Parameshwar BL.SC.U4AIE24008 (was 24019), Jagan BL.SC.U4AIE24023 (was 24020); Hemalatha BL.SC.U4AIE24018 unchanged.

• Stride-1 timing re-done with a trained-model proxy (replaces the withdrawn dummy-scorer projection).

Open items before the results freeze (20 November 2026): final test evaluation of each detector (once, beside the earlier figures, with a statement that the test sets had already been inspected); error analysis; the CO4 spectrogram fragment-level results (precision, recall, F1 with support, false-positive/false-negative patterns, comparison with the 1-D model, the controls and the fusion); per-length detection analysis; the preliminary report.

Requests: None.

## 7. Bundle Verification

D1 bundle: verified at the readiness check (SHA-256 522127833…, 11,900,797 B).

D2 bundle: present on this machine and verified — every file in Release_Bundle_RevD2 matches its SHA256SUMS.txt manifest (shasum -a 256 -c: all OK). The four I1 data files (OPSSAT_nominal_dev_segments.csv and the SMAP/MSL train/test .npy files for P-1 and C-1) are byte-identical to D1. The D2 bundle archive (announced SHA-256 86ccaa25…a2a6, 11,903,184 B) is not retained as a single archive on this machine; integrity was verified on the extracted contents against the manifest.

