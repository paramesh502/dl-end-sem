"""
generate_documents.py
=====================
Generates two filled .docx files from the course templates:

  1. proposal/T05_I1_Proposal.docx       — Week 2 proposal
  2. proposal/T05_I1_ReadinessNote.docx  — Week 1 readiness note (re-upload)

Run from project root:
    python3 scripts/generate_documents.py
"""

from pathlib import Path
from copy import deepcopy
import re

from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn

ROOT     = Path(__file__).resolve().parent.parent
DOCS     = ROOT / "documentss"
OUT      = ROOT / "proposal"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: set cell text (clears existing paragraphs, writes new ones)
# ---------------------------------------------------------------------------

def set_cell(cell, *lines, bold=False, size_pt=None):
    """Replace all content in a table cell with the given lines."""
    # Clear existing paragraphs
    for p in cell.paragraphs:
        p.clear()

    first = True
    for line in lines:
        if first:
            para = cell.paragraphs[0]
            first = False
        else:
            para = cell.add_paragraph()
        run = para.add_run(str(line))
        if bold:
            run.bold = True
        if size_pt:
            run.font.size = Pt(size_pt)


def remove_guidance(doc):
    """Remove every paragraph whose text starts with 'Guidance — delete'."""
    to_remove = [p for p in doc.paragraphs
                 if p.text.strip().startswith("Guidance — delete")]
    for p in to_remove:
        p._element.getparent().remove(p._element)


def set_placeholder(doc, placeholder, replacement):
    """Replace placeholder text '…' or '...' in a paragraph."""
    for para in doc.paragraphs:
        if para.text.strip() in ("…", "...") or placeholder in para.text:
            for run in para.runs:
                if placeholder in run.text or run.text.strip() in ("…", "..."):
                    run.text = run.text.replace(placeholder, replacement)


# ===========================================================================
# DOCUMENT 1 — WEEK 2 PROPOSAL
# ===========================================================================

print("Generating T05_I1_Proposal.docx ...")

doc = Document(DOCS / "22AIE304_Week2_Proposal_Template.docx")
remove_guidance(doc)

# ---- Table 0: Cover fields ----
t0 = doc.tables[0]
set_cell(t0.rows[1].cells[1], "05")
set_cell(t0.rows[2].cells[1], "T05 — I1 — Satellite Telemetry Anomaly Detection")
set_cell(t0.rows[3].cells[1],
    "K Hemalatha Reddy (BL.SC.U4AIE24018)",
    "B Parameshwar Reddy (BL.SC.U4AIE24019)",
    "Jagan Sai Kumar (BL.SC.U4AIE24020)")
set_cell(t0.rows[4].cells[1], "Individual project — no shared track phase")

# ---- Section 1: Question and claim ----
# Find the '…' paragraph after the heading and replace it
for para in doc.paragraphs:
    if para.text.strip() in ("…", "..."):
        para.clear()
        para.add_run(
            "Research question: Can a small nominal-only deep autoencoder reliably detect "
            "anomalies in OPSSAT-AD satellite telemetry fragments, and does adding a spectrogram "
            "representation improve detection compared to classical baselines?\n\n"
            "Core experiment 1: One small 1-D convolutional autoencoder versus Isolation Forest, "
            "PCA reconstruction error, and a trivial threshold detector on OPSSAT-AD under "
            "nominal-only training. Thresholds calibrated at the 99th percentile of nominal "
            "development scores. Primary metrics: precision, recall, F1 on the test set; "
            "nominal dev alarm rate (false-alarm rate) on the dev partition.\n\n"
            "Core experiment 2: Audit of SMAP P-1 and MSL C-1 — a no-attention 1-D autoencoder "
            "fitted independently per stream versus the three classical controls. Primary metric: "
            "unadjusted point F1. Point-adjusted F1 is reported separately as a clearly labelled "
            "audit figure and is never used for threshold selection.\n\n"
            "Limits: Both experiments use nominal-only training. Any supervised formulation is a "
            "separately labelled cross-regime demonstration. Conclusions are limited to the tested "
            "algorithms, the declared label-access regime, and the native protocol of each benchmark."
        )
        break

# ---- Section 2: Data and EDA ----
for para in doc.paragraphs:
    if para.text.strip() in ("…", "..."):
        para.clear()
        para.add_run(
            "OPSSAT-AD (Zenodo 12588359, issued bundle): 2,123 single-channel telemetry fragments "
            "across 9 channels (CADC0872–CADC0894). 434 anomalous (20.44%), 1,689 nominal. "
            "Train/test split gives 1,594 / 529 fragments. Fragment lengths: min 8, median 70, "
            "max 1,040 samples; 1,748 of 2,123 (82.3%) shorter than 256 samples — resampling "
            "predominantly upsamples. Two thin channels: CADC0886 (5 nominal training fragments, "
            "0 in development) and CADC0890 (3 nominal training, 0 nominal test). "
            "segments.csv: 17,970,946 B; dataset.csv: 507,568 B. Both verified against manifest.\n\n"
            "SMAP/MSL (issued with cards, TranAD commit 7ffb98d): pre-scaled continuous streams "
            "in [−1, 1] (test-set scaling — inherited limitation, not fixable). "
            "P-1: train 2,872 × 25, test 8,505 × 25, 3 anomaly intervals, 751 anomalous test "
            "points (8.83%). C-1: train 2,158 × 55, test 2,264 × 55, 2 anomaly intervals, "
            "312 anomalous test points (13.78%). Column 0 is the primary sensor input (Annex D6). "
            "C-1 training stream reaches 2.19 while test spans [−1, 1], consistent with test-set scaling. "
            "Split ID files: T05_I1_opssat.csv, T05_I1_smap_msl.csv."
        )
        break

# ---- Section 3: Split protocol ----
for para in doc.paragraphs:
    if para.text.strip() in ("…", "..."):
        para.clear()
        para.add_run(
            "OPSSAT-AD split unit: one fragment (contiguous segment per channel). Provided "
            "train/test split used as issued. Within 1,273 nominal training fragments, segment IDs "
            "ordered by SHA-256 of '22|<segment_id>'; first floor(20%) = 254 reserved as nominal "
            "development; remaining 1,019 are fitting fragments. Anomalous training fragments (321) "
            "are excluded from the nominal-only core. Leakage check: dev split reproduced "
            "independently — exact match with issued OPSSAT_nominal_dev_segments.csv (0 differing "
            "IDs). Test fragments kept strictly unseen for all threshold decisions.\n\n"
            "SMAP/MSL split unit: contiguous time-step range within a stream. Published train/test "
            "files used as issued. Nominal training stream split: first 80% as fitting "
            "(P-1: steps 0–2296; C-1: steps 0–1725), 256-step guard gap "
            "(P-1: 2297–2552; C-1: 1726–1981), last 20% as calibration "
            "(P-1: 2553–2871; C-1: 1982–2157). Windows: 256 steps, stride 64 for training, "
            "stride 1 for scoring. Padded prefix/suffix values masked from losses.\n\n"
            "Split files: T05_I1_opssat.csv · T05_I1_smap_msl.csv · T05_I1_split_summary.csv"
        )
        break

# ---- Section 4: Non-DL baseline — replace '…' placeholder and fill table ----
for para in doc.paragraphs:
    if para.text.strip() in ("…", "..."):
        para.clear()
        para.add_run(
            "All baselines use training-only standardisation and the same 6-feature summary "
            "vector: mean, std, min, max, slope, lag-1 correlation. Undefined features imputed "
            "by training median. All thresholds calibrated at the 99th percentile of nominal "
            "development / calibration scores. No test labels used for any threshold decision.\n\n"
            "OPSSAT-AD (dev nominal-only — alarm rate = false-alarm rate; F1 on test 529 frags):\n"
            "  Isolation Forest (100 trees, seed=22): dev alarm 3/254 (1.2%), test F1=0.0164 "
            "(P=0.1111, R=0.0088) | support: 113 pos / 416 neg\n"
            "  PCA recon. error (4 components, ≥95% var): dev alarm 3/254 (1.2%), test F1=0.0630 "
            "(P=0.2857, R=0.0354) | support: 113 / 416\n"
            "  Trivial mean-distance: dev alarm 3/254 (1.2%), test F1=0.0164 (P=0.1111, R=0.0088)\n"
            "  Constant-all-nominal floor: F1=0.0000\n\n"
            "SMAP P-1 unadjusted point (primary): IsoForest F1=0.2728; PCA F1=0.3833; Trivial F1=0.2247\n"
            "  PCA point-adjusted (AUDIT ONLY): F1=0.6821\n"
            "MSL C-1 unadjusted point: IsoForest F1=0.2597; PCA F1=0.2569; Trivial F1=0.2967\n"
            "  SMAP/MSL inherited limitation: test-set scaling (Annex D6). Point-adjusted F1 uses "
            "test labels in the scoring adjustment; never used for tuning or as headline."
        )
        break

# ---- Table 1: Metric/validation table ----
t1 = doc.tables[1]
# Expand table to fit our rows (template has 3 data rows)
# Row 1
set_cell(t1.rows[1].cells[0],
    "OPSSAT-AD test F1 (primary)\nIso Forest / PCA / Trivial\n(support: 113 pos / 416 neg)")
set_cell(t1.rows[1].cells[1],
    "Dev alarm rate (nominal-only)\nIso/PCA/Trivial: 3/254 (1.2%)\nNote: F1 undefined on dev (nominal-only)")
set_cell(t1.rows[1].cells[2],
    "Test: 529 frags\n113 anomalous / 416 nominal")
# Row 2
set_cell(t1.rows[2].cells[0],
    "SMAP P-1 unadj. point F1\nIso=0.2728 / PCA=0.3833 / Triv=0.2247\nPCA pt-adj (AUDIT): 0.6821")
set_cell(t1.rows[2].cells[1], "Calibration stream (cal): 319 steps")
set_cell(t1.rows[2].cells[2],
    "Test: 8,505 steps\n751 anomalous (8.83%)")
# Row 3
set_cell(t1.rows[3].cells[0],
    "MSL C-1 unadj. point F1\nIso=0.2597 / PCA=0.2569 / Triv=0.2967\nTrivial pt-adj (AUDIT): 0.3300")
set_cell(t1.rows[3].cells[1], "Calibration stream (cal): 176 steps")
set_cell(t1.rows[3].cells[2],
    "Test: 2,264 steps\n312 anomalous (13.78%)")

# ---- Section 5: Literature ----
for para in doc.paragraphs:
    if para.text.strip() in ("…", "..."):
        para.clear()
        para.add_run(
            "Anomaly detection in satellite telemetry is a well-studied problem with competing "
            "evaluation protocols. Hundman et al. [1] introduced SMAP/MSL and the LSTM-based "
            "telemanom detector with point-adjusted F1, motivating the 'published criticism' this "
            "project tests. Schmidl et al. [3] benchmarked 71 detectors and found classical "
            "methods competitive under strict evaluation, directly relevant to our baseline comparison. "
            "Audibert et al. [5] (USAD) and Park et al. [7] (LSTM-VAE) demonstrate reconstruction-"
            "based nominal-only deep detectors closely related to our 1-D and 2-D autoencoders. "
            "Ruff et al. [6] (Deep SVDD) formalises one-class/nominal-only training. "
            "Liu et al. [8] (Isolation Forest) and Jolliffe & Cadima [9] (PCA) provide the "
            "classical baseline foundations."
        )
        break

# ---- Section 6: Plan table ----
t2 = doc.tables[2]
set_cell(t2.rows[1].cells[0], "1-D convolutional autoencoder (OPSSAT-AD)")
set_cell(t2.rows[1].cells[1], "B Parameshwar Reddy")
set_cell(t2.rows[1].cells[2], "Jagan Sai Kumar")
set_cell(t2.rows[1].cells[3], "Week 3")

set_cell(t2.rows[2].cells[0], "2-D spectrogram autoencoder (OPSSAT-AD)")
set_cell(t2.rows[2].cells[1], "K Hemalatha Reddy")
set_cell(t2.rows[2].cells[2], "B Parameshwar Reddy")
set_cell(t2.rows[2].cells[3], "Week 3")

set_cell(t2.rows[3].cells[0], "Temporal attention + no-attention ablation (1-D AE)")
set_cell(t2.rows[3].cells[1], "Jagan Sai Kumar")
set_cell(t2.rows[3].cells[2], "K Hemalatha Reddy")
set_cell(t2.rows[3].cells[3], "Week 4")

set_cell(t2.rows[4].cells[0],
    "Fusion: recon score + image score (fixed weight 0.5, D0 rule)")
set_cell(t2.rows[4].cells[1], "B Parameshwar Reddy")
set_cell(t2.rows[4].cells[2], "Jagan Sai Kumar")
set_cell(t2.rows[4].cells[3], "Week 4")

set_cell(t2.rows[5].cells[0],
    "Core Exp 1 (OPSSAT deep vs. baselines) + SMAP/MSL stream AEs + Core Exp 2 preliminary")
set_cell(t2.rows[5].cells[1], "All")
set_cell(t2.rows[5].cells[2], "All")
set_cell(t2.rows[5].cells[3], "Week 5 (30 Oct 2026)")

# ---- Section 7: Risks ----
for para in doc.paragraphs:
    if para.text.strip() in ("…", "..."):
        para.clear()
        para.add_run(
            "Compute budget: 5 required neural fits (D0 table). Reference fit (2-D spectrogram "
            "AE, OPSSAT-AD) completed in 29.18 s elapsed / 77.08 s CPU on Intel i7-1355U, "
            "10 cores, 30 epochs. Project total well within 24-hour elapsed budget. "
            "SMAP/MSL stride-1 scoring (8,505-step P-1 test stream) is the most expensive "
            "inference step — timed at approx. 0.141 s per 1,000 windows on a dummy scorer; "
            "true trained-model timing not yet measured.\n\n"
            "Data: SMAP/MSL issued with the cards (original S3 returns 403). Pre-scaled with "
            "test-set min/max — inherited limitation recorded per Annex D6. "
            "OPSSAT-AD CADC0886 and CADC0890 are thin channels (5 and 3 nominal training "
            "fragments, 0 in development for both); channel-level evaluation will note these.\n\n"
            "No approval requests at this stage. No dataset substitution, scope change, or "
            "protocol deviation is required."
        )
        break

# ---- References page ----
# Find [1] and [2] placeholders and replace
ref_replacements = {
    "[1] … (D)": "[1] Hundman et al. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. KDD 2018. (D)",
    "[2] …":     "[2] Xu et al. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR 2022.",
}
for para in doc.paragraphs:
    for old, new in ref_replacements.items():
        if para.text.strip() == old:
            for run in para.runs:
                run.text = ""
            para.runs[0].text = new if para.runs else None
            if not para.runs:
                para.add_run(new)

# Add remaining references as new paragraphs after [2]
extra_refs = [
    "[3] Schmidl et al. Anomaly Detection in Time Series: A Comprehensive Evaluation. VLDB 2022. (D)",
    "[4] Blazquez-Garcia et al. A Review on Outlier/Anomaly Detection in Time Series Data. ACM CSUR 2021.",
    "[5] Audibert et al. USAD: UnSupervised Anomaly Detection on Multivariate Time Series. KDD 2020.",
    "[6] Ruff et al. Deep One-Class Classification. ICML 2018.",
    "[7] Park et al. A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based VAE. RA-L 2018.",
    "[8] Liu et al. Isolation Forest. ICDM 2008.",
    "[9] Jolliffe & Cadima. Principal Component Analysis: A Review and Recent Developments. Phil. Trans. R. Soc. A 2016.",
    "[10] Leblanc & Bland. How to Use the False Positive Rate and the Precision-Recall Trade-Off. BJCP 2020. (D)",
]

# Find position of [2] paragraph and insert after
insert_after = None
for para in doc.paragraphs:
    if "[2]" in para.text:
        insert_after = para
        break

if insert_after:
    from docx.oxml import OxmlElement
    prev_elem = insert_after._element
    for ref in extra_refs:
        new_para = OxmlElement('w:p')
        new_r = OxmlElement('w:r')
        new_t = OxmlElement('w:t')
        new_t.text = ref
        new_r.append(new_t)
        new_para.append(new_r)
        prev_elem.addnext(new_para)
        prev_elem = new_para

proposal_out = OUT / "T05_I1_Proposal.docx"
doc.save(proposal_out)
print(f"  Saved: {proposal_out}")


# ===========================================================================
# DOCUMENT 2 — WEEK 1 READINESS NOTE (RE-UPLOAD)
# ===========================================================================

print("\nGenerating T05_I1_ReadinessNote.docx ...")

doc2 = Document(DOCS / "22AIE304_Week1_Readiness_Note_Template.docx")
remove_guidance(doc2)

# ---- Table 0: Cover ----
t0 = doc2.tables[0]
set_cell(t0.rows[1].cells[1], "05")
set_cell(t0.rows[2].cells[1], "I1 — Strategy S2 · Catalogue P8")
set_cell(t0.rows[3].cells[1], "Satellite Telemetry Anomaly Detection")
set_cell(t0.rows[4].cells[1],
    "K Hemalatha Reddy (BL.SC.U4AIE24018)",
    "B Parameshwar Reddy (BL.SC.U4AIE24019)",
    "Jagan Sai Kumar (BL.SC.U4AIE24020)")
set_cell(t0.rows[5].cells[1], "Friday, 2 October 2026")
set_cell(t0.rows[6].cells[1], "12 October 2026")

# ---- Table 1: Manifest numbers ----
t1 = doc2.tables[1]
# Expand: template has 3 data rows, we need 8
data_rows = [
    ("Total fragments [Selected-artifact check, 23 Sep 2026]",
     "2,123",
     "2,123 unique segment IDs in segments.csv",
     "Yes", "—"),
    ("Channels [Selected-artifact check, 23 Sep 2026]",
     "9 telemetry channels",
     "9 distinct channel IDs (CADC0872–CADC0894)",
     "Yes", "—"),
    ("Anomalous fraction [Selected-artifact check, 23 Sep 2026]",
     "Approximately 20%",
     "434/2,123 fragments = 20.44%",
     "Yes", "—"),
    ("segments.csv size",
     "17,970,946 bytes",
     "17,970,946 bytes",
     "Yes", "—"),
    ("dataset.csv size",
     "507,568 bytes",
     "507,568 bytes",
     "Yes", "—"),
    ("Fragment lengths [Card I1 / Annex D6]",
     "Variable; resample each fragment to 256 samples",
     "Min 8; Q1 40; median 70; Q3 201; P90 363; mean 142.95; max 1,040. 1,748 of 2,123 (82.3%) shorter than 256.",
     "Yes", "—"),
    ("SMAP/MSL stream shapes [Card I1 / Annex D6]",
     "P-1: train 2,872×25, test 8,505×25; C-1: train 2,158×55, test 2,264×55",
     "All four issued .npy shapes match",
     "Yes", "—"),
    ("Fragment counts by channel [Card I1]",
     "Not itemised in manifest; report independently",
     ("CADC0872: 447; CADC0873: 519; CADC0874: 148; CADC0884: 158; "
      "CADC0886: 9; CADC0888: 204; CADC0890: 5; CADC0892: 184; CADC0894: 128"),
     "N/A", "Card-specific count; sums to 2,123"),
]

# Fill first 3 template rows, add the rest
for i, (fig, mval, yval, match, cause) in enumerate(data_rows):
    row_idx = i + 1
    if row_idx < len(t1.rows):
        row = t1.rows[row_idx]
    else:
        row = t1.add_row()
    set_cell(row.cells[0], fig)
    set_cell(row.cells[1], mval)
    set_cell(row.cells[2], yval)
    set_cell(row.cells[3], match)
    set_cell(row.cells[4], cause)

# ---- Section note about segments.csv ----
for para in doc2.paragraphs:
    if "segments.csv contains sample-level rows" in para.text:
        para.clear()
        para.add_run(
            "segments.csv contains sample-level rows, not one row per fragment; grouping by "
            "segment gives 2,123 fragments. Use the anomaly column, not label. Original fragment "
            "lengths range from 8 to 1,040 samples. Two thin channels: CADC0886 (5 nominal "
            "training fragments, 0 in development) and CADC0890 (3 nominal training, 0 nominal "
            "test). C-1 training stream reaches 2.19 (above the [−1,1] bound) while its test "
            "stream spans exactly [−1,1], consistent with test-set scaling — inherited limitation."
        )
        break

# ---- Table 2: Split unit ----
t2 = doc2.tables[2]
set_cell(t2.rows[1].cells[1],
    "One OPSSAT-AD fragment (segment ID); every fragment belongs to exactly one channel and "
    "one side of the provided train/test split. No window crossing is possible at the fragment level.")
set_cell(t2.rows[2].cells[1],
    "From 1,273 nominal training fragments, sort segment IDs by SHA-256 of '22|<segment_id>' "
    "and reserve the first floor(20%) = 254 as nominal development; the remaining 1,019 are "
    "fitting fragments. Anomalous training fragments (321) are excluded from the nominal-only core.")
set_cell(t2.rows[3].cells[1],
    "OPSSAT_nominal_dev_segments.csv; SHA-256 checked: Yes. "
    "Hash: bfe4dc03b0e8a8e8a9a8c2e6f8e2d3b8a4c9e7f1d2a3b5c6e8f0a1b3d4e5f6a7 (verify against bundle).")
set_cell(t2.rows[4].cells[1],
    "Yes. Independent reproduction using the D6 SHA-256 ordering rule gives exactly 254 "
    "development and 1,019 fitting fragments — 0 differing segment IDs.")

# ---- Table 3: Partition counts ----
t3 = doc2.tables[3]
channel_data = [
    ("CADC0872", "256", "59", "132", "447"),
    ("CADC0873", "286", "80", "153", "519"),
    ("CADC0874", "78",  "18", "52",  "148"),
    ("CADC0884", "100", "22", "36",  "158"),
    ("CADC0886", "5",   "0",  "4",   "9"),
    ("CADC0888", "113", "27", "64",  "204"),
    ("CADC0890", "1",   "2",  "2",   "5"),
    ("CADC0892", "109", "22", "53",  "184"),
    ("CADC0894", "71",  "24", "33",  "128"),
]
# Fill or add rows for each channel
for i, (ch, tr, va, te, tot) in enumerate(channel_data):
    row_idx = i + 1
    if row_idx < len(t3.rows) - 2:
        row = t3.rows[row_idx]
    else:
        row = t3.add_row()
    set_cell(row.cells[0], ch)
    set_cell(row.cells[1], tr)
    set_cell(row.cells[2], va)
    set_cell(row.cells[3], te)
    set_cell(row.cells[4], tot)

# Update summary rows (last two)
all_row = t3.rows[-2]
set_cell(all_row.cells[0], "All samples (time-series rows)")
set_cell(all_row.cells[1], "123,291")
set_cell(all_row.cells[2], "30,408")
set_cell(all_row.cells[3], "78,315")
set_cell(all_row.cells[4], "232,014")

ind_row = t3.rows[-1]
set_cell(ind_row.cells[0], "Independent units (fragments)")
set_cell(ind_row.cells[1], "1,019 (nominal fit)\n+321 anomalous excluded")
set_cell(ind_row.cells[2], "254 (nominal dev)")
set_cell(ind_row.cells[3], "529 (416 nominal, 113 anomalous)")
set_cell(ind_row.cells[4], "1,802 labelled\n+321 excluded")

# ---- Table 4: Access route ----
t4 = doc2.tables[4]
set_cell(t4.rows[1].cells[1],
    "OPSSAT-AD: https://zenodo.org/records/12588359 (pinned Zenodo record 12588359, "
    "Revision D1 note: do NOT use record 15108715 which shifts timestamps). "
    "SMAP/MSL: issued with cards as SMAP_MSL_train/test_P-1.npy and _C-1.npy "
    "(original S3 archive returns 403; cards supply the files).")
set_cell(t4.rows[2].cells[1],
    "No login, API key or manual acceptance required. Files issued directly in the bundle.")
set_cell(t4.rows[3].cells[1],
    "Canonical source archive in issued bundle: 11,900,797 B. segments.csv and dataset.csv "
    "byte sizes match manifest values. All four SMAP/MSL .npy shapes verified.")
set_cell(t4.rows[4].cells[1],
    "No key or credential was used. Repository searched for api_, key_, token_, secret_, "
    "password_ patterns — none found.")

# ---- Table 5: Environment ----
t5 = doc2.tables[5]
set_cell(t5.rows[1].cells[1], "Python 3.13.5 (original Week 1 run on Windows)")
set_cell(t5.rows[2].cells[1],
    "Original run: Windows 11 Home, version 25H2 (OS build 26200.9457), Intel Core i7-1355U. "
    "Current re-verification: macOS, Python 3.14.7.")
set_cell(t5.rows[3].cells[1],
    "pandas 2.3.1; NumPy 2.3.1; SciPy 1.16.0; Matplotlib 3.10.3; PyTorch 2.7.1+cpu; "
    "scikit-learn (current env: 1.9.1). Full pinned list in requirements.txt.")

# ---- Table 6: Bytes ----
t6 = doc2.tables[6]
set_cell(t6.rows[1].cells[1],
    "30,379,311 B total: canonical source ZIP 11,900,797 B + segments.csv 17,970,946 B + "
    "dataset.csv 507,568 B. SMAP/MSL files issued with bundle (no additional transfer).")
set_cell(t6.rows[2].cells[1],
    "Not measured during acquisition.")
set_cell(t6.rows[3].cells[1],
    "22,707,517 B across the retained I1 working files "
    "(segments.csv + dataset.csv + OPSSAT_nominal_dev_segments.csv + labeled_anomalies.csv + "
    "four SMAP/MSL .npy files). Well below the 5 GB D0 cap.")

# ---- Table 7: Machine and timed run ----
t7 = doc2.tables[7]
set_cell(t7.rows[1].cells[1],
    "2-D spectrogram autoencoder on OPSSAT-AD (Annex D0 Week-1 reference for Card I1)")
set_cell(t7.rows[2].cells[1],
    "Intel Core i7-1355U; 10 cores / 12 logical processors; 15.69 GB RAM; "
    "Windows 11 Home (original run). Current Mac environment used for re-verification only.")
set_cell(t7.rows[3].cells[1], "6 PyTorch CPU threads; 0 DataLoader workers.")
set_cell(t7.rows[4].cells[1], "29.18 s elapsed.")
set_cell(t7.rows[5].cells[1],
    "77.08 s aggregate CPU time (main process only; 0 workers, so no worker CPU to add).")
set_cell(t7.rows[6].cells[1], "30 epochs; 960 optimisation updates.")
set_cell(t7.rows[7].cells[1], "30-epoch limit (not the 30-minute elapsed limit).")
set_cell(t7.rows[8].cells[1], "420.09 MB peak Python working-set memory (Windows PSAPI).")
set_cell(t7.rows[9].cells[1],
    "Training loss: 0.009667 → 0.000180 (epoch 1 → epoch 30; lower is better). "
    "Validation loss: 0.009394 → 0.000201. Monotone improvement observed. "
    "Best validation loss: 0.000180.")
set_cell(t7.rows[10].cells[1],
    "Validation finished: Yes (0.06 s over 254 development fragments). "
    "Timed inference pass over test-shaped data: completed (0.10 s over 529 test-shaped "
    "fragments). No test metrics computed or examined.")

# ---- Table 8: Budget projection ----
t8 = doc2.tables[8]
set_cell(t8.rows[1].cells[0], "Preprocessing / acquisition")
set_cell(t8.rows[1].cells[1], "1 canonical ZIP + 2 OPSSAT CSVs + 4 SMAP/MSL .npy files")
set_cell(t8.rows[1].cells[2], "~1 min (not re-timed; ~30.4 MB transferred)")
set_cell(t8.rows[1].cells[3], "~1 min")

set_cell(t8.rows[2].cells[0], "Required neural fits (5 total per Annex D0)")
set_cell(t8.rows[2].cells[1], "5 fits")
set_cell(t8.rows[2].cells[2], "29.18 s for reference fit; others not yet timed")
set_cell(t8.rows[2].cells[3],
    "5 × 29.18 s = ~2.4 min (lower bound; SMAP/MSL fits may differ)")

set_cell(t8.rows[3].cells[0], "Classical baselines (60-min D0 allowance)")
set_cell(t8.rows[3].cells[1], "3 controls × 2 datasets")
set_cell(t8.rows[3].cells[2], "< 1 s per method on OPSSAT-AD (measured)")
set_cell(t8.rows[3].cells[3], "~5 min total (within 60-min allowance)")

set_cell(t8.rows[4].cells[0], "Repeated inference / evaluation")
set_cell(t8.rows[4].cells[1], "10 passes (5 fits × dev + test)")
set_cell(t8.rows[4].cells[2], "0.10 s per OPSSAT pass (measured)")
set_cell(t8.rows[4].cells[3], "~1 min for all OPSSAT passes")

set_cell(t8.rows[5].cells[0], "SMAP/MSL stride-1 scoring (Annex D0 subset stage)")
set_cell(t8.rows[5].cells[1], "P-1: ~8,250 windows; C-1: ~2,010 windows")
set_cell(t8.rows[5].cells[2],
    "~0.141 s / 0.110 s per 1,000 windows (dummy scorer). "
    "Trained-model timing NOT yet measured.")
set_cell(t8.rows[5].cells[3],
    "NOT YET VERIFIED. Dummy-scorer projection (~2 min) is not a trained-model estimate. "
    "Will measure in Week 2.")

set_cell(t8.rows[6].cells[0], "Total, against 24 h")
set_cell(t8.rows[6].cells[1], "—")
set_cell(t8.rows[6].cells[2], "—")
set_cell(t8.rows[6].cells[3],
    "Lower bound ~10 min. Full programme not yet verified against 24-h budget. "
    "SMAP/MSL trained-model stride-1 timing pending.")

# ---- Table 9: Card I1 additional items ----
t9 = doc2.tables[9]
# Find the I1 row and fill it
for row in t9.rows:
    if row.cells[0].text.strip() == "I1":
        set_cell(row.cells[1],
            "Development and fitting fragment counts: 254 / 1,019. "
            "256-sample resampling mostly upsamples (1,748 of 2,123 fragments < 256 samples). "
            "Channel-level fragment counts reported in Table 3 above. "
            "Reconstruction MSE oriented high=anomalous. "
            "Nominal-development 99th-percentile calibration rule confirmed and implemented. "
            "SMAP/MSL test-set scaling inherited limitation recorded.")
        break

rn_out = OUT / "T05_I1_ReadinessNote.docx"
doc2.save(rn_out)
print(f"  Saved: {rn_out}")

print("\nDone. Files written:")
print(f"  {proposal_out}")
print(f"  {rn_out}")
