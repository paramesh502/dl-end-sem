"""
generate_splits.py
==================
Generates the three split-ID CSV files required for Team 05, Card I1
Week 2 submission:

    T05_I1_opssat.csv
    T05_I1_smap_msl.csv
    T05_I1_split_summary.csv

Follows Split_ID_File_Format.md and Annex D6 exactly.
Run from the project root:

    python3 scripts/generate_splits.py
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Release_Bundle_RevD1" / "Release_Bundle_RevD1" / "I1"
OUT  = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: sha-256 hex of a file
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# PART 1 — OPSSAT-AD split-ID file
# ===========================================================================
# Protocol (Annex D6, I1 telemetry section):
#   - Use the provided train/test split.
#   - Within nominal training fragments, order segment IDs by SHA-256 of
#     '22|<segment>' and reserve the first floor(20%) as nominal development.
#   - Anomalous training fragments: unused in nominal-only core.
#   - Sample ID (Handout §13): the segment ID only, e.g. "1" (not "1_CADC0872").
#   - Unit ID: <segment_id>.
#   - reason: "anomalous_train_unused" on excluded rows, "" on every other row.
# ===========================================================================

print("=" * 60)
print("PART 1: OPSSAT-AD split")
print("=" * 60)

segs = pd.read_csv(DATA / "segments.csv")
issued_dev = pd.read_csv(DATA / "OPSSAT_nominal_dev_segments.csv")
labels_csv = pd.read_csv(DATA / "labeled_anomalies.csv")

# Fragment-level attributes (one row per segment)
frag = segs.groupby("segment").agg(
    anomaly=("anomaly", "first"),
    train=("train", "first"),
    channel=("channel", "first"),
).reset_index()
frag["segment"] = frag["segment"].astype(str)
frag["train"] = frag["train"].astype(str)

# Issued dev IDs
issued_dev_ids = set(issued_dev["segment"].astype(str))

# Reproduce split to verify
def sha_key(sid: str) -> str:
    return hashlib.sha256(f"22|{sid}".encode()).hexdigest()

nominal_train_frags = frag[(frag["anomaly"] == 0) & (frag["train"] == "1")].copy()
nom_sorted = sorted(nominal_train_frags["segment"].tolist(), key=sha_key)
n_dev_expected = int(len(nom_sorted) * 0.2)
reproduced_dev_ids = set(nom_sorted[:n_dev_expected])

assert reproduced_dev_ids == issued_dev_ids, (
    f"Dev split mismatch: {len(reproduced_dev_ids.symmetric_difference(issued_dev_ids))} "
    f"differing IDs"
)
print(f"Dev split verified: {len(issued_dev_ids)} dev, "
      f"{len(nom_sorted) - len(issued_dev_ids)} fitting fragments")

# Build split rows — one per fragment
rows = []
for _, row in frag.iterrows():
    sid = row["segment"]
    anom = int(row["anomaly"])
    in_train_side = row["train"] == "1"

    # Handout §13: sample_id is the segment ID only (e.g. "1").
    sample_id = str(sid)
    unit_id   = str(sid)

    if in_train_side:
        if anom == 1:
            # Anomalous training — excluded from nominal-only core
            partition = "excluded"
            label     = 1
            reason    = "anomalous_train_unused"
        elif sid in issued_dev_ids:
            partition = "val"
            label     = 0
            reason    = ""
        else:
            partition = "train"
            label     = 0
            reason    = ""
    else:
        # Test side — labelled by actual anomaly status
        partition = "test"
        label     = anom
        reason    = ""

    rows.append({
        "sample_id": sample_id,
        "unit_id":   unit_id,
        "partition": partition,
        "label":     label,
        "reason":    reason,
    })

opssat_df = pd.DataFrame(rows, columns=[
    "sample_id", "unit_id", "partition", "label", "reason"
])

# Sanity checks
assert opssat_df["sample_id"].nunique() == len(opssat_df), "Duplicate sample IDs in OPSSAT file"
assert set(opssat_df["partition"]).issubset({"train", "val", "test", "excluded"}), "Bad partition value"

print("OPSSAT-AD partition counts:")
print(opssat_df["partition"].value_counts().sort_index().to_string())
print("Total rows:", len(opssat_df))

opssat_path = OUT / "T05_I1_opssat.csv"
opssat_df.to_csv(opssat_path, index=False)
print(f"Written: {opssat_path}")


# ===========================================================================
# PART 2 — SMAP/MSL split-ID file
# ===========================================================================
# Protocol (Annex D6, SMAP/MSL section):
#   - Use published train/test files for P-1 and C-1.
#   - Split the nominal training stream: first 80% fit, last 20% calibration,
#     with a 256-step guard gap between them.
#   - Scoring uses 256-step windows, stride 1 for scoring (stride 64 for
#     training).
#   - The split-ID file records native index ranges, not one row per window.
#   - Sample ID (Handout §13): half-open range "<stream>:<file>:<start>-<end>",
#     e.g. "P-1:train:0-2297" (fit), "P-1:train:2297-2553" (guard),
#     "P-1:train:2553-2872" (cal), "P-1:test:0-8505" (test). End is exclusive.
#   - Unit ID: "P-1" or "C-1".
#   - label: empty string (labels are interval-based, held in labeled_anomalies.csv).
#   - reason: "guard" on guard rows, "" everywhere else.
# ===========================================================================

print()
print("=" * 60)
print("PART 2: SMAP/MSL split")
print("=" * 60)

STREAMS = [
    ("P-1", "SMAP"),
    ("C-1", "MSL"),
]
GUARD = 256  # guard gap in steps

smap_rows = []

for stream_id, benchmark in STREAMS:
    train_arr = np.load(DATA / f"SMAP_MSL_train_{stream_id}.npy", allow_pickle=False)
    test_arr  = np.load(DATA / f"SMAP_MSL_test_{stream_id}.npy",  allow_pickle=False)

    N_train = len(train_arr)
    N_test  = len(test_arr)

    # Fit: first 80% of training stream
    fit_end   = int(N_train * 0.8)   # exclusive
    # Guard gap: [fit_end, fit_end + GUARD)
    cal_start = fit_end + GUARD       # calibration starts after guard
    cal_end   = N_train               # exclusive

    print(f"\nStream {benchmark} {stream_id}:")
    print(f"  Train file length: {N_train}")
    print(f"  Test  file length: {N_test}")
    print(f"  Fit  range [0, {fit_end - 1}]  ({fit_end} steps)")
    print(f"  Guard range [{fit_end}, {fit_end + GUARD - 1}]  ({GUARD} steps)")
    print(f"  Cal  range [{cal_start}, {cal_end - 1}]  ({cal_end - cal_start} steps)")
    print(f"  Test range [0, {N_test - 1}]  ({N_test} steps)")

    unit_id = stream_id   # Handout §13: "P-1" or "C-1"

    # Half-open ranges "<stream>:<file>:<start>-<end>" (end exclusive)
    smap_rows.append({
        "sample_id": f"{stream_id}:train:0-{fit_end}",
        "unit_id":   unit_id,
        "partition": "train",
        "label":     "",
        "reason":    "",
    })
    smap_rows.append({
        "sample_id": f"{stream_id}:train:{fit_end}-{cal_start}",
        "unit_id":   unit_id,
        "partition": "excluded",
        "label":     "",
        "reason":    "guard",
    })
    smap_rows.append({
        "sample_id": f"{stream_id}:train:{cal_start}-{cal_end}",
        "unit_id":   unit_id,
        "partition": "val",
        "label":     "",
        "reason":    "",
    })
    smap_rows.append({
        "sample_id": f"{stream_id}:test:0-{N_test}",
        "unit_id":   unit_id,
        "partition": "test",
        "label":     "",
        "reason":    "",
    })

smap_df = pd.DataFrame(smap_rows, columns=[
    "sample_id", "unit_id", "partition", "label", "reason"
])

assert smap_df["sample_id"].nunique() == len(smap_df), "Duplicate sample IDs in SMAP file"

print("\nSMAP/MSL partition counts:")
print(smap_df["partition"].value_counts().sort_index().to_string())

smap_path = OUT / "T05_I1_smap_msl.csv"
smap_df.to_csv(smap_path, index=False)
print(f"Written: {smap_path}")


# ===========================================================================
# PART 3 — Summary CSV
# ===========================================================================

print()
print("=" * 60)
print("PART 3: Split summary")
print("=" * 60)

summary_rows = []

for csv_path, df in [(opssat_path, opssat_df), (smap_path, smap_df)]:
    sha = file_sha256(csv_path)
    filename = csv_path.name
    for part in ["train", "val", "test", "excluded"]:
        sub = df[df["partition"] == part]
        n_samples = len(sub)
        n_units   = sub["unit_id"].nunique()
        summary_rows.append({
            "file":       filename,
            "partition":  part,
            "n_samples":  n_samples,
            "n_units":    n_units,
            "sha256":     sha,
        })

summary_df = pd.DataFrame(summary_rows, columns=[
    "file", "partition", "n_samples", "n_units", "sha256"
])

summary_path = OUT / "T05_I1_split_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("Split summary:")
print(summary_df.to_string(index=False))
print(f"\nWritten: {summary_path}")
print()
print("=" * 60)
print("All split files generated successfully.")
print("=" * 60)
