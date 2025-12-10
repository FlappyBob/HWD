import os
import csv
from hwd.datasets import FolderDataset, GeneratedDataset
from hwd.scores import (
    HWDScore, FIDScore, BFIDScore,
    KIDScore, BKIDScore,
    LPIPSScore, IntraLPIPSScore,
    CERScore
)

# ============================================================
# Evaluation pairs: (fake_dataset_path, real_dataset_key)
# ============================================================
DATASETS = [
    ("small_emuru_words", "iam_words__reference"),
    ("small_emuru_lines", "iam_lines__reference"),
    ("IAMWords_my_model", "iam_words__reference"),
    ("IAMLines_my_model", "iam_lines__reference"),
]

# ============================================================
# Where to save CSV
# ============================================================
CSV_PATH = "eval_results.csv"

# ============================================================
# Helper function to evaluate one dataset
# ============================================================
def evaluate_dataset(fake_path, real_key):

    print(f"\n======================================")
    print(f" Evaluating {fake_path}")
    print("======================================")

    fakes = FolderDataset(fake_path)
    reals = GeneratedDataset(real_key)

    print(f"Loaded {len(fakes)} generated samples")
    print(f"Loaded {len(reals)} reference samples")

    results = {}

    # HWD
    print("\n[1] HWD Score:")
    hwd = HWDScore(height=32)
    results["HWD"] = hwd(fakes, reals)
    print(" →", results["HWD"])

    # FID
    print("\n[2] FID Score:")
    fid = FIDScore(height=32)
    results["FID"] = fid(fakes, reals)
    print(" →", results["FID"])

    # BFID
    print("\n[3] BFID Score:")
    bfid = BFIDScore(height=32)
    results["BFID"] = bfid(fakes, reals)
    print(" →", results["BFID"])

    # KID
    print("\n[4] KID Score:")
    kid = KIDScore(height=32)
    results["KID"] = kid(fakes, reals)
    print(" →", results["KID"])

    # CER
    trans_json = os.path.join(fake_path, "transcriptions.json")
    if os.path.isfile(trans_json):
        print("\n[5] CER Score:")
        cer = CERScore(height=64)
        results["CER"] = cer(fakes)
        print(" →", results["CER"])
    else:
        print("\n[5] CER Score skipped (no transcriptions.json)")
        results["CER"] = None

    return results


# ============================================================
# Master evaluation loop
# ============================================================
all_results = []

for fake_path, real_key in DATASETS:
    row = {"dataset_name": fake_path}
    result = evaluate_dataset(fake_path, real_key)

    # merge scores
    row.update(result)
    all_results.append(row)


# ============================================================
# Write CSV
# ============================================================
print("\n======================================")
print(f"Saving results to {CSV_PATH}")
print("======================================")

fieldnames = ["dataset_name", "HWD", "FID", "BFID", "KID", "CER"]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

print("🎉 CSV saved successfully!")
print(f"Path: {CSV_PATH}")
