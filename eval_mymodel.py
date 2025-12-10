import os
from hwd.datasets import FolderDataset,GeneratedDataset
from hwd.scores import (
    HWDScore, FIDScore, BFIDScore,
    KIDScore, BKIDScore,
    LPIPSScore, IntraLPIPSScore,
    CERScore
)

DATA_FAKE = "/workspace/HWD/IAMLines_my_model"
DATA_REAL = "iam_lines__reference"  # 自动从 HWD Releases 加载官方参考集

print("Loading datasets...")
fakes = FolderDataset(DATA_FAKE)
reals = GeneratedDataset(DATA_REAL)

print(f"Loaded {len(fakes)} generated samples")
print(f"Loaded {len(reals)} reference samples")


# ======================================
# RUN SCORES
# ======================================

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


# CER（仅当存在 transcriptions.json）
trans_json = os.path.join(DATA_FAKE, "transcriptions.json")
if os.path.isfile(trans_json):
    print("\n[8] CER Score:")
    cer = CERScore(height=64)
    results["CER"] = cer(fakes)
    print(" →", results["CER"])
else:
    print("\n[8] CER Score skipped (no transcriptions.json found)")



# ======================================
# FINAL SUMMARY
# ======================================
print("\n===============================")
print(" FINAL EVALUATION RESULTS")
print("===============================")
for k, v in results.items():
    print(f"{k}: {v}")
