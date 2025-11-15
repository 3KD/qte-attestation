#!/usr/bin/env python
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import List, Tuple


def find_u58_attestation(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*_U58_attestation.json"))
    if not candidates:
        raise SystemExit("No *_U58_attestation.json files found in runs/")
    return candidates[-1]


def compute_auc(roc: List[dict]) -> float:
    pts: List[Tuple[float, float]] = []
    for row in roc:
        fpr = float(row.get("FPR", 0.0))
        tpr = float(row.get("TPR", 0.0))
        pts.append((fpr, tpr))
    pts.sort(key=lambda x: x[0])

    area = 0.0
    last_fpr, last_tpr = 0.0, 0.0
    for fpr, tpr in pts:
        dx = fpr - last_fpr
        area += dx * (tpr + last_tpr) / 2.0
        last_fpr, last_tpr = fpr, tpr
    return area


def tpr_at_fpr(roc: List[dict], target_fpr: float) -> float:
    best = 0.0
    for row in roc:
        fpr = float(row.get("FPR", 0.0))
        tpr = float(row.get("TPR", 0.0))
        if fpr <= target_fpr and tpr > best:
            best = tpr
    return best


def canonical_sha256(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs"
    att_path = find_u58_attestation(runs_dir)

    print(f"File: {att_path.relative_to(repo_root)}")

    att = json.loads(att_path.read_text())
    roc = att["ROC"]

    auc_stored = float(att.get("AUC", 0.0))
    tpr1_stored = float(att.get("TPR_at_1pct_FPR", 0.0))

    auc_re = compute_auc(roc)
    tpr1_re = tpr_at_fpr(roc, 0.01)

    payload = {k: v for k, v in att.items() if k != "provenance"}
    sha_re = canonical_sha256(payload)

    prov = att.get("provenance", {}) or {}
    sha_stored = prov.get("sha256", "")

    print(f"AUC stored     : {auc_stored:.16f}")
    print(f"AUC recomputed : {auc_re:.9f}")
    print(f"AUC diff       : {abs(auc_stored - auc_re):.3e}")
    print()
    print(f"TPR@1% FPR stored    : {tpr1_stored:.6f}")
    print(f"TPR@1% FPR recomputed: {tpr1_re:.6f}")
    print()
    print(f"Stored sha256  : {sha_stored}")
    print(f"Recomp. sha256 : {sha_re}")
    print(f"SHA256 MATCH   : {sha_stored == sha_re}")

    ok_auc = abs(auc_stored - auc_re) <= 1e-6
    ok_tpr = abs(tpr1_stored - tpr1_re) <= 1e-6
    ok_sha = (sha_stored == sha_re)

    print()
    print("PASS_U31_U58   :", ok_auc and ok_tpr and ok_sha)


if __name__ == "__main__":
    main()
