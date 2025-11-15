#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def find_u58_attestation(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*_U58_attestation.json"))
    if not candidates:
        raise SystemExit("No *_U58_attestation.json files found in runs/")
    return candidates[-1]


def compute_auc_from_rates(fprs: np.ndarray, tprs: np.ndarray) -> float:
    pts = sorted(zip(fprs.tolist(), tprs.tolist()), key=lambda x: x[0])
    area = 0.0
    last_fpr, last_tpr = 0.0, 0.0
    for fpr, tpr in pts:
        dx = fpr - last_fpr
        area += dx * (tpr + last_tpr) / 2.0
        last_fpr, last_tpr = fpr, tpr
    return area


def derive_totals_and_null_probs(roc: List[dict]) -> Tuple[int, int, np.ndarray]:
    keyed_succ = np.array([int(row.get("keyed_succ", 0)) for row in roc], dtype=float)
    forg_succ = np.array([int(row.get("forger_succ", 0)) for row in roc], dtype=float)
    s_key = int(keyed_succ.max())
    s_forg = int(forg_succ.max())
    total = s_key + s_forg
    if total == 0:
        raise SystemExit("Zero total shots inferred from ROC; cannot build null model.")
    p = (keyed_succ + forg_succ) / float(total)
    return s_key, s_forg, p


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs"
    att_path = find_u58_attestation(runs_dir)

    print(f"File: {att_path.relative_to(repo_root)}")

    att = json.loads(att_path.read_text())
    roc = att["ROC"]

    auc_obs = float(att.get("AUC", 0.0))

    s_key, s_forg, p = derive_totals_and_null_probs(roc)

    np.random.seed(20251113)
    n_mc = 2000
    auc_null = np.empty(n_mc, dtype=float)

    for i in range(n_mc):
        keyed_counts = np.random.binomial(s_key, p)
        forg_counts = np.random.binomial(s_forg, p)
        tprs = keyed_counts.astype(float) / float(s_key)
        fprs = forg_counts.astype(float) / float(s_forg)
        auc_null[i] = compute_auc_from_rates(fprs, tprs)

    mean_null = float(auc_null.mean())
    std_null = float(auc_null.std(ddof=1))
    q95 = float(np.percentile(auc_null, 95))
    q99 = float(np.percentile(auc_null, 99))

    p_val = float((np.sum(auc_null >= auc_obs) + 1.0) / (n_mc + 1.0))

    print(f"AUC_obs        : {auc_obs:.9f}")
    print(f"AUC_null_mean  : {mean_null:.9f}")
    print(f"AUC_null_std   : {std_null:.9f}")
    print(f"AUC_null_q95   : {q95:.9f}")
    print(f"AUC_null_q99   : {q99:.9f}")
    print(f"p_value_null   : {p_val:.3e}")
    print()
    print("PASS_U56_null  :", p_val < 0.01)


if __name__ == "__main__":
    main()
