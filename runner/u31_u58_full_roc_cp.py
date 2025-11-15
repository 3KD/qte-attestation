#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple
from math import isfinite

try:
    from scipy.stats import beta
except ImportError as e:
    raise SystemExit("scipy is required for Clopper–Pearson (pip install scipy in qiskit-fresh)") from e


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    if k < 0 or k > n:
        raise ValueError(f"invalid k={k} for n={n}")
    if k == 0:
        low = 0.0
    else:
        low = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    if k == n:
        high = 1.0
    else:
        high = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    if not isfinite(low):
        low = 0.0
    if not isfinite(high):
        high = 1.0
    return low, high


def find_u58_attestation(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*_U58_attestation.json"))
    if not candidates:
        raise SystemExit("No *_U58_attestation.json files found in runs/")
    return candidates[-1]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs"
    att_path = find_u58_attestation(runs_dir)

    att = json.loads(att_path.read_text())
    shots_per_pair = int(att.get("shots_per_pair", 0))
    num_pairs = int(att.get("num_pairs", 0))
    n = shots_per_pair * num_pairs

    roc = att["ROC"]

    alpha = 0.05
    tpr_target = 0.99
    fpr_target = 0.01

    print("File:", att_path.relative_to(repo_root))
    print("n_shots_per_hyp:", n)
    print()

    print("r  keyed_hat  keyed_low  keyed_high  forger_hat  forger_low  forger_high")

    passing_rs = []

    for row in roc:
        r = int(row.get("r", 0))
        k_keyed = int(row.get("keyed_succ", 0))
        k_forg = int(row.get("forger_succ", 0))

        p_keyed = k_keyed / n if n > 0 else 0.0
        p_forg = k_forg / n if n > 0 else 0.0

        kl, kh = clopper_pearson(k_keyed, n, alpha)
        fl, fh = clopper_pearson(k_forg, n, alpha)

        print(
            f"{r:2d} {p_keyed:10.6f} {kl:10.6f} {kh:10.6f} "
            f"{p_forg:11.6f} {fl:11.6f} {fh:11.6f}"
        )

        if kl >= tpr_target and fh <= fpr_target:
            passing_rs.append(r)

    print()
    if passing_rs:
        r_min = min(passing_rs)
        r_max = max(passing_rs)
        print("PASS_ROC_CP:", True)
        print("good_r_range:", r_min, r_max)
        print("good_r_list:", passing_rs)
    else:
        print("PASS_ROC_CP:", False)


if __name__ == "__main__":
    main()
