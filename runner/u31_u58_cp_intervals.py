#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

def load_u58():
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs"
    cands = sorted(runs_dir.glob("*_U58_attestation.json"))
    if not cands:
        raise SystemExit("no *_U58_attestation.json files found in runs/")
    return json.loads(cands[-1].read_text()), cands[-1].relative_to(repo_root)

def binom_cdf(k, n, p):
    if p <= 0.0:
        return 1.0 if k >= 0 else 0.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    q = 1.0 - p
    term = q**n
    s = term
    i = 0
    while i < k:
        i += 1
        term = term * (n - i + 1) / i * p / q
        s += term
    return s

def clopper_pearson(k, n, alpha=0.05):
    if k == 0:
        lower = 0.0
        upper = 1.0 - (alpha / 2.0) ** (1.0 / n)
        return lower, upper
    if k == n:
        lower = (alpha / 2.0) ** (1.0 / n)
        upper = 1.0
        return lower, upper
    target_low = alpha / 2.0
    lo, hi = 0.0, float(k) / n
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = binom_cdf(k - 1, n, mid)
        if cdf > target_low:
            hi = mid
        else:
            lo = mid
    lower = 0.5 * (lo + hi)
    target_high = alpha / 2.0
    lo, hi = float(k) / n, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf_k = binom_cdf(k, n, mid)
        if 1.0 - cdf_k > target_high:
            lo = mid
        else:
            hi = mid
    upper = 0.5 * (lo + hi)
    return lower, upper

def main():
    att, rel_path = load_u58()
    r_pass = att.get("min_PASS_r", None)
    if r_pass is None:
        raise SystemExit("min_PASS_r not found in U58 attestation JSON")
    roc = att["ROC"]
    keyed_succ = None
    forger_succ = None
    for row in roc:
        if int(row.get("r")) == int(r_pass):
            keyed_succ = int(row.get("keyed_succ"))
            forger_succ = int(row.get("forger_succ"))
            break
    if keyed_succ is None:
        raise SystemExit(f"no ROC entry with r={r_pass}")
    shots_per_pair = int(att.get("shots_per_pair"))
    num_pairs = int(att.get("num_pairs"))
    n = shots_per_pair * num_pairs
    k1 = keyed_succ
    k0 = forger_succ
    p1_hat = float(k1) / n
    p0_hat = float(k0) / n
    l1, u1 = clopper_pearson(k1, n, 0.05)
    l0, u0 = clopper_pearson(k0, n, 0.05)
    print("File            :", rel_path)
    print("n_shots_per_hyp :", n)
    print("min_PASS_r      :", r_pass)
    print("keyed_succ      :", k1)
    print("forger_succ     :", k0)
    print()
    print("keyed_hat       :", f"{p1_hat:.6f}")
    print("keyed_CI95_low  :", f"{l1:.6f}")
    print("keyed_CI95_high :", f"{u1:.6f}")
    print()
    print("forger_hat      :", f"{p0_hat:.6f}")
    print("forger_CI95_low :", f"{l0:.6f}")
    print("forger_CI95_high:", f"{u0:.6f}")
    ok_tpr = l1 >= 0.99
    ok_fpr = u0 <= 0.01
    print()
    print("PASS_CP_spec    :", ok_tpr and ok_fpr)

if __name__ == "__main__":
    main()
