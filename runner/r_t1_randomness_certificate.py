#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class RandomnessCertificate:
    unit: str
    version: str
    n_bits: int
    n_samples: int
    delta: float

    empirical_entropy: float
    empirical_min_entropy: float

    epsilon_infty: float
    epsilon_L1: float
    epsilon_H: float
    epsilon_Hmin: float

    entropy_lower_bound: float
    min_entropy_lower_bound: float

    raw_data_hash: str
    analysis_code_hash: str | None
    provenance: Dict[str, str]


def _binary_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def _load_bits(path: Path, n_bits: int) -> Tuple[int, Counter]:
    text = path.read_text(encoding="utf-8")
    counts: Counter = Counter()
    total = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s) != n_bits or any(c not in "01" for c in s):
            raise SystemExit(f"Invalid bitstring '{s}' (len={len(s)}) in {path}")
        idx = int(s, 2)
        counts[idx] += 1
        total += 1
    if total == 0:
        raise SystemExit(f"No valid samples found in {path}")
    return total, counts


def _compute_certificate(
    n_bits: int, n_samples: int, counts: Counter, delta: float
) -> RandomnessCertificate:
    K = 2**n_bits
    N = n_samples

    # empirical distribution
    max_p = 0.0
    H_hat = 0.0
    for idx, c in counts.items():
        p = c / N
        if p <= 0.0:
            continue
        H_hat -= p * math.log2(p)
        if p > max_p:
            max_p = p
    H_min_hat = -math.log2(max_p)

    # Hoeffding + union bound for sup-norm on probabilities
    # P( max_k |p_hat(k) - p(k)| > eps ) <= 2K exp(-2 N eps^2) <= delta
    # => eps = sqrt( (ln(2K) + ln(1/delta)) / (2N) )
    epsilon_infty = math.sqrt(
        (math.log(2 * K) + math.log(1.0 / delta)) / (2.0 * N)
    )

    # L1 ≤ K * sup-norm, clamp to [0,1]
    epsilon_L1 = min(1.0, K * epsilon_infty)

    # entropy deviation bound (very standard conservative form)
    # |H(p_hat) - H(p)| <= eps_H = epsilon_L1 * log2(K/epsilon_L1) + H_b(epsilon_L1)
    if epsilon_L1 > 0.0 and epsilon_L1 < 1.0:
        epsilon_H = epsilon_L1 * math.log2(K / epsilon_L1) + _binary_entropy(
            epsilon_L1
        )
    else:
        epsilon_H = 0.0

    # min-entropy deviation: p_max_true <= p_max_hat + epsilon_infty
    # so H_min_true >= -log2(p_max_hat + epsilon_infty)
    # => drop relative to H_min_hat is at most log2((p_max_hat + eps)/p_max_hat)
    epsilon_Hmin = 0.0
    if max_p > 0.0:
        epsilon_Hmin = math.log2((max_p + epsilon_infty) / max_p)

    H_lb = H_hat - epsilon_H
    Hmin_lb = H_min_hat - epsilon_Hmin

    return RandomnessCertificate(
        unit="R1_randomness_certificate",
        version="0.1",
        n_bits=n_bits,
        n_samples=N,
        delta=delta,
        empirical_entropy=H_hat,
        empirical_min_entropy=H_min_hat,
        epsilon_infty=epsilon_infty,
        epsilon_L1=epsilon_L1,
        epsilon_H=epsilon_H,
        epsilon_Hmin=epsilon_Hmin,
        entropy_lower_bound=H_lb,
        min_entropy_lower_bound=Hmin_lb,
        raw_data_hash="",
        analysis_code_hash=None,
        provenance={},
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unit R1 – randomness certificate for a raw bitstring file"
    )
    ap.add_argument(
        "--bits-file",
        required=True,
        help="Path to text file with one n-bit 0/1 string per line",
    )
    ap.add_argument(
        "--n-bits",
        type=int,
        required=True,
        help="Number of bits per sample (line length in file)",
    )
    ap.add_argument(
        "--delta",
        type=float,
        default=1e-6,
        help="Failure probability for the concentration inequality (default: 1e-6)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path for the randomness certificate",
    )
    args = ap.parse_args()

    bits_path = Path(args.bits_file)
    out_path = Path(args.out)

    N, counts = _load_bits(bits_path, args.n_bits)
    cert = _compute_certificate(args.n_bits, N, counts, args.delta)

    # hash the raw data
    raw_bytes = bits_path.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()

    # hash this analysis code for provenance
    try:
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        code_hash = None

    cert.raw_data_hash = raw_hash
    cert.analysis_code_hash = code_hash
    cert.provenance = {
        "bits_file": str(bits_path),
        "generated_by": "runner/r_t1_randomness_certificate.py",
        "notes": "Demo R1 run on a raw bit file.",
    }

    out_path.write_text(json.dumps(asdict(cert), indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  n_bits: {cert.n_bits}")
    print(f"  n_samples: {cert.n_samples}")
    print(f"  delta: {cert.delta}")
    print(f"  empirical_entropy:        {cert.empirical_entropy:.6f} bits")
    print(f"  entropy_lower_bound:      {cert.entropy_lower_bound:.6f} bits")
    print(f"  empirical_min_entropy:    {cert.empirical_min_entropy:.6f} bits")
    print(f"  min_entropy_lower_bound:  {cert.min_entropy_lower_bound:.6f} bits")
    print(f"  raw_data_hash: {cert.raw_data_hash[:16]}...")
    if cert.analysis_code_hash:
        print(f"  analysis_code_hash: {cert.analysis_code_hash[:16]}...")


if __name__ == "__main__":
    main()
