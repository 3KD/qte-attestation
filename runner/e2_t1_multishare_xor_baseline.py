#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class E2T1Result:
    label: str
    message_bits: int
    n_keys: int
    n_trials: int
    seed: int
    honest_bit_error_rate: float
    partial_bit_error_rates: List[float]
    avg_partial_bit_error_rate: float


def run_e2_t1(
    message_bits: int,
    n_keys: int,
    n_trials: int,
    seed: int,
) -> E2T1Result:
    if message_bits <= 0:
        raise ValueError("message_bits must be positive")
    if n_keys < 2:
        raise ValueError("n_keys must be at least 2 for a multi-share scheme")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    rng = np.random.default_rng(seed)

    # Counters
    honest_bit_errors = 0
    partial_bit_errors = np.zeros(n_keys, dtype=np.int64)

    total_bits_checked = n_trials * message_bits

    for _ in range(n_trials):
        # Random message m in {0,1}^{message_bits}
        m = rng.integers(0, 2, size=message_bits, dtype=np.int8)

        # Random key shares k_i in {0,1}^{message_bits}
        keys = rng.integers(0, 2, size=(n_keys, message_bits), dtype=np.int8)

        # Aggregate key K_total = XOR_i k_i
        k_total = np.bitwise_xor.reduce(keys, axis=0)

        # Ciphertext c = m XOR K_total
        c = np.bitwise_xor(m, k_total)

        # Honest decrypt with all keys: m_rec = c XOR K_total
        m_rec = np.bitwise_xor(c, k_total)
        honest_bit_errors += int(np.count_nonzero(m_rec != m))

        # Partial decrypts: missing each key j in turn
        # If we omit key j, we compute K_partial = XOR_{i≠j} k_i
        # Then m_partial = c XOR K_partial = m XOR k_j (still masked).
        for j in range(n_keys):
            if n_keys == 1:
                continue
            # XOR all keys except j
            if j == 0:
                k_partial = np.bitwise_xor.reduce(keys[1:], axis=0)
            elif j == n_keys - 1:
                k_partial = np.bitwise_xor.reduce(keys[:-1], axis=0)
            else:
                k_partial = np.bitwise_xor.reduce(
                    np.concatenate((keys[:j], keys[j + 1 :]), axis=0),
                    axis=0,
                )

            m_partial = np.bitwise_xor(c, k_partial)
            partial_bit_errors[j] += int(np.count_nonzero(m_partial != m))

    honest_bit_error_rate = honest_bit_errors / float(total_bits_checked)
    partial_bit_error_rates = (partial_bit_errors / float(total_bits_checked)).tolist()
    avg_partial_bit_error_rate = float(np.mean(partial_bit_error_rates))

    return E2T1Result(
        label="E2-T1_multishare_xor_baseline",
        message_bits=message_bits,
        n_keys=n_keys,
        n_trials=n_trials,
        seed=seed,
        honest_bit_error_rate=honest_bit_error_rate,
        partial_bit_error_rates=partial_bit_error_rates,
        avg_partial_bit_error_rate=avg_partial_bit_error_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "E2-T1: baseline multi-share XOR scheme.\n\n"
            "Models r classical key shares k_i ∈ {0,1}^m with ciphertext "
            "c = m ⊕ (⊕_i k_i). Honest decrypt uses all shares; partial "
            "decrypt omits one share and stays information-theoretically masked."
        )
    )
    parser.add_argument(
        "--message-bits",
        type=int,
        required=True,
        help="Number of message bits m.",
    )
    parser.add_argument(
        "--n-keys",
        type=int,
        required=True,
        help="Number of independent key shares (r ≥ 2).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10000,
        help="Number of random (message, keys) trials.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=424242,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path for summary metrics.",
    )
    args = parser.parse_args()

    res = run_e2_t1(
        message_bits=args.message_bits,
        n_keys=args.n_keys,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2, sort_keys=True)

    print(f"wrote {out_path}")
    print("  label:", res.label)
    print("  message_bits:", res.message_bits)
    print("  n_keys:", res.n_keys)
    print("  n_trials:", res.n_trials)
    print("  honest_bit_error_rate:    ", res.honest_bit_error_rate)
    print("  partial_bit_error_rates:  ", res.partial_bit_error_rates)
    print("  avg_partial_bit_error_rate:", res.avg_partial_bit_error_rate)


if __name__ == "__main__":
    main()
