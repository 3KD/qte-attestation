#!/usr/bin/env python3
"""
E1-T2: NVade-style phase encryption correctness test.

- Key state: truncated Ramanujan π series encoded as an amplitude vector.
- Encryption: apply a diagonal ±1 phase pattern determined by message bits.
- Decryption: apply the same keyed unitary again (Z-phase is its own inverse).
- Test: check that decrypt∘encrypt(|ψ_key>) ≈ |ψ_key> with high fidelity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from math import factorial
from typing import Sequence

import numpy as np

MAX_TERMS = 25  # max Ramanujan series terms used for amplitudes


@dataclass
class E1T2Result:
    label: str
    n_qubits: int
    message_bits: int
    n_trials: int
    seed: int
    average_fidelity: float
    min_fidelity: float


def ramanujan_pi_terms(num_terms: int) -> np.ndarray:
    """Return the first `num_terms` coefficients of Ramanujan's 1/pi series."""
    if num_terms <= 0:
        return np.zeros(0, dtype=float)

    coeffs = np.zeros(num_terms, dtype=float)
    for k in range(num_terms):
        num = factorial(4 * k) * (1103 + 26390 * k)
        den = (factorial(k) ** 4) * (396.0 ** (4 * k))
        coeffs[k] = num / den
    return coeffs


def build_ramanujan_state(n_qubits: int, max_terms: int = MAX_TERMS) -> np.ndarray:
    """Build normalized |psi_key> from truncated Ramanujan coefficients."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")

    dim = 1 << n_qubits
    t = ramanujan_pi_terms(min(dim, max_terms))

    state = np.zeros(dim, dtype=complex)
    state[: t.size] = t.astype(complex)

    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise ValueError("Ramanujan state has zero norm (unexpected).")

    state /= norm
    return state


def phase_mask_for_message(n_qubits: int, message: Sequence[int]) -> np.ndarray:
    """
    Construct a ±1 phase mask for a given message bitstring.

    For each computational basis index x in [0, 2^n-1], the phase is:
        (-1)^{<x, m>}
    where <x, m> is bitwise inner product mod 2 between x and the message bits
    (padded with zeros beyond len(m)).
    """
    m_bits = np.array(message, dtype=int).ravel()
    if m_bits.size > n_qubits:
        raise ValueError("message_bits must be <= n_qubits")

    dim = 1 << n_qubits
    phases = np.ones(dim, dtype=complex)

    mask_int = 0
    for j, bit in enumerate(m_bits):
        if bit not in (0, 1):
            raise ValueError("message bits must be 0/1")
        if bit == 1:
            mask_int |= (1 << j)

    for x in range(dim):
        if bin(x & mask_int).count("1") % 2 == 1:
            phases[x] = -1.0

    return phases


def apply_message_unitary(state: np.ndarray, message: Sequence[int]) -> np.ndarray:
    """Apply the keyed diagonal unitary U_m = diag((-1)^{<x, m>}) to a state."""
    dim = state.shape[0]
    n_qubits = int(round(math.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError("state dimension must be a power of 2")

    phases = phase_mask_for_message(n_qubits, message)
    return state * phases


def run_e1_t2(
    n_qubits: int,
    message_bits: int,
    n_trials: int,
    seed: int,
) -> E1T2Result:
    if message_bits <= 0:
        raise ValueError("message_bits must be positive")
    if message_bits > n_qubits:
        raise ValueError("message_bits must be <= n_qubits")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    key_state = build_ramanujan_state(n_qubits)
    rng = np.random.default_rng(seed)

    fidelities = np.empty(n_trials, dtype=float)

    for _ in range(n_trials):
        msg = rng.integers(0, 2, size=message_bits, dtype=int)
        cipher = apply_message_unitary(key_state, msg)
        recovered = apply_message_unitary(cipher, msg)
        overlap = np.vdot(key_state, recovered)
        fidelities[_] = float(abs(overlap) ** 2)

    return E1T2Result(
        label=f"E1-T2_nvade_phase_ramanujan_n{n_qubits}_m{message_bits}",
        n_qubits=n_qubits,
        message_bits=message_bits,
        n_trials=n_trials,
        seed=seed,
        average_fidelity=float(fidelities.mean()),
        min_fidelity=float(fidelities.min()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1-T2: NVade-style phase encryption correctness test.",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=6,
        help="Number of qubits in the Ramanujan key state (default: 6).",
    )
    parser.add_argument(
        "--message-bits",
        type=int,
        default=6,
        help="Number of message bits (<= n-qubits). Default: 6.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5000,
        help="Number of random messages to test (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=424242,
        help="Random seed for reproducibility (default: 424242).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to JSON output file.",
    )

    args = parser.parse_args()

    res = run_e1_t2(
        n_qubits=args.n_qubits,
        message_bits=args.message_bits,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    payload = asdict(res)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"wrote {args.out}")
    print("  label:", res.label)
    print("  n_qubits:", res.n_qubits)
    print("  message_bits:", res.message_bits)
    print("  n_trials:", res.n_trials)
    print("  average_fidelity:   ", res.average_fidelity)
    print("  min_fidelity:       ", res.min_fidelity)


if __name__ == "__main__":
    main()
