#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class E2T2Result:
    label: str
    n_qubits: int
    message_bits: int
    n_keys: int
    n_trials: int
    seed: int
    avg_honest_fidelity: float
    avg_partial_fidelities: List[float]
    avg_partial_fidelity: float


def make_basis_bits(n_qubits: int) -> np.ndarray:
    """
    basis_bits[b, i] = i-th bit of basis index b (LSB-first), in {0,1}.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    N = 1 << n_qubits
    bits = np.zeros((N, n_qubits), dtype=np.int8)
    for b in range(N):
        for i in range(n_qubits):
            bits[b, i] = (b >> i) & 1
    return bits


def make_key_state(n_qubits: int) -> np.ndarray:
    """
    Construct a fixed, non-trivial 'NVADE-like' key state |psi_key> in C^{2^n}:

        psi[b] ∝ exp(-((b - center)^2)/(2 sigma^2)) * exp(i 2π b / N)

    so it has both amplitude structure and phase structure.
    """
    N = 1 << n_qubits
    idx = np.arange(N, dtype=np.float64)
    center = (N - 1) / 2.0
    sigma = max(1.0, N / 8.0)

    envelope = np.exp(-((idx - center) ** 2) / (2.0 * sigma ** 2))
    phase = np.exp(1j * 2.0 * np.pi * idx / float(N))
    psi = envelope * phase

    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise RuntimeError("constructed zero-norm key state")
    psi = (psi / norm).astype(np.complex128)
    return psi


def run_e2_t2(
    n_qubits: int,
    message_bits: int,
    n_keys: int,
    n_trials: int,
    seed: int,
) -> E2T2Result:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if message_bits <= 0:
        raise ValueError("message_bits must be positive")
    if n_keys < 2:
        raise ValueError("n_keys must be at least 2 for a multi-share scheme")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    # For this model we tie message bits to per-qubit Z masks:
    if message_bits != n_qubits:
        raise ValueError(
            f"for this E2-T2 model, require message_bits == n_qubits "
            f"(got message_bits={message_bits}, n_qubits={n_qubits})"
        )

    rng = np.random.default_rng(seed)
    basis_bits = make_basis_bits(n_qubits)
    psi_key = make_key_state(n_qubits)
    N = psi_key.shape[0]
    if N != (1 << n_qubits):
        raise RuntimeError("internal key state dimension mismatch")

    total_honest_fid = 0.0
    total_partial_fids = np.zeros(n_keys, dtype=np.float64)

    for _ in range(n_trials):
        # Random message m ∈ {0,1}^n and r key shares k_j ∈ {0,1}^n.
        m = rng.integers(0, 2, size=message_bits, dtype=np.int8)
        keys = rng.integers(0, 2, size=(n_keys, message_bits), dtype=np.int8)

        # Total phase mask bitstring a_total = m ⊕ k_1 ⊕ ... ⊕ k_r.
        a_total = m.copy()
        for j in range(n_keys):
            a_total ^= keys[j]

        # phase_total[b] = (-1)^{a_total · b}  using precomputed basis_bits.
        parity_total = (basis_bits @ a_total) & 1  # shape (N,)
        phase_total = np.where(parity_total == 0, 1.0, -1.0).astype(np.complex128)

        # Encryption: |psi_enc> = U_total |psi_key>.
        psi_enc = phase_total * psi_key

        # Honest decrypt: apply U_total again, since U_total^2 = I.
        psi_dec = phase_total * psi_enc
        norm_dec = np.linalg.norm(psi_dec)
        if norm_dec == 0.0:
            raise RuntimeError("zero-norm decrypted state")
        psi_dec /= norm_dec

        inner = np.vdot(psi_key, psi_dec)
        F_honest = float(np.abs(inner) ** 2)
        total_honest_fid += F_honest

        # Partial decrypts: omit share j, so leftover mask is Z^{k_j}.
        for j in range(n_keys):
            a_partial = m.copy()
            for k in range(n_keys):
                if k == j:
                    continue
                a_partial ^= keys[k]

            parity_partial = (basis_bits @ a_partial) & 1
            phase_partial = np.where(parity_partial == 0, 1.0, -1.0).astype(np.complex128)

            psi_partial = phase_partial * psi_enc
            norm_partial = np.linalg.norm(psi_partial)
            if norm_partial == 0.0:
                # Extremely unlikely; skip contribution if it happens.
                continue
            psi_partial /= norm_partial

            inner_p = np.vdot(psi_key, psi_partial)
            F_partial = float(np.abs(inner_p) ** 2)
            total_partial_fids[j] += F_partial

    avg_honest = total_honest_fid / float(n_trials)
    avg_partial_list = (total_partial_fids / float(n_trials)).tolist()
    avg_partial = float(np.mean(avg_partial_list))

    return E2T2Result(
        label="E2-T2_nvade_multishare_phase_numpy",
        n_qubits=n_qubits,
        message_bits=message_bits,
        n_keys=n_keys,
        n_trials=n_trials,
        seed=seed,
        avg_honest_fidelity=avg_honest,
        avg_partial_fidelities=avg_partial_list,
        avg_partial_fidelity=avg_partial,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "E2-T2: NVADE-style multishare phase-mask model (numpy).\n"
            "Key: a structured amplitude state |psi_key> ∈ C^{2^n}.\n"
            "Encryption: diagonal phase U_total(m, k_1,...,k_r) with bits "
            "a_total = m ⊕ k_1 ⊕ ... ⊕ k_r applied as (-1)^{a_total·b} on each basis index b.\n"
            "Honest decrypt: apply U_total again (U_total^2 = I). Missing any share leaves "
            "a residual phase Z^{k_j}, making |psi_partial> nearly orthogonal to |psi_key>."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        required=True,
        help="Number of qubits n (state dimension 2^n).",
    )
    parser.add_argument(
        "--message-bits",
        type=int,
        required=True,
        help="Number of message bits; in this model must equal n_qubits.",
    )
    parser.add_argument(
        "--n-keys",
        type=int,
        required=True,
        help="Number of key shares r (>= 2).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=2000,
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

    res = run_e2_t2(
        n_qubits=args.n_qubits,
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
    print("  n_qubits:", res.n_qubits)
    print("  message_bits:", res.message_bits)
    print("  n_keys:", res.n_keys)
    print("  n_trials:", res.n_trials)
    print("  avg_honest_fidelity:     ", res.avg_honest_fidelity)
    print("  avg_partial_fidelities:  ", res.avg_partial_fidelities)
    print("  avg_partial_fidelity:    ", res.avg_partial_fidelity)


if __name__ == "__main__":
    main()
