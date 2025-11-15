from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


def product_state_amplitudes(angles: np.ndarray) -> np.ndarray:
    """Return amplitudes for an n-qubit product state defined by angles.

    Each qubit i is prepared as cos(theta_i)|0> + sin(theta_i)|1>.
    """
    angles = np.asarray(angles, dtype=float).reshape(-1)
    n = angles.size
    dim = 1 << n
    vec = np.zeros(dim, dtype=np.complex128)
    for i in range(dim):
        b = format(i, f"0{n}b")
        amp = 1.0
        # qubit 0 = least significant bit
        for q, bit in enumerate(reversed(b)):
            theta = angles[q]
            if bit == "0":
                amp *= math.cos(theta)
            else:
                amp *= math.sin(theta)
        vec[i] = amp
    vec = vec / np.linalg.norm(vec)
    return vec


def build_prob_map(vec: np.ndarray) -> Dict[str, float]:
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    n = int(round(math.log2(vec.size)))
    if 1 << n != vec.size:
        raise ValueError(f"Vector length {vec.size} is not a power of two.")
    probs = np.abs(vec) ** 2
    probs = probs / probs.sum()
    out: Dict[str, float] = {}
    for i, p in enumerate(probs):
        b = format(i, f"0{n}b")
        out[b] = float(p)
    return out


def scores_from_counts_llr(
    counts: Dict[str, int],
    p_ref: Dict[str, float],
    p_alt: Dict[str, float],
) -> np.ndarray:
    """Turn observed counts into log-likelihood ratio scores.

    S(x) = log( p_ref(x) / p_alt(x) ), replicated 'count' times.
    """
    scores: List[float] = []
    eps = 1e-15
    for bitstring, c in counts.items():
        pr = p_ref.get(bitstring, eps)
        pa = p_alt.get(bitstring, eps)
        s = math.log((pr + eps) / (pa + eps))
        scores.extend([s] * int(c))
    return np.asarray(scores, dtype=float)


@dataclass
class RocResult:
    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc: float
    tpr_at_1pct_fpr: float


def compute_roc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> RocResult:
    """Compute ROC and AUC from positive/negative score samples."""
    pos = np.asarray(scores_pos, dtype=float).reshape(-1)
    neg = np.asarray(scores_neg, dtype=float).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Need at least one positive and one negative score.")

    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])

    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    P = float((labels == 1).sum())
    N = float((labels == 0).sum())

    tpr_list: List[float] = [0.0]
    fpr_list: List[float] = [0.0]
    thr_list: List[float] = [float("inf")]

    tp = 0.0
    fp = 0.0

    for s, y in zip(scores, labels):
        if y == 1:
            tp += 1.0
        else:
            fp += 1.0
        tpr = tp / P
        fpr = fp / N
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        thr_list.append(float(s))

    if tpr_list[-1] != 1.0 or fpr_list[-1] != 1.0:
        tpr_list.append(1.0)
        fpr_list.append(1.0)
        thr_list.append(-float("inf"))

    auc = 0.0
    for i in range(1, len(fpr_list)):
        dx = fpr_list[i] - fpr_list[i - 1]
        auc += dx * 0.5 * (tpr_list[i] + tpr_list[i - 1])

    target = 0.01
    tpr_at = 0.0
    for f, t in zip(fpr_list, tpr_list):
        if f <= target and t > tpr_at:
            tpr_at = t

    return RocResult(
        fpr=tpr_list,
        tpr=tpr_list,
        thresholds=thr_list,
        auc=float(auc),
        tpr_at_1pct_fpr=float(tpr_at),
    )


def build_product_circuit(angles: np.ndarray):
    """Build an n-qubit product-state circuit with RY rotations and full measurement."""
    from qiskit import QuantumCircuit  # type: ignore

    angles = np.asarray(angles, dtype=float).reshape(-1)
    n = angles.size
    qc = QuantumCircuit(n, n)
    for q, theta in enumerate(angles):
        qc.ry(2 * theta, q)
    qc.measure(range(n), range(n))
    return qc


def choose_backend(backend_kind: str, backend_name: Optional[str], n_qubits: int, send_ibm: bool):
    """Return (backend, backend_name_str)."""
    if backend_kind == "aer":
        try:
            from qiskit_aer import Aer, AerSimulator  # type: ignore
        except Exception as exc:
            raise RuntimeError("qiskit-aer is required for backend=aer") from exc
        try:
            bk = Aer.get_backend("aer_simulator")
        except Exception:
            bk = AerSimulator()
        return bk, getattr(bk, "name", "aer_simulator")

    # IBM Runtime
    if not send_ibm:
        raise SystemExit(
            "Refusing to use IBM backend without --send-ibm flag. "
            "This is deliberate to avoid accidental cloud runs."
        )

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
    except Exception as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for backend=ibm") from exc

    svc = QiskitRuntimeService()
    if backend_name:
        bk = svc.backend(backend_name)
    else:
        cands = [
            b
            for b in svc.backends(simulator=False)
            if getattr(b.configuration(), "num_qubits", 0) >= n_qubits
        ]
        if not cands:
            raise RuntimeError(f"No IBM backends with >= {n_qubits} qubits.")
        cands.sort(key=lambda b: getattr(b.status(), "pending_jobs", 0))
        bk = cands[0]
    return bk, getattr(bk, "name", "ibm_unknown")


def run_counts(qc, backend, backend_kind: str, shots: int, seed: Optional[int]) -> Dict[str, int]:
    """Run qc on Aer or IBM Runtime and return counts."""
    from qiskit import transpile  # type: ignore
    try:
        from qiskit_ibm_runtime import SamplerV2 as Sampler, IBMBackend  # type: ignore
    except Exception:
        Sampler = None
        IBMBackend = None

    if backend_kind == "ibm" and Sampler is not None and IBMBackend is not None and isinstance(backend, IBMBackend):
        tqc = transpile(qc, backend, seed_transpiler=seed)
        sampler = Sampler(mode=backend)
        job = sampler.run([tqc], shots=shots)
        pub_result = job.result()[0]
        counts = pub_result.join_data().get_counts()
        return {str(k): int(v) for k, v in dict(counts).items()}

    tqc = transpile(qc, backend, seed_transpiler=seed)
    try:
        job = backend.run(tqc, shots=shots, seed_simulator=seed)
    except TypeError:
        job = backend.run(tqc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return {str(k): int(v) for k, v in dict(counts).items()}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="U31-T2: wrong-key trapdoor witness with log-likelihood ratio scores."
    )
    parser.add_argument(
        "--backend",
        choices=["aer", "ibm"],
        default="aer",
        help="Backend kind.",
    )
    parser.add_argument(
        "--backend-name",
        default=None,
        help="Optional explicit backend name (e.g. ibm_torino).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Shots per class (honest / impostor).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=424242,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--send-ibm",
        action="store_true",
        help="Required when --backend ibm is used; opt-in guardrail.",
    )
    parser.add_argument(
        "--out",
        default="runs/u31_t2_wrong_key_results.json",
        help="Output JSON path.",
    )

    args = parser.parse_args(argv)

    # Honest "key" angles and wrong-key impostor angles (3-qubit product states)
    angles_ref = np.array([math.pi / 3.0, math.pi / 4.0, math.pi / 6.0], dtype=float)
    angles_alt = np.array(
        [math.pi / 3.0 + 0.25, math.pi / 4.0 - 0.2, math.pi / 6.0 + 0.15],
        dtype=float,
    )

    ref_vec = product_state_amplitudes(angles_ref)
    alt_vec = product_state_amplitudes(angles_alt)

    p_ref = build_prob_map(ref_vec)
    p_alt = build_prob_map(alt_vec)

    n_qubits = angles_ref.size

    qc_ref = build_product_circuit(angles_ref)
    qc_alt = build_product_circuit(angles_alt)

    backend, backend_name = choose_backend(
        backend_kind=args.backend,
        backend_name=args.backend_name,
        n_qubits=n_qubits,
        send_ibm=args.send_ibm,
    )

    counts_ref = run_counts(qc_ref, backend, backend_kind=args.backend, shots=args.shots, seed=args.seed)
    counts_alt = run_counts(qc_alt, backend, backend_kind=args.backend, shots=args.shots, seed=args.seed + 1)

    scores_pos = scores_from_counts_llr(counts_ref, p_ref, p_alt)
    scores_neg = scores_from_counts_llr(counts_alt, p_ref, p_alt)

    roc = compute_roc(scores_pos, scores_neg)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "u31_version": "U31-T2-wrong-key-LLR",
        "backend_kind": args.backend,
        "backend_name": backend_name,
        "shots_per_class": int(args.shots),
        "seed": int(args.seed),
        "n_qubits": int(n_qubits),
        "angles_ref": angles_ref.tolist(),
        "angles_alt": angles_alt.tolist(),
        "counts": {
            "honest": counts_ref,
            "impostor": counts_alt,
        },
        "scores_summary": {
            "honest": {
                "count": int(scores_pos.size),
                "mean": float(scores_pos.mean()),
                "std": float(scores_pos.std()),
            },
            "impostor": {
                "count": int(scores_neg.size),
                "mean": float(scores_neg.mean()),
                "std": float(scores_neg.std()),
            },
        },
        "roc": asdict(roc),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
