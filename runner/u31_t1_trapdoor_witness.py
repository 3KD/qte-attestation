from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


def _ensure_unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(vec)
    if nrm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return vec / nrm


def _build_prob_map(vec: np.ndarray) -> Dict[str, float]:
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


def _scores_from_counts(counts: Dict[str, int], prob_map: Dict[str, float]) -> np.ndarray:
    scores: List[float] = []
    eps = 1e-15
    for bitstring, c in counts.items():
        p = prob_map.get(bitstring, eps)
        s = math.log(p + eps)
        scores.extend([s] * int(c))
    return np.asarray(scores, dtype=float)


@dataclass
class RocResult:
    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc: float
    tpr_at_1pct_fpr: float


def _compute_roc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> RocResult:
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


def _build_circuit_from_vector(vec: np.ndarray):
    """Aer path: prepare |psi> from vec with initialize()."""
    from qiskit import QuantumCircuit  # type: ignore

    vec = _ensure_unit(vec)
    n = int(round(math.log2(vec.size)))
    if 1 << n != vec.size:
        raise ValueError(f"Vector length {vec.size} is not a power of two.")
    qc = QuantumCircuit(n, n)
    qc.initialize(vec, range(n))
    qc.measure(range(n), range(n))
    return qc


def _build_circuit_product3():
    """IBM path: hard-coded 3-qubit product state via RY gates.

    |psi> = (cos t0 |0> + sin t0 |1>) ⊗ (cos t1 |0> + sin t1 |1>) ⊗ (cos t2 |0> + sin t2 |1>)
    Angles must match the generator used for the .npy ref state.
    """
    from qiskit import QuantumCircuit  # type: ignore

    n = 3
    qc = QuantumCircuit(n, n)
    t0 = math.pi / 3.0
    t1 = math.pi / 4.0
    t2 = math.pi / 6.0
    qc.ry(2 * t0, 0)
    qc.ry(2 * t1, 1)
    qc.ry(2 * t2, 2)
    qc.measure(range(n), range(n))
    return qc


def _choose_backend(backend_kind: str, n_qubits: int, backend_name: Optional[str], send_ibm: bool):
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


def _run_counts(qc, backend, backend_kind: str, shots: int, seed: Optional[int]) -> Dict[str, int]:
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
    p = argparse.ArgumentParser(description="U31-T1 trapdoor witness ROC harness (Aer / IBM).")
    p.add_argument(
        "--ref-psi",
        required=True,
        help=".npy file with complex amplitudes for the honest key state.",
    )
    p.add_argument(
        "--alt-psi",
        required=False,
        help=".npy file with amplitudes for the impostor state. "
             "If omitted, a permuted/rephased version of ref-psi is used.",
    )
    p.add_argument(
        "--backend",
        choices=["aer", "ibm"],
        default="aer",
        help="Backend kind.",
    )
    p.add_argument(
        "--backend-name",
        default=None,
        help="Optional explicit backend name (e.g. ibm_torino).",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Shots per class (honest / impostor).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=424242,
        help="Base RNG seed.",
    )
    p.add_argument(
        "--send-ibm",
        action="store_true",
        help="Required when --backend ibm is used; opt-in guardrail.",
    )
    p.add_argument(
        "--out",
        default="runs/u31_t1_results.json",
        help="Output JSON path.",
    )

    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    ref_vec = np.load(args.ref_psi)
    ref_vec = np.asarray(ref_vec, dtype=complex).reshape(-1)

    if args.alt_psi:
        alt_vec = np.load(args.alt_psi)
        alt_vec = np.asarray(alt_vec, dtype=complex).reshape(-1)
        if alt_vec.size != ref_vec.size:
            raise ValueError("alt-psi must have the same length as ref-psi.")
    else:
        perm = rng.permutation(ref_vec.size)
        phases = np.exp(1j * 2 * np.pi * rng.random(ref_vec.size))
        alt_vec = ref_vec[perm] * phases

    prob_map = _build_prob_map(ref_vec)

    n_qubits = int(round(math.log2(ref_vec.size)))
    if 1 << n_qubits != ref_vec.size:
        raise ValueError("ref-psi length is not a power of two.")

    # Build circuits
    if args.backend == "aer":
        qc_ref = _build_circuit_from_vector(ref_vec)
        qc_alt = _build_circuit_from_vector(alt_vec)
    else:
        if n_qubits != 3:
            raise SystemExit("IBM path currently implemented only for 3-qubit product state.")
        qc_ref = _build_circuit_product3()
        qc_alt = _build_circuit_product3()

    backend, backend_name = _choose_backend(
        backend_kind=args.backend,
        n_qubits=n_qubits,
        backend_name=args.backend_name,
        send_ibm=args.send_ibm,
    )

    counts_ref = _run_counts(qc_ref, backend, backend_kind=args.backend, shots=args.shots, seed=args.seed)
    counts_alt = _run_counts(qc_alt, backend, backend_kind=args.backend, shots=args.shots, seed=args.seed + 1)

    scores_pos = _scores_from_counts(counts_ref, prob_map)
    scores_neg = _scores_from_counts(counts_alt, prob_map)

    roc = _compute_roc(scores_pos, scores_neg)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "u31_version": "U31-T1",
        "backend_kind": args.backend,
        "backend_name": backend_name,
        "shots_per_class": int(args.shots),
        "seed": int(args.seed),
        "n_qubits": int(n_qubits),
        "ref_psi": os.path.abspath(args.ref_psi),
        "alt_psi": os.path.abspath(args.alt_psi) if args.alt_psi else None,
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
