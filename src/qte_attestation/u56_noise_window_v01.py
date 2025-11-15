# file: src/qte_attestation/u56_noise_window_v01.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService  # uses IBM Quantum Cloud env / saved account


Counts = Mapping[str, int]


@dataclass
class NoisePointResult:
    lambda_: float
    backend_kind: str
    shots: int
    auc: float
    tpr_at_1pct_fpr: float
    roc: List[Tuple[float, float, float]]  # (threshold, fpr, tpr)


def _build_depolarizing_noise_model(lambda_: float) -> NoiseModel:
    """
    Simple global depolarizing NoiseModel for Aer.

    lambda_ in [0,1]. lambda_=0 returns an "empty" model (effectively noiseless).
    """
    if lambda_ < 0 or lambda_ > 1:
        raise ValueError(f"lambda must be in [0,1], got {lambda_}")
    noise_model = NoiseModel()
    if lambda_ == 0.0:
        return noise_model

    # 1- and 2-qubit depolarizing errors on common gate sets
    error_1q = depolarizing_error(lambda_, 1)
    error_2q = depolarizing_error(lambda_, 2)

    one_qubit_gates = ["id", "rz", "sx", "x", "ry", "rx", "u1", "u2", "u3"]
    two_qubit_gates = ["cx", "cz"]

    noise_model.add_all_qubit_quantum_error(error_1q, one_qubit_gates)
    noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)
    return noise_model


def _get_aer_backend(lambda_: float, seed: int | None = None) -> AerSimulator:
    noise_model = _build_depolarizing_noise_model(lambda_)
    simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    return simulator


def _get_ibm_backend(backend_name: str | None = None):
    """
    IBM Quantum Cloud backend via QiskitRuntimeService.

    Expects your IBM Runtime config to be set via environment variables or saved
    account, e.g. QISKIT_IBM_CHANNEL, QISKIT_IBM_TOKEN, QISKIT_IBM_INSTANCE.
    """
    service = QiskitRuntimeService()  # reads config from env or ~/.qiskit/qiskit-ibm.json
    if backend_name is None:
        raise ValueError("backend_name must be provided when using IBM backends")
    backend = service.backend(backend_name)
    return backend


def _run_circuits_on_backend(
    circuits: Sequence[QuantumCircuit],
    backend_kind: str,
    shots: int,
    lambda_: float,
    ibm_backend_name: str | None = None,
    seed: int | None = None,
) -> List[Counts]:
    """
    Run the provided circuits on either Aer (with depolarizing noise) or an IBM device.
    """
    backend_kind = backend_kind.lower()
    if backend_kind == "aer":
        backend = _get_aer_backend(lambda_, seed=seed)
    elif backend_kind == "ibm":
        # Real hardware noise isn't parameterized by lambda; enforce lambda=0 here.
        if lambda_ not in (0.0, 0):
            raise ValueError("lambda must be 0 when running on real IBM hardware")
        backend = _get_ibm_backend(backend_name=ibm_backend_name)
    else:
        raise ValueError(f"unknown backend_kind {backend_kind!r}, expected 'aer' or 'ibm'")

    t_circuits = transpile(circuits, backend=backend, optimization_level=1)
    job = backend.run(t_circuits, shots=shots)
    result = job.result()
    # Determine number of experiments from the Result, not from the Circuit object,
    # because a single QuantumCircuit can still have len()>1 (instructions).
    try:
        num_experiments = len(result.results)
    except Exception:
        num_experiments = 1
    if num_experiments == 1:
        counts_list: List[Counts] = [result.get_counts()]
    else:
        counts_list = [result.get_counts(i) for i in range(num_experiments)]
    return counts_list


def _aggregate_counts(counts_list: Sequence[Counts]) -> Dict[str, int]:
    agg: Dict[str, int] = {}
    for counts in counts_list:
        for bitstring, n in counts.items():
            agg[bitstring] = agg.get(bitstring, 0) + int(n)
    return agg


def _compute_llr_scores(
    keyed_agg: Counts,
    forger_agg: Counts,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Build per-bitstring log-likelihood ratio scores:

        score(b) = log( (p_key(b) + eps) / (p_forg(b) + eps) )

    using aggregate keyed/forger histograms.
    """
    total_keyed = sum(int(v) for v in keyed_agg.values())
    total_forg = sum(int(v) for v in forger_agg.values())
    if total_keyed == 0 or total_forg == 0:
        raise ValueError("Need non-zero counts for both keyed and forger to compute LLR.")

    all_bits = set(keyed_agg.keys()) | set(forger_agg.keys())

    import math

    scores: Dict[str, float] = {}
    for b in all_bits:
        p_k = keyed_agg.get(b, 0) / total_keyed
        p_f = forger_agg.get(b, 0) / total_forg
        scores[b] = math.log((p_k + eps) / (p_f + eps))
    return scores


def _build_labeled_scores(
    keyed_agg: Counts,
    forger_agg: Counts,
    scores: Mapping[str, float],
) -> Tuple[List[float], List[int]]:
    """
    Expand aggregate histograms into (score, label) samples.

    label = 1 for keyed, 0 for forger.
    """
    xs: List[float] = []
    ys: List[int] = []
    for b, n in keyed_agg.items():
        s = scores[b]
        xs.extend([s] * int(n))
        ys.extend([1] * int(n))
    for b, n in forger_agg.items():
        s = scores[b]
        xs.extend([s] * int(n))
        ys.extend([0] * int(n))
    return xs, ys


def _compute_roc_auc(
    scores: Sequence[float],
    labels: Sequence[int],
) -> Tuple[float, float, List[Tuple[float, float, float]]]:
    """
    Compute ROC, AUC, and TPR at FPR <= 1% via a standard threshold sweep.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have same length")
    n = len(scores)
    if n == 0:
        raise ValueError("no samples for ROC/AUC")

    # Sort by score descending
    idx = sorted(range(n), key=lambda i: scores[i], reverse=True)
    sorted_scores = [scores[i] for i in idx]
    sorted_labels = [labels[i] for i in idx]

    # Count positives/negatives
    P = sum(sorted_labels)
    N = n - P
    if P == 0 or N == 0:
        raise ValueError("need both positive and negative samples for ROC/AUC")

    roc: List[Tuple[float, float, float]] = []
    tp = fp = 0
    best_tpr_at_1pct = 0.0

    for i, (s, y) in enumerate(zip(sorted_scores, sorted_labels)):
        if y == 1:
            tp += 1
        else:
            fp += 1

        # Record at score changes
        next_score = sorted_scores[i + 1] if i + 1 < n else None
        if next_score != s:
            tpr = tp / P
            fpr = fp / N
            roc.append((s, fpr, tpr))
            if fpr <= 0.01:
                best_tpr_at_1pct = max(best_tpr_at_1pct, tpr)

    # AUC via trapezoidal rule over FPR
    roc_sorted_by_fpr = sorted(roc, key=lambda t: t[1])
    auc = 0.0
    prev_fpr, prev_tpr = 0.0, 0.0
    for _, fpr, tpr in roc_sorted_by_fpr:
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr

    return auc, best_tpr_at_1pct, roc_sorted_by_fpr


def run_u56_noise_window(
    keyed_circuits: Sequence[QuantumCircuit],
    forger_circuits: Sequence[QuantumCircuit],
    lambdas: Sequence[float],
    shots: int,
    backend_kind: str = "aer",
    ibm_backend_name: str | None = None,
    seed: int | None = 1234,
) -> List[NoisePointResult]:
    """
    Run a U56-v0.1 style noise-window scan for a fixed U58-style card.

    Parameters
    ----------
    keyed_circuits:
        Circuits encoding the keyed secret.
    forger_circuits:
        Corresponding forger circuits (same layout/measurement).
    lambdas:
        Iterable of depolarizing noise rates in [0,1]. For backend_kind=='ibm',
        all lambdas must be 0.
    shots:
        Shot count per circuit.
    backend_kind:
        'aer' for local simulation with a depolarizing NoiseModel;
        'ibm' for IBM Quantum Cloud backends via QiskitRuntimeService.
    ibm_backend_name:
        Name of the IBM backend when backend_kind=='ibm'.
    seed:
        Optional simulator seed for reproducibility (Aer only).

    Returns
    -------
    List[NoisePointResult]
        One entry per lambda in `lambdas`, containing AUC, TPR@1% FPR,
        and the full ROC curve for that configuration.
    """
    if len(keyed_circuits) != len(forger_circuits):
        raise ValueError("keyed_circuits and forger_circuits must have same length")

    results: List[NoisePointResult] = []

    for lambda_ in lambdas:
        # 1) Run keyed + forger circuits for this (lambda, backend, shots)
        keyed_counts_list = _run_circuits_on_backend(
            keyed_circuits,
            backend_kind=backend_kind,
            shots=shots,
            lambda_=lambda_,
            ibm_backend_name=ibm_backend_name,
            seed=seed,
        )
        forger_counts_list = _run_circuits_on_backend(
            forger_circuits,
            backend_kind=backend_kind,
            shots=shots,
            lambda_=lambda_,
            ibm_backend_name=ibm_backend_name,
            seed=seed,
        )

        # 2) Aggregate counts over circuit pairs
        keyed_agg = _aggregate_counts(keyed_counts_list)
        forger_agg = _aggregate_counts(forger_counts_list)

        # 3) Build LLR witness and compute ROC/AUC + TPR@1% FPR
        scores = _compute_llr_scores(keyed_agg, forger_agg)
        xs, ys = _build_labeled_scores(keyed_agg, forger_agg, scores)
        auc, tpr_at_1pct, roc = _compute_roc_auc(xs, ys)

        results.append(
            NoisePointResult(
                lambda_=lambda_,
                backend_kind=backend_kind,
                shots=shots,
                auc=auc,
                tpr_at_1pct_fpr=tpr_at_1pct,
                roc=roc,
            )
        )

    return results

