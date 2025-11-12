"""Unit 01 helpers: deterministic NVE bundle construction."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Optional

DEFAULT_SEED = 38192
N_QUBITS = 3
AMPLITUDE_PRECISION = 12


def _sample_unit_vector(dim: int, rng: random.Random) -> List[complex]:
    samples = [complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(dim)]
    norm = math.sqrt(sum(abs(z) ** 2 for z in samples))
    if norm == 0:
        raise ValueError("Random sampling produced zero vector; retry with different seed")
    return [z / norm for z in samples]


def _format_index(i: int, *, n_qubits: int) -> str:
    return format(i, f"0{n_qubits}b")


def _round_complex(z: complex, digits: int) -> Dict[str, float]:
    return {"real": round(z.real, digits), "imag": round(z.imag, digits)}


def _bundle_state(vector: Iterable[complex], *, n_qubits: int, precision: int) -> Dict[str, object]:
    amplitudes = []
    for idx, amp in enumerate(vector):
        amplitudes.append(
            {
                "index": _format_index(idx, n_qubits=n_qubits),
                "amplitude": _round_complex(amp, precision),
            }
        )
    return {
        "precision": precision,
        "amplitudes": amplitudes,
        "encoding": {
            "endianness": "little",
            "rail_mode": "binary",
            "qft_kernel_sign": "+",
        },
    }


def package_nve(*, seed: Optional[int] = None) -> Dict[str, object]:
    """Produce the canonical Unit-01 bundle for the provided ``seed``.

    The bundle encodes a deterministic complex amplitude vector ``psi`` with
    little-endian ordering and explicit metadata capturing the loader
    invariants required by later units.
    """

    rng = random.Random(DEFAULT_SEED if seed is None else seed)
    n_qubits = N_QUBITS
    dim = 2 ** n_qubits
    vector = _sample_unit_vector(dim, rng)
    metadata = {
        "rail_mode": "binary",
        "endianness": "little",
        "qft_kernel_sign": "+",
    }
    state = _bundle_state(vector, n_qubits=n_qubits, precision=AMPLITUDE_PRECISION)

    norm = math.sqrt(sum(abs(z) ** 2 for z in vector))
    bundle: Dict[str, object] = {
        "unit": "U01",
        "bundle_version": 1,
        "seed": DEFAULT_SEED if seed is None else seed,
        "n_qubits": n_qubits,
        "dimension": dim,
        "state": state,
        "metadata": metadata,
        "norm_l2": round(norm, 12),
    }
    return bundle


def _extract_vector(bundle: Mapping[str, object]) -> List[complex]:
    state = bundle.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("bundle missing state mapping")
    amplitudes = state.get("amplitudes")
    if not isinstance(amplitudes, list) or not amplitudes:
        raise ValueError("bundle.state.amplitudes must be a non-empty list")
    vector: List[complex] = []
    for entry in amplitudes:
        if not isinstance(entry, Mapping):
            raise ValueError("amplitude entry must be a mapping")
        amp = entry.get("amplitude")
        if not isinstance(amp, Mapping):
            raise ValueError("amplitude entry missing amplitude mapping")
        real = amp.get("real")
        imag = amp.get("imag")
        if not isinstance(real, (int, float)) or not isinstance(imag, (int, float)):
            raise ValueError("amplitude real/imag must be numbers")
        vector.append(complex(float(real), float(imag)))
    return vector


def validate_nve(bundle: Mapping[str, object], *, atol: float = 1e-9) -> None:
    """Validate the structural and norm invariants for a Unit-01 bundle."""

    if bundle.get("unit") != "U01":
        raise ValueError("bundle.unit must be 'U01'")
    register_n = bundle.get("n_qubits")
    if not isinstance(register_n, int) or register_n <= 0:
        raise ValueError("n_qubits must be a positive integer")
    if bundle.get("dimension") != 2 ** register_n:
        raise ValueError("dimension must equal 2**n_qubits")

    state = bundle.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("bundle.state must be a mapping")
    encoding = state.get("encoding")
    if not isinstance(encoding, Mapping):
        raise ValueError("state.encoding must be a mapping")
    expected = {"endianness": "little", "rail_mode": "binary", "qft_kernel_sign": "+"}
    for key, value in expected.items():
        if encoding.get(key) != value:
            raise ValueError(f"state.encoding.{key} must be '{value}'")

    metadata = bundle.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("bundle.metadata must be a mapping")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata.{key} must be '{value}'")

    precision = state.get("precision")
    if precision != AMPLITUDE_PRECISION:
        raise ValueError(f"state.precision must be {AMPLITUDE_PRECISION}")

    vector = _extract_vector(bundle)
    if len(vector) != bundle.get("dimension"):
        raise ValueError("amplitude vector length mismatch")

    norm = math.sqrt(sum(abs(z) ** 2 for z in vector))
    if abs(norm - 1.0) > atol:
        raise ValueError(f"psi L2 norm {norm} violates tolerance {atol}")

    recorded_norm = bundle.get("norm_l2")
    if not isinstance(recorded_norm, (int, float)):
        raise ValueError("norm_l2 must be numeric")
    if abs(float(recorded_norm) - norm) > max(atol, 1e-12):
        raise ValueError("recorded norm_l2 does not match actual norm")
