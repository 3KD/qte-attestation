from __future__ import annotations

import copy
import math
from typing import Callable

import pytest

from qte_attestation.unit01 import package_nve, validate_nve


def mutate(bundle_factory: Callable[[], dict], mutator: Callable[[dict], None]) -> dict:
    """Return a deep-copied bundle with ``mutator`` applied."""

    bundle = copy.deepcopy(bundle_factory())
    mutator(bundle)
    return bundle


def test_package_nve_deterministic():
    assert package_nve(seed=1234) == package_nve(seed=1234)


def test_package_nve_default_seed_is_stable():
    assert package_nve() == package_nve()


def test_package_nve_norm_and_structure():
    bundle = package_nve(seed=42)
    validate_nve(bundle)

    amplitudes = [entry["amplitude"] for entry in bundle["state"]["amplitudes"]]
    norm = math.sqrt(
        sum(float(amp["real"]) ** 2 + float(amp["imag"]) ** 2 for amp in amplitudes)
    )
    assert math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9)
    expected_meta = {
        "endianness": "little",
        "rail_mode": "binary",
        "qft_kernel_sign": "+",
    }
    assert bundle["metadata"] == expected_meta
    assert bundle["state"]["encoding"] == expected_meta


def test_validate_nve_rejects_wrong_unit():
    bad_bundle = mutate(package_nve, lambda bundle: bundle.__setitem__("unit", "U00"))

    with pytest.raises(ValueError):
        validate_nve(bad_bundle)


def test_validate_nve_rejects_encoding_mismatch():
    bad_bundle = mutate(
        package_nve, lambda bundle: bundle["state"]["encoding"].__setitem__("rail_mode", "ternary")
    )

    with pytest.raises(ValueError):
        validate_nve(bad_bundle)


def test_validate_nve_rejects_metadata_mismatch():
    bad_bundle = mutate(
        package_nve, lambda bundle: bundle["metadata"].__setitem__("endianness", "big")
    )

    with pytest.raises(ValueError):
        validate_nve(bad_bundle)


def test_validate_nve_rejects_norm_mismatch():
    bad_bundle = mutate(package_nve, lambda bundle: bundle.__setitem__("norm_l2", 0.5))

    with pytest.raises(ValueError):
        validate_nve(bad_bundle)


def test_validate_nve_rejects_amplitude_length_mismatch():
    bad_bundle = mutate(
        package_nve, lambda bundle: bundle["state"]["amplitudes"].pop()
    )

    with pytest.raises(ValueError):
        validate_nve(bad_bundle)
