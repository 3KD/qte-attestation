from __future__ import annotations

import json

from runner.provenance import (
    attach_provenance,
    canonical_payload,
    compute_hashes,
    stable_dumps,
    verify_provenance,
)


def test_stable_dumps_is_canonical():
    payload_a = {"b": 1, "a": [3, 2, 1]}
    payload_b = {"a": [3, 2, 1], "b": 1}

    dumped_a = stable_dumps(payload_a)
    dumped_b = stable_dumps(payload_b)

    assert dumped_a == dumped_b
    # Canonical form should not include whitespace
    assert " " not in dumped_a


def test_attach_and_verify_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    payload = {"alpha": 7, "nested": {"beta": 3}}
    path.write_text(json.dumps(payload, indent=2))

    monkeypatch.setenv("PROV_HMAC_KEY", "shared-secret")

    sha = attach_provenance(path, key_id="test-key")
    stored = json.loads(path.read_text())

    assert stored["provenance"]["sha256"] == sha
    canon = canonical_payload(stored)
    recomputed_sha, recomputed_hmac = compute_hashes(canon, "shared-secret")
    assert recomputed_sha == sha
    assert stored["provenance"]["hmac"] == recomputed_hmac

    verify = verify_provenance(path)
    assert verify["sha256_matches"] is True
    assert verify["hmac_matches"] is True

    monkeypatch.delenv("PROV_HMAC_KEY", raising=False)


def test_verify_detects_mutation(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    payload = {"alpha": 7, "nested": {"beta": 3}}
    path.write_text(json.dumps(payload, indent=2))

    attach_provenance(path)

    mutated = json.loads(path.read_text())
    mutated["nested"]["beta"] = 4
    path.write_text(json.dumps(mutated, indent=2))

    result = verify_provenance(path)
    assert result["sha256_matches"] is False
    assert result["sha256_claimed"] != result["sha256_actual"]
