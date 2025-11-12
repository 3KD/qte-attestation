# Test Suite Overview

This directory contains the regression coverage for the provenance helper and the Unit 01 bundle contracts.

* `test_provenance.py` exercises attach/verify hashing and optional HMAC coverage using the canonical JSON rules from `runner/provenance.py`.
* `test_unit01.py` validates deterministic bundle generation, metadata constraints, and the L2 norm checks implemented in `src/qte_attestation/unit01.py`.

Run the full suite from the repository root with:

```bash
pytest
```

If this folder is absent in a local checkout, ensure you have synced to a branch that includes the latest work (e.g., `git fetch origin && git switch work && git pull`).
