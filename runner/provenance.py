#!/usr/bin/env python3
"""Utility helpers for canonical JSON provenance handling.

This module powers both the provenance CLI (``python runner/provenance.py``)
and lightweight helpers that other tooling can import.  Canonicalisation is
performed with ``sort_keys=True`` and compact separators so that hashing is
stable across platforms.  The hash domain explicitly excludes the top-level
``provenance`` key allowing refreshes without perturbing the canonical payload.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import json
import os
import pathlib
import sys
from typing import Dict, Mapping, Optional, Tuple

CANONICAL_SEPARATORS = (",", ":")
CANONICAL_SORT_KEYS = True
CANONICAL_EXCLUDE = ("provenance",)


def stable_dumps(data: object) -> str:
    """Return the canonical JSON representation for *data*."""

    return json.dumps(
        data,
        sort_keys=CANONICAL_SORT_KEYS,
        separators=CANONICAL_SEPARATORS,
        ensure_ascii=False,
    )


def sha256_hex(payload: str) -> str:
    """Return the SHA-256 hex digest for ``payload``."""

    return hashlib.sha256(payload.encode()).hexdigest()


def hmac_sha256_hex(payload: str, key: str) -> str:
    """Return the HMAC-SHA256 hex digest for ``payload`` using ``key``."""

    return hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()


def canonical_payload(document: Mapping) -> str:
    """Serialise *document* into its canonical form excluding provenance."""

    payload = {k: v for k, v in document.items() if k not in CANONICAL_EXCLUDE}
    return stable_dumps(payload)


def compute_hashes(canon: str, hmac_key: Optional[str]) -> Tuple[str, Optional[str]]:
    sha = sha256_hex(canon)
    hm = hmac_sha256_hex(canon, hmac_key) if hmac_key else None
    return sha, hm


def attach_provenance(
    path: pathlib.Path,
    *,
    key_id: Optional[str] = None,
    hmac_env: str = "PROV_HMAC_KEY",
) -> str:
    """Attach or refresh provenance for ``path`` and return the SHA-256 claim."""

    document = json.loads(path.read_text())
    canon = canonical_payload(document)
    sha, hm = compute_hashes(canon, os.getenv(hmac_env))

    provenance: Dict[str, object] = dict(document.get("provenance", {}))
    provenance.update(
        {
            "sha256": sha,
            **({"hmac": hm} if hm else {}),
            **({"key_id": key_id} if key_id else {}),
            "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "canon": {
                "separators": list(CANONICAL_SEPARATORS),
                "sort_keys": CANONICAL_SORT_KEYS,
                "exclude": list(CANONICAL_EXCLUDE),
            },
        }
    )
    document["provenance"] = provenance
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"[prov] attached: {path}")
    print(f"sha256: {sha}")
    return sha


def verify_provenance(path: pathlib.Path, *, hmac_env: str = "PROV_HMAC_KEY") -> Dict[str, object]:
    """Verify that the stored provenance claims match the canonical payload."""

    document = json.loads(path.read_text())
    claim = (document.get("provenance") or {}).get("sha256")
    claim_hmac = (document.get("provenance") or {}).get("hmac")
    canon = canonical_payload(document)
    sha, hm = compute_hashes(canon, os.getenv(hmac_env))
    result = {
        "sha256_matches": (claim == sha),
        "hmac_matches": (None if claim_hmac is None else claim_hmac == hm),
        "sha256_actual": sha,
        "sha256_claimed": claim,
    }
    print(f"sha256: {sha}")
    print(json.dumps(result, indent=2))
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path to JSON artifact")
    p.add_argument("--attach", action="store_true", help="attach/refresh provenance and write back")
    p.add_argument("--verify", action="store_true", help="verify against stored provenance claims")
    p.add_argument("--key-id", default=None, help="optional key identifier to record in provenance")
    p.add_argument("--hmac-env", default="PROV_HMAC_KEY", help="env var name that holds HMAC key")
    args = p.parse_args()

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"[prov] ERROR: not found: {path}", file=sys.stderr)
        sys.exit(2)

    if args.attach:
        attach_provenance(path, key_id=args.key_id, hmac_env=args.hmac_env)
    # Default to verify if not attaching (or do both if both flags set)
    if args.verify or not args.attach:
        verify_provenance(path, hmac_env=args.hmac_env)

if __name__ == "__main__":
    main()
