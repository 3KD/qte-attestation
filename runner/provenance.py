#!/usr/bin/env python3
import json, hashlib, hmac, base64, os, subprocess, time, sys
from typing import Any, Tuple

# Canonical float/complex formatting (stable across runs)
_FLOAT_FMT = "%.17g"

def _canon_val(x: Any):
    import numpy as _np
    if isinstance(x, float):
        # stringify floats to lock repr
        return {"__float__": _FLOAT_FMT % x}
    if isinstance(x, complex):
        return {"__complex__":[_FLOAT_FMT % x.real, _FLOAT_FMT % x.imag]}
    if isinstance(x, (_np.floating,)):
        return {"__float__": _FLOAT_FMT % float(x)}
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.complexfloating,)):
        z = complex(x)
        return {"__complex__":[_FLOAT_FMT % z.real, _FLOAT_FMT % z.imag]}
    if isinstance(x, (list, tuple)):
        return [_canon_val(v) for v in x]
    if isinstance(x, dict):
        return {k:_canon_val(x[k]) for k in sorted(x.keys())}
    return x

def stable_dumps(obj: Any) -> str:
    """Deterministic JSON string for hashing/signature."""
    canon = _canon_val(obj)
    return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hmac_sha256_b64(s: str, key: bytes) -> str:
    return base64.b64encode(hmac.new(key, s.encode("utf-8"), hashlib.sha256).digest()).decode("ascii")

def _openssl_sign(det_json: str, key_path: str) -> Tuple[str,str]:
    """
    Uses OpenSSL to produce a detached signature over the canonical JSON bytes.
    Returns (sig_algo, sig_b64). Works with RSA/EC/Ed25519 private keys in PEM.
    """
    # Write temp file to sign
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(det_json.encode("utf-8"))
        tmp = tf.name
    try:
        # Try generic pkeyutl (supports Ed25519/Ed448/RSA/ECDSA depending on key)
        sig = subprocess.check_output(
            ["openssl","pkeyutl","-sign","-inkey",key_path,"-pkeyopt","digest:sha256","-in",tmp],
            stderr=subprocess.DEVNULL
        )
        # Probe key type for labeling
        pub = subprocess.check_output(["openssl","pkey","-in",key_path,"-pubout"], stderr=subprocess.DEVNULL)
        algo = "openssl-pkeyutl-sha256"
        sig_b64 = base64.b64encode(sig).decode("ascii")
        pub_pem = pub.decode("utf-8")
        return (f"{algo}|{pub_pem.strip().splitlines()[0]}", sig_b64)
    finally:
        try: os.unlink(tmp)
        except: pass

def attach_provenance(bundle: dict, *, key_id: str=None, hmac_key_env="PROV_HMAC_KEY",
                      openssl_key_env="PROV_OPENSSL_KEY") -> dict:
    """
    Adds/overwrites bundle['provenance'] with canonical hash and optional signature.
    - Always includes 'sha256' over canonical JSON.
    - If PROV_HMAC_KEY is set, also includes 'hmac_sha256' with 'hmac_key_id'.
    - If PROV_OPENSSL_KEY is set (path to PEM private key), also includes 'sig' and 'sig_algo'.
    """
    det = stable_dumps(bundle)
    prov = {
        "canonicalization": {"float_fmt": _FLOAT_FMT, "complex_tag": "__complex__", "float_tag": "__float__"},
        "sha256": sha256_hex(det),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "provenance.py/v1"
    }
    hkey = os.environ.get(hmac_key_env)
    if hkey:
        prov["hmac_sha256"] = hmac_sha256_b64(det, hkey.encode("utf-8"))
        if key_id: prov["hmac_key_id"] = key_id
    pkey = os.environ.get(openssl_key_env)
    if pkey and os.path.exists(pkey):
        algo, sig = _openssl_sign(det, pkey)
        prov["sig_algo"] = algo
        prov["sig_b64"] = sig
    bundle["provenance"] = prov
    return bundle

def verify_provenance(bundle: dict) -> dict:
    """Recompute sha256 and compare with embedded; returns dict with booleans."""
    det = stable_dumps(bundle)
    out = {"sha256_matches": False, "hmac_matches": None}
    prov = bundle.get("provenance") or {}
    out["sha256_actual"] = sha256_hex(det)
    out["sha256_claimed"] = prov.get("sha256")
    out["sha256_matches"] = (out["sha256_actual"] == out["sha256_claimed"])
    # HMAC check (if present and env provides key)
    if "hmac_sha256" in prov:
        key = os.environ.get("PROV_HMAC_KEY")
        if key:
            out["hmac_matches"] = (hmac_sha256_b64(det, key.encode("utf-8")) == prov["hmac_sha256"])
        else:
            out["hmac_matches"] = False
    return out

def cli():
    import argparse, pathlib
    ap = argparse.ArgumentParser(description="Provenance: canonical hash / attach / verify")
    ap.add_argument("path", help="JSON file to process (input; will be updated in-place if --attach)")
    ap.add_argument("--attach", action="store_true", help="Attach provenance into the JSON (in-place)")
    ap.add_argument("--key-id", default=None, help="Label for HMAC key (metadata only)")
    ap.add_argument("--verify", action="store_true", help="Verify embedded provenance")
    args = ap.parse_args()

    p = pathlib.Path(args.path)
    data = json.loads(p.read_text())
    if args.attach:
        attach_provenance(data, key_id=args.key_id)
        p.write_text(json.dumps(data, indent=2))
        print("[prov] attached:", p)
    det = stable_dumps(data)
    print("sha256:", sha256_hex(det))
    if args.verify:
        res = verify_provenance(data)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    cli()
