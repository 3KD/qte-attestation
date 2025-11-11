#!/usr/bin/env python3
"""
Round-trip determinism checker for Unit-1 packaging.

Modes:
  1) --module MOD --func package_nve --seed 123
     -> import MOD, call func(seed=...), twice, compare canonical dumps + hashes.

  2) --file bundle.json
     -> load file twice (simulated), canonicalize, confirm stable hash.

Returns exit code 0 on success; prints details.
"""
import importlib, json, sys, time
from pathlib import Path
from provenance import stable_dumps, sha256_hex

def _ok(msg): print("[u01-rt]", msg)

def rt_from_module(mod_name, func_name, seed):
    m = importlib.import_module(mod_name)
    f = getattr(m, func_name)
    b1 = f(seed=seed) if "seed" in f.__code__.co_varnames else f()
    time.sleep(0.01)
    b2 = f(seed=seed) if "seed" in f.__code__.co_varnames else f()
    d1, d2 = stable_dumps(b1), stable_dumps(b2)
    h1, h2 = sha256_hex(d1), sha256_hex(d2)
    same = (d1 == d2 and h1 == h2)
    _ok(f"module={mod_name}.{func_name} seed={seed}  canon_equal={same}  h1={h1}  h2={h2}")
    return 0 if same else 2

def rt_from_file(path):
    p = Path(path)
    data = json.loads(p.read_text())
    d1, d2 = stable_dumps(data), stable_dumps(json.loads(p.read_text()))
    h1, h2 = sha256_hex(d1), sha256_hex(d2)
    same = (d1 == d2 and h1 == h2)
    _ok(f"file={p.name} canon_equal={same} h={h1}")
    return 0 if same else 2

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", help="python module path (e.g., program5_branch_series.unit1)")
    ap.add_argument("--func", default="package_nve", help="callable that returns a Unit-1 bundle")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--file", help="Path to existing Unit-1 JSON bundle")
    args = ap.parse_args()

    if args.module:
        sys.exit(rt_from_module(args.module, args.func, args.seed))
    elif args.file:
        sys.exit(rt_from_file(args.file))
    else:
        print("Usage: u01-rt --module MOD [--func package_nve] [--seed 123]  |  u01-rt --file bundle.json")
        sys.exit(1)

if __name__ == "__main__":
    main()
