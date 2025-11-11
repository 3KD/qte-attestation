from __future__ import annotations
import argparse, sys, json
from dataclasses import dataclass
from qte_attestation.metrics.cp import cp_lower, cp_upper

@dataclass
class Targets:
    min_accept: float = 0.99
    max_forge: float  = 0.01
    alpha: float      = 0.05

def verify_counts(args: argparse.Namespace) -> int:
    t = Targets(args.min_accept, args.max_forge, args.alpha)
    acc_lo = cp_lower(args.keyed_succ, args.keyed_trials, t.alpha)
    forg_hi = cp_upper(args.forg_succ, args.forg_trials, t.alpha)
    ok_keyed = acc_lo >= t.min_accept
    ok_forg  = forg_hi <= t.max_forge
    print(json.dumps({
        "alpha": t.alpha,
        "targets": {"min_accept": t.min_accept, "max_forge": t.max_forge},
        "keyed":   {"succ": args.keyed_succ, "trials": args.keyed_trials, "cp_lower": round(acc_lo, 6), "pass": ok_keyed},
        "forger":  {"succ": args.forg_succ,  "trials": args.forg_trials,  "cp_upper": round(forg_hi, 6), "pass": ok_forg},
        "PASS": bool(ok_keyed and ok_forg)
    }, indent=2))
    return 0 if (ok_keyed and ok_forg) else 1

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="qte-attestation")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="Verify keyed/forger pass rates with CP intervals.")
    v.add_argument("--keyed-succ",   type=int, required=True)
    v.add_argument("--keyed-trials", type=int, required=True)
    v.add_argument("--forg-succ",    type=int, required=True)
    v.add_argument("--forg-trials",  type=int, required=True)
    v.add_argument("--alpha",        type=float, default=0.05)
    v.add_argument("--min-accept",   type=float, default=0.99)
    v.add_argument("--max-forge",    type=float, default=0.01)
    v.set_defaults(func=verify_counts)
    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
