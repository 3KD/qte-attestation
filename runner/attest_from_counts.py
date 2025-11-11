#!/usr/bin/env python3
import sys, json, pathlib, subprocess, math, argparse
from collections import Counter
from datetime import datetime
from scipy.stats import beta  # exact Clopper–Pearson

RUNS = pathlib.Path("/Users/erik/Documents/qte-attestation/runs")

def cp_lower(x, n, alpha=0.05):
    return 0.0 if x==0 else beta.ppf(alpha/2, x, n-x+1)

def cp_upper(x, n, alpha=0.05):
    return 1.0 if x==n else beta.ppf(1-alpha/2, x+1, n-x)

def hamming_radius_succ(counts, r):
    return sum(v for b,v in counts.items() if b.count("1") <= r)

def load_counts(jsonl_path: pathlib.Path):
    recs = [json.loads(line) for line in open(jsonl_path)]
    assert len(recs)==2 and {recs[0]["tag"],recs[1]["tag"]}=={"keyed","forger"}
    n     = recs[0]["n_qubits"]; shots = int(recs[0]["shots"])
    back  = recs[0]["backend"];  mode  = recs[0]["mode"]
    return n, shots, back, mode, recs[0]["counts"], recs[1]["counts"]

def choose_radius(n, shots, kc, fc):
    chosen = None
    rows   = []
    for r in range(n+1):
        ks = hamming_radius_succ(kc, r)
        fs = hamming_radius_succ(fc, r)
        lo = cp_lower(ks, shots)
        hi = cp_upper(fs, shots)
        ok = (lo >= 0.99) and (hi <= 0.01)
        rows.append((r, ks, fs, ks/shots, fs/shots, lo, hi, ok))
        if ok and chosen is None:
            chosen = r
    return chosen, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="path to *_submit_counts.jsonl (default = newest in runs/)")
    args = ap.parse_args()
    if args.jsonl:
        jsonl = pathlib.Path(args.jsonl)
    else:
        jsonls = sorted(RUNS.glob("*_submit_counts.jsonl"))
        if not jsonls: print("No *_submit_counts.jsonl found.", file=sys.stderr); sys.exit(2)
        jsonl = jsonls[-1]

    n, shots, backend, mode, kc, fc = load_counts(jsonl)
    r, rows = choose_radius(n, shots, kc, fc)

    print(f"\nAttestation sweep on {backend} ({mode}), n={n}, shots={shots}")
    print("r  TPR      FPR      keyed  forger   CP_lo(keyed)  CP_hi(forg)  PASS")
    for (rr, ks, fs, tpr, fpr, lo, hi, ok) in rows:
        print(f"{rr:<2} {tpr:0.4f}  {fpr:0.4f}   {ks:>4}/{shots} {fs:>4}/{shots}   {lo:0.5f}       {hi:0.5f}    {'PASS' if ok else 'fail'}")

    if r is None:
        print("\nNo radius meets the prereg thresholds with 95% CP. Exiting nonzero.", file=sys.stderr)
        sys.exit(1)

    # Verify with chosen radius
    ksucc = hamming_radius_succ(kc, r)
    fsucc = hamming_radius_succ(fc, r)
    exe   = shutil.which("qte-attestation") if (shutil:=__import__("shutil")) else None
    exe   = exe or str(pathlib.Path("/Users/erik/Documents/qte-attestation/.venv/bin/qte-attestation"))
    cmd   = [exe,"verify",
             "--keyed-succ",str(ksucc),"--keyed-trials",str(shots),
             "--forg-succ", str(fsucc), "--forg-trials", str(shots),
             "--alpha","0.05","--min-accept","0.99","--max-forge","0.01"]
    print(f"\nUsing smallest PASS radius r={r} (acceptor=hamming_leq:{r})")
    print("Running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode

    # Write a compact PASS artifact
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out = RUNS / f"{ts}_attestation_r{r}.json"
    with open(out,"w") as f:
        json.dump({
            "backend": backend, "mode": mode, "n_qubits": n, "shots": shots,
            "acceptor": f"hamming_leq:{r}",
            "keyed_succ": ksucc, "forger_succ": fsucc,
            "cp_lower_keyed": cp_lower(ksucc, shots),
            "cp_upper_forg":  cp_upper(fsucc, shots),
            "alpha": 0.05, "min_accept": 0.99, "max_forge": 0.01,
            "pass": (rc==0)
        }, f, indent=2)
    print("Wrote:", out)
    sys.exit(rc)

if __name__ == "__main__":
    main()
