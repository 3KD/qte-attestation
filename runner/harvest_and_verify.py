from __future__ import annotations
import json, sys, pathlib, subprocess
# TOML shim for 3.9
try: import tomllib
except Exception: import tomli as tomllib

BASE = pathlib.Path("/Users/erik/Documents/qte-attestation")
RUNS = BASE / "runs"
CFG  = BASE / "runner" / "config.toml"

def _cfg(): 
    with open(CFG, "rb") as f: 
        return tomllib.load(f)

def _secret_bits(n, seed):
    import numpy as np
    return np.random.default_rng(seed).integers(0, 2, size=n).tolist()

def _secret_str(n, seed):
    # Qiskit prints bitstrings as msb..lsb = c_{n-1}...c_0,
    # while we X() qubits in ascending order; reverse for display.
    return "".join(map(str, _secret_bits(n, seed)[::-1]))

def _successes(counts: dict, n: int, cfg) -> int:
    mode = str(cfg.get("acceptor", "secret")).lower()
    if mode == "allzeros":
        return int(counts.get("0"*n, 0))
    # default: secret-bit acceptance
    s = _secret_str(n, int(cfg["key_seed"]))
    return int(counts.get(s, 0))

def main():
    cfg = _cfg()
    n   = int(cfg["n_qubits"])
    files = sorted(RUNS.glob("*_submit_counts.jsonl"))
    if not files:
        print("No submit_counts.jsonl found.", file=sys.stderr)
        sys.exit(2)
    fpath = files[-1]

    keyed_succ = keyed_trials = forg_succ = forg_trials = 0
    with open(fpath) as f:
        for line in f:
            rec = json.loads(line)
            shots = int(rec["shots"])
            succ  = _successes(rec["counts"], n, cfg)
            if rec["tag"] == "keyed":
                keyed_succ, keyed_trials = succ, shots
            else:
                forg_succ,  forg_trials  = succ, shots

    cmd = [
      str(BASE/".venv/bin/qte-attestation"), "verify",
      "--keyed-succ", str(keyed_succ), "--keyed-trials", str(keyed_trials),
      "--forg-succ",  str(forg_succ),  "--forg-trials",  str(forg_trials),
      "--alpha", "0.05", "--min-accept", "0.99", "--max-forge", "0.01"
    ]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == "__main__":
    main()
