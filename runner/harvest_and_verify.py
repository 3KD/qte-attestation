from __future__ import annotations
import json, sys, pathlib, subprocess

BASE = pathlib.Path("/Users/erik/Documents/qte-attestation")
RUNS = BASE / "runs"

def accept_all_zeros(counts: dict) -> int:
    if not counts:
        return 0
    bitlen = max(len(k.replace(' ','')) for k in counts.keys())
    return counts.get("0"*bitlen, 0)

def main():
    files = sorted(RUNS.glob("*_submit_counts.jsonl"))
    if not files:
        print("No submit_counts.jsonl found.", file=sys.stderr); sys.exit(2)
    fpath = files[-1]
    keyed_succ = keyed_trials = forg_succ = forg_trials = 0
    with open(fpath) as f:
        for line in f:
            rec = json.loads(line)
            shots = int(rec["shots"])
            if rec["tag"] == "keyed":
                keyed_succ  = accept_all_zeros(rec["counts"]); keyed_trials = shots
            else:
                forg_succ   = accept_all_zeros(rec["counts"]);  forg_trials = shots

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
