from __future__ import annotations
import json, time, pathlib
from typing import Any, Dict

# Python 3.9 TOML shim
try:
    import tomllib  # py311+
except Exception:
    import tomli as tomllib  # py39/py310

from qiskit import transpile
from circuits import keyed_and_forger  # circuits.py lives in same folder

BASE = pathlib.Path("/Users/erik/Documents/qte-attestation")
RUNS = BASE / "runs"; RUNS.mkdir(exist_ok=True)
RDIR = BASE / "runner"

def _load_cfg() -> Dict[str, Any]:
    with open(RDIR / "config.toml", "rb") as f:
        return tomllib.load(f)

def _get_backend(cfg):
    # Try Qiskit Runtime first
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(channel="ibm_quantum", instance=cfg.get("instance"))
        return svc.backend(cfg["backend"]), "runtime"
    except Exception:
        pass
    # Try old Provider
    try:
        from qiskit_ibm_provider import IBMProvider
        provider = IBMProvider()
        return provider.get_backend(cfg["backend"]), "provider"
    except Exception:
        pass
    # Fallback Aer
    from qiskit_aer import Aer
    return Aer.get_backend("aer_simulator"), "aer"

def _quasi_to_counts(quasi: dict, shots: int) -> dict:
    s = float(sum(quasi.values()))
    if s <= 1.0001:  # normalized probabilities
        return {k: int(round(v * shots)) for k, v in quasi.items()}
    return {k: int(round(v)) for k, v in quasi.items()}

def main():
    cfg   = _load_cfg()
    n     = int(cfg["n_qubits"])
    seed  = int(cfg["key_seed"])
    shots = int(cfg["shots"])

    keyed, forger = keyed_and_forger(n, seed)
    backend, mode = _get_backend(cfg)

    if mode in ("runtime", "provider"):
        keyed_t  = transpile(keyed,  backend=backend, optimization_level=1)
        forger_t = transpile(forger, backend=backend, optimization_level=1)
    else:
        keyed_t, forger_t = keyed, forger

    ts = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    out_jsonl = RUNS / f"{ts}_submit_counts.jsonl"
    runlog    = RUNS / f"{ts}_runlog.json"
    job_id    = "local"

    if mode == "runtime":
        # Prefer Runtime SamplerV2; if not available, fall back to provider path
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler, Session
            with Session(backend=backend) as session:
                sampler = Sampler(session=session)
                j1 = sampler.run([keyed_t],  shots=shots); r1 = j1.result()[0]
                j2 = sampler.run([forger_t], shots=shots); r2 = j2.result()[0]
                c1 = _quasi_to_counts(getattr(r1.data, "meas", {}).get("0x0", getattr(r1.data, "quasi_dists", {})), shots)
                c2 = _quasi_to_counts(getattr(r2.data, "meas", {}).get("0x0", getattr(r2.data, "quasi_dists", {})), shots)
                job_id = f"{j1.job_id()}|{j2.job_id()}"
        except Exception:
            mode = "provider"

    if mode == "provider":
        job1 = backend.run(keyed_t,  shots=shots)
        job2 = backend.run(forger_t, shots=shots)
        res1 = job1.result(); res2 = job2.result()
        c1   = res1.get_counts()
        c2   = res2.get_counts()
        job_id = f"{job1.job_id()}|{job2.job_id()}"

    if mode == "aer":
        job1 = backend.run(keyed_t,  shots=shots)
        job2 = backend.run(forger_t, shots=shots)
        c1   = job1.result().get_counts()
        c2   = job2.result().get_counts()

    with open(out_jsonl, "w") as f:
        for tag, counts in (("keyed", c1), ("forger", c2)):
            f.write(json.dumps({
                "tag": tag,
                "backend": str(cfg["backend"]),
                "mode": mode,
                "shots": shots,
                "n_qubits": n,
                "counts": counts
            }) + "\n")

    with open(runlog, "w") as f:
        f.write(json.dumps({
            "ts": ts, "backend": str(cfg["backend"]), "mode": mode, "shots": shots,
            "job_id": job_id, "n_qubits": n
        }, indent=2))

    print(str(out_jsonl)); print(str(runlog))

if __name__ == "__main__":
    main()
