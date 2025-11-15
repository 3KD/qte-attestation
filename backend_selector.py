
import os
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    QiskitRuntimeService = None

def get_backend(default_name: str = "aer_simulator", ibm_backend: str | None = None):
    \"\"\"Return a Qiskit backend.

    Default: AerSimulator (no network).
    If QTE_USE_IBM=1 and IBM credentials are configured, return an IBM backend
    without changing any test code.

    Args:
        default_name: name for Aer backend (ignored for IBM case).
        ibm_backend: IBM backend name (e.g. "ibm_torino"); if None, use the
                     account's default or the first usable backend.

    Raises:
        RuntimeError if IBM is requested but not available.
    \"\"\"
    use_ibm = os.getenv("QTE_USE_IBM", "0") == "1"

    if not use_ibm:
        # Local simulator path (default)
        return AerSimulator()

    # IBM path
    if QiskitRuntimeService is None:
        raise RuntimeError("QTE_USE_IBM=1 but qiskit_ibm_runtime is not installed")

    try:
        svc = QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(f"QTE_USE_IBM=1 but cannot load IBM account: {exc}") from exc

    if ibm_backend is not None:
        return svc.backend(ibm_backend)

    # Fallback: pick default / first backend
    backends = svc.backends()
    if not backends:
        raise RuntimeError("QTE_USE_IBM=1 but no IBM backends are available")
    return backends[0]
