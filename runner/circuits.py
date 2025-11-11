from __future__ import annotations
from typing import Tuple
from qiskit import QuantumCircuit
import numpy as np

def keyed_and_forger(n: int, seed: int) -> Tuple[QuantumCircuit, QuantumCircuit]:
    # Keyed: H Rz(0) H = I → all-zeros with prob ~1
    keyed = QuantumCircuit(n, n)
    for q in range(n):
        keyed.h(q); keyed.rz(0.0, q); keyed.h(q)
    keyed.measure(range(n), range(n))

    # Forger: H Rz(pi) H = X → all-zeros with prob ~0
    forger = QuantumCircuit(n, n)
    for q in range(n):
        forger.h(q); forger.rz(np.pi, q); forger.h(q)
    forger.measure(range(n), range(n))
    return keyed, forger
