from __future__ import annotations
from typing import Tuple
from scipy.stats import beta

def cp_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha/2, k, n-k+1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha/2, k+1, n-k))
    return (lo, hi)

def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    return cp_interval(k, n, alpha)[0]

def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return cp_interval(k, n, alpha)[1]
