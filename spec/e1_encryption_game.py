"""
E1 -- Encryption semantics baseline (T1): keyed vs no-key decryptor.

We model a simple *classical* encryption game using the NVADE key bundle as the
source of key material. This version does not touch quantum hardware; it just
establishes message/key logic and success metrics.

Key idea:
  - Fix a NVADE key bundle (E0).
  - Derive a deterministic bitstring key from the bundle metadata.
  - Messages are bitstrings of fixed length m.
  - Encryption is XOR with the derived key.
  - Honest decryptor knows the key; adversary does not and must guess.

This is intentionally minimal: it gives us a concrete E1 game with an honest
vs. no-key baseline gap, while reusing the same NVADE bundle objects that are
used for attestation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from spec.e0_nvade_key_channel import NVADEKeyBundle


def _derive_key_bits(bundle: NVADEKeyBundle, message_bits: int) -> np.ndarray:
    """
    Deterministically derive a key bitstring K from the NVADE key bundle.

    We hash together the family_id and key label, reduce to a 32-bit seed,
    and use it to drive a numpy RNG. This keeps key derivation reproducible
    without exposing any internal theta representation.
    """
    s = f"{bundle.family.family_id}|{bundle.secret.label}"
    # Simple deterministic reduction of the UTF-8 bytes to a 32-bit seed.
    raw = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
    seed = int(raw.sum(dtype=np.uint64) % (2**32))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=message_bits, dtype=np.int8)


def _sample_message(rng: np.random.Generator, message_bits: int) -> np.ndarray:
    """Sample a random message bitstring of length message_bits."""
    return rng.integers(0, 2, size=message_bits, dtype=np.int8)


def _xor_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise XOR of two {0,1}-valued arrays."""
    return (a ^ b).astype(np.int8)


def encrypt(message: np.ndarray, key_bits: np.ndarray) -> np.ndarray:
    """Encrypt by XOR with the key."""
    if message.shape != key_bits.shape:
        raise ValueError("Message and key_bits must have the same shape.")
    return _xor_bits(message, key_bits)


def decrypt_with_key(ciphertext: np.ndarray, key_bits: np.ndarray) -> np.ndarray:
    """Honest decryptor: XOR again with the same key."""
    if ciphertext.shape != key_bits.shape:
        raise ValueError("Ciphertext and key_bits must have the same shape.")
    return _xor_bits(ciphertext, key_bits)


def decrypt_without_key(ciphertext: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Adversary baseline: guess a random bitstring of the same length.

    This ignores the ciphertext structure entirely and serves as a no-key
    baseline for success probability.
    """
    return rng.integers(0, 2, size=ciphertext.shape[0], dtype=np.int8)


@dataclass
class EncryptionGameSpec:
    """
    Specification for the E1-T1 encryption baseline game.

    Fields:
      - message_bits: length of each message in bits (m).
      - n_trials:    number of independent message encrypt/decrypt trials.
      - bundle:      NVADE key bundle providing the key material (E0).
      - seed:        RNG seed for reproducibility.
    """

    message_bits: int
    n_trials: int
    bundle: NVADEKeyBundle
    seed: int = 424242


@dataclass
class EncryptionGameStats:
    """Aggregate results for the E1-T1 encryption game."""

    message_bits: int
    n_trials: int
    honest_correct: int
    adversary_correct: int

    @property
    def honest_success_rate(self) -> float:
        return self.honest_correct / self.n_trials if self.n_trials > 0 else 0.0

    @property
    def adversary_success_rate(self) -> float:
        return self.adversary_correct / self.n_trials if self.n_trials > 0 else 0.0

    def as_tuple(self) -> Tuple[float, float]:
        """Return (honest_success_rate, adversary_success_rate)."""
        return (self.honest_success_rate, self.adversary_success_rate)


def run_e1_t1(spec: EncryptionGameSpec) -> EncryptionGameStats:
    """
    Run the E1-T1 encryption baseline game for the given spec.

    Procedure per trial:
      1. Sample random message M in {0,1}^m.
      2. Encrypt C = M XOR K, where K is derived from the NVADE bundle.
      3. Honest decryptor outputs M_honest = C XOR K.
      4. Adversary outputs M_adv = random bitstring of length m (no key).
      5. Record whether M_honest == M and whether M_adv == M.

    Returns aggregate success counts and rates.
    """
    rng = np.random.default_rng(spec.seed)
    key_bits = _derive_key_bits(spec.bundle, spec.message_bits)

    honest_correct = 0
    adversary_correct = 0

    for _ in range(spec.n_trials):
        msg = _sample_message(rng, spec.message_bits)
        ct = encrypt(msg, key_bits)
        dec_honest = decrypt_with_key(ct, key_bits)
        dec_adv = decrypt_without_key(ct, rng)

        if np.array_equal(dec_honest, msg):
            honest_correct += 1
        if np.array_equal(dec_adv, msg):
            adversary_correct += 1

    return EncryptionGameStats(
        message_bits=spec.message_bits,
        n_trials=spec.n_trials,
        honest_correct=honest_correct,
        adversary_correct=adversary_correct,
    )
