"""
E0 — NVADE key, channel, and threat-model specification.

We treat the *encoder* as an existing black box provided by QTE, e.g.:

    psi, meta = generate_series_encoding(series_id=..., params=..., truncation=..., weighting=...)

This file does NOT re-implement encoding or Qiskit plumbing.
Instead, it gives a precise *classical* specification of:

  • What a NVADE key family is.
  • What a secret key is.
  • How an honest channel is specified over Aer vs IBM Cloud.
  • What information an adversary is assumed to see.

Mathematical view (informal but precise enough for E0):

  • Let H_n be the n-qubit Hilbert space with basis {|x> : x in {0,1}^n}.
  • A NVADE encoder family F is a map

        F : Θ -> ℂ^{2^n},   θ ↦ ψ(θ)

    together with metadata describing truncation, weighting, and index order.

  • For any θ ∈ Θ, the normalized statevector is

        |ψ(θ)> = ψ(θ) / ||ψ(θ)||_2

    and the induced measurement distribution in the computational basis is

        p_θ(x) = |⟨x | ψ(θ)⟩|^2,    x ∈ {0,1}^n.

  • An honest channel implementation over a backend B (Aer or IBM Cloud) attempts to realize
    samples from p_θ (possibly distorted by noise).

This module encodes that structure as Python dataclasses suitable for use by U31/U56/U58.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class BackendKind(str, Enum):
    """Which physical / simulated stack implements the NVADE channel.

    This is purely descriptive here; actual Qiskit objects live in runner scripts.

    • AER:     qiskit-aer "aer_simulator" or similar
    • IBM:     IBM Cloud / Qiskit Runtime backend (e.g. "ibm_torino")
    """

    AER = "aer"
    IBM = "ibm"


@dataclass(frozen=True)
class NVADEPublicFamily:
    """Public description of a NVADE key family F.

    This is the part that can be written in a spec / paper and shared openly.

    Fields correspond to the encoder metadata already used in QTE. We do *not*
    fix the exact argument names here; this is a semantic wrapper that U01/U00
    are expected to honor.

    Mathematically, this describes:

        F : Θ -> ℂ^{2^n}

    where:
      • 'series_id' selects a particular analytic family (e.g. "ramanujan_pi_1_over_pi").
      • 'truncation_n' fixes n (number of qubits) and the number of kept coefficients.
      • 'weighting_scheme' encodes how coefficients are re-weighted before normalization.
      • 'index_order' fixes how term indices map to computational basis states.
      • 'amp_norm' specifies the L2 normalization convention.
    """

    family_id: str               # high-level label, e.g. "nvade_ramanujan_pi"
    series_id: str               # QTE series handle / name
    truncation_n: int            # number of qubits n; Hilbert space dimension is 2^n
    max_terms: int               # number of series coefficients used before padding
    weighting_scheme: str        # e.g. "plain", "egf", "custom_w_k"
    index_order: str             # description of basis ordering, e.g. "binary_lex_lsb_first"
    amp_norm: str                # e.g. "l2_unit", "l2_scaled"
    extra_metadata: Mapping[str, str]  # free-form key→value metadata (QTE-compatible)

    def hilbert_dimension(self) -> int:
        """Return dim(H) = 2^n.

        This is the dimension of the complex statevector ψ(θ) in this family.
        """
        return 1 << self.truncation_n


@dataclass(frozen=True)
class NVADESecretKey:
    """Secret key θ for a given NVADE key family.

    Conceptually, θ ∈ Θ is a parameter vector controlling the encoder F.

    In code, we keep θ as an opaque payload, plus a human-readable label.
    The *only* thing we assert at E0 is that different θ typically yield
    different normalized statevectors ψ(θ) up to global phase.

    Examples of what 'theta_payload' might encode (QTE-level semantics):

      • Hidden series parameters (e.g. choice of branch, hidden offset).
      • Seeds controlling randomized constructions inside the encoder.
      • Selection among a finite set of admissible key instances.
    """

    family: NVADEPublicFamily
    label: str                   # e.g. "key_01_ramanujan_pi_variant_A"
    theta_payload: Mapping[str, str]  # opaque parameters; interpreted by QTE encoder

    def describe(self) -> str:
        """Short human-readable description, safe to log locally (not public)."""
        return f"NVADESecretKey(label={self.label}, family_id={self.family.family_id})"


@dataclass(frozen=True)
class NVADEChannelSpec:
    """Honest NVADE channel implementation over a particular backend.

    This is the *intended* physical or simulated realization of p_θ(x):

        p_θ(x) = |⟨x | ψ(θ)⟩|^2

    possibly composed with a noise channel Λ:

        p_θ^Λ(x) = Tr[ Λ(|ψ(θ)⟩⟨ψ(θ)|) |x⟩⟨x| ].

    At E0 we do not fix Λ explicitly; we only record the backend and any
    high-level noise/shot budget hints.

    Fields:

      • backend_kind  — "aer" or "ibm"
      • backend_name  — simulator/hardware identifier, e.g. "aer_simulator", "ibm_torino"
      • n_qubits      — must match family.truncation_n for honest implementations
      • shots         — number of measurement shots per experimental call
      • measurement_basis — usually "computational", could be extended later
      • noise_tag     — high-level description: "native", "twirled", "mitigated"
      • extra_config  — free-form key→value config (e.g. "ibm_instance", "layout_strategy")
    """

    family: NVADEPublicFamily
    backend_kind: BackendKind
    backend_name: str
    n_qubits: int
    shots: int
    measurement_basis: str  # e.g. "computational"
    noise_tag: str          # e.g. "native", "ideal", "mitigated"
    extra_config: Mapping[str, str]

    def check_consistency(self) -> None:
        """Raise ValueError if basic constraints are violated.

        In particular, require that n_qubits matches the family's truncation_n.
        """
        if self.n_qubits != self.family.truncation_n:
            raise ValueError(
                f"ChannelSpec n_qubits={self.n_qubits} does not match "
                f"family.truncation_n={self.family.truncation_n}"
            )


class AdversaryAccess(str, Enum):
    """What can the adversary observe / query?

    This is a *classical* description for E0; actual security games (U31/U56)
    instantiate these options more concretely.

    Options:

      • PUBLIC_SPEC_ONLY
          Adversary sees only the public family spec (NVADEPublicFamily) and
          any associated documentation.

      • PUBLIC_SPEC_AND_SAMPLES
          Adversary sees the public spec and can query the honest channel to
          obtain (bounded) samples from p_θ^Λ(x).

      • PUBLIC_SPEC_AND_NOISY_MODEL
          Adversary knows an explicit approximate noise model Λ_approx in addition
          to the public spec, and can simulate approximate p_θ^Λ_approx(x).

      • FULL_CHANNEL_TRANSCRIPT
          Adversary sees everything the verifier sees in an attestation protocol
          (e.g. full measurement transcript), but does not know θ.
    """

    PUBLIC_SPEC_ONLY = "public_spec_only"
    PUBLIC_SPEC_AND_SAMPLES = "public_spec_and_samples"
    PUBLIC_SPEC_AND_NOISY_MODEL = "public_spec_and_noisy_model"
    FULL_CHANNEL_TRANSCRIPT = "full_channel_transcript"


@dataclass(frozen=True)
class ThreatModelE0:
    """Classical threat-model shell for NVADE-SEC E0.

    This does NOT attempt to prove security; it pins down the knobs that
    U31/U56/U58 will later use.

    Fields:

      • public_family     — what is known to everyone.
      • channel_spec      — how the honest prover realizes p_θ^Λ on Aer/IBM.
      • adversary_access  — what view the adversary is assumed to have.
      • notes             — free-form text for additional assumptions.

    Example instantiations:

      • "Classical attacker with public spec + limited samples":
          adversary_access = PUBLIC_SPEC_AND_SAMPLES

      • "Benchmarking setting where everyone sees the full transcript":
          adversary_access = FULL_CHANNEL_TRANSCRIPT
    """

    public_family: NVADEPublicFamily
    channel_spec: NVADEChannelSpec
    adversary_access: AdversaryAccess
    notes: str

    def short_summary(self) -> str:
        """Return a one-line summary suitable for logs or filenames."""
        return (
            f"family={self.public_family.family_id}, "
            f"backend={self.channel_spec.backend_kind.value}:{self.channel_spec.backend_name}, "
            f"adv={self.adversary_access.value}"
        )


@dataclass(frozen=True)
class NVADEKeyBundle:
    """Convenience container bundling (public family, secret key, channel, threat model).

    This is the E0 object that U31 / U56 / U58 will typically consume:

      • family:   defines F : Θ -> ℂ^{2^n}
      • secret:   chooses θ ∈ Θ (the actual key)
      • channel:  commits to a concrete Aer/IBM implementation
      • threat:   fixes what the adversary is assumed to see

    No quantum code is run here; this is a purely classical specification
    suitable for:

      • embedding in JSON manifests,
      • logging provenance,
      • constructing security / attestation experiments downstream.
    """

    family: NVADEPublicFamily
    secret: NVADESecretKey
    channel: NVADEChannelSpec
    threat: ThreatModelE0

    def sanity_check(self) -> None:
        """Run a few simple internal consistency checks.

        Checks:

          • channel.family == family
          • secret.family  == family
          • threat.public_family == family
          • channel qubit count matches family.truncation_n
        """
        if self.secret.family is not self.family:
            raise ValueError("Secret key family does not match bundle family.")
        if self.channel.family is not self.family:
            raise ValueError("Channel family does not match bundle family.")
        if self.threat.public_family is not self.family:
            raise ValueError("ThreatModelE0 public_family does not match bundle family.")
        self.channel.check_consistency()
