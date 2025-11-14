"""
U58 -- Attestation card specification and helpers.

This module defines a classical "attestation card" object that records one
experiment using a NVADE key bundle (E0) together with:

  • the unit/test identifiers (e.g. U31-T2-wrong-key-LLR),
  • backend + measurement configuration,
  • witness metrics (ROC, AUC, TPR@1%FPR),
  • provenance fields (run JSON paths, commit hash, timestamp),
  • a canonical SHA-256 hash of the card for integrity.

It is designed to wrap existing U31 run artifacts in runs/*.json and the
E0 NVADEKeyBundle defined in spec.e0_nvade_key_channel.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from spec.e0_nvade_key_channel import NVADEKeyBundle


@dataclass
class WitnessMetrics:
    """Summary witness metrics recorded on the attestation card."""

    auc: float
    tpr_at_1pct_fpr: float


@dataclass
class AttestationCard:
    """
    U58 attestation card.

    This is the object we serialize to JSON and (optionally) sign or publish.

    Fields:

      • schema_version   — version tag for the card format.
      • unit_id          — unit identifier, e.g. "U31".
      • test_id          — test identifier, e.g. "T2-wrong-key-LLR".
      • backend_kind     — "aer" or "ibm".
      • backend_name     — simulator/hardware identifier, e.g. "aer_simulator", "ibm_torino".
      • family_id        — NVADE family id from E0 (public).
      • series_id        — underlying series identifier (public).
      • key_label        — secret key label (does not reveal theta payload).
      • adversary_access — E0 access mode, e.g. "public_spec_and_samples".
      • n_qubits         — number of qubits used in the experiment.
      • shots            — shot count.
      • noise_tag        — high-level noise descriptor from E0 channel spec.
      • witness_id       — identifier for the witness family, e.g. U31 version string.
      • witness_params   — key→value map for witness configuration parameters.
      • witness_metrics  — ROC summary metrics (AUC, TPR@1%FPR).
      • roc_points       — optional ROC curve points: {"fpr": [...], "tpr": [...], "thresholds": [...]}.
      • runs_paths       — list of underlying run JSON file paths.
      • commit_hash      — optional VCS commit hash of the code used.
      • timestamp_utc    — ISO-8601 timestamp (UTC) for when the card was created.
      • card_hash        — SHA-256 hash of the card (excluding this field) in hex.
      • notes            — free-form notes for additional assumptions/context.
    """

    schema_version: str
    unit_id: str
    test_id: str
    backend_kind: str
    backend_name: str
    family_id: str
    series_id: str
    key_label: str
    adversary_access: str
    n_qubits: int
    shots: int
    noise_tag: str
    witness_id: str
    witness_params: Dict[str, str]
    witness_metrics: WitnessMetrics
    roc_points: Dict[str, List[float]]
    runs_paths: List[str]
    commit_hash: Optional[str]
    timestamp_utc: str
    card_hash: Optional[str] = None
    notes: str = ""

    def to_payload(self, include_hash: bool = False) -> Dict[str, Any]:
        """Return a dict ready for JSON serialization.

        When include_hash=False, the card_hash field is set to None in the payload,
        so that compute_hash() is taken over a representation that does not depend
        on its own hash.
        """
        payload = asdict(self)
        if not include_hash:
            payload["card_hash"] = None
        return payload

    def compute_hash(self) -> str:
        """Compute SHA-256 hex digest of the canonical hashable payload."""
        payload = self.to_payload(include_hash=False)
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def attach_hash(self) -> None:
        """Compute and set card_hash in-place."""
        self.card_hash = self.compute_hash()


def _now_iso_utc() -> str:
    """Return current UTC time as ISO-8601 string with 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_attestation_card_from_run(
    run_path: Path,
    bundle: NVADEKeyBundle,
    unit_id: str,
    test_id: str,
    witness_id: Optional[str] = None,
    witness_params: Optional[Mapping[str, str]] = None,
    commit_hash: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
    notes: str = "",
) -> AttestationCard:
    """
    Construct an attestation card from:

      • a single U31-style run JSON (run_path),
      • an E0 NVADEKeyBundle (bundle),
      • unit/test identifiers and optional witness/commit metadata.

    The run JSON is expected to contain at least:

      • "backend_kind", "backend_name"
      • "n_qubits"
      • "shots" or "shots_per_class"
      • "roc" with fields "auc" and "tpr_at_1pct_fpr"
      • optionally "roc.fpr", "roc.tpr", "roc.thresholds"
      • optionally "u31_version" for witness_id

    Any missing configuration fields are filled from the E0 bundle.
    """
    data = json.loads(run_path.read_text(encoding="utf-8"))
    roc = data.get("roc", {})
    if "auc" not in roc or "tpr_at_1pct_fpr" not in roc:
        raise ValueError(f"Run JSON at {run_path} is missing ROC summary fields.")

    auc = float(roc["auc"])
    tpr_1pct = float(roc["tpr_at_1pct_fpr"])

    backend_kind = str(data.get("backend_kind", bundle.channel.backend_kind.value))
    backend_name = str(data.get("backend_name", bundle.channel.backend_name))
    n_qubits = int(data.get("n_qubits", bundle.channel.n_qubits))
    shots = int(data.get("shots_per_class", data.get("shots", bundle.channel.shots)))

    if timestamp_utc is None:
        timestamp_utc = _now_iso_utc()

    if witness_id is None:
        witness_id = str(data.get("u31_version", test_id))

    if witness_params is None:
        witness_params_dict: Dict[str, str] = {}
    else:
        witness_params_dict = {str(k): str(v) for k, v in dict(witness_params).items()}

    roc_points = {
        "fpr": list(roc.get("fpr", [])),
        "tpr": list(roc.get("tpr", [])),
        "thresholds": list(roc.get("thresholds", [])),
    }

    card = AttestationCard(
        schema_version="U58-attestation-v0.1",
        unit_id=unit_id,
        test_id=test_id,
        backend_kind=backend_kind,
        backend_name=backend_name,
        family_id=bundle.family.family_id,
        series_id=bundle.family.series_id,
        key_label=bundle.secret.label,
        adversary_access=bundle.threat.adversary_access.value,
        n_qubits=n_qubits,
        shots=shots,
        noise_tag=bundle.channel.noise_tag,
        witness_id=witness_id,
        witness_params=witness_params_dict,
        witness_metrics=WitnessMetrics(
            auc=auc,
            tpr_at_1pct_fpr=tpr_1pct,
        ),
        roc_points=roc_points,
        runs_paths=[str(run_path)],
        commit_hash=commit_hash,
        timestamp_utc=timestamp_utc,
        card_hash=None,
        notes=notes,
    )
    return card


def write_attestation_card(card: AttestationCard, out_path: Path) -> None:
    """
    Attach a hash to the card and write it as pretty JSON to out_path.

    The payload written includes the computed card_hash. The JSON uses
    sorted keys and indentation for readability.
    """
    card.attach_hash()
    payload = card.to_payload(include_hash=True)
    text = json.dumps(payload, sort_keys=True, indent=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
