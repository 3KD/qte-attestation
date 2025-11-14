#!/usr/bin/env python

"""
Runner: E1-T1 encryption baseline (keyed vs no-key decryptor).

This script:
  - constructs a default NVADE key bundle (Ramanujan pi, n=3) via E0,
  - runs the E1-T1 encryption game for a given number of trials and message length,
  - writes a JSON summary to runs/e1_t1_*.json,
  - prints a short summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `spec` can be imported when running from runner/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.e0_nvade_key_channel import (
    NVADEPublicFamily,
    NVADESecretKey,
    NVADEChannelSpec,
    NVADEKeyBundle,
    ThreatModelE0,
    BackendKind,
    AdversaryAccess,
)
from spec.e1_encryption_game import EncryptionGameSpec, run_e1_t1


def _make_default_bundle() -> NVADEKeyBundle:
    """
    Construct a default NVADE key bundle for Ramanujan pi (n=3) for E1.

    This mirrors the family we have been using in E0/E3, but the channel
    parameters here are purely informational for E1 (no hardware access).
    """
    family = NVADEPublicFamily(
        family_id="nvade_ramanujan_pi_n3",
        series_id="ramanujan_pi_1_over_pi",
        truncation_n=3,
        max_terms=8,
        weighting_scheme="plain",
        index_order="binary_lex_lsb_first",
        amp_norm="l2_unit",
        extra_metadata={"source": "QTE", "version": "E1-T1-demo"},
    )

    secret = NVADESecretKey(
        family=family,
        label="e1_demo_key_01",
        theta_payload={"variant": "A", "seed": "424242"},
    )

    # Channel spec is here mainly to keep the bundle structure consistent.
    channel = NVADEChannelSpec(
        family=family,
        backend_kind=BackendKind.AER,
        backend_name="logical_encryption_only",
        n_qubits=3,
        shots=0,
        measurement_basis="none",
        noise_tag="not_applicable",
        extra_config={"note": "E1-T1 classical encryption game"},
    )

    threat = ThreatModelE0(
        public_family=family,
        channel_spec=channel,
        adversary_access=AdversaryAccess.PUBLIC_SPEC_ONLY,
        notes="Adversary learns only the public family/channel; no key material.",
    )

    bundle = NVADEKeyBundle(
        family=family,
        secret=secret,
        channel=channel,
        threat=threat,
    )
    bundle.sanity_check()
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1-T1 encryption baseline: keyed vs no-key decryptor."
    )
    parser.add_argument(
        "--message-bits",
        type=int,
        default=3,
        help="Length of each message in bits (default: 3).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="Number of encrypt/decrypt trials to run (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=424242,
        help="Random seed for reproducibility (default: 424242).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/e1_t1_baseline.json",
        help="Path to output JSON summary (default: runs/e1_t1_baseline.json).",
    )

    args = parser.parse_args()

    bundle = _make_default_bundle()
    spec = EncryptionGameSpec(
        message_bits=args.message_bits,
        n_trials=args.n_trials,
        bundle=bundle,
        seed=args.seed,
    )
    stats = run_e1_t1(spec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "E1-T1-encryption-baseline",
        "message_bits": stats.message_bits,
        "n_trials": stats.n_trials,
        "honest_correct": stats.honest_correct,
        "adversary_correct": stats.adversary_correct,
        "honest_success_rate": stats.honest_success_rate,
        "adversary_success_rate": stats.adversary_success_rate,
        "seed": args.seed,
        "key": {
            "family_id": bundle.family.family_id,
            "series_id": bundle.family.series_id,
            "key_label": bundle.secret.label,
        },
        "threat_model": {
            "adversary_access": bundle.threat.adversary_access.value,
            "notes": bundle.threat.notes,
        },
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  message_bits: {stats.message_bits}")
    print(f"  n_trials:     {stats.n_trials}")
    print(f"  honest_success_rate:    {stats.honest_success_rate:.4f}")
    print(f"  adversary_success_rate: {stats.adversary_success_rate:.4f}")


if __name__ == "__main__":
    main()
