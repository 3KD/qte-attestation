#!/usr/bin/env python
import argparse, json, pathlib, sys

def main() -> int:
    p = argparse.ArgumentParser(
        description="U56 seed check: AUC_gap = AUC_Q - 0.5 from a U58 attestation JSON."
    )
    p.add_argument(
        "u58_path",
        nargs="?",
        default="runs/2025-11-11T19-55-37Z_U58_attestation.json",
        help="Path to U58 attestation JSON (default: runs/2025-11-11T19-55-37Z_U58_attestation.json)",
    )
    args = p.parse_args()

    path = pathlib.Path(args.u58_path)
    if not path.is_file():
        print(f"[U56] error: attestation file not found: {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text())
    try:
        auc_q = float(data["AUC"])
    except Exception as e:
        print(f"[U56] error: could not read AUC from {path}: {e}", file=sys.stderr)
        return 1

    # U56 v0.1: treat classical baseline as chance-level ROC, AUC_C = 0.5
    auc_c = 0.5
    auc_gap = auc_q - auc_c

    print(f"[U56] U58 file   : {path}")
    print(f"[U56] AUC_Q (U58): {auc_q:.9f}")
    print(f"[U56] AUC_C (toy baseline): {auc_c:.3f}")
    print(f"[U56] AUC_gap    : {auc_gap:.9f}")

    # Pre-registered seed window: AUC_gap >= 0.03
    if auc_gap >= 0.03:
        print("[U56] PASS: U58 lies inside the (toy) security window (gap >= 0.03).")
        return 0
    else:
        print("[U56] FAIL: U58 does NOT clear the (toy) window threshold (gap < 0.03).")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
