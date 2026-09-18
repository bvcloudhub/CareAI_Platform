"""Create a patient-bound wearable credential for the CareAI iOS companion app.

Example:
    python scripts/pair_wearable.py NL-CR-003 --label "Pieter iPhone"
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.db import init_db
from services.wearable_sync import create_pairing_token


def main():
    parser = argparse.ArgumentParser(description="Pair a wearable credential to one CareAI patient")
    parser.add_argument("patient_ref", help="CareAI patient external reference, for example NL-CR-003")
    parser.add_argument("--provider", default="apple_watch", help="Provider key (default: apple_watch)")
    parser.add_argument("--label", default=None, help="Friendly device/account label")
    args = parser.parse_args()

    init_db()
    result = create_pairing_token(args.patient_ref, args.provider, args.label)
    print("Wearable pairing created")
    print(f"Patient: {result['patient_ref']} - {result['patient_name']}")
    print(f"Provider: {result['provider']}")
    print(f"Wearable device row: {result['wearable_device_id']}")
    print("\nCOPY THIS TOKEN NOW. It is not stored in plaintext and cannot be shown again:")
    print(result["token"])


if __name__ == "__main__":
    main()
