"""Safe additive Patient Management migration for the existing CareAI SQLite database."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("CAREAI_DB_PATH", str(ROOT / "careai.db")))

COLUMNS = {
    "phone": "TEXT",
    "email": "TEXT",
    "address_line1": "TEXT",
    "postal_code": "TEXT",
    "profile_photo_path": "TEXT",
    "mobility_status": "TEXT",
    "allergies": "TEXT",
    "clinical_notes": "TEXT",
    "assigned_nurse_id": "INTEGER",
    "assigned_clinician_id": "INTEGER",
    "created_at": "TEXT",
    "updated_at": "TEXT",
    "updated_by": "INTEGER",
    "archived_at": "TEXT",
    "archived_by": "INTEGER",
}


def migrate() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(patients)")}
        for name, ddl in COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE patients ADD COLUMN {name} {ddl}")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS patient_contacts(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
              display_name TEXT NOT NULL,
              relationship TEXT,
              phone TEXT,
              email TEXT,
              authorised_for_updates INTEGER NOT NULL DEFAULT 0,
              notification_consent INTEGER NOT NULL DEFAULT 0,
              consent_reference TEXT,
              expires_at TEXT,
              active INTEGER NOT NULL DEFAULT 1,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_patient_contacts_patient_active
              ON patient_contacts(patient_id,active);
            CREATE INDEX IF NOT EXISTS idx_patients_active_name
              ON patients(active,last_name,first_name);
            CREATE INDEX IF NOT EXISTS idx_patients_assigned_nurse
              ON patients(assigned_nurse_id,active);
            """
        )
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute("UPDATE patients SET created_at=COALESCE(created_at,?), updated_at=COALESCE(updated_at,?)", (now, now))
        conn.commit()
        print(f"Patient Management migration applied safely to: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
