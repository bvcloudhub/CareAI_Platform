"""Add dedicated ECG waveform storage for the Generic ECG patch adapter.

Safe to run repeatedly. Existing patient, vital, device-adapter, wearable and
Agentic Care records are not changed or renumbered.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("CAREAI_DB_PATH", ROOT / "careai.db"))

DDL = """
CREATE TABLE IF NOT EXISTS ecg_recordings(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  device_adapter_record_id INTEGER UNIQUE REFERENCES device_adapter_records(id) ON DELETE SET NULL,
  device_identifier TEXT NOT NULL,
  lead_name TEXT NOT NULL DEFAULT 'Lead I',
  sampling_rate_hz INTEGER NOT NULL,
  duration_seconds REAL NOT NULL,
  amplitude_unit TEXT NOT NULL DEFAULT 'mV',
  waveform_json TEXT NOT NULL,
  heart_rate_bpm REAL,
  rr_interval_ms REAL,
  pr_interval_ms REAL,
  qrs_duration_ms REAL,
  qt_interval_ms REAL,
  qtc_ms REAL,
  rhythm_label TEXT,
  measured_at TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'device_adapter:generic_ecg_patch',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ecg_recordings_patient_time
  ON ecg_recordings(patient_id,measured_at DESC);
"""


def migrate(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(DDL)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
    print(f"Enhanced ECG recording migration complete: {DB_PATH}")
