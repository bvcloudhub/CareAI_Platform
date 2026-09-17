"""Additive migration for CareAI JSON device adapters.

Safe to run more than once. It does not alter or renumber patient, vital,
wearable, Agentic Care, or HealthKit records.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("CAREAI_DB_PATH", ROOT / "careai.db"))

DDL = """
CREATE TABLE IF NOT EXISTS device_adapter_records(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  adapter_key TEXT NOT NULL,
  patient_id INTEGER REFERENCES patients(id) ON DELETE SET NULL,
  source_mode TEXT NOT NULL DEFAULT 'upload',
  source_identifier TEXT,
  device_identifier TEXT,
  original_filename TEXT,
  payload_json TEXT,
  payload_hash TEXT NOT NULL,
  validation_status TEXT NOT NULL DEFAULT 'valid' CHECK(validation_status IN ('valid','invalid')),
  processing_status TEXT NOT NULL DEFAULT 'generated' CHECK(processing_status IN ('generated','processed','failed')),
  error_message TEXT,
  generated_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
  generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  processed_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
  processed_at TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_device_adapter_payload_hash
  ON device_adapter_records(adapter_key,payload_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_device_adapter_source_record
  ON device_adapter_records(adapter_key,patient_id,source_identifier)
  WHERE patient_id IS NOT NULL AND source_identifier IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_device_adapter_history
  ON device_adapter_records(adapter_key,id DESC);
CREATE INDEX IF NOT EXISTS idx_device_adapter_patient
  ON device_adapter_records(patient_id,adapter_key,id DESC);

CREATE TABLE IF NOT EXISTS device_adapter_measurements(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  record_id INTEGER NOT NULL REFERENCES device_adapter_records(id) ON DELETE CASCADE,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  adapter_key TEXT NOT NULL,
  metric TEXT NOT NULL,
  value_real REAL,
  value_text TEXT,
  unit TEXT,
  measured_at TEXT NOT NULL,
  vital_id INTEGER REFERENCES vitals(id) ON DELETE SET NULL,
  device_event_id INTEGER REFERENCES device_events(id) ON DELETE SET NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(record_id,metric)
);
CREATE INDEX IF NOT EXISTS idx_device_adapter_measurement_patient
  ON device_adapter_measurements(patient_id,metric,measured_at DESC);
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
    print(f"Device JSON adapter migration complete: {DB_PATH}")
