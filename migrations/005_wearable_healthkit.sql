-- Additive CareAI wearable/HealthKit schema. No existing table or patient data is removed.
CREATE TABLE IF NOT EXISTS wearable_devices(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  installation_id TEXT,
  device_id TEXT,
  device_name TEXT,
  account_label TEXT,
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','revoked')),
  connected_at TEXT DEFAULT CURRENT_TIMESTAMP,
  last_seen_at TEXT,
  last_synced_at TEXT,
  last_sync_summary TEXT,
  UNIQUE(provider,installation_id)
);
CREATE INDEX IF NOT EXISTS idx_wearable_devices_patient ON wearable_devices(patient_id,status);

CREATE TABLE IF NOT EXISTS wearable_measurements(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  wearable_device_id INTEGER REFERENCES wearable_devices(id) ON DELETE SET NULL,
  provider TEXT NOT NULL,
  source_record_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL NOT NULL,
  unit TEXT NOT NULL,
  raw_value REAL,
  raw_unit TEXT,
  source TEXT,
  device_name TEXT,
  measured_at TEXT NOT NULL,
  received_at TEXT DEFAULT CURRENT_TIMESTAMP,
  vital_id INTEGER REFERENCES vitals(id) ON DELETE SET NULL,
  UNIQUE(patient_id,provider,source_record_id)
);
CREATE INDEX IF NOT EXISTS idx_wearable_measurements_patient_time
  ON wearable_measurements(patient_id,measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_wearable_measurements_metric_time
  ON wearable_measurements(patient_id,metric,measured_at DESC);

CREATE TABLE IF NOT EXISTS wearable_sync_state(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  wearable_device_id INTEGER NOT NULL UNIQUE REFERENCES wearable_devices(id) ON DELETE CASCADE,
  last_request_at TEXT,
  last_success_at TEXT,
  last_measurement_at TEXT,
  last_error TEXT
);
