-- Care.AI dynamic patient/event agentic workflow context.
-- Safe additive migration. No existing table, column or patient record is dropped or overwritten.

PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS agentic_event_context(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL UNIQUE REFERENCES agentic_runs(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  event_label TEXT NOT NULL,
  event_detected_at TEXT NOT NULL,
  location TEXT,
  simulation_profile TEXT,
  source TEXT DEFAULT 'synthetic_demo',
  impact_g REAL,
  orientation_change_deg REAL,
  immobility_seconds INTEGER,
  sensor_confidence INTEGER,
  sensor_posture TEXT,
  sensor_vitals_json TEXT,
  assigned_user_id INTEGER REFERENCES users(id),
  mobility_context TEXT,
  recent_trend TEXT,
  triage_score INTEGER,
  triage_at TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agentic_event_context_run
  ON agentic_event_context(run_id);
CREATE INDEX IF NOT EXISTS idx_agentic_event_context_patient_time
  ON agentic_event_context(event_detected_at DESC);
