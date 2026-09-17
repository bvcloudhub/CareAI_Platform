-- CareAI Hospital Hybrid AI additive migration.
-- Safe to run repeatedly. No existing CareAI table or data is dropped/overwritten.

CREATE TABLE IF NOT EXISTS hospital_admissions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  encounter_ref TEXT UNIQUE NOT NULL,
  hospital_name TEXT DEFAULT 'Care.AI Demo University Hospital',
  ward TEXT NOT NULL,
  room TEXT,
  bed TEXT,
  admission_reason TEXT,
  attending_physician TEXT,
  admission_at TEXT NOT NULL,
  expected_discharge_at TEXT,
  status TEXT DEFAULT 'admitted',
  code_status TEXT DEFAULT 'Full escalation per local policy',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ehr_vitals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER NOT NULL REFERENCES hospital_admissions(id) ON DELETE CASCADE,
  measured_at TEXT NOT NULL,
  heart_rate REAL, spo2 REAL, resp_rate REAL,
  systolic_bp REAL, diastolic_bp REAL, map REAL,
  temperature REAL, oxygen_lpm REAL DEFAULT 0,
  consciousness TEXT DEFAULT 'Alert',
  source TEXT DEFAULT 'bedside_monitor'
);

CREATE TABLE IF NOT EXISTS ehr_labs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER NOT NULL REFERENCES hospital_admissions(id) ON DELETE CASCADE,
  collected_at TEXT NOT NULL,
  test_code TEXT NOT NULL,
  test_name TEXT NOT NULL,
  value REAL, unit TEXT, reference_low REAL, reference_high REAL,
  abnormal_flag TEXT, source TEXT DEFAULT 'laboratory'
);

CREATE TABLE IF NOT EXISTS ehr_med_admin(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER NOT NULL REFERENCES hospital_admissions(id) ON DELETE CASCADE,
  medication_name TEXT NOT NULL,
  dose TEXT,
  scheduled_at TEXT NOT NULL,
  administered_at TEXT,
  status TEXT DEFAULT 'administered',
  reason TEXT,
  high_risk INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ehr_nursing_notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER NOT NULL REFERENCES hospital_admissions(id) ON DELETE CASCADE,
  observed_at TEXT NOT NULL,
  note_type TEXT DEFAULT 'nursing_observation',
  note_text TEXT NOT NULL,
  mobility_status TEXT,
  pain_score INTEGER,
  intake_ml INTEGER,
  urine_output_ml INTEGER,
  fall_event INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hospital_scans(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_type TEXT NOT NULL,
  status TEXT DEFAULT 'completed',
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  completed_at TEXT,
  patients_scanned INTEGER DEFAULT 0,
  anomalies_found INTEGER DEFAULT 0,
  runs_created INTEGER DEFAULT 0,
  triggered_by TEXT DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS hospital_anomalies(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_id INTEGER REFERENCES hospital_scans(id) ON DELETE SET NULL,
  admission_id INTEGER NOT NULL REFERENCES hospital_admissions(id) ON DELETE CASCADE,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  anomaly_type TEXT NOT NULL,
  severity TEXT NOT NULL,
  score INTEGER NOT NULL,
  headline TEXT NOT NULL,
  evidence TEXT NOT NULL,
  recommended_review TEXT,
  status TEXT DEFAULT 'open',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_agent_results(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES agentic_runs(id) ON DELETE CASCADE,
  step_no INTEGER NOT NULL,
  agent_name TEXT NOT NULL,
  execution_mode TEXT NOT NULL,
  rule_score INTEGER,
  ml_score INTEGER,
  llm_model TEXT,
  confidence REAL,
  output_json TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hospital_agentic_context(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL UNIQUE REFERENCES agentic_runs(id) ON DELETE CASCADE,
  admission_id INTEGER NOT NULL REFERENCES hospital_admissions(id) ON DELETE CASCADE,
  anomaly_id INTEGER REFERENCES hospital_anomalies(id) ON DELETE SET NULL,
  anomaly_type TEXT NOT NULL,
  rule_score INTEGER NOT NULL,
  ml_score INTEGER DEFAULT 0,
  hybrid_score INTEGER NOT NULL,
  assigned_user_id INTEGER REFERENCES users(id),
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hospital_admissions_patient_status ON hospital_admissions(patient_id,status);
CREATE INDEX IF NOT EXISTS idx_ehr_vitals_admission_time ON ehr_vitals(admission_id,measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_ehr_labs_admission_time ON ehr_labs(admission_id,collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_ehr_med_admin_admission_time ON ehr_med_admin(admission_id,scheduled_at DESC);
CREATE INDEX IF NOT EXISTS idx_ehr_notes_admission_time ON ehr_nursing_notes(admission_id,observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hospital_anomalies_admission_status_score ON hospital_anomalies(admission_id,status,score DESC);
CREATE INDEX IF NOT EXISTS idx_ai_agent_results_run ON ai_agent_results(run_id,step_no);
CREATE INDEX IF NOT EXISTS idx_hospital_agentic_context_admission ON hospital_agentic_context(admission_id,created_at DESC);
