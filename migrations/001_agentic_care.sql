-- CareAI additive migration: isolated Agentic Care workflow.
-- Safe for the current SQLite CareAI database. No existing table is dropped,
-- renamed or overwritten.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agentic_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  module_key TEXT NOT NULL DEFAULT 'general',
  scenario TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'running',
  severity TEXT NOT NULL DEFAULT 'pending',
  started_at TEXT DEFAULT CURRENT_TIMESTAMP,
  approved_at TEXT,
  completed_at TEXT,
  approved_by INTEGER REFERENCES users(id),
  created_by INTEGER REFERENCES users(id),
  summary TEXT
);

CREATE TABLE IF NOT EXISTS agentic_steps(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES agentic_runs(id) ON DELETE CASCADE,
  step_no INTEGER NOT NULL,
  agent_name TEXT NOT NULL,
  problem_text TEXT,
  action_text TEXT NOT NULL,
  evidence_text TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  started_at TEXT,
  completed_at TEXT,
  requires_human_approval INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT,
  UNIQUE(run_id, step_no)
);

CREATE TABLE IF NOT EXISTS care_outcomes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  run_id INTEGER REFERENCES agentic_runs(id) ON DELETE SET NULL,
  response_time_seconds INTEGER,
  intervention TEXT,
  outcome TEXT,
  avoidable_visit INTEGER NOT NULL DEFAULT 0,
  escalation_type TEXT,
  recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agentic_runs_patient_started
  ON agentic_runs(patient_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_agentic_steps_run_step
  ON agentic_steps(run_id, step_no);
CREATE INDEX IF NOT EXISTS idx_care_outcomes_run
  ON care_outcomes(run_id);
