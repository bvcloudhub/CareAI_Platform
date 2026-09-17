"""Additive database support for the optional Hospital Hybrid AI module.

The core CareAI schema is intentionally left untouched. This module creates only
hospital/EHR tables and can seed an explicitly requested synthetic inpatient demo.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from services.db import get_conn


HOSPITAL_SCHEMA = r"""
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
  heart_rate REAL,
  spo2 REAL,
  resp_rate REAL,
  systolic_bp REAL,
  diastolic_bp REAL,
  map REAL,
  temperature REAL,
  oxygen_lpm REAL DEFAULT 0,
  consciousness TEXT DEFAULT 'Alert',
  source TEXT DEFAULT 'bedside_monitor'
);

CREATE TABLE IF NOT EXISTS ehr_labs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  admission_id INTEGER NOT NULL REFERENCES hospital_admissions(id) ON DELETE CASCADE,
  collected_at TEXT NOT NULL,
  test_code TEXT NOT NULL,
  test_name TEXT NOT NULL,
  value REAL,
  unit TEXT,
  reference_low REAL,
  reference_high REAL,
  abnormal_flag TEXT,
  source TEXT DEFAULT 'laboratory'
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

CREATE INDEX IF NOT EXISTS idx_hospital_admissions_patient_status
  ON hospital_admissions(patient_id,status);
CREATE INDEX IF NOT EXISTS idx_ehr_vitals_admission_time
  ON ehr_vitals(admission_id,measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_ehr_labs_admission_time
  ON ehr_labs(admission_id,collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_ehr_med_admin_admission_time
  ON ehr_med_admin(admission_id,scheduled_at DESC);
CREATE INDEX IF NOT EXISTS idx_ehr_notes_admission_time
  ON ehr_nursing_notes(admission_id,observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hospital_anomalies_admission_status_score
  ON hospital_anomalies(admission_id,status,score DESC);
CREATE INDEX IF NOT EXISTS idx_ai_agent_results_run
  ON ai_agent_results(run_id,step_no);
CREATE INDEX IF NOT EXISTS idx_hospital_agentic_context_admission
  ON hospital_agentic_context(admission_id,created_at DESC);
"""


def init_hospital_schema() -> None:
    """Create hospital-only tables. Safe to call repeatedly."""
    conn = get_conn()
    try:
        conn.executescript(HOSPITAL_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def hospital_demo_status() -> dict:
    init_hospital_schema()
    conn = get_conn()
    try:
        admissions = conn.execute("SELECT COUNT(*) c FROM hospital_admissions").fetchone()["c"]
        vitals = conn.execute("SELECT COUNT(*) c FROM ehr_vitals").fetchone()["c"]
        return {"loaded": admissions > 0, "admissions": admissions, "vital_rows": vitals}
    finally:
        conn.close()


def seed_hospital_demo_data() -> dict:
    """Explicitly seed a synthetic inpatient demo using existing active patients.

    This never creates or edits core patient identities. It adds hospital encounter
    records that reference existing demo patients and is idempotent once admissions
    exist. The UI labels this data as synthetic.
    """
    init_hospital_schema()
    conn = get_conn()
    try:
        existing_count = conn.execute("SELECT COUNT(*) c FROM hospital_admissions").fetchone()["c"]
        if existing_count:
            return {"created": False, "admissions": existing_count, "reason": "Hospital demo data already exists"}

        patients = conn.execute(
            "SELECT id,external_ref,first_name,last_name FROM patients WHERE active=1 ORDER BY id LIMIT 8"
        ).fetchall()
        if not patients:
            return {"created": False, "admissions": 0, "reason": "No active patients are available"}

        now = datetime.now().replace(microsecond=0)
        wards = [
            "Respiratory", "Internal Medicine", "Cardiology", "Geriatrics",
            "Respiratory", "Surgical Ward", "Geriatrics", "Internal Medicine",
        ]
        reasons = [
            "COPD exacerbation", "Fever and weakness", "Heart failure monitoring",
            "Fall observation", "Pneumonia", "Post-operative observation",
            "Frailty and dehydration", "Diabetes monitoring",
        ]
        physicians = [
            "Dr. van Dijk", "Dr. Bakker", "Dr. Smit", "Dr. Visser",
            "Dr. van Dijk", "Dr. de Boer", "Dr. Smit", "Dr. Bakker",
        ]

        admission_ids = []
        for i, patient in enumerate(patients):
            encounter = f"HDEMO-{now.strftime('%Y%m%d')}-{i + 1:03d}"
            cur = conn.execute(
                """INSERT INTO hospital_admissions(
                     patient_id,encounter_ref,ward,room,bed,admission_reason,attending_physician,
                     admission_at,expected_discharge_at,status
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    patient["id"], encounter, wards[i % len(wards)], f"{2 + i // 4}{10 + i % 4}",
                    chr(65 + i % 2), reasons[i % len(reasons)], physicians[i % len(physicians)],
                    (now - timedelta(days=2 + i % 3)).isoformat(timespec="seconds"),
                    (now + timedelta(days=2 + i % 4)).isoformat(timespec="seconds"), "admitted",
                ),
            )
            admission_ids.append(cur.lastrowid)

        # 24 hours of synthetic bedside observations, with several distinct patterns.
        for idx, admission_id in enumerate(admission_ids):
            for hours_ago in range(24, -1, -4):
                measured = now - timedelta(hours=hours_ago)
                hr, spo2, rr, sbp, dbp, temp, oxygen = 76, 96, 17, 132, 78, 36.8, 0
                if idx == 0 and hours_ago <= 8:  # respiratory deterioration
                    progress = (8 - hours_ago) / 8
                    spo2 = 93 - progress * 3
                    rr = 21 + progress * 7
                    hr = 88 + progress * 14
                    oxygen = 1 + progress * 2
                elif idx == 1 and hours_ago <= 8:  # infection/sepsis-like pattern
                    progress = (8 - hours_ago) / 8
                    temp = 37.6 + progress * 1.3
                    hr = 90 + progress * 20
                    rr = 19 + progress * 6
                    sbp = 118 - progress * 22
                    dbp = 70 - progress * 12
                elif idx == 2 and hours_ago <= 4:  # hemodynamic concern
                    hr, sbp, dbp, rr = 105, 88, 54, 22
                elif idx == 3 and hours_ago <= 4:  # inpatient fall context
                    hr, rr = 92, 20
                map_value = (sbp + 2 * dbp) / 3
                conn.execute(
                    """INSERT INTO ehr_vitals(
                         admission_id,measured_at,heart_rate,spo2,resp_rate,systolic_bp,diastolic_bp,map,
                         temperature,oxygen_lpm,consciousness,source
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        admission_id, measured.isoformat(timespec="seconds"), hr, spo2, rr, sbp, dbp,
                        map_value, temp, oxygen, "Alert", "synthetic_bedside_monitor",
                    ),
                )

        if admission_ids:
            conn.executemany(
                """INSERT INTO ehr_labs(
                     admission_id,collected_at,test_code,test_name,value,unit,reference_low,reference_high,abnormal_flag
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                [
                    (admission_ids[0], (now - timedelta(hours=2)).isoformat(timespec="seconds"), "CRP", "C-reactive protein", 68, "mg/L", 0, 10, "H"),
                    (admission_ids[0], (now - timedelta(hours=2)).isoformat(timespec="seconds"), "WBC", "White blood cells", 13.2, "10^9/L", 4, 11, "H"),
                ],
            )
        if len(admission_ids) > 1:
            conn.executemany(
                """INSERT INTO ehr_labs(
                     admission_id,collected_at,test_code,test_name,value,unit,reference_low,reference_high,abnormal_flag
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                [
                    (admission_ids[1], (now - timedelta(hours=1)).isoformat(timespec="seconds"), "LAC", "Lactate", 3.1, "mmol/L", 0.5, 2.0, "H"),
                    (admission_ids[1], (now - timedelta(hours=1)).isoformat(timespec="seconds"), "WBC", "White blood cells", 15.8, "10^9/L", 4, 11, "H"),
                    (admission_ids[1], (now - timedelta(hours=1)).isoformat(timespec="seconds"), "CRP", "C-reactive protein", 112, "mg/L", 0, 10, "H"),
                ],
            )
        if len(admission_ids) > 4:
            conn.executemany(
                """INSERT INTO ehr_labs(
                     admission_id,collected_at,test_code,test_name,value,unit,reference_low,reference_high,abnormal_flag
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                [
                    (admission_ids[4], (now - timedelta(days=1)).isoformat(timespec="seconds"), "CREA", "Creatinine", 82, "umol/L", 45, 90, ""),
                    (admission_ids[4], (now - timedelta(hours=1)).isoformat(timespec="seconds"), "CREA", "Creatinine", 126, "umol/L", 45, 90, "H"),
                ],
            )

        for idx, admission_id in enumerate(admission_ids):
            for dose_idx in range(3):
                scheduled = now - timedelta(hours=12 - dose_idx * 4)
                status, administered, reason = "administered", scheduled + timedelta(minutes=8), None
                if idx == 5 and dose_idx in (1, 2):
                    status, administered, reason = "omitted", None, "Patient unavailable / procedure"
                conn.execute(
                    """INSERT INTO ehr_med_admin(
                         admission_id,medication_name,dose,scheduled_at,administered_at,status,reason,high_risk
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        admission_id, "Amlodipine" if idx % 2 else "Tiotropium",
                        "5 mg" if idx % 2 else "18 mcg", scheduled.isoformat(timespec="seconds"),
                        administered.isoformat(timespec="seconds") if administered else None,
                        status, reason, 0,
                    ),
                )

        for idx, admission_id in enumerate(admission_ids):
            note = "Patient mobilised with assistance; oral intake adequate."
            mobility, fall_event = "assisted", 0
            if idx == 3:
                note = "Synthetic unwitnessed bathroom fall event recorded by the demo EHR; human post-fall review required."
                mobility, fall_event = "high_fall_risk", 1
            conn.execute(
                """INSERT INTO ehr_nursing_notes(
                     admission_id,observed_at,note_type,note_text,mobility_status,pain_score,intake_ml,urine_output_ml,fall_event
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    admission_id, (now - timedelta(hours=1)).isoformat(timespec="seconds"),
                    "nursing_observation", note, mobility, 2, 650, 500, fall_event,
                ),
            )

        conn.commit()
        return {"created": True, "admissions": len(admission_ids), "reason": "Synthetic hospital demo loaded"}
    finally:
        conn.close()
