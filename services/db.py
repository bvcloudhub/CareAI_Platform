import sqlite3, os
from pathlib import Path
from werkzeug.security import generate_password_hash
from data.seed_data import seed_database

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "careai.db"
DB_PATH = Path(os.getenv("CAREAI_DB_PATH", str(DEFAULT_PATH)))

def get_conn():
    conn=sqlite3.connect(DB_PATH)
    conn.row_factory=sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def query_db(sql,args=(),one=False):
    conn=get_conn()
    try:
        rows=conn.execute(sql,args).fetchall()
        return (rows[0] if rows else None) if one else rows
    finally:
        conn.close()

def execute_db(sql,args=()):
    conn=get_conn()
    try:
        cur=conn.execute(sql,args)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def init_db(force_seed=False):
    conn=get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      role TEXT NOT NULL CHECK(role IN ('admin','nurse','gp')),
      display_name TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS patients(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      external_ref TEXT UNIQUE NOT NULL,
      first_name TEXT NOT NULL,
      last_name TEXT NOT NULL,
      birth_date TEXT NOT NULL,
      sex TEXT,
      city TEXT,
      country TEXT DEFAULT 'Netherlands',
      preferred_language TEXT DEFAULT 'nl',
      living_setting TEXT,
      gp_name TEXT,
      emergency_contact_name TEXT,
      emergency_contact_relation TEXT,
      consent_monitoring INTEGER DEFAULT 1,
      consent_updated_at TEXT,
      lawful_basis TEXT DEFAULT 'demo_synthetic',
      purpose_code TEXT DEFAULT 'remote_care_demo',
      data_residency TEXT DEFAULT 'EU',
      retention_until TEXT,
      dpa_reference TEXT DEFAULT 'DEMO-DPA',
      active INTEGER DEFAULT 1,
      current_status TEXT DEFAULT 'stable'
    );

    CREATE TABLE IF NOT EXISTS conditions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      code TEXT, display TEXT, onset_date TEXT, active INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS medications(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      name TEXT, dose TEXT, schedule TEXT, active INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS medication_events(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      medication_id INTEGER REFERENCES medications(id) ON DELETE CASCADE,
      scheduled_at TEXT NOT NULL,
      status TEXT DEFAULT 'pending',
      recorded_by INTEGER REFERENCES users(id),
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS vitals(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      kind TEXT NOT NULL,
      value REAL NOT NULL,
      unit TEXT NOT NULL,
      source TEXT DEFAULT 'simulator',
      measured_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_vitals_pk ON vitals(patient_id,kind,measured_at DESC);

    CREATE TABLE IF NOT EXISTS device_events(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      device_type TEXT DEFAULT 'home_sensor',
      event_type TEXT NOT NULL,
      value TEXT,
      severity TEXT DEFAULT 'info',
      location TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS risk_scores(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      risk_type TEXT NOT NULL,
      score INTEGER NOT NULL,
      level TEXT NOT NULL,
      explanation TEXT,
      evidence TEXT,
      model_version TEXT DEFAULT 'rules-v0.3',
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS fall_assessments(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      assessment_type TEXT NOT NULL CHECK(assessment_type IN ('prediction','detection')),
      score INTEGER NOT NULL,
      level TEXT NOT NULL,
      status TEXT NOT NULL,
      impact_g REAL,
      orientation_change_deg REAL,
      camera_posture TEXT,
      immobility_seconds INTEGER,
      location TEXT,
      evidence TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS alerts(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      alert_type TEXT,
      severity TEXT,
      status TEXT DEFAULT 'open',
      message TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS care_tasks(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      task_type TEXT,
      priority TEXT,
      status TEXT DEFAULT 'open',
      assigned_role TEXT,
      rationale TEXT,
      created_by INTEGER REFERENCES users(id),
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS care_events(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      event_type TEXT,
      source TEXT,
      description TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS notifications(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      audience TEXT,
      channel TEXT,
      message TEXT,
      status TEXT DEFAULT 'queued',
      source TEXT DEFAULT 'agent',
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS audit_logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER REFERENCES users(id),
      action TEXT NOT NULL,
      patient_id INTEGER REFERENCES patients(id),
      details TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );


    CREATE TABLE IF NOT EXISTS chat_threads(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
      thread_type TEXT NOT NULL CHECK(thread_type IN ('clinician_copilot','patient_chat')),
      created_by_user_id INTEGER REFERENCES users(id),
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS chat_messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      thread_id INTEGER NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
      sender_type TEXT NOT NULL CHECK(sender_type IN ('user','patient','assistant','nurse','gp','system')),
      sender_id INTEGER,
      content TEXT NOT NULL,
      safety_label TEXT DEFAULT 'normal',
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS patient_questions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      question TEXT NOT NULL,
      status TEXT DEFAULT 'open',
      assigned_role TEXT DEFAULT 'nurse',
      response TEXT,
      responded_by INTEGER REFERENCES users(id),
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      responded_at TEXT
    );


    CREATE TABLE IF NOT EXISTS family_users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      display_name TEXT NOT NULL,
      relationship TEXT NOT NULL,
      access_scope TEXT DEFAULT 'summary,alerts,messages,teleconsult',
      consent_granted INTEGER DEFAULT 1,
      consent_reference TEXT DEFAULT 'DEMO-FAMILY-CONSENT',
      expires_at TEXT,
      active INTEGER DEFAULT 1,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS teleconsultations(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      requested_by_type TEXT NOT NULL,
      requested_by_id INTEGER,
      clinician_role TEXT DEFAULT 'nurse',
      clinician_name TEXT,
      status TEXT DEFAULT 'requested',
      reason TEXT,
      room_code TEXT UNIQUE NOT NULL,
      scheduled_at TEXT,
      started_at TEXT,
      ended_at TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS teleconsult_messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      teleconsultation_id INTEGER NOT NULL REFERENCES teleconsultations(id) ON DELETE CASCADE,
      sender_type TEXT NOT NULL,
      sender_name TEXT,
      message TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

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

    CREATE INDEX IF NOT EXISTS idx_agentic_runs_patient_started ON agentic_runs(patient_id,started_at DESC);
    CREATE INDEX IF NOT EXISTS idx_agentic_steps_run_step ON agentic_steps(run_id,step_no);
    CREATE INDEX IF NOT EXISTS idx_care_outcomes_run ON care_outcomes(run_id);

    CREATE TABLE IF NOT EXISTS agentic_approvals(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER NOT NULL REFERENCES agentic_runs(id) ON DELETE CASCADE,
      reviewer_user_id INTEGER NOT NULL REFERENCES users(id),
      decision TEXT NOT NULL CHECK(decision IN ('approved','rejected')),
      note TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agentic_actions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER NOT NULL REFERENCES agentic_runs(id) ON DELETE CASCADE,
      action_type TEXT NOT NULL,
      status TEXT NOT NULL,
      reference_table TEXT,
      reference_id INTEGER,
      message TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      completed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS agentic_workflow_metrics(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id INTEGER NOT NULL UNIQUE REFERENCES agentic_runs(id) ON DELETE CASCADE,
      event_detected_at TEXT,
      nurse_review_at TEXT,
      approval_at TEXT,
      response_started_at TEXT,
      family_notification_at TEXT,
      response_time_seconds INTEGER,
      intervention TEXT,
      outcome_status TEXT,
      workflow_completion_status TEXT,
      teleconsultation_status TEXT,
      operational_impact TEXT,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_agentic_approvals_run ON agentic_approvals(run_id,created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_agentic_actions_run_type ON agentic_actions(run_id,action_type);
    CREATE INDEX IF NOT EXISTS idx_agentic_metrics_run ON agentic_workflow_metrics(run_id);

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
    CREATE INDEX IF NOT EXISTS idx_agentic_event_context_run ON agentic_event_context(run_id);
    CREATE INDEX IF NOT EXISTS idx_agentic_event_context_patient_time ON agentic_event_context(event_detected_at DESC);

    CREATE TABLE IF NOT EXISTS care_bot_sessions(
      id TEXT PRIMARY KEY,
      patient_id INTEGER REFERENCES patients(id) ON DELETE SET NULL,
      family_user_id INTEGER REFERENCES family_users(id) ON DELETE SET NULL,
      user_id INTEGER REFERENCES users(id),
      lang TEXT DEFAULT 'en',
      state_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS care_bot_messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL REFERENCES care_bot_sessions(id) ON DELETE CASCADE,
      role TEXT NOT NULL CHECK(role IN ('user','bot')),
      content TEXT NOT NULL,
      payload_json TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_care_bot_messages_session ON care_bot_messages(session_id, id);

    CREATE TABLE IF NOT EXISTS device_integrations(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      category TEXT NOT NULL,
      region TEXT DEFAULT 'NL/EU',
      connection_method TEXT,
      data_types TEXT,
      status TEXT DEFAULT 'available_option',
      notes TEXT
    );
    """)
    conn.commit()

    # Backfill the new metrics shell for Agentic Care runs created by the earlier
    # integration. This is additive and leaves prior workflow records unchanged.
    conn.execute("""
      INSERT OR IGNORE INTO agentic_workflow_metrics(
        run_id,event_detected_at,workflow_completion_status,teleconsultation_status,
        outcome_status,operational_impact
      )
      SELECT id,
             started_at,
             status,'not_recorded',
             CASE WHEN status='completed' THEN 'legacy_completed' ELSE 'pending' END,
             'Legacy agentic run created before dynamic agentic workflow context was added'
      FROM agentic_runs
    """)
    conn.commit()

    # Additive migration: record where each Copilot answer came from
    # (record lookup, grounded OpenAI answer, or a safety rule).
    chat_columns = {r["name"] for r in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}
    if "source" not in chat_columns:
        conn.execute("ALTER TABLE chat_messages ADD COLUMN source TEXT")
        conn.commit()

    if conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]==0:
        for email,role,name in [
            ("admin@care.ai","admin","Demo Administrator"),
            ("nurse@care.ai","nurse","Sophie de Jong"),
            ("gp@care.ai","gp","Dr. Eva van Dijk")
        ]:
            conn.execute("INSERT INTO users(email,password_hash,role,display_name) VALUES(?,?,?,?)",
                         (email,generate_password_hash("demo123"),role,name))
        conn.commit()

    if conn.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"]==0:
        seed_database(conn)
        conn.commit()

    if conn.execute("SELECT COUNT(*) c FROM device_integrations").fetchone()["c"]==0:
        integrations = [
          ("Apple Watch","Wearable","NL/EU","iPhone companion app + Apple Health / HealthKit","heart rate, activity, sleep, ECG where permitted, mobility metrics","available_option","Treat Apple Fall Detection as an independent Apple safety feature unless Apple explicitly exposes the needed event to your app."),
          ("Withings","Medical devices","NL/EU","Vendor cloud/API or Apple Health / Health Connect","BP, HR, weight, body composition, ECG depending on device","available_option","Good fit for home monitoring; confirm API and commercial terms for each pilot."),
          ("OMRON connect","Medical devices","NL/EU","OMRON ecosystem + Apple Health / Google Health Connect","BP, HR, weight/body metrics depending on device","available_option","Useful for hypertension pathways; confirm enterprise integration method."),
          ("Sensara / radar living-pattern monitoring","Smart home sensors","Netherlands","Vendor gateway/API subject to commercial agreement","room activity, living patterns, radar-based motion events","available_option","Strong elderly-care fit; commercial integration to be agreed with vendor."),
          ("Generic BLE pulse oximeter","Medical devices","EU","BLE gateway/mobile SDK","SpO2, pulse rate","available_option","Prefer CE-marked devices and vendor SDKs for pilots."),
          ("Generic BLE glucometer","Medical devices","EU","BLE gateway/mobile SDK","glucose","available_option","Use validated/CE-marked devices; units configurable to mmol/L."),
          ("Generic ECG patch","Medical devices","EU","Vendor API/mobile gateway","ECG, HR, rhythm events","available_option","Clinical claims depend on chosen certified device."),
          ("IP camera / edge camera","Computer vision","NL/EU","RTSP/ONVIF to local edge gateway","posture, person-on-floor event, room occupancy","available_option","Prefer edge inference and event-only export to minimise privacy exposure."),
          ("mmWave radar sensor","Contactless sensor","NL/EU","MQTT/REST/vendor gateway","presence, motion, respiration proxy, fall-like movement","available_option","Privacy-preserving alternative to video where suitable."),
          ("Smart medication dispenser","Medication","NL/EU","Vendor API/MQTT/webhook","dispense events, missed dose events","available_option","Integrate events into care task orchestration.")
        ]
        conn.executemany("""INSERT INTO device_integrations(name,category,region,connection_method,data_types,status,notes)
                            VALUES(?,?,?,?,?,?,?)""", integrations)
        conn.commit()


    # Demo delegated family access for NL-CR-003 Pieter Bakker.
    pieter=conn.execute("SELECT id FROM patients WHERE external_ref='NL-CR-003'").fetchone()
    if pieter and conn.execute("SELECT COUNT(*) c FROM family_users WHERE patient_id=?",(pieter["id"],)).fetchone()["c"]==0:
        from datetime import datetime, timedelta
        expires=(datetime.now()+timedelta(days=180)).isoformat(timespec="seconds")
        rows=[
          (pieter["id"],"laura.bakker@demo.nl",generate_password_hash("family123"),"Laura Bakker","Daughter",
           "summary,alerts,messages,teleconsult",1,"DEMO-CONSENT-PIETER-001",expires,1),
          (pieter["id"],"thomas.bakker@demo.nl",generate_password_hash("family123"),"Thomas Bakker","Son",
           "summary,alerts,messages",1,"DEMO-CONSENT-PIETER-002",expires,1)
        ]
        conn.executemany("""INSERT INTO family_users(
          patient_id,email,password_hash,display_name,relationship,access_scope,consent_granted,
          consent_reference,expires_at,active
        ) VALUES(?,?,?,?,?,?,?,?,?,?)""",rows)
        conn.commit()

    conn.close()
