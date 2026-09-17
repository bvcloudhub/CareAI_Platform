import os, json, secrets
from datetime import timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Shell environment variables still work if python-dotenv is unavailable.
    pass
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for, jsonify, flash, abort, session,
                   send_from_directory, make_response)
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash

from services.db import init_db, query_db, execute_db
from services.risk_engine import compute_all_risks, get_latest_vitals, latest_risks
from services.simulator import tick_all_patients, run_named_scenario
from services.fhir_service import patient_bundle
from services.population_service import population_metrics
from services.dashboard_metrics_service import (
    dashboard_metrics,
    landing_patient_overview,
    landing_featured_patient,
    normalise_patient_filter,
    patient_overview_rows,
)
from agents.orchestrator import orchestrate_patient, summarize_patient
from agents.notification_agent import create_notification
from services.copilot_service import clinician_answer, patient_answer, get_or_create_thread, save_message, thread_messages
from services.clinical_copilot import SOURCE_LABELS as COPILOT_SOURCE_LABELS, copilot_status
from services.i18n import translate
from services.care_bot_service import (
    ai_available as care_bot_ai_available,
    handle_message as care_bot_handle,
    handoff as care_bot_send_handoff,
    open_session as care_bot_open,
    restart as care_bot_restart_session,
    transcript as care_bot_transcript,
    ui_strings as care_bot_ui_strings,
)
from services.diagnostics_service import analyse_demo
from services.diagnostic_report import (build_pdf as build_report_pdf, filename_for as report_filename,
                                        load_result as load_report_result, save_result as save_report_result)
from services.agentic_care_service import (
    AGENT_CATALOG,
    EVENT_TYPES,
    LOCATION_OPTIONS,
    MEDICAL_DISCLAIMER,
    MODULE_CATALOG,
    SIMULATION_PROFILES,
    advance_run,
    approve_run,
    available_nurses,
    create_patient_event_run,
    create_patient_fall_run,
    dashboard_agentic_cases,
    outcome_detail,
    patient_options,
    recent_runs,
    reject_run,
    run_state,
)
from services.lung_imaging_service import (
    REGISTRY as LUNG_MODEL_REGISTRY,
    CANCER_MODULES as LUNG_CANCER_MODULES,
    UPLOAD_DIR as LUNG_UPLOAD_DIR,
    analyse_upload as analyse_lung_image,
    runtime_status as lung_imaging_runtime_status,
)
from services.wound_imaging_service import (
    WOUND_CONTEXTS,
    WoundAnalysisRejected,
    analyse_upload as analyse_wound_image,
    runtime_status as wound_imaging_runtime_status,
)
from services.hospital_db import init_hospital_schema, hospital_demo_status, seed_hospital_demo_data
from services.hospital_ehr_service import (
    admission_detail as hospital_admission_detail,
    agent_results as hospital_agent_results,
    hospital_dashboard_rows,
    scan_admissions,
)
from services.hospital_agentic_service import (
    approve_hospital_run,
    hospital_run_state,
    reject_hospital_run,
)
from services.openai_clinical_ai import ai_status as hospital_ai_status

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("CAREAI_SECRET_KEY", "dev-only-change-me"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("CAREAI_ENV") == "production",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
)
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User:
    def __init__(self, row):
        self.id = row["id"]
        self.email = row["email"]
        self.role = row["role"]
        self.password_hash = row["password_hash"]
        self.active = bool(row["active"])
        self.display_name = row["display_name"]
    @property
    def is_authenticated(self): return True
    @property
    def is_active(self): return self.active
    @property
    def is_anonymous(self): return False
    def get_id(self): return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    row = query_db("SELECT * FROM users WHERE id = ?", (user_id,), one=True)
    return User(row) if row else None

def audit(action, patient_id=None, details=""):
    if current_user.is_authenticated:
        execute_db(
            "INSERT INTO audit_logs(user_id,action,patient_id,details) VALUES(?,?,?,?)",
            (current_user.id, action, patient_id, str(details)[:1500])
        )

def role_required(*roles):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("login"))
            if current_user.role not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapped
    return deco


def current_family():
    fid=session.get("family_user_id")
    if not fid:
        return None
    return query_db("SELECT * FROM family_users WHERE id=? AND active=1",(fid,),one=True)

def family_access_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        fam=current_family()
        if not fam:
            return redirect(url_for("family_login"))
        return fn(*args, **kwargs)
    return wrapped

def new_room_code():
    return "CARE-" + secrets.token_hex(4).upper()

@app.before_request
def bootstrap():
    init_db()
    # Hospital Hybrid AI uses an isolated additive schema.
    init_hospital_schema()

@app.context_processor
def inject_global():
    lang=session.get("lang","en")
    return {
        "app_name":"Care.AI",
        "lang":lang,
        "t":lambda key: translate(lang,key),
        # Nav previously hardcoded patient_id=3, which 404s on any dataset that
        # does not happen to have that row. Resolve a real patient instead.
        "default_twin_patient_id":default_twin_patient_id(),
    }


def default_twin_patient_id():
    row=query_db("SELECT id FROM patients WHERE active=1 ORDER BY id LIMIT 1",one=True)
    return row["id"] if row else 1


@app.route("/language/<lang_code>")
def set_language(lang_code):
    if lang_code not in ("en","nl","de"):
        lang_code="en"
    session["lang"]=lang_code
    return redirect(request.referrer or url_for("landing"))

@app.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        row = query_db("SELECT * FROM users WHERE email=? AND active=1",(email,),one=True)
        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row))
            session.permanent = True
            audit("LOGIN")
            return redirect(url_for("dashboard"))
        flash("Invalid demo credentials.", "danger")
    return render_template("login.html", title="Care.AI Login")

@app.route("/logout")
@login_required
def logout():
    audit("LOGOUT")
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
def landing():
    patient_filter = normalise_patient_filter(request.args.get("filter"))
    metrics = dashboard_metrics()
    landing_patients = landing_patient_overview(limit=None, filter_key=patient_filter)
    # Keep the Patient 360 preview aligned with the current Patient Overview filter.
    featured_patient = landing_featured_patient(landing_patients[0]["id"]) if landing_patients else None
    return render_template(
        "landing.html",
        title="Care.AI",
        dashboard_metrics=metrics,
        landing_patients=landing_patients,
        featured_patient=featured_patient,
        patient_filter=patient_filter,
    )

@app.route("/command-centre")
@login_required
def dashboard():
    raw_rows = query_db("""
        SELECT p.*,
          (SELECT value FROM vitals v WHERE v.patient_id=p.id AND v.kind='spo2' ORDER BY measured_at DESC LIMIT 1) spo2,
          (SELECT value FROM vitals v WHERE v.patient_id=p.id AND v.kind='heart_rate' ORDER BY measured_at DESC LIMIT 1) heart_rate,
          (SELECT value FROM vitals v WHERE v.patient_id=p.id AND v.kind='activity' ORDER BY measured_at DESC LIMIT 1) activity,
          (SELECT score FROM risk_scores r WHERE r.patient_id=p.id AND r.risk_type='overall' ORDER BY created_at DESC LIMIT 1) risk_score,
          (SELECT level FROM risk_scores r WHERE r.patient_id=p.id AND r.risk_type='overall' ORDER BY created_at DESC LIMIT 1) risk_level,
          (SELECT explanation FROM risk_scores r WHERE r.patient_id=p.id AND r.risk_type='overall' ORDER BY created_at DESC LIMIT 1) risk_reason
        FROM patients p WHERE p.active=1
    """)
    agentic_cases = dashboard_agentic_cases()
    latest_case_by_patient = {}
    for case in agentic_cases:
        latest_case_by_patient.setdefault(case["patient_id"], case)

    risk_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    rows = []
    agentic_extra_high_critical = set()
    for raw in raw_rows:
        row = dict(raw)
        case = latest_case_by_patient.get(row["id"])
        row["agentic_case"] = case
        if case and case["status"] != "rejected":
            existing_level = (row.get("risk_level") or "low").lower()
            case_level = (case.get("severity") or "low").lower()
            if risk_rank.get(case_level, 1) > risk_rank.get(existing_level, 1):
                row["risk_level"] = case_level
            if case_level in ("critical", "high") and existing_level not in ("critical", "high"):
                agentic_extra_high_critical.add(row["id"])
            row["risk_score"] = max(
                int(row.get("risk_score") or 0),
                int(case.get("triage_score") or case.get("confidence") or 0),
            )
            row["risk_reason"] = (
                f"Agentic event: {case['alert_type']} · {case['confidence']}% sensor confidence · "
                f"triage {case_level} · {case['status'].replace('_',' ')}"
            )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            risk_rank.get((row.get("risk_level") or "low").lower(), 1),
            int(row.get("risk_score") or 0),
            1 if row.get("agentic_case") and row["agentic_case"]["status"] != "rejected" else 0,
        ),
        reverse=True,
    )

    metrics = population_metrics()
    alerts = query_db("""
        SELECT a.*, p.first_name, p.last_name FROM alerts a
        JOIN patients p ON p.id=a.patient_id
        WHERE a.status='open' ORDER BY a.created_at DESC LIMIT 8
    """)
    tasks = query_db("""
        SELECT t.*, p.first_name, p.last_name FROM care_tasks t
        JOIN patients p ON p.id=t.patient_id
        WHERE t.status='open' ORDER BY
        CASE t.priority WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
        t.created_at DESC LIMIT 8
    """)
    audit("VIEW_COMMAND_CENTRE")
    return render_template(
        "dashboard.html",
        patients=rows,
        metrics=metrics,
        alerts=alerts,
        tasks=tasks,
        agentic_cases=agentic_cases,
        agentic_extra_critical=len(agentic_extra_high_critical),
    )

@app.route("/patient/<int:patient_id>")
@login_required
def patient(patient_id):
    p = query_db("SELECT * FROM patients WHERE id=?", (patient_id,), one=True)
    if not p:
        abort(404)

    valid_tabs = {"overview", "vitals", "digital_twin", "diagnostics", "medication", "care_plan"}
    active_tab = (request.args.get("tab") or "overview").strip().lower()
    if active_tab not in valid_tabs:
        active_tab = "overview"

    data = {
        "p": p,
        "active_tab": active_tab,
        "latest": get_latest_vitals(patient_id),
        "risks": latest_risks(patient_id),
        "conditions": query_db("SELECT * FROM conditions WHERE patient_id=? AND active=1", (patient_id,)),
        "meds": query_db("SELECT * FROM medications WHERE patient_id=? AND active=1 ORDER BY name", (patient_id,)),
        "med_events": query_db("SELECT * FROM medication_events WHERE patient_id=? ORDER BY scheduled_at DESC LIMIT 30", (patient_id,)),
        "tasks": query_db("SELECT * FROM care_tasks WHERE patient_id=? ORDER BY created_at DESC LIMIT 20", (patient_id,)),
        "timeline": query_db("SELECT * FROM care_events WHERE patient_id=? ORDER BY created_at DESC LIMIT 40", (patient_id,)),
        "notifications": query_db("SELECT * FROM notifications WHERE patient_id=? ORDER BY created_at DESC LIMIT 20", (patient_id,)),
        "vital_history": query_db("SELECT * FROM vitals WHERE patient_id=? ORDER BY measured_at DESC LIMIT 120", (patient_id,)),
        "device_events": query_db("SELECT * FROM device_events WHERE patient_id=? ORDER BY created_at DESC LIMIT 30", (patient_id,)),
        "alerts": query_db("SELECT * FROM alerts WHERE patient_id=? ORDER BY created_at DESC LIMIT 30", (patient_id,)),
        "agent_summary": summarize_patient(patient_id),
    }
    audit("VIEW_PATIENT_360", patient_id, f"tab={active_tab}")
    return render_template("patient.html", **data)

@app.route("/patient/<int:patient_id>/orchestrate", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def orchestrate(patient_id):
    compute_all_risks(patient_id)
    result = orchestrate_patient(patient_id, actor_user_id=current_user.id)
    audit("RUN_AGENTIC_CARE_FLOW", patient_id, json.dumps(result, default=str))
    flash("Care.AI agents reassessed the patient and updated the care plan.", "success")
    return redirect(url_for("patient", patient_id=patient_id))

@app.route("/patient/<int:patient_id>/task", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def task(patient_id):
    t = request.form.get("task_type","nurse_review")
    note = request.form.get("note","")[:500]
    priority = request.form.get("priority","medium")
    execute_db("""INSERT INTO care_tasks(patient_id,task_type,priority,status,assigned_role,rationale,created_by)
                  VALUES(?,?,?,?,?,?,?)""",
               (patient_id,t,priority,"open",current_user.role,note or "Human initiated task",current_user.id))
    execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
               (patient_id,"human_action","care_team",f"{current_user.display_name} created {t}"))
    audit("CREATE_TASK", patient_id, t)
    return redirect(url_for("patient", patient_id=patient_id))

@app.route("/patient/<int:patient_id>/medication/<int:event_id>/<decision>", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def medication_decision(patient_id,event_id,decision):
    if decision not in ("taken","missed","declined"):
        abort(400)
    execute_db("UPDATE medication_events SET status=?, recorded_by=? WHERE id=? AND patient_id=?",
               (decision,current_user.id,event_id,patient_id))
    execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
               (patient_id,"medication_update","care_team",f"Medication event marked {decision}"))
    audit("MEDICATION_EVENT_UPDATE",patient_id,f"{event_id}:{decision}")
    return redirect(url_for("patient",patient_id=patient_id))

@app.route("/simulate/tick", methods=["POST"])
@login_required
@role_required("admin","nurse")
def simulate_tick():
    tick_all_patients()
    for p in query_db("SELECT id FROM patients WHERE active=1"):
        compute_all_risks(p["id"])
    audit("SIMULATION_TICK")
    flash("Simulation advanced by one monitoring interval.", "success")
    return redirect(request.referrer or url_for("dashboard"))

@app.route("/simulate/scenario/<int:patient_id>/<scenario>", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def simulate_scenario(patient_id,scenario):
    allowed={"copd","fall","medication","recovery"}
    if scenario not in allowed: abort(400)
    out = run_named_scenario(patient_id, scenario)
    compute_all_risks(patient_id)
    orchestrate_patient(patient_id, actor_user_id=current_user.id)
    audit("RUN_SCENARIO",patient_id,scenario)
    flash(f"{scenario.upper()} scenario simulated.", "success")
    return redirect(url_for("patient",patient_id=patient_id))


@app.route("/simulate/fall-live/<int:patient_id>", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def simulate_fall_live(patient_id):
    run_named_scenario(patient_id,"fall")
    compute_all_risks(patient_id)
    orchestrate_patient(patient_id, actor_user_id=current_user.id)
    audit("RUN_LIVE_FALL_SIMULATION",patient_id,"sensor-fusion fall demo")
    return redirect(url_for("digital_twin",patient_id=patient_id,play="fall"))


@app.route("/copilot")
@login_required
def copilot_home():
    patients=query_db("SELECT id,external_ref,first_name,last_name,city,current_status FROM patients WHERE active=1 ORDER BY last_name")
    return render_template("copilot.html",patients=patients,selected=None,messages=[],
                           copilot_ai=copilot_status(),source_labels=COPILOT_SOURCE_LABELS)

@app.route("/copilot/<int:patient_id>", methods=["GET","POST"])
@login_required
def copilot_patient(patient_id):
    p=query_db("SELECT * FROM patients WHERE id=?",(patient_id,),one=True)
    if not p: abort(404)
    thread_id=get_or_create_thread(patient_id,"clinician_copilot",current_user.id)
    if request.method=="POST":
        question=request.form.get("question","").strip()[:1000]
        if question:
            history=thread_messages(thread_id,limit=6)
            save_message(thread_id,"user",question,current_user.id)
            ans=clinician_answer(patient_id,question,history=history)
            save_message(thread_id,"assistant",ans["answer"],None,ans["safety"],ans.get("source"))
            audit("COPILOT_QUERY",patient_id,f"{ans.get('source')}: {question[:280]}")
        return redirect(url_for("copilot_patient",patient_id=patient_id))
    messages=thread_messages(thread_id)
    patients=query_db("SELECT id,external_ref,first_name,last_name,city,current_status FROM patients WHERE active=1 ORDER BY last_name")
    return render_template("copilot.html",patients=patients,selected=p,messages=messages,
                           copilot_ai=copilot_status(),source_labels=COPILOT_SOURCE_LABELS)


@app.route("/copilot/<int:patient_id>/ask", methods=["POST"])
@login_required
def copilot_ask(patient_id):
    """JSON endpoint behind the in-page Copilot chat (static/js/copilot.js)."""
    if not query_db("SELECT 1 FROM patients WHERE id=?",(patient_id,),one=True):
        abort(404)
    payload=request.get_json(silent=True)
    if not isinstance(payload,dict):
        return jsonify({"error":"Expected a JSON body."}),415
    question=(payload.get("question") or "").strip()[:1000]
    if not question:
        return jsonify({"error":"Type a question first."}),400
    thread_id=get_or_create_thread(patient_id,"clinician_copilot",current_user.id)
    history=thread_messages(thread_id,limit=6)
    save_message(thread_id,"user",question,current_user.id)
    ans=clinician_answer(patient_id,question,history=history)
    save_message(thread_id,"assistant",ans["answer"],None,ans["safety"],ans.get("source"))
    audit("COPILOT_QUERY",patient_id,f"{ans.get('source')}: {question[:280]}")
    latest=query_db("SELECT created_at FROM chat_messages WHERE thread_id=? ORDER BY id DESC LIMIT 1",(thread_id,),one=True)
    return jsonify({**ans,"question":question,"created_at":latest["created_at"] if latest else None})

@app.route("/patient-portal/<int:patient_id>", methods=["GET","POST"])
def patient_portal(patient_id):
    # Demo-only patient portal without authentication. Production requires strong patient identity / MFA.
    p=query_db("SELECT * FROM patients WHERE id=?",(patient_id,),one=True)
    if not p: abort(404)
    thread_id=get_or_create_thread(patient_id,"patient_chat",None)
    if request.method=="POST":
        question=request.form.get("question","").strip()[:1000]
        if question:
            save_message(thread_id,"patient",question,None)
            ans=patient_answer(patient_id,question)
            save_message(thread_id,"assistant",ans["answer"],None,ans["safety"])
        return redirect(url_for("patient_portal",patient_id=patient_id))
    messages=thread_messages(thread_id)
    latest=get_latest_vitals(patient_id)
    risks=latest_risks(patient_id)
    upcoming=query_db("""SELECT * FROM teleconsultations WHERE patient_id=? AND status IN ('requested','scheduled','active')
                         ORDER BY created_at DESC LIMIT 5""",(patient_id,))
    return render_template("patient_portal.html",p=p,messages=messages,latest=latest,risks=risks,upcoming=upcoming)

@app.route("/patient-portal/<int:patient_id>/ask-care-team", methods=["POST"])
def patient_ask_care_team(patient_id):
    question=request.form.get("question","").strip()[:1000]
    role=request.form.get("assigned_role","nurse")
    if role not in ("nurse","gp"): role="nurse"
    if question:
        execute_db("""INSERT INTO patient_questions(patient_id,question,status,assigned_role)
                      VALUES(?,?,?,?)""",(patient_id,question,"open",role))
        execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                   (patient_id,"patient_question","patient_portal",f"Question sent to {role}: {question[:250]}"))
    return redirect(url_for("patient_portal",patient_id=patient_id))


@app.route("/family/login", methods=["GET","POST"])
def family_login():
    if request.method=="POST":
        email=request.form.get("email","").strip().lower()
        password=request.form.get("password","")
        fam=query_db("SELECT * FROM family_users WHERE email=? AND active=1",(email,),one=True)
        if fam and check_password_hash(fam["password_hash"],password) and fam["consent_granted"]:
            session["family_user_id"]=fam["id"]
            return redirect(url_for("family_dashboard"))
        flash("Invalid family demo credentials or access is not active.","danger")
    return render_template("family_login.html")

@app.route("/family/logout")
def family_logout():
    session.pop("family_user_id",None)
    return redirect(url_for("family_login"))

@app.route("/family")
@family_access_required
def family_dashboard():
    fam=current_family()
    p=query_db("SELECT * FROM patients WHERE id=?",(fam["patient_id"],),one=True)
    latest=get_latest_vitals(p["id"])
    risks=latest_risks(p["id"])
    tasks=query_db("""SELECT task_type,priority,status,rationale,created_at FROM care_tasks
                      WHERE patient_id=? ORDER BY created_at DESC LIMIT 8""",(p["id"],))
    events=query_db("""SELECT event_type,source,description,created_at FROM care_events
                       WHERE patient_id=? ORDER BY created_at DESC LIMIT 10""",(p["id"],))
    upcoming=query_db("""SELECT * FROM teleconsultations WHERE patient_id=? ORDER BY created_at DESC LIMIT 5""",(p["id"],))
    family_notifications=[]
    if "alerts" in (fam["access_scope"] or "") or "messages" in (fam["access_scope"] or ""):
        family_notifications=query_db("""SELECT * FROM notifications
                                         WHERE patient_id=? AND audience='family' AND status='sent'
                                         ORDER BY created_at DESC LIMIT 8""",(p["id"],))
    thread_id=get_or_create_thread(p["id"],"patient_chat",None)
    messages=thread_messages(thread_id,limit=20)
    return render_template("family_dashboard.html",fam=fam,p=p,latest=latest,risks=risks,tasks=tasks,
                           events=events,upcoming=upcoming,messages=messages,
                           family_notifications=family_notifications)

@app.route("/family/ask", methods=["POST"])
@family_access_required
def family_ask():
    fam=current_family()
    question=request.form.get("question","").strip()[:1000]
    if question:
        thread_id=get_or_create_thread(fam["patient_id"],"patient_chat",None)
        save_message(thread_id,"patient",f"[Family: {fam['display_name']}] {question}",fam["id"])
        ans=patient_answer(fam["patient_id"],question)
        save_message(thread_id,"assistant",ans["answer"],None,ans["safety"])
    return redirect(url_for("family_dashboard"))

@app.route("/family/contact-care-team", methods=["POST"])
@family_access_required
def family_contact_care_team():
    fam=current_family()
    question=request.form.get("question","").strip()[:1000]
    role=request.form.get("assigned_role","nurse")
    if role not in ("nurse","gp"): role="nurse"
    if question:
        execute_db("""INSERT INTO patient_questions(patient_id,question,status,assigned_role)
                      VALUES(?,?,?,?)""",
                   (fam["patient_id"],f"Family ({fam['display_name']}, {fam['relationship']}): {question}","open",role))
        execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                   (fam["patient_id"],"family_question","family_portal",f"{fam['display_name']} sent question to {role}"))
    return redirect(url_for("family_dashboard"))

@app.route("/teleconsult/request/<int:patient_id>", methods=["POST"])
def request_teleconsult(patient_id):
    # Prototype endpoint is available from patient portal; production requires authenticated patient identity.
    p=query_db("SELECT * FROM patients WHERE id=?",(patient_id,),one=True)
    if not p: abort(404)
    clinician_role=request.form.get("clinician_role","nurse")
    if clinician_role not in ("nurse","gp"): clinician_role="nurse"
    reason=request.form.get("reason","Remote care consultation")[:500]
    requested_by_type="family" if session.get("family_user_id") else "patient"
    requested_by_id=session.get("family_user_id")
    room=new_room_code()
    nurse_row=query_db("SELECT display_name FROM users WHERE role='nurse' AND active=1 ORDER BY id LIMIT 1",one=True) if clinician_role=="nurse" else None
    clinician_name=(nurse_row["display_name"] if nurse_row else "Responsible nurse") if clinician_role=="nurse" else p["gp_name"]
    tid=execute_db("""INSERT INTO teleconsultations(
        patient_id,requested_by_type,requested_by_id,clinician_role,clinician_name,status,reason,room_code
    ) VALUES(?,?,?,?,?,?,?,?)""",
    (patient_id,requested_by_type,requested_by_id,clinician_role,clinician_name,"requested",reason,room))
    execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
               (patient_id,"teleconsult_requested","teleconsult",f"{clinician_role} video consultation requested: {reason}"))
    return redirect(url_for("teleconsult_room",consult_id=tid))

@app.route("/teleconsult/<int:consult_id>")
def teleconsult_room(consult_id):
    consult=query_db("""SELECT t.*,p.first_name,p.last_name,p.external_ref,p.city,p.gp_name
                        FROM teleconsultations t JOIN patients p ON p.id=t.patient_id
                        WHERE t.id=?""",(consult_id,),one=True)
    if not consult: abort(404)
    latest=get_latest_vitals(consult["patient_id"])
    risks=latest_risks(consult["patient_id"])
    msgs=query_db("SELECT * FROM teleconsult_messages WHERE teleconsultation_id=? ORDER BY id",(consult_id,))
    return render_template("teleconsult.html",consult=consult,latest=latest,risks=risks,msgs=msgs)

@app.route("/teleconsult/<int:consult_id>/status", methods=["POST"])
def teleconsult_status(consult_id):
    status=request.form.get("status","active")
    if status not in ("requested","scheduled","active","ended"): abort(400)
    if status=="active":
        execute_db("UPDATE teleconsultations SET status='active',started_at=CURRENT_TIMESTAMP WHERE id=?",(consult_id,))
    elif status=="ended":
        execute_db("UPDATE teleconsultations SET status='ended',ended_at=CURRENT_TIMESTAMP WHERE id=?",(consult_id,))
    else:
        execute_db("UPDATE teleconsultations SET status=? WHERE id=?",(status,consult_id))
    return redirect(url_for("teleconsult_room",consult_id=consult_id))

@app.route("/teleconsult/<int:consult_id>/message", methods=["POST"])
def teleconsult_message(consult_id):
    consult=query_db("SELECT * FROM teleconsultations WHERE id=?",(consult_id,),one=True)
    if not consult: abort(404)
    msg=request.form.get("message","").strip()[:1000]
    if msg:
        fam=current_family()
        if fam:
            sender_type="family"; sender_name=fam["display_name"]
        elif current_user.is_authenticated:
            sender_type=current_user.role; sender_name=current_user.display_name
        else:
            sender_type="patient"; sender_name="Patient"
        execute_db("""INSERT INTO teleconsult_messages(teleconsultation_id,sender_type,sender_name,message)
                      VALUES(?,?,?,?)""",(consult_id,sender_type,sender_name,msg))
    return redirect(url_for("teleconsult_room",consult_id=consult_id))

@app.route("/consultations")
@login_required
@role_required("admin", "nurse", "gp")
def consultations_page():
    date_filter = (request.args.get("date") or "today").strip().lower()
    if date_filter == "today":
        consultations = query_db("""
            SELECT t.*, p.external_ref, p.first_name, p.last_name, p.city
            FROM teleconsultations t
            JOIN patients p ON p.id=t.patient_id
            WHERE date(COALESCE(t.scheduled_at,t.created_at))=date('now','localtime')
            ORDER BY COALESCE(t.scheduled_at,t.created_at) DESC
        """)
    else:
        date_filter = "all"
        consultations = query_db("""
            SELECT t.*, p.external_ref, p.first_name, p.last_name, p.city
            FROM teleconsultations t
            JOIN patients p ON p.id=t.patient_id
            ORDER BY COALESCE(t.scheduled_at,t.created_at) DESC
            LIMIT 100
        """)
    audit("VIEW_CONSULTATIONS", details=f"date={date_filter}")
    return render_template("consultations.html", consultations=consultations, date_filter=date_filter)


@app.route("/clinician-inbox")
@login_required
@role_required("admin","nurse","gp")
def clinician_inbox():
    selected_id=request.args.get("patient_id", type=int)
    patients=query_db("""
      SELECT p.*,
        (SELECT question FROM patient_questions q WHERE q.patient_id=p.id ORDER BY q.created_at DESC LIMIT 1) latest_question,
        (SELECT created_at FROM patient_questions q WHERE q.patient_id=p.id ORDER BY q.created_at DESC LIMIT 1) latest_question_at,
        (SELECT COUNT(*) FROM patient_questions q WHERE q.patient_id=p.id AND q.status='open') open_questions
      FROM patients p WHERE p.active=1
      ORDER BY COALESCE(latest_question_at,'1900-01-01') DESC, p.last_name
    """)
    if not selected_id and patients:
        selected_id=next((r["id"] for r in patients if (r["open_questions"] or 0)>0), patients[0]["id"])
    selected=None; messages=[]; latest={}; risks=[]; conditions=[]
    if selected_id:
        selected=query_db("SELECT * FROM patients WHERE id=?",(selected_id,),one=True)
        thread_id=get_or_create_thread(selected_id,"patient_chat",None)
        messages=thread_messages(thread_id,limit=60)
        latest=get_latest_vitals(selected_id)
        risks=latest_risks(selected_id)
        conditions=query_db("SELECT * FROM conditions WHERE patient_id=? AND active=1",(selected_id,))
    return render_template("clinician_inbox.html",patients=patients,selected=selected,messages=messages,
                           latest=latest,risks=risks,conditions=conditions)

@app.route("/clinician-inbox/<int:patient_id>/message", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def clinician_chat_message(patient_id):
    message=request.form.get("message","").strip()[:1500]
    if message:
        thread_id=get_or_create_thread(patient_id,"patient_chat",None)
        save_message(thread_id,current_user.role,message,current_user.id)
        q=query_db("SELECT id FROM patient_questions WHERE patient_id=? AND status='open' ORDER BY created_at ASC LIMIT 1",(patient_id,),one=True)
        if q:
            execute_db("UPDATE patient_questions SET status='answered',response=?,responded_by=?,responded_at=CURRENT_TIMESTAMP WHERE id=?",(message,current_user.id,q["id"]))
        execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",(patient_id,"care_team_message","clinician_inbox",f"{current_user.role} sent a patient chat message"))
        audit("CLINICIAN_CHAT_MESSAGE",patient_id,message[:250])
    return redirect(url_for("clinician_inbox",patient_id=patient_id))

@app.route("/clinician-inbox/<int:patient_id>/quick-action", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def clinician_quick_action(patient_id):
    action=request.form.get("action","nurse_review")
    label={"call_patient":"Call patient","video_call":"Start video consultation","nurse_review":"Nurse review","gp_review":"GP review"}.get(action,action)
    execute_db("""INSERT INTO care_tasks(patient_id,task_type,priority,status,assigned_role,rationale,created_by)
                  VALUES(?,?,?,?,?,?,?)""",(patient_id,action,"medium","open",current_user.role,f"{label} initiated from Patient Inbox",current_user.id))
    execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",(patient_id,"quick_action","clinician_inbox",label))
    audit("PATIENT_INBOX_QUICK_ACTION",patient_id,action)
    return redirect(url_for("clinician_inbox",patient_id=patient_id))

@app.route("/clinician-inbox/<int:question_id>/respond", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def clinician_respond(question_id):
    q=query_db("SELECT * FROM patient_questions WHERE id=?",(question_id,),one=True)
    if not q: abort(404)
    response=request.form.get("response","").strip()[:1500]
    if response:
        execute_db("""UPDATE patient_questions SET status='answered',response=?,responded_by=?,responded_at=CURRENT_TIMESTAMP
                      WHERE id=?""",(response,current_user.id,question_id))
        thread_id=get_or_create_thread(q["patient_id"],"patient_chat",None)
        save_message(thread_id,current_user.role,response,current_user.id)
        execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                   (q["patient_id"],"care_team_response","patient_portal",f"{current_user.role} answered patient question"))
        audit("ANSWER_PATIENT_QUESTION",q["patient_id"],str(question_id))
    return redirect(url_for("clinician_inbox"))

@app.route("/digital-twin/<int:patient_id>")
@login_required
def digital_twin(patient_id):
    p=query_db("SELECT * FROM patients WHERE id=?",(patient_id,),one=True)
    if not p: abort(404)
    latest=get_latest_vitals(patient_id)
    events=query_db("SELECT * FROM device_events WHERE patient_id=? ORDER BY created_at DESC LIMIT 12",(patient_id,))
    risks=latest_risks(patient_id)
    fall=query_db("SELECT * FROM fall_assessments WHERE patient_id=? ORDER BY id DESC LIMIT 1",(patient_id,),one=True)
    audit("VIEW_DIGITAL_TWIN",patient_id)
    return render_template("digital_twin.html",p=p,latest=latest,events=events,risks=risks,fall=fall)


DEMO_MODULES = ("skin",)
WOUND_MODULE = "wound"


def _diagnostics_context(**overrides):
    ctx = {
        "result": None,
        "demo_result": None,
        "selected_module": None,
        "image_name": None,
        "error": None,
        "rejection": None,
        "wound_context": "unknown",
        "wound_contexts": WOUND_CONTEXTS,
        "runtime": lung_imaging_runtime_status(),
        "wound_runtime": wound_imaging_runtime_status(),
        "registry": LUNG_MODEL_REGISTRY,
        "cancer_modules": LUNG_CANCER_MODULES,
        "demo_modules": DEMO_MODULES,
    }
    ctx.update(overrides)
    return ctx


@app.route("/ai-diagnostics")
@login_required
def ai_diagnostics():
    return render_template("ai_diagnostics.html", **_diagnostics_context())


@app.route("/ai-diagnostics/<module>", methods=["GET","POST"])
@login_required
def ai_diagnostics_module(module):
    # 'lung' was the old single demo module; it now resolves to the CT model.
    if module == "lung":
        return redirect(url_for("ai_diagnostics_module", module="ct"))
    if module not in DEMO_MODULES and module not in LUNG_MODEL_REGISTRY and module != WOUND_MODULE:
        abort(404)

    result = demo_result = error = image_name = rejection = None
    wound_context = "unknown"
    if request.method == "POST":
        f = request.files.get("image")
        if module in LUNG_MODEL_REGISTRY:
            try:
                result = analyse_lung_image(module, f)
                image_name = result["source_filename"]
                audit("AI_IMAGING_ANALYSIS", None,
                      f"{module}:{image_name}:{result['prediction']}@{result['confidence_pct']}%")
            except ValueError as exc:
                error = str(exc)
            except RuntimeError as exc:
                error = str(exc)
                audit("AI_IMAGING_UNAVAILABLE", None, f"{module}:{exc}")
        elif module == WOUND_MODULE:
            wound_context = request.form.get("wound_context", "unknown")
            if wound_context not in WOUND_CONTEXTS:
                wound_context = "unknown"
            try:
                result = analyse_wound_image(f, wound_context)
                image_name = result["source_filename"]
                risk = result.get("surgical_risk")
                if result["mode"] == "wound_type":
                    summary = f"{result['prediction']}@{result['confidence_pct']}%"
                    if risk:
                        summary += f"+risk:{risk['category']}@{risk['ensemble_score']:.2f}"
                else:
                    summary = f"{risk['category']}@{risk['ensemble_score']:.2f}"
                audit("AI_WOUND_ANALYSIS", None, f"{result['mode']}:{image_name}:{summary}")
            except WoundAnalysisRejected as exc:
                error = str(exc)
                rejection = {"stage": exc.stage}
                audit("AI_WOUND_SCREENED_OUT", None, f"{exc.stage}:{getattr(f, 'filename', '')}")
            except ValueError as exc:
                error = str(exc)
            except RuntimeError as exc:
                error = str(exc)
                audit("AI_WOUND_UNAVAILABLE", None, str(exc)[:300])
        elif f and f.filename:
            demo_result = analyse_demo(module, f.filename, f.read())
            image_name = f.filename
            audit("AI_DIAGNOSTIC_DEMO", None, f"{module}:{f.filename}")

    # Keep the finished result so its PDF report can be built on download.
    save_report_result(result or demo_result)

    return render_template("ai_diagnostics.html", **_diagnostics_context(
        result=result, demo_result=demo_result, selected_module=module,
        image_name=image_name, error=error, rejection=rejection, wound_context=wound_context))


@app.route("/ai-diagnostics/report/<run_id>.pdf")
@login_required
def ai_diagnostics_report(run_id):
    """Download one AI analysis as a PDF report."""
    result=load_report_result(run_id)
    if not result: abort(404)
    response=make_response(build_report_pdf(result, clinician=current_user.display_name))
    response.headers["Content-Type"]="application/pdf"
    response.headers["Content-Disposition"]=f'attachment; filename="{report_filename(result)}"'
    audit("AI_REPORT_DOWNLOAD", None, f"{result.get('module')}:{run_id}")
    return response


@app.route("/diagnostics-image/<path:filename>")
@login_required
def diagnostics_image(filename):
    """Serve an uploaded scan back to the clinician who submitted it."""
    if "/" in filename or "\\" in filename or ".." in filename:
        abort(404)
    return send_from_directory(LUNG_UPLOAD_DIR, filename)


@app.route("/hospital")
@login_required
def hospital_command_centre():
    demo = hospital_demo_status()
    rows = hospital_dashboard_rows() if demo["loaded"] else []
    last_scan = query_db("SELECT * FROM hospital_scans ORDER BY id DESC LIMIT 1", one=True)
    active_runs = query_db(
        """SELECT r.*,p.first_name,p.last_name,p.external_ref
           FROM agentic_runs r
           JOIN patients p ON p.id=r.patient_id
           WHERE r.module_key='hospital'
           ORDER BY CASE r.status WHEN 'awaiting_approval' THEN 1 WHEN 'responding' THEN 2 ELSE 3 END,
                    r.id DESC LIMIT 20"""
    ) if demo["loaded"] else []
    audit("VIEW_HOSPITAL_AI")
    return render_template(
        "hospital_dashboard.html",
        title="Hospital Hybrid AI",
        rows=rows,
        last_scan=last_scan,
        active_runs=active_runs,
        ai=hospital_ai_status(),
        demo=demo,
    )


@app.route("/hospital/demo-seed", methods=["POST"])
@login_required
@role_required("admin")
def hospital_demo_seed():
    result = seed_hospital_demo_data()
    audit("HOSPITAL_DEMO_SEED", None, result)
    category = "success" if result.get("created") else "warning"
    flash(result.get("reason", "Hospital demo seed completed."), category)
    return redirect(url_for("hospital_command_centre"))


@app.route("/hospital/scan", methods=["POST"])
@login_required
@role_required("admin", "nurse", "gp")
def hospital_scan():
    if not hospital_demo_status()["loaded"]:
        flash("No hospital encounter data is configured. Load the synthetic hospital demo first.", "warning")
        return redirect(url_for("hospital_command_centre"))
    scan_type = (request.form.get("scan_type") or "adhoc").strip().lower()
    if scan_type not in ("adhoc", "daily"):
        scan_type = "adhoc"
    scan_id = scan_admissions(
        scan_type=scan_type,
        triggered_by=current_user.email,
        actor_user_id=current_user.id,
    )
    audit("HOSPITAL_EHR_SCAN", None, f"scan={scan_id};type={scan_type}")
    flash("Hospital EHR scan completed. High/critical findings were routed to human-controlled workflows.", "success")
    return redirect(url_for("hospital_command_centre"))


@app.route("/hospital/admission/<int:admission_id>")
@login_required
def hospital_admission(admission_id):
    detail = hospital_admission_detail(admission_id)
    if not detail:
        abort(404)
    ctx, vitals, labs, meds, notes, anomalies, runs = detail
    audit("VIEW_HOSPITAL_ADMISSION", ctx["admission"]["patient_id"], f"admission={admission_id}")
    return render_template(
        "hospital_admission.html",
        title="Hospital Admission",
        ctx=ctx,
        vitals=vitals,
        labs=labs,
        meds=meds,
        notes=notes,
        anomalies=anomalies,
        runs=runs,
    )


@app.route("/hospital/run/<int:run_id>")
@login_required
def hospital_agent_run(run_id):
    state = hospital_run_state(run_id)
    if not state:
        abort(404)
    run = state["run"]
    audit("VIEW_HOSPITAL_AGENT_RUN", run["patient_id"], f"run={run_id}")
    return render_template(
        "hospital_agent_run.html",
        title="Hospital Hybrid AI Workflow",
        run=run,
        ctx=state["context"],
        steps=state["steps"],
        outcome=state["outcome"],
        approval=state["approval"],
        actions=state["actions"],
        metrics=state["metrics"],
        ai_results=hospital_agent_results(run_id),
        ai=hospital_ai_status(),
    )


@app.route("/hospital/run/<int:run_id>/approve", methods=["POST"])
@login_required
@role_required("admin", "nurse")
def hospital_agent_approve(run_id):
    try:
        state = approve_hospital_run(run_id, current_user.id, request.form.get("note", ""))
    except PermissionError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("hospital_agent_run", run_id=run_id))
    patient_id = state["run"]["patient_id"] if state else None
    audit("HOSPITAL_AGENT_APPROVED", patient_id, f"run={run_id}")
    flash("Human review approved. Care coordination, consent-aware family handling and outcome tracking completed.", "success")
    return redirect(url_for("hospital_agent_run", run_id=run_id))


@app.route("/hospital/run/<int:run_id>/reject", methods=["POST"])
@login_required
@role_required("admin", "nurse")
def hospital_agent_reject(run_id):
    try:
        state = reject_hospital_run(run_id, current_user.id, request.form.get("note", ""))
    except PermissionError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("hospital_agent_run", run_id=run_id))
    patient_id = state["run"]["patient_id"] if state else None
    audit("HOSPITAL_AGENT_REJECTED", patient_id, f"run={run_id}")
    flash("Hospital workflow stopped by the human reviewer. No downstream actions were executed.", "warning")
    return redirect(url_for("hospital_agent_run", run_id=run_id))


@app.route("/api/hospital-ai/status")
@login_required
def api_hospital_ai_status():
    return jsonify(hospital_ai_status())


@app.route("/agentic-care")
@login_required
@role_required("admin", "nurse", "gp")
def agentic_care():
    patients = patient_options()
    nurses = available_nurses()
    audit("VIEW_AGENTIC_CARE")
    return render_template(
        "agentic_care.html",
        catalog=AGENT_CATALOG,
        patients=patients,
        nurses=nurses,
        event_types=EVENT_TYPES,
        locations=LOCATION_OPTIONS,
        simulation_profiles=SIMULATION_PROFILES,
        recent=recent_runs(),
        disclaimer=MEDICAL_DISCLAIMER,
    )


@app.route("/agentic-care/simulate", methods=["POST"])
@login_required
@role_required("admin", "nurse", "gp")
def simulate_agentic_event():
    try:
        patient_id = int(request.form.get("patient_id", "0"))
    except ValueError:
        patient_id = 0
    event_type = (request.form.get("event_type") or "fall").strip().lower()
    location = (request.form.get("location") or "Unknown location").strip()
    simulation_profile = (request.form.get("simulation_profile") or "high_confidence").strip().lower()
    try:
        assigned_user_id = int(request.form.get("assigned_user_id") or 0) or None
    except ValueError:
        assigned_user_id = None
    try:
        run_id = create_patient_event_run(
            patient_id,
            current_user.id,
            event_type=event_type,
            location=location,
            simulation_profile=simulation_profile,
            assigned_user_id=assigned_user_id,
        )
    except RuntimeError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("agentic_care"))
    audit(
        "AGENTIC_EVENT_SIMULATION_STARTED",
        patient_id,
        f"run={run_id}; event={event_type}; location={location}; profile={simulation_profile}",
    )
    return redirect(url_for("agentic_run", run_id=run_id, autorun=1))


@app.route("/agentic-care/module/<module_key>")
@login_required
@role_required("admin", "nurse", "gp")
def agentic_module_route(module_key):
    targets = {
        "general": ("copilot_home", {}),
        "lung": ("ai_diagnostics_module", {"module": "ct"}),
        "wound": ("ai_diagnostics_module", {"module": "wound"}),
        "skin": ("ai_diagnostics_module", {"module": "skin"}),
    }
    target = targets.get(module_key)
    if not target:
        abort(404)
    endpoint, kwargs = target
    audit("AGENTIC_MODULE_ROUTE", details=module_key)
    return redirect(url_for(endpoint, **kwargs))


@app.route("/agentic-care/patient/<int:patient_id>/simulate", methods=["POST"])
@login_required
@role_required("admin", "nurse", "gp")
def simulate_patient_agentic_fall(patient_id):
    event_type = (request.form.get("event_type") or "fall").strip().lower()
    location = (request.form.get("location") or "Unknown location").strip()
    simulation_profile = (request.form.get("simulation_profile") or "high_confidence").strip().lower()
    try:
        run_id = create_patient_event_run(
            patient_id, current_user.id, event_type=event_type, location=location,
            simulation_profile=simulation_profile
        )
    except RuntimeError:
        abort(404)
    audit("AGENTIC_SIMULATION_STARTED", patient_id, f"run={run_id}; event={event_type}")
    return redirect(url_for("agentic_run", run_id=run_id, autorun=1))


@app.route("/agentic-care/run/<int:run_id>")
@login_required
@role_required("admin", "nurse", "gp")
def agentic_run(run_id):
    try:
        state = run_state(run_id)
    except RuntimeError:
        abort(404)
    return render_template(
        "agentic_run.html",
        run=state["run"],
        steps=state["steps"],
        outcome=state["outcome"],
        approval=state["approval"],
        actions=state["actions"],
        metrics=state["metrics"],
        disclaimer=MEDICAL_DISCLAIMER,
    )


@app.route("/agentic-care/run/<int:run_id>/advance", methods=["POST"])
@login_required
@role_required("admin", "nurse", "gp")
def advance_agentic_run(run_id):
    try:
        state = advance_run(run_id, current_user.id)
    except RuntimeError:
        abort(404)
    return jsonify(state)


@app.route("/agentic-care/run/<int:run_id>/status")
@login_required
@role_required("admin", "nurse", "gp")
def agentic_run_status(run_id):
    try:
        return jsonify(run_state(run_id))
    except RuntimeError:
        abort(404)


@app.route("/agentic-care/run/<int:run_id>/approve", methods=["POST"])
@login_required
@role_required("admin", "nurse")
def approve_agentic_run(run_id):
    note = request.form.get("note", "")[:500]
    try:
        state = approve_run(run_id, current_user.id, note)
    except RuntimeError:
        abort(404)
    except PermissionError:
        abort(403)
    audit("AGENTIC_SIMULATION_APPROVED", state["run"]["patient_id"], f"run={run_id}; note={note[:200]}")
    return redirect(url_for("agentic_run", run_id=run_id, autorun=1))


@app.route("/agentic-care/run/<int:run_id>/reject", methods=["POST"])
@login_required
@role_required("admin", "nurse")
def reject_agentic_run(run_id):
    note = request.form.get("note", "")[:500]
    try:
        state = reject_run(run_id, current_user.id, note)
    except RuntimeError:
        abort(404)
    except PermissionError:
        abort(403)
    audit("AGENTIC_SIMULATION_REJECTED", state["run"]["patient_id"], f"run={run_id}; note={note[:200]}")
    flash("The proposed response was rejected. No post-approval workflow actions were executed.", "success")
    return redirect(url_for("agentic_run", run_id=run_id))


@app.route("/agentic-care/run/<int:run_id>/outcome")
@login_required
@role_required("admin", "nurse", "gp")
def agentic_outcome(run_id):
    try:
        state = outcome_detail(run_id)
    except RuntimeError:
        abort(404)
    return render_template(
        "agentic_outcome.html", state=state, run=state["run"], event=state["event"],
        metrics=state["metrics"], actions=state["actions"], outcome=state["outcome"], approval=state["approval"]
    )


@app.context_processor
def inject_care_bot():
    """The floating care assistant is on every page; on the patient portal and
    the family view it also knows which patient it is for."""
    patient_id = None
    if request.endpoint == "patient_portal":
        patient_id = (request.view_args or {}).get("patient_id")
    elif request.endpoint == "family_dashboard":
        fam = current_family()
        patient_id = fam["patient_id"] if fam else None
    return {"care_bot_ui": care_bot_ui_strings(session.get("lang", "en")), "care_bot_patient_id": patient_id}


def _care_bot_context(payload):
    """Who the chat is for. Family access comes from the family session. The demo
    patient portal has no login, so its page supplies the patient id - the same
    trust level as the portal's existing "Ask your nurse" form."""
    fam = current_family()
    if fam:
        return {"patient_id": fam["patient_id"], "family_user_id": fam["id"], "user_id": None}, fam
    patient_id = None
    raw = (payload or {}).get("patient_id")
    if raw not in (None, ""):
        try:
            candidate = int(raw)
        except (TypeError, ValueError):
            candidate = None
        if candidate and query_db("SELECT 1 FROM patients WHERE id=? AND active=1", (candidate,), one=True):
            patient_id = candidate
    user_id = current_user.id if current_user.is_authenticated else None
    return {"patient_id": patient_id, "family_user_id": None, "user_id": user_id}, None


def _care_bot_json():
    # get_json(silent=True) only accepts an application/json body, which a
    # cross-site HTML form cannot send - the app has no CSRF tokens.
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else None


@app.route("/care-bot/session")
def care_bot_session():
    context, _ = _care_bot_context({"patient_id": request.args.get("patient_id")})
    bot = care_bot_open(session.get("care_bot_session"), session.get("lang", "en"), **context)
    session["care_bot_session"] = bot["id"]
    return jsonify({"session_id": bot["id"], "messages": care_bot_transcript(bot["id"]),
                    "ai": care_bot_ai_available(), "linked_patient": bool(bot["patient_id"])})


@app.route("/care-bot/message", methods=["POST"])
def care_bot_message():
    payload = _care_bot_json()
    if payload is None:
        return jsonify({"error": "Expected a JSON body."}), 415
    context, _ = _care_bot_context(payload)
    try:
        result = care_bot_handle(session.get("care_bot_session"), payload.get("text", ""),
                                 session.get("lang", "en"), **context)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    session["care_bot_session"] = result["session_id"]
    return jsonify(result)


@app.route("/care-bot/restart", methods=["POST"])
def care_bot_restart():
    payload = _care_bot_json()
    if payload is None:
        return jsonify({"error": "Expected a JSON body."}), 415
    context, _ = _care_bot_context(payload)
    bot = care_bot_restart_session(session.get("care_bot_session"), session.get("lang", "en"), **context)
    session["care_bot_session"] = bot["id"]
    return jsonify({"session_id": bot["id"], "messages": care_bot_transcript(bot["id"])})


@app.route("/care-bot/handoff", methods=["POST"])
def care_bot_handoff():
    payload = _care_bot_json()
    if payload is None:
        return jsonify({"error": "Expected a JSON body."}), 415
    _, fam = _care_bot_context(payload)
    try:
        result = care_bot_send_handoff(session.get("care_bot_session"), session.get("lang", "en"),
                                       family_name=fam["display_name"] if fam else None)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/patients")
@login_required
def patients_page():
    patient_filter = normalise_patient_filter(request.args.get("filter"))
    patients = patient_overview_rows(filter_key=patient_filter, limit=None)
    return render_template("patients.html", patients=patients, patient_filter=patient_filter)

@app.route("/reports")
@login_required
def reports():
    metrics=population_metrics()
    return render_template("reports.html",metrics=metrics)

@app.route("/population")
@login_required
def population():
    metrics=population_metrics()
    risk_dist=query_db("""
        SELECT COALESCE(current_status,'stable') label, COUNT(*) value
        FROM patients WHERE active=1 GROUP BY current_status
    """)
    city_dist=query_db("SELECT city label, COUNT(*) value FROM patients WHERE active=1 GROUP BY city ORDER BY value DESC")
    condition_dist=query_db("""
        SELECT display label, COUNT(*) value FROM conditions WHERE active=1
        GROUP BY display ORDER BY value DESC LIMIT 10
    """)
    interventions=query_db("""
        SELECT date(created_at) day, COUNT(*) value FROM care_tasks
        GROUP BY date(created_at) ORDER BY day DESC LIMIT 14
    """)
    audit("VIEW_POPULATION_ANALYTICS")
    return render_template("population.html",metrics=metrics,risk_dist=risk_dist,city_dist=city_dist,
                           condition_dist=condition_dist,interventions=interventions)

@app.route("/integrations")
@login_required
def integrations():
    items=query_db("SELECT * FROM device_integrations ORDER BY category,name")
    audit("VIEW_INTEGRATIONS")
    return render_template("integrations.html", items=items)

@app.route("/security")
@login_required
def security():
    audit("VIEW_SECURITY_GOVERNANCE")
    return render_template("security.html")

@app.route("/audit")
@login_required
@role_required("admin")
def audit_view():
    rows=query_db("""SELECT a.*,u.email,u.display_name,p.external_ref
                     FROM audit_logs a LEFT JOIN users u ON a.user_id=u.id
                     LEFT JOIN patients p ON a.patient_id=p.id
                     ORDER BY a.created_at DESC LIMIT 500""")
    return render_template("audit.html",rows=rows)

@app.route("/api/patient/<int:patient_id>/vitals")
@login_required
def api_vitals(patient_id):
    rows=query_db("""SELECT kind,value,unit,source,measured_at FROM vitals
                     WHERE patient_id=? ORDER BY measured_at DESC LIMIT 240""",(patient_id,))
    audit("API_READ_VITALS",patient_id)
    return jsonify([dict(r) for r in rows])

@app.route("/api/patient/<int:patient_id>/twin")
@login_required
def api_twin(patient_id):
    latest=get_latest_vitals(patient_id)
    risks=query_db("SELECT risk_type,score,level,explanation,created_at FROM risk_scores WHERE patient_id=? ORDER BY created_at DESC LIMIT 8",(patient_id,))
    return jsonify({"latest":latest,"risks":[dict(r) for r in risks]})

@app.route("/api/patient/<int:patient_id>/fhir")
@login_required
@role_required("admin","nurse","gp")
def api_fhir(patient_id):
    audit("FHIR_EXPORT",patient_id,"Synthetic FHIR R4-style demo bundle")
    return jsonify(patient_bundle(patient_id))

@app.route("/api/notifications/<int:patient_id>")
@login_required
def api_notifications(patient_id):
    rows=query_db("SELECT * FROM notifications WHERE patient_id=? ORDER BY created_at DESC LIMIT 20",(patient_id,))
    return jsonify([dict(r) for r in rows])

@app.route("/notify/<int:patient_id>", methods=["POST"])
@login_required
@role_required("admin","nurse","gp")
def notify(patient_id):
    channel=request.form.get("channel","app")
    audience=request.form.get("audience","patient")
    message=request.form.get("message","Care team would like to check in.")[:500]
    create_notification(patient_id,audience,channel,message,"human")
    audit("CREATE_NOTIFICATION",patient_id,f"{audience}/{channel}")
    return redirect(url_for("patient",patient_id=patient_id))

def scheduled_hospital_scan():
    try:
        init_hospital_schema()
        if hospital_demo_status()["loaded"]:
            scan_admissions(scan_type="daily", triggered_by="scheduler", actor_user_id=None)
    except Exception as exc:
        app.logger.warning("Scheduled hospital scan failed: %s", exc)


def start_optional_hospital_scheduler():
    if os.getenv("CAREAI_ENABLE_SCHEDULER", "0") != "1":
        return None
    # Flask's development reloader launches two processes. Start the scheduler
    # only in the serving child process so a daily scan cannot be duplicated.
    if os.getenv("CAREAI_ENV", "development") == "development" and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as exc:
        app.logger.warning("Hospital scheduler requested but APScheduler is unavailable: %s", exc)
        return None
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(scheduled_hospital_scan, "cron", hour=6, minute=0, id="daily_hospital_scan", replace_existing=True)
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    init_db(force_seed=True)
    # Initialize first risk snapshot for demo
    for p in query_db("SELECT id FROM patients WHERE active=1"):
        if not query_db("SELECT 1 FROM risk_scores WHERE patient_id=?",(p["id"],),one=True):
            compute_all_risks(p["id"])
    init_hospital_schema()
    start_optional_hospital_scheduler()
    app.run(host="127.0.0.1", port=5002, debug=True)
