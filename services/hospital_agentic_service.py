"""Human-controlled post-triage actions for Hospital Hybrid AI workflows."""

from __future__ import annotations

import json
import secrets
from datetime import datetime

from services.db import execute_db, query_db
from services.hospital_db import init_hospital_schema


def _actor(user_id):
    return query_db(
        "SELECT id,display_name,role,active FROM users WHERE id=? AND active=1",
        (user_id,), one=True,
    )


def _context(run_id):
    return query_db(
        """SELECT h.*,a.patient_id,a.ward,a.room,a.bed,a.encounter_ref,a.admission_reason,
                  r.status run_status,r.severity,r.started_at,r.approved_at,r.completed_at,r.summary,
                  p.first_name,p.last_name,p.external_ref,
                  an.severity anomaly_severity,an.score anomaly_score,an.headline,an.evidence,an.recommended_review,
                  u.display_name assigned_professional
           FROM hospital_agentic_context h
           JOIN agentic_runs r ON r.id=h.run_id
           JOIN hospital_admissions a ON a.id=h.admission_id
           JOIN patients p ON p.id=a.patient_id
           LEFT JOIN hospital_anomalies an ON an.id=h.anomaly_id
           LEFT JOIN users u ON u.id=h.assigned_user_id
           WHERE h.run_id=? AND r.module_key='hospital'""",
        (run_id,), one=True,
    )


def _family_eligibility(patient_id):
    family = query_db(
        """SELECT * FROM family_users
           WHERE patient_id=? AND active=1 ORDER BY id LIMIT 1""",
        (patient_id,), one=True,
    )
    if not family:
        return {"eligible": False, "reason": "No authorised family contact available", "contact": None}
    if not family["consent_granted"]:
        return {"eligible": False, "reason": "Family notification skipped because consent is not available", "contact": family}
    if family["expires_at"]:
        try:
            expiry = datetime.fromisoformat(str(family["expires_at"]).replace("Z", "+00:00"))
            now = datetime.now(expiry.tzinfo) if expiry.tzinfo else datetime.now()
            if expiry <= now:
                return {"eligible": False, "reason": "Family notification skipped because delegated consent has expired", "contact": family}
        except ValueError:
            return {"eligible": False, "reason": "Family notification skipped because consent expiry could not be validated", "contact": family}
    scope = {item.strip().lower() for item in (family["access_scope"] or "").split(",") if item.strip()}
    if not ({"alerts", "messages", "summary"} & scope):
        return {"eligible": False, "reason": "Family notification skipped because delegated access scope does not allow this update", "contact": family}
    return {"eligible": True, "reason": "Authorised family contact and consent verified", "contact": family}


def _new_room_code():
    return "HOSP-" + secrets.token_hex(4).upper()


def _seconds_between(start_value, end_value):
    def parse(value):
        if not value:
            return None
        text = str(value).replace("Z", "+00:00")
        for candidate in (text, text.replace(" ", "T")):
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                pass
        return None

    start, end = parse(start_value), parse(end_value)
    if not start or not end:
        return None
    try:
        return max(0, int((end - start).total_seconds()))
    except TypeError:
        # Mixed timezone-aware/naive data should not break a demo outcome record.
        return None


def _approval_row(run_id):
    return query_db(
        """SELECT a.*,u.display_name reviewer_name,u.role reviewer_role
           FROM agentic_approvals a JOIN users u ON u.id=a.reviewer_user_id
           WHERE a.run_id=? ORDER BY a.id DESC LIMIT 1""",
        (run_id,), one=True,
    )


def approve_hospital_run(run_id, user_id, note=""):
    init_hospital_schema()
    context = _context(run_id)
    if not context:
        raise RuntimeError("Hospital workflow not found")
    if context["run_status"] != "awaiting_approval":
        return hospital_run_state(run_id)

    actor = _actor(user_id)
    if not actor or actor["role"] not in ("nurse", "admin"):
        raise PermissionError("Only an authorised nurse or admin can approve this hospital workflow")
    if actor["role"] == "nurse" and context["assigned_user_id"] and actor["id"] != context["assigned_user_id"]:
        raise PermissionError(f"This hospital workflow is assigned to {context['assigned_professional']}")

    clean_note = (note or "").strip()[:500]
    patient_name = f"{context['first_name']} {context['last_name']}"

    execute_db(
        "INSERT INTO agentic_approvals(run_id,reviewer_user_id,decision,note) VALUES(?,?,?,?)",
        (run_id, user_id, "approved", clean_note),
    )
    approval = _approval_row(run_id)
    approval_time = approval["created_at"] if approval else None
    evidence = f"{actor['display_name']} reviewed the evidence and approved the proposed response."
    if clean_note:
        evidence += f" Reviewer note: {clean_note}"
    step7 = query_db("SELECT metadata_json FROM agentic_steps WHERE run_id=? AND step_no=7", (run_id,), one=True)
    try:
        step_meta = json.loads(step7["metadata_json"] or "{}") if step7 else {}
    except (TypeError, ValueError):
        step_meta = {}
    step_meta.update({"decision": "approved", "approved_by": actor["display_name"], "reviewer_note": clean_note})
    execute_db(
        """UPDATE agentic_steps
           SET status='completed',evidence_text=?,started_at=COALESCE(started_at,CURRENT_TIMESTAMP),completed_at=CURRENT_TIMESTAMP,
               metadata_json=?
           WHERE run_id=? AND step_no=7""",
        (evidence, json.dumps(step_meta, ensure_ascii=False), run_id),
    )
    execute_db(
        "UPDATE agentic_runs SET approved_by=?,approved_at=CURRENT_TIMESTAMP,status='responding' WHERE id=?",
        (user_id, run_id),
    )
    execute_db(
        """UPDATE agentic_workflow_metrics
           SET nurse_review_at=COALESCE(nurse_review_at,?),approval_at=?,workflow_completion_status='responding',updated_at=CURRENT_TIMESTAMP
           WHERE run_id=?""",
        (approval_time, approval_time, run_id),
    )

    # Step 8: Care coordination + demo-safe virtual/specialist consultation.
    execute_db(
        "UPDATE agentic_steps SET status='active',started_at=CURRENT_TIMESTAMP WHERE run_id=? AND step_no=8",
        (run_id,),
    )
    task_type = "urgent_hospital_review" if context["severity"] == "critical" else "hospital_clinical_review"
    task_id = execute_db(
        """INSERT INTO care_tasks(patient_id,task_type,priority,status,assigned_role,rationale,created_by)
           VALUES(?,?,?,?,?,?,?)""",
        (
            context["patient_id"], task_type, context["severity"], "open", "nurse",
            context["recommended_review"] or "Human inpatient clinical review required.", user_id,
        ),
    )
    tele_id = execute_db(
        """INSERT INTO teleconsultations(
             patient_id,requested_by_type,requested_by_id,clinician_role,clinician_name,status,reason,room_code,scheduled_at
           ) VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
        (
            context["patient_id"], "clinician", user_id, "gp", "Hospital specialist / virtual care",
            "requested", f"Demo hospital review linked to {context['encounter_ref']} · {context['headline'] or context['anomaly_type']}",
            _new_room_code(),
        ),
    )
    execute_db(
        "INSERT INTO agentic_actions(run_id,action_type,status,reference_table,reference_id,message,completed_at) VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        (run_id, "care_task", "created", "care_tasks", task_id, "Human-approved hospital clinical review task created"),
    )
    execute_db(
        "INSERT INTO agentic_actions(run_id,action_type,status,reference_table,reference_id,message,completed_at) VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        (run_id, "teleconsultation", "initiated", "teleconsultations", tele_id, "Demo specialist/virtual-care consultation initiated after human approval"),
    )
    execute_db(
        """UPDATE agentic_steps
           SET status='completed',completed_at=CURRENT_TIMESTAMP,evidence_text=?
           WHERE run_id=? AND step_no=8""",
        (
            f"Clinical review task #{task_id} created for {context['ward']} Room {context['room']}. "
            f"Demo specialist/virtual-care consultation #{tele_id} initiated after human approval.",
            run_id,
        ),
    )
    response_started = query_db(
        "SELECT completed_at FROM agentic_steps WHERE run_id=? AND step_no=8", (run_id,), one=True
    )["completed_at"]

    # Step 9: Medication context + consent-based family update.
    execute_db(
        "UPDATE agentic_steps SET status='active',started_at=CURRENT_TIMESTAMP WHERE run_id=? AND step_no=9",
        (run_id,),
    )
    omission_row = query_db(
        """SELECT COUNT(*) c FROM ehr_med_admin
           WHERE admission_id=? AND status='omitted' AND scheduled_at>=datetime('now','-24 hours')""",
        (context["admission_id"],), one=True,
    )
    omissions = int(omission_row["c"] if omission_row else 0)
    family = _family_eligibility(context["patient_id"])
    family_time = None
    if family["eligible"]:
        contact = family["contact"]
        message = (
            f"Care.AI has identified a significant change in {patient_name}'s hospital monitoring data. "
            "A nurse has reviewed the alert and initiated follow-up. Further updates will be shared if required."
        )
        notification_id = execute_db(
            """INSERT INTO notifications(patient_id,audience,channel,message,status,source)
               VALUES(?,?,?,?,?,?)""",
            (context["patient_id"], "family", "app", message, "queued", "Hospital Family Engagement Agent"),
        )
        execute_db(
            "INSERT INTO agentic_actions(run_id,action_type,status,reference_table,reference_id,message,completed_at) VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)",
            (run_id, "family_notification", "queued", "notifications", notification_id, f"Authorised update queued for {contact['display_name']}"),
        )
        family_time = query_db("SELECT created_at FROM notifications WHERE id=?", (notification_id,), one=True)["created_at"]
        family_text = f"Authorised family update queued for {contact['display_name']}."
    else:
        execute_db(
            "INSERT INTO agentic_actions(run_id,action_type,status,message,completed_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP)",
            (run_id, "family_notification", "skipped", family["reason"]),
        )
        family_text = family["reason"] + "."

    execute_db(
        """UPDATE agentic_steps
           SET status='completed',completed_at=CURRENT_TIMESTAMP,evidence_text=?
           WHERE run_id=? AND step_no=9""",
        (f"Medication review context: {omissions} omission(s) in the last 24 hours. {family_text}", run_id),
    )

    # Step 10: actual timing/outcome metrics, not a hardcoded response time.
    execute_db(
        "UPDATE agentic_steps SET status='active',started_at=CURRENT_TIMESTAMP WHERE run_id=? AND step_no=10",
        (run_id,),
    )
    response_seconds = _seconds_between(context["started_at"], response_started)
    intervention = "Human-approved inpatient review task + demo specialist/virtual-care consultation"
    outcome_text = "Human clinical review initiated; final clinical outcome pending"
    outcome_id = execute_db(
        """INSERT INTO care_outcomes(
             patient_id,run_id,response_time_seconds,intervention,outcome,avoidable_visit,escalation_type
           ) VALUES(?,?,?,?,?,?,?)""",
        (context["patient_id"], run_id, response_seconds, intervention, outcome_text, 0, "inpatient_review"),
    )
    execute_db(
        """UPDATE agentic_workflow_metrics
           SET event_detected_at=COALESCE(event_detected_at,?),nurse_review_at=COALESCE(nurse_review_at,?),
               approval_at=COALESCE(approval_at,?),response_started_at=?,family_notification_at=?,
               response_time_seconds=?,intervention=?,outcome_status='human_review_started',
               workflow_completion_status='completed',teleconsultation_status='initiated',
               operational_impact='Hospital anomaly prioritised with rules + ML + optional LLM/RAG; human-approved follow-up started',
               updated_at=CURRENT_TIMESTAMP
           WHERE run_id=?""",
        (
            context["started_at"], approval_time, approval_time, response_started, family_time,
            response_seconds, intervention, run_id,
        ),
    )
    execute_db(
        """UPDATE agentic_steps
           SET status='completed',completed_at=CURRENT_TIMESTAMP,evidence_text=?
           WHERE run_id=? AND step_no=10""",
        (
            f"Response started after {response_seconds if response_seconds is not None else 'unavailable'} seconds. "
            f"Outcome #{outcome_id} recorded; final clinical outcome remains pending human care-team assessment.",
            run_id,
        ),
    )
    execute_db(
        """UPDATE agentic_runs
           SET status='completed',completed_at=CURRENT_TIMESTAMP,
               summary=?
           WHERE id=?""",
        (
            f"Hospital Hybrid AI workflow completed for {patient_name}: rules/ML/optional AI support led to a human-approved inpatient follow-up workflow.",
            run_id,
        ),
    )
    if context["anomaly_id"]:
        execute_db("UPDATE hospital_anomalies SET status='human_reviewed' WHERE id=?", (context["anomaly_id"],))
    execute_db(
        "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
        (context["patient_id"], "hospital_hybrid_ai_completed", "Hospital AI", "Human-approved Hospital Hybrid AI workflow completed"),
    )
    return hospital_run_state(run_id)


def reject_hospital_run(run_id, user_id, note=""):
    init_hospital_schema()
    context = _context(run_id)
    if not context:
        raise RuntimeError("Hospital workflow not found")
    if context["run_status"] != "awaiting_approval":
        return hospital_run_state(run_id)

    actor = _actor(user_id)
    if not actor or actor["role"] not in ("nurse", "admin"):
        raise PermissionError("Only an authorised nurse or admin can reject this hospital workflow")
    if actor["role"] == "nurse" and context["assigned_user_id"] and actor["id"] != context["assigned_user_id"]:
        raise PermissionError(f"This hospital workflow is assigned to {context['assigned_professional']}")

    clean_note = (note or "").strip()[:500]
    execute_db(
        "INSERT INTO agentic_approvals(run_id,reviewer_user_id,decision,note) VALUES(?,?,?,?)",
        (run_id, user_id, "rejected", clean_note),
    )
    evidence = f"{actor['display_name']} rejected the proposed continuation."
    if clean_note:
        evidence += f" Reviewer note: {clean_note}"
    evidence += " No hospital task, teleconsultation or family update was created by this workflow."
    execute_db(
        """UPDATE agentic_steps
           SET status='rejected',evidence_text=?,completed_at=CURRENT_TIMESTAMP
           WHERE run_id=? AND step_no=7""",
        (evidence, run_id),
    )
    execute_db(
        """UPDATE agentic_steps
           SET status='cancelled',evidence_text='Cancelled because the responsible human reviewer rejected the proposed response.'
           WHERE run_id=? AND step_no IN (8,9,10)""",
        (run_id,),
    )
    execute_db(
        "UPDATE agentic_runs SET status='rejected',completed_at=CURRENT_TIMESTAMP,summary=? WHERE id=?",
        ("Hospital Hybrid AI recommendation rejected by the responsible human reviewer; no downstream action executed.", run_id),
    )
    execute_db(
        """UPDATE agentic_workflow_metrics
           SET nurse_review_at=CURRENT_TIMESTAMP,outcome_status='stopped_by_human',
               workflow_completion_status='rejected',teleconsultation_status='not_started',
               operational_impact='Human reviewer stopped the hospital workflow before downstream actions',updated_at=CURRENT_TIMESTAMP
           WHERE run_id=?""",
        (run_id,),
    )
    if context["anomaly_id"]:
        execute_db("UPDATE hospital_anomalies SET status='reviewed_no_action' WHERE id=?", (context["anomaly_id"],))
    return hospital_run_state(run_id)


def hospital_run_state(run_id):
    context = _context(run_id)
    if not context:
        return None
    run = query_db(
        """SELECT r.*,p.first_name,p.last_name,p.external_ref
           FROM agentic_runs r JOIN patients p ON p.id=r.patient_id WHERE r.id=?""",
        (run_id,), one=True,
    )
    steps = query_db("SELECT * FROM agentic_steps WHERE run_id=? ORDER BY step_no", (run_id,))
    outcome = query_db("SELECT * FROM care_outcomes WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,), one=True)
    approval = _approval_row(run_id)
    actions = query_db("SELECT * FROM agentic_actions WHERE run_id=? ORDER BY id", (run_id,))
    metrics = query_db("SELECT * FROM agentic_workflow_metrics WHERE run_id=?", (run_id,), one=True)
    return {
        "run": dict(run),
        "context": dict(context),
        "steps": [dict(step) for step in steps],
        "outcome": dict(outcome) if outcome else None,
        "approval": dict(approval) if approval else None,
        "actions": [dict(action) for action in actions],
        "metrics": dict(metrics) if metrics else {},
    }
