"""Hospital EHR monitoring and Hybrid AI workflow creation.

This service is isolated from the existing home-care Agentic Care engine. It reads
hospital-specific synthetic EHR tables, applies deterministic rules, adds an
optional ML anomaly signal, retrieves local demo protocols, and optionally uses
OpenAI for contextual triage/copilot explanation. All operational actions remain
blocked until a human approves the hospital run.
"""

from __future__ import annotations

import json
from services.db import query_db, execute_db
from services.hospital_db import init_hospital_schema
from services.ml_anomaly_service import ml_anomaly_signal
from services.rag_service import retrieve_protocols
from services.openai_clinical_ai import triage_agent, copilot_agent, ai_status


def latest_vitals(admission_id):
    row = query_db(
        "SELECT * FROM ehr_vitals WHERE admission_id=? ORDER BY measured_at DESC,id DESC LIMIT 1",
        (admission_id,), one=True,
    )
    return dict(row) if row else {}


def latest_lab(admission_id, code):
    row = query_db(
        """SELECT * FROM ehr_labs WHERE admission_id=? AND test_code=?
           ORDER BY collected_at DESC,id DESC LIMIT 1""",
        (admission_id, code), one=True,
    )
    return dict(row) if row else None


def creatinine_series(admission_id):
    return [
        dict(row) for row in query_db(
            """SELECT * FROM ehr_labs WHERE admission_id=? AND test_code='CREA'
               ORDER BY collected_at DESC,id DESC LIMIT 3""",
            (admission_id,),
        )
    ]


def med_omissions(admission_id):
    row = query_db(
        """SELECT COUNT(*) c FROM ehr_med_admin
           WHERE admission_id=? AND status='omitted'
             AND scheduled_at>=datetime('now','-24 hours')""",
        (admission_id,), one=True,
    )
    return int(row["c"] if row else 0)


def recent_fall(admission_id):
    return query_db(
        """SELECT * FROM ehr_nursing_notes WHERE admission_id=? AND fall_event=1
           ORDER BY observed_at DESC,id DESC LIMIT 1""",
        (admission_id,), one=True,
    )


def patient_context(admission_id):
    admission = query_db(
        """SELECT a.*,p.first_name,p.last_name,p.external_ref,p.birth_date,p.sex,p.city,p.id patient_id
           FROM hospital_admissions a JOIN patients p ON p.id=a.patient_id
           WHERE a.id=?""",
        (admission_id,), one=True,
    )
    if not admission:
        return None
    conditions = [
        dict(row) for row in query_db(
            "SELECT display,code FROM conditions WHERE patient_id=? AND active=1 ORDER BY display",
            (admission["patient_id"],),
        )
    ]
    medications = [
        dict(row) for row in query_db(
            "SELECT name,dose,schedule FROM medications WHERE patient_id=? AND active=1 ORDER BY id",
            (admission["patient_id"],),
        )
    ]
    return {"admission": dict(admission), "conditions": conditions, "medications": medications}


def ehr_context_for_ai(admission_id, hours=24):
    context = patient_context(admission_id)
    if not context:
        return None
    vitals = [
        dict(row) for row in query_db(
            """SELECT measured_at,heart_rate,spo2,resp_rate,systolic_bp,diastolic_bp,map,
                      temperature,oxygen_lpm,consciousness,source
               FROM ehr_vitals WHERE admission_id=?
               ORDER BY measured_at DESC,id DESC LIMIT 12""",
            (admission_id,),
        )
    ]
    labs = [
        dict(row) for row in query_db(
            """SELECT collected_at,test_code,test_name,value,unit,reference_low,reference_high,abnormal_flag
               FROM ehr_labs WHERE admission_id=? ORDER BY collected_at DESC,id DESC LIMIT 20""",
            (admission_id,),
        )
    ]
    meds = [
        dict(row) for row in query_db(
            """SELECT medication_name,dose,scheduled_at,administered_at,status,reason,high_risk
               FROM ehr_med_admin WHERE admission_id=? ORDER BY scheduled_at DESC,id DESC LIMIT 20""",
            (admission_id,),
        )
    ]
    notes = [
        dict(row) for row in query_db(
            """SELECT observed_at,note_type,note_text,mobility_status,pain_score,intake_ml,urine_output_ml,fall_event
               FROM ehr_nursing_notes WHERE admission_id=? ORDER BY observed_at DESC,id DESC LIMIT 10""",
            (admission_id,),
        )
    ]
    admission = context["admission"]
    return {
        "patient": {
            "id": admission["patient_id"],
            "external_ref": admission["external_ref"],
            "name": f"{admission['first_name']} {admission['last_name']}",
            "birth_date": admission["birth_date"],
            "sex": admission["sex"],
        },
        "encounter": {
            "encounter_ref": admission["encounter_ref"],
            "ward": admission["ward"],
            "room": admission["room"],
            "bed": admission["bed"],
            "admission_reason": admission["admission_reason"],
            "attending_physician": admission["attending_physician"],
            "admission_at": admission["admission_at"],
        },
        "conditions": context["conditions"],
        "home_medications": context["medications"],
        "vital_trend_latest_first": vitals,
        "labs_latest_first": labs,
        "medication_administration_latest_first": meds,
        "nursing_notes_latest_first": notes,
        "data_note": "Synthetic hospital demo EHR only; not real patient data.",
    }


def detect_anomalies(admission_id):
    if not patient_context(admission_id):
        return []
    vitals = latest_vitals(admission_id)
    if not vitals:
        return []
    findings = []

    def add(kind, severity, score, headline, evidence, recommendation, rule_inputs=None):
        findings.append({
            "type": kind,
            "severity": severity,
            "score": int(score),
            "headline": headline,
            "evidence": evidence,
            "recommendation": recommendation,
            "rule_inputs": rule_inputs or {},
        })

    spo2 = float(vitals.get("spo2", 100) or 100)
    rr = float(vitals.get("resp_rate", 16) or 16)
    oxygen = float(vitals.get("oxygen_lpm", 0) or 0)
    if spo2 < 92 or rr >= 24 or oxygen >= 2:
        score = min(100, max(70, 55 + max(0, 92 - spo2) * 8 + max(0, rr - 22) * 4 + oxygen * 4))
        severity = "critical" if score >= 85 else "high"
        add(
            "respiratory_deterioration", severity, score, "Respiratory deterioration pattern",
            f"SpO₂ {spo2:.0f}%, RR {rr:.0f}/min, oxygen {oxygen:.1f} L/min.",
            "Immediate human review; validate measurements and assess respiratory status according to local protocol.",
            {"spo2": spo2, "resp_rate": rr, "oxygen_lpm": oxygen},
        )

    temp = float(vitals.get("temperature", 36.8) or 36.8)
    hr = float(vitals.get("heart_rate", 75) or 75)
    sbp = float(vitals.get("systolic_bp", 130) or 130)
    lactate = latest_lab(admission_id, "LAC")
    wbc = latest_lab(admission_id, "WBC")
    infection_points, evidence_parts = 0, []
    if temp >= 38.3:
        infection_points += 25; evidence_parts.append(f"temperature {temp:.1f}°C")
    if hr >= 100:
        infection_points += 18; evidence_parts.append(f"HR {hr:.0f}")
    if rr >= 22:
        infection_points += 18; evidence_parts.append(f"RR {rr:.0f}")
    if sbp < 100:
        infection_points += 20; evidence_parts.append(f"SBP {sbp:.0f}")
    if lactate and lactate["value"] > 2:
        infection_points += 25; evidence_parts.append(f"lactate {lactate['value']:.1f}")
    if wbc and (wbc["value"] > 12 or wbc["value"] < 4):
        infection_points += 12; evidence_parts.append(f"WBC {wbc['value']:.1f}")
    if infection_points >= 45:
        score = min(100, infection_points)
        add(
            "infection_sepsis_pattern", "critical" if score >= 85 else "high", score,
            "Possible infection / sepsis deterioration pattern", ", ".join(evidence_parts),
            "Urgent human clinical review against the hospital's approved infection/sepsis pathway.",
            {
                "temperature": temp, "heart_rate": hr, "resp_rate": rr, "systolic_bp": sbp,
                "lactate": lactate["value"] if lactate else None,
                "wbc": wbc["value"] if wbc else None,
            },
        )

    map_value = float(vitals.get("map", 80) or 80)
    if sbp < 90 or map_value < 65:
        score = 92 if map_value < 60 else 82
        add(
            "hemodynamic_instability", "critical" if score >= 85 else "high", score,
            "Hemodynamic instability", f"SBP {sbp:.0f} mmHg, MAP {map_value:.0f} mmHg.",
            "Immediate bedside validation and human review; use local rapid-response criteria.",
            {"systolic_bp": sbp, "map": map_value},
        )

    creatinine = creatinine_series(admission_id)
    if len(creatinine) >= 2:
        latest_value = float(creatinine[0]["value"])
        previous = float(creatinine[1]["value"])
        delta = latest_value - previous
        if delta >= 26.5 or latest_value >= 1.5 * previous:
            score = 78 if delta < 60 else 88
            add(
                "aki_pattern", "critical" if score >= 85 else "high", score,
                "Acute kidney injury pattern",
                f"Creatinine increased from {previous:.0f} to {latest_value:.0f} µmol/L.",
                "Human review of renal trend, fluid balance, medication exposure and urine output.",
                {"creatinine_latest": latest_value, "creatinine_previous": previous, "delta": delta},
            )

    omissions = med_omissions(admission_id)
    if omissions >= 2:
        add(
            "medication_administration", "medium", 58, "Repeated medication omissions",
            f"{omissions} scheduled doses omitted in the last 24 hours.",
            "Medication administration reconciliation by the responsible nurse/pharmacist.",
            {"omissions_24h": omissions},
        )

    fall = recent_fall(admission_id)
    if fall:
        add(
            "inpatient_fall", "high", 76, "Recent inpatient fall event",
            f"{fall['note_text']} Mobility: {fall['mobility_status']}.",
            "Human post-fall assessment and review under the local inpatient fall pathway.",
            {"fall_event": True, "mobility_status": fall["mobility_status"]},
        )

    return sorted(findings, key=lambda row: row["score"], reverse=True)


def _insert_step(run_id, no, agent, action, evidence, status="pending", approval=0, meta=None):
    execute_db(
        """INSERT INTO agentic_steps(
             run_id,step_no,agent_name,problem_text,action_text,evidence_text,status,
             requires_human_approval,metadata_json,started_at,completed_at
           ) VALUES(?,?,?,?,?,?,?,?,?,
             CASE WHEN ?='completed' THEN CURRENT_TIMESTAMP ELSE NULL END,
             CASE WHEN ?='completed' THEN CURRENT_TIMESTAMP ELSE NULL END)""",
        (
            run_id, no, agent, "Hospital admitted-patient monitoring", action, evidence,
            status, approval, json.dumps(meta or {}, ensure_ascii=False, default=str), status, status,
        ),
    )


def _save_ai_result(run_id, step_no, agent_name, execution_mode, rule_score, ml_score, llm_model, confidence, output):
    execute_db(
        """INSERT INTO ai_agent_results(
             run_id,step_no,agent_name,execution_mode,rule_score,ml_score,llm_model,confidence,output_json
           ) VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            run_id, step_no, agent_name, execution_mode, rule_score, ml_score,
            llm_model, confidence, json.dumps(output, ensure_ascii=False, default=str),
        ),
    )


def _fmt_list(items):
    if not items:
        return "None documented."
    if isinstance(items, str):
        return items
    return " • ".join(str(item) for item in items)


def _assigned_nurse(actor_user_id=None):
    if actor_user_id:
        actor = query_db("SELECT id,display_name,role FROM users WHERE id=? AND active=1", (actor_user_id,), one=True)
        if actor and actor["role"] == "nurse":
            return actor
    return query_db(
        "SELECT id,display_name,role FROM users WHERE role='nurse' AND active=1 ORDER BY id LIMIT 1",
        one=True,
    )


def create_hospital_agent_run(admission_id, anomaly, anomaly_id=None, actor_user_id=None):
    context = patient_context(admission_id)
    if not context:
        raise RuntimeError("Hospital admission not found")
    admission = context["admission"]
    patient_id = admission["patient_id"]
    ehr = ehr_context_for_ai(admission_id)
    ml = ml_anomaly_signal(admission_id)
    query = f"{anomaly['type']} {anomaly['headline']} {anomaly['evidence']} {admission['admission_reason']}"
    protocols = retrieve_protocols(query, top_k=3)
    triage = triage_agent(ehr, anomaly, ml, protocols)
    copilot = copilot_agent(ehr, anomaly, ml, triage, protocols)

    rule_score = int(anomaly["score"])
    ml_score = int(ml.get("score", 0) or 0)
    hybrid_score = max(rule_score, int(round(0.7 * rule_score + 0.3 * ml_score)))
    severity = "critical" if hybrid_score >= 85 else "high" if hybrid_score >= 70 else "medium" if hybrid_score >= 45 else "low"
    if anomaly["severity"] == "critical":
        severity = "critical"
    elif anomaly["severity"] == "high" and severity != "critical":
        severity = "high"

    copilot_summary = copilot.get("one_line_summary") or f"{anomaly['headline']}: {anomaly['evidence']}"
    summary = f"{copilot_summary} Recommended human clinical review is required before downstream actions."
    assigned = _assigned_nurse(actor_user_id)
    run_id = execute_db(
        """INSERT INTO agentic_runs(
             patient_id,module_key,scenario,status,severity,created_by,summary
           ) VALUES(?,?,?,?,?,?,?)""",
        (
            patient_id, "hospital", f"Hospital EHR anomaly: {anomaly['type']}",
            "awaiting_approval", severity, actor_user_id, summary,
        ),
    )

    execute_db(
        """INSERT INTO hospital_agentic_context(
             run_id,admission_id,anomaly_id,anomaly_type,rule_score,ml_score,hybrid_score,assigned_user_id
           ) VALUES(?,?,?,?,?,?,?,?)""",
        (
            run_id, admission_id, anomaly_id, anomaly["type"], rule_score, ml_score,
            hybrid_score, assigned["id"] if assigned else None,
        ),
    )
    execute_db(
        """INSERT OR IGNORE INTO agentic_workflow_metrics(
             run_id,event_detected_at,workflow_completion_status,teleconsultation_status,outcome_status,operational_impact
           ) VALUES(?,CURRENT_TIMESTAMP,'awaiting_approval','not_started','pending','Hospital Hybrid AI workflow awaiting human review')""",
        (run_id,),
    )

    conditions = ", ".join(row["display"] for row in context["conditions"]) or "none recorded"
    medications = ", ".join(row["name"] for row in context["medications"]) or "none recorded"
    vitals = latest_vitals(admission_id)
    protocol_ids = ", ".join(str(p.get("id")) for p in protocols if p.get("id")) or "No protocol retrieved"
    ml_evidence = (
        f"Isolation Forest anomaly score {ml_score}/100 ({ml.get('label')}). Top deviations: {ml.get('explanation','n/a')}"
        if ml.get("available") else ml.get("explanation", "ML unavailable")
    )
    triage_evidence = (
        f"RULE {rule_score}/100 · ML {ml_score}/100 · HYBRID {hybrid_score}/100. "
        f"LLM priority: {triage.get('priority_suggestion','n/a')} · confidence {triage.get('confidence',0)}. "
        f"Why: {triage.get('why_priority','')}. Data quality: {_fmt_list(triage.get('contradictions_or_data_quality',[]))}. "
        f"Retrieved demo protocol evidence: {protocol_ids}."
    )
    copilot_evidence = (
        f"Summary: {copilot.get('one_line_summary','')}. What changed: {_fmt_list(copilot.get('what_changed',[]))}. "
        f"Relevant context: {_fmt_list(copilot.get('relevant_context',[]))}. "
        f"Evidence for escalation: {_fmt_list(copilot.get('evidence_for_escalation',[]))}. "
        f"Recommended HUMAN next action: {copilot.get('recommended_next_human_action','')}. "
        f"Questions: {_fmt_list(copilot.get('questions_for_clinician',[]))}. "
        f"Data gaps: {_fmt_list(copilot.get('data_gaps',[]))}. Sources: {protocol_ids}."
    )
    shared = {
        "admission_id": admission_id,
        "anomaly_id": anomaly_id,
        "anomaly": anomaly,
        "ml_signal": ml,
        "triage": triage,
        "copilot": copilot,
        "protocols": protocols,
        "hybrid_score": hybrid_score,
        "assigned_user_id": assigned["id"] if assigned else None,
        "assigned_professional": assigned["display_name"] if assigned else "Responsible nurse",
        "ai_status": ai_status(),
    }

    steps = [
        (1, "Hospital Monitoring Agent", "Aggregate bedside vitals, labs, medication administration and nursing observations.",
         f"Latest vitals: SpO₂ {vitals.get('spo2')}%, HR {vitals.get('heart_rate')}, RR {vitals.get('resp_rate')}, BP {vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')}, temperature {vitals.get('temperature')}°C, oxygen {vitals.get('oxygen_lpm')} L/min.", 0),
        (2, "Hospital Digital Twin", "Resolve ward, room, bed and current encounter context.",
         f"{admission['ward']} · Room {admission['room']} · Bed {admission['bed']} · Encounter {admission['encounter_ref']} · Attending {admission['attending_physician']}.", 0),
        (3, "Clinical Anomaly Agent", "Apply deterministic clinical rules and interpretable ML anomaly detection.",
         f"Rule finding: {anomaly['headline']} — {anomaly['evidence']} Rule score {rule_score}/100. {ml_evidence}", 0),
        (4, "Patient Twin", "Add diagnoses, medications, admission reason and recent EHR context.",
         f"Admission: {admission['admission_reason']}. Conditions: {conditions}. Medications: {medications}. 24h EHR context assembled.", 0),
        (5, "AI Triage Agent", "Combine deterministic safety floor + ML signal + optional contextual LLM analysis.", triage_evidence, 0),
        (6, "Clinical Copilot", "Use retrieved demo protocol context and optional LLM analysis to prepare an evidence-grounded handoff.", copilot_evidence, 0),
        (7, "Responsible Nurse", "Review source evidence, rules, ML signal and AI output; approve or reject the proposed response.",
         f"Human approval required. Assigned reviewer: {assigned['display_name'] if assigned else 'responsible nurse'}. AI output is decision support only.", 1),
        (8, "Care Coordination + Virtual Care", "After approval, create a clinical review task and a demo specialist/virtual-care consultation.",
         "Prepared; blocked pending human approval.", 0),
        (9, "Medication + Family Engagement", "After approval, review medication-administration exceptions and notify authorised family only when consent permits.",
         "Prepared; blocked pending human approval.", 0),
        (10, "Outcome Agent", "Record actual workflow timestamps, response time, intervention and operational status.",
         "Prepared; blocked pending human approval.", 0),
    ]
    for no, agent, action, evidence, approval in steps:
        status = "completed" if no <= 6 else ("awaiting_approval" if no == 7 else "pending")
        _insert_step(run_id, no, agent, action, evidence, status, approval, shared)

    _save_ai_result(run_id, 3, "Clinical Anomaly Agent", "rules+ml", rule_score, ml_score, None, None, {"rule": anomaly, "ml": ml})
    _save_ai_result(run_id, 5, "AI Triage Agent", triage.get("mode", "unknown"), rule_score, ml_score, triage.get("model"), triage.get("confidence"), triage)
    _save_ai_result(run_id, 6, "Clinical Copilot", copilot.get("mode", "unknown"), rule_score, ml_score, copilot.get("model"), None, {"copilot": copilot, "retrieved_protocols": protocols})
    execute_db(
        "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
        (patient_id, "hospital_hybrid_ai_started", "Hospital AI", f"Hospital Hybrid AI workflow started for {anomaly['headline']}"),
    )
    return run_id


def scan_admissions(scan_type="adhoc", triggered_by="manual", actor_user_id=None):
    init_hospital_schema()
    scan_id = execute_db(
        "INSERT INTO hospital_scans(scan_type,status,triggered_by) VALUES(?,?,?)",
        (scan_type, "running", triggered_by),
    )
    admissions = query_db("SELECT * FROM hospital_admissions WHERE status='admitted' ORDER BY id")
    anomaly_count = 0
    run_count = 0
    for admission in admissions:
        findings = detect_anomalies(admission["id"])
        # Supersede prior open results for this admission so the dashboard reflects
        # the latest scan rather than accumulating stale alerts.
        execute_db(
            "UPDATE hospital_anomalies SET status='superseded' WHERE admission_id=? AND status='open'",
            (admission["id"],),
        )
        inserted = []
        for finding in findings:
            anomaly_count += 1
            anomaly_id = execute_db(
                """INSERT INTO hospital_anomalies(
                     scan_id,admission_id,patient_id,anomaly_type,severity,score,headline,evidence,recommended_review
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    scan_id, admission["id"], admission["patient_id"], finding["type"],
                    finding["severity"], finding["score"], finding["headline"],
                    finding["evidence"], finding["recommendation"],
                ),
            )
            inserted.append((finding, anomaly_id))

        if inserted and inserted[0][0]["score"] >= 70:
            top_finding, anomaly_id = inserted[0]
            duplicate = query_db(
                """SELECT 1
                   FROM hospital_agentic_context h
                   JOIN agentic_runs r ON r.id=h.run_id
                   WHERE h.admission_id=? AND h.anomaly_type=?
                     AND r.status IN ('awaiting_approval','responding','running')
                   ORDER BY r.id DESC LIMIT 1""",
                (admission["id"], top_finding["type"]), one=True,
            )
            if not duplicate:
                create_hospital_agent_run(
                    admission["id"], top_finding, anomaly_id=anomaly_id,
                    actor_user_id=actor_user_id,
                )
                run_count += 1

    execute_db(
        """UPDATE hospital_scans
           SET status='completed',completed_at=CURRENT_TIMESTAMP,patients_scanned=?,anomalies_found=?,runs_created=?
           WHERE id=?""",
        (len(admissions), anomaly_count, run_count, scan_id),
    )
    return scan_id


def hospital_dashboard_rows():
    init_hospital_schema()
    return query_db(
        """SELECT a.*,p.first_name,p.last_name,p.external_ref,
             (SELECT spo2 FROM ehr_vitals v WHERE v.admission_id=a.id ORDER BY v.measured_at DESC,v.id DESC LIMIT 1) spo2,
             (SELECT heart_rate FROM ehr_vitals v WHERE v.admission_id=a.id ORDER BY v.measured_at DESC,v.id DESC LIMIT 1) heart_rate,
             (SELECT resp_rate FROM ehr_vitals v WHERE v.admission_id=a.id ORDER BY v.measured_at DESC,v.id DESC LIMIT 1) resp_rate,
             (SELECT systolic_bp FROM ehr_vitals v WHERE v.admission_id=a.id ORDER BY v.measured_at DESC,v.id DESC LIMIT 1) systolic_bp,
             (SELECT diastolic_bp FROM ehr_vitals v WHERE v.admission_id=a.id ORDER BY v.measured_at DESC,v.id DESC LIMIT 1) diastolic_bp,
             (SELECT severity FROM hospital_anomalies h WHERE h.admission_id=a.id AND h.status='open' ORDER BY h.score DESC,h.id DESC LIMIT 1) anomaly_severity,
             (SELECT score FROM hospital_anomalies h WHERE h.admission_id=a.id AND h.status='open' ORDER BY h.score DESC,h.id DESC LIMIT 1) anomaly_score,
             (SELECT headline FROM hospital_anomalies h WHERE h.admission_id=a.id AND h.status='open' ORDER BY h.score DESC,h.id DESC LIMIT 1) anomaly_headline
           FROM hospital_admissions a JOIN patients p ON p.id=a.patient_id
           WHERE a.status='admitted'
           ORDER BY CASE COALESCE(anomaly_severity,'low')
             WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
             COALESCE(anomaly_score,0) DESC,p.last_name"""
    )


def admission_detail(admission_id):
    context = patient_context(admission_id)
    if not context:
        return None
    vitals = query_db("SELECT * FROM ehr_vitals WHERE admission_id=? ORDER BY measured_at DESC,id DESC LIMIT 16", (admission_id,))
    labs = query_db("SELECT * FROM ehr_labs WHERE admission_id=? ORDER BY collected_at DESC,id DESC LIMIT 20", (admission_id,))
    meds = query_db("SELECT * FROM ehr_med_admin WHERE admission_id=? ORDER BY scheduled_at DESC,id DESC LIMIT 20", (admission_id,))
    notes = query_db("SELECT * FROM ehr_nursing_notes WHERE admission_id=? ORDER BY observed_at DESC,id DESC LIMIT 12", (admission_id,))
    anomalies = query_db("SELECT * FROM hospital_anomalies WHERE admission_id=? ORDER BY created_at DESC,score DESC,id DESC LIMIT 20", (admission_id,))
    runs = query_db(
        """SELECT r.* FROM agentic_runs r
           JOIN hospital_agentic_context h ON h.run_id=r.id
           WHERE h.admission_id=? ORDER BY r.id DESC LIMIT 10""",
        (admission_id,),
    )
    return context, vitals, labs, meds, notes, anomalies, runs


def agent_results(run_id):
    rows = []
    for row in query_db("SELECT * FROM ai_agent_results WHERE run_id=? ORDER BY step_no,id", (run_id,)):
        item = dict(row)
        try:
            item["output"] = json.loads(item.get("output_json") or "{}")
        except Exception:
            item["output"] = {}
        rows.append(item)
    return rows
