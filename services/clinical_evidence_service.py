"""Shared clinical-review evidence for CareAI clinician-facing pages.

This module does not create clinical scores. It reads the scores, workflow results,
EHR/wearable values and decision traces already produced by CareAI, then presents
one consistent explanation model to Patient 360, Hospital AI and Command Centre.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

from services.db import query_db
from services.risk_engine import (
    RISK_LEVEL_THRESHOLDS,
    latest_vital_row,
    risk_rule_metadata,
)

_PRIORITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}
_STALE_AFTER_HOURS = 24  # UI freshness marker only, not a clinical threshold.


def _dict(row):
    return dict(row) if row else None


def _loads(value):
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _parse_dt(value):
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T")):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is not None:
                return parsed.astimezone().replace(tzinfo=None)
            return parsed
        except ValueError:
            pass
    return None


def _freshness(value):
    dt = _parse_dt(value)
    if not dt:
        return {"state": "missing", "label": "Timestamp unavailable", "age_hours": None}
    hours = max(0.0, (datetime.now() - dt).total_seconds() / 3600.0)
    if hours > _STALE_AFTER_HOURS:
        return {"state": "stale", "label": f"Stale · {hours:.0f}h old", "age_hours": round(hours, 1)}
    return {"state": "current", "label": f"Current · {hours:.1f}h old", "age_hours": round(hours, 1)}


def _age(birth_date):
    try:
        born = datetime.strptime(birth_date or "", "%Y-%m-%d").date()
    except ValueError:
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _patient(patient_id):
    return query_db(
        """SELECT p.*,n.display_name assigned_nurse_name,c.display_name assigned_clinician_name
           FROM patients p
           LEFT JOIN users n ON n.id=p.assigned_nurse_id
           LEFT JOIN users c ON c.id=p.assigned_clinician_id
           WHERE p.id=?""",
        (patient_id,), one=True,
    )


def _latest_vital_row(patient_id, kind, as_of=None):
    # Shared with the Risk Engine so explanation and calculation
    # always refer to the same measurement.
    return latest_vital_row(patient_id, kind, as_of=as_of)
def _metric(row, label):
    if not row:
        return {
            "key": None, "label": label, "value": None, "unit": "", "source": "No data",
            "measured_at": None, "freshness": {"state": "missing", "label": "Missing", "age_hours": None},
        }
    item = dict(row)
    item.update({"label": label, "freshness": _freshness(item.get("measured_at"))})
    return item


def _home_vitals(patient_id, as_of=None):
    wanted = [
        ("spo2", "SpO₂"), ("heart_rate", "Heart rate"), ("resp_rate", "Respiratory rate"),
        ("bp_sys", "Systolic BP"), ("bp_dia", "Diastolic BP"), ("temperature", "Temperature"),
        ("activity", "Activity"), ("sleep", "Sleep"),
    ]
    result = []
    for kind, label in wanted:
        row = _latest_vital_row(patient_id, kind, as_of=as_of)
        item = _metric(row, label)
        item["key"] = kind
        result.append(item)
    return result


def _conditions(patient_id):
    return [
        dict(row) for row in query_db(
            "SELECT code,display,onset_date FROM conditions WHERE patient_id=? AND active=1 ORDER BY display",
            (patient_id,),
        )
    ]


def _medications(patient_id):
    return [
        dict(row) for row in query_db(
            "SELECT name,dose,schedule FROM medications WHERE patient_id=? AND active=1 ORDER BY name",
            (patient_id,),
        )
    ]


def _risk_rows(patient_id):
    return query_db(
        """SELECT r.* FROM risk_scores r
           JOIN (SELECT risk_type,MAX(id) max_id FROM risk_scores WHERE patient_id=? GROUP BY risk_type) x
             ON x.max_id=r.id
           WHERE r.patient_id=?
           ORDER BY CASE r.risk_type
             WHEN 'overall' THEN 0 WHEN 'fall_prediction' THEN 1 WHEN 'fall_event' THEN 2
             WHEN 'copd' THEN 3 WHEN 'medication' THEN 4 WHEN 'cardiac' THEN 5
             WHEN 'infection' THEN 6 WHEN 'frailty' THEN 7 ELSE 8 END""",
        (patient_id, patient_id),
    )


def _risk_input_snapshot(patient_id, risk_type, as_of):
    vitals = {item["key"]: item for item in _home_vitals(patient_id, as_of=as_of)}
    conditions = [row["display"] for row in _conditions(patient_id)]
    condition_text = " | ".join(conditions) if conditions else "None recorded"

    def v(kind):
        return vitals.get(kind, {})

    def show(kind):
        item = v(kind)
        if item.get("value") is None:
            return "Missing"
        return f"{item['value']:g} {item.get('unit') or ''}".strip()

    inputs = []
    if risk_type == "fall_prediction":
        inputs = [
            {"label": "Conditions", "value": condition_text, "source": "conditions"},
            {"label": "Activity", "value": show("activity"), "source": v("activity").get("source") or "vitals"},
            {"label": "Sleep", "value": show("sleep"), "source": v("sleep").get("source") or "vitals"},
        ]
        recent = query_db(
            """SELECT event_type,location,created_at FROM device_events
               WHERE patient_id=? AND event_type='fall'
                 AND datetime(created_at)<datetime(?,'-2 minutes')
                 AND datetime(created_at)>=datetime(?,'-30 days')
               ORDER BY datetime(created_at) DESC,id DESC LIMIT 1""",
            (patient_id, as_of, as_of), one=True,
        )
        inputs.append({
            "label": "Recent fall history",
            "value": f"Recorded {recent['created_at']}" if recent else "None recorded in device events",
            "source": "device_events",
        })
    elif risk_type == "fall_event":
        fall = query_db(
            """SELECT * FROM fall_assessments WHERE patient_id=? AND assessment_type='detection'
               AND status!='cleared'
               AND datetime(created_at)<=datetime(?)
               AND datetime(created_at)>=datetime(?,'-30 minutes')
               ORDER BY datetime(created_at) DESC,id DESC LIMIT 1""",
            (patient_id, as_of, as_of), one=True,
        )
        if fall:
            inputs = [
                {"label": "Fall assessment", "value": f"{fall['score']}/100 · {fall['status']}", "source": "fall_assessments"},
                {"label": "Impact", "value": f"{fall['impact_g']} g" if fall['impact_g'] is not None else "Not recorded", "source": "fall_assessments"},
                {"label": "Orientation", "value": f"{fall['orientation_change_deg']}°" if fall['orientation_change_deg'] is not None else "Not recorded", "source": "fall_assessments"},
                {"label": "Immobility", "value": f"{fall['immobility_seconds']} s" if fall['immobility_seconds'] is not None else "Not recorded", "source": "fall_assessments"},
                {"label": "Posture", "value": fall['camera_posture'] or "Not recorded", "source": "fall_assessments"},
            ]
        else:
            inputs = [{"label": "Fall assessment", "value": "No detection assessment available", "source": "fall_assessments"}]
    elif risk_type == "copd":
        inputs = [
            {"label": "Conditions", "value": condition_text, "source": "conditions"},
            {"label": "SpO₂", "value": show("spo2"), "source": v("spo2").get("source") or "vitals"},
            {"label": "Respiratory rate", "value": show("resp_rate"), "source": v("resp_rate").get("source") or "vitals"},
            {"label": "Heart rate", "value": show("heart_rate"), "source": v("heart_rate").get("source") or "vitals"},
            {"label": "Activity", "value": show("activity"), "source": v("activity").get("source") or "vitals"},
        ]
    elif risk_type == "medication":
        missed = query_db(
            """SELECT COUNT(*) c FROM medication_events WHERE patient_id=? AND status='missed'
               AND datetime(scheduled_at)>=datetime(?,'-3 days') AND datetime(scheduled_at)<=datetime(?)""",
            (patient_id, as_of, as_of), one=True,
        )
        inputs = [{"label": "Missed doses in prior 3 days", "value": str(int(missed["c"] if missed else 0)), "source": "medication_events"}]
    elif risk_type == "cardiac":
        inputs = [
            {"label": "Conditions", "value": condition_text, "source": "conditions"},
            {"label": "Heart rate", "value": show("heart_rate"), "source": v("heart_rate").get("source") or "vitals"},
            {"label": "Systolic BP", "value": show("bp_sys"), "source": v("bp_sys").get("source") or "vitals"},
            {"label": "Activity", "value": show("activity"), "source": v("activity").get("source") or "vitals"},
        ]
    elif risk_type == "infection":
        inputs = [
            {"label": "Temperature", "value": show("temperature"), "source": v("temperature").get("source") or "vitals"},
            {"label": "Respiratory rate", "value": show("resp_rate"), "source": v("resp_rate").get("source") or "vitals"},
            {"label": "Heart rate", "value": show("heart_rate"), "source": v("heart_rate").get("source") or "vitals"},
            {"label": "Activity", "value": show("activity"), "source": v("activity").get("source") or "vitals"},
        ]
    elif risk_type == "frailty":
        inputs = [
            {"label": "Conditions", "value": condition_text, "source": "conditions"},
            {"label": "Activity", "value": show("activity"), "source": v("activity").get("source") or "vitals"},
            {"label": "Sleep", "value": show("sleep"), "source": v("sleep").get("source") or "vitals"},
        ]
    return inputs


def risk_breakdown(patient_id):
    patient = query_db("SELECT lawful_basis FROM patients WHERE id=?", (patient_id,), one=True)
    patient_is_demo = bool(patient and (patient["lawful_basis"] or "").lower() == "demo_synthetic")
    result = []
    for row in _risk_rows(patient_id):
        item = dict(row)
        if item["risk_type"] == "overall":
            continue
        meta = risk_rule_metadata(item["risk_type"])
        inputs = _risk_input_snapshot(patient_id, item["risk_type"], item["created_at"])
        source_values = " ".join(str(x.get("source") or "").lower() for x in inputs)
        demo = patient_is_demo or any(token in source_values for token in ("simulator", "scenario", "synthetic", "demo"))
        item.update({
            "label": meta["label"],
            "agent": meta["agent"],
            "calculation": meta["calculation"],
            "implementation": meta["kind"],
            "data_sources": meta["sources"],
            "inputs": inputs,
            "origin_label": "Demo / synthetic inputs" if demo else "Recorded data / rule calculation",
            "freshness": _freshness(item["created_at"]),
        })
        result.append(item)
    return result


def _workflow_row(patient_id, run_id=None, active_only=False):
    where = "r.id=? AND r.patient_id=?" if run_id else "r.patient_id=?"
    args = (run_id, patient_id) if run_id else (patient_id,)
    status_clause = " AND r.status IN ('running','awaiting_approval','responding')" if active_only else ""
    order_clause = (
        "ORDER BY CASE r.severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC, "
        "COALESCE(h.hybrid_score,e.triage_score,e.sensor_confidence,0) DESC, r.id DESC"
        if active_only else "ORDER BY r.id DESC"
    )
    try:
        return query_db(
            f"""SELECT r.*,h.admission_id,h.anomaly_id,h.anomaly_type,h.rule_score,h.ml_score,h.hybrid_score,
                       h.assigned_user_id,u.display_name assigned_professional,u.role assigned_role,
                       e.triage_score,e.sensor_confidence,e.event_label,e.location event_location
                FROM agentic_runs r
                LEFT JOIN hospital_agentic_context h ON h.run_id=r.id
                LEFT JOIN users u ON u.id=h.assigned_user_id
                LEFT JOIN agentic_event_context e ON e.run_id=r.id
                WHERE {where} {status_clause}
                {order_clause} LIMIT 1""",
            args, one=True,
        )
    except sqlite3.OperationalError:
        return query_db(
            f"""SELECT r.*,NULL admission_id,NULL anomaly_id,NULL anomaly_type,NULL rule_score,NULL ml_score,NULL hybrid_score,
                       NULL assigned_user_id,NULL assigned_professional,NULL assigned_role,
                       e.triage_score,e.sensor_confidence,e.event_label,e.location event_location
                FROM agentic_runs r
                LEFT JOIN agentic_event_context e ON e.run_id=r.id
                WHERE {where} {status_clause}
                {order_clause} LIMIT 1""",
            args, one=True,
        )


def priority_snapshot(patient_id):
    home = query_db(
        """SELECT score,level,explanation,evidence,created_at FROM risk_scores
           WHERE patient_id=? AND risk_type='overall' ORDER BY id DESC LIMIT 1""",
        (patient_id,), one=True,
    )
    candidates = []
    if home:
        candidates.append({
            "level": (home["level"] or "low").lower(), "score": int(home["score"] or 0),
            "reason": home["explanation"] or home["evidence"] or "CareAI risk assessment",
            "source": "Care intelligence", "run_id": None, "module_key": None,
            "updated_at": home["created_at"],
        })
    workflow = _workflow_row(patient_id, active_only=True)
    if workflow and (workflow["severity"] or "").lower() in _PRIORITY_RANK:
        if workflow["module_key"] == "hospital":
            score = int(workflow["hybrid_score"] or workflow["rule_score"] or 0)
            reason = workflow["summary"] or f"Hospital AI {workflow['anomaly_type'] or 'clinical finding'}"
            source = "Hospital AI workflow"
        else:
            score = int(workflow["triage_score"] or workflow["sensor_confidence"] or 0)
            reason = workflow["summary"] or workflow["scenario"]
            source = "Agentic Care workflow"
        candidates.append({
            "level": workflow["severity"].lower(), "score": score, "reason": reason,
            "source": source, "run_id": workflow["id"], "module_key": workflow["module_key"],
            "updated_at": workflow["started_at"],
        })
    if not candidates:
        return {"level": "low", "score": 0, "reason": "No current risk assessment", "source": "No assessment", "run_id": None, "module_key": None, "updated_at": None}
    return max(candidates, key=lambda row: (_PRIORITY_RANK.get(row["level"], 1), row["score"]))


def _hospital_detail(workflow):
    if not workflow or workflow["module_key"] != "hospital" or not workflow["admission_id"]:
        return None
    admission = query_db(
        """SELECT a.*,p.first_name,p.last_name,p.external_ref,p.birth_date,p.sex,p.city
           FROM hospital_admissions a JOIN patients p ON p.id=a.patient_id WHERE a.id=?""",
        (workflow["admission_id"],), one=True,
    )
    if not admission:
        return None
    vitals = query_db(
        "SELECT * FROM ehr_vitals WHERE admission_id=? ORDER BY datetime(measured_at) DESC,id DESC LIMIT 1",
        (workflow["admission_id"],), one=True,
    )
    labs = [dict(r) for r in query_db(
        "SELECT * FROM ehr_labs WHERE admission_id=? ORDER BY datetime(collected_at) DESC,id DESC LIMIT 12",
        (workflow["admission_id"],),
    )]
    anomaly = query_db("SELECT * FROM hospital_anomalies WHERE id=?", (workflow["anomaly_id"],), one=True) if workflow["anomaly_id"] else None
    if not anomaly:
        anomaly = query_db(
            "SELECT * FROM hospital_anomalies WHERE admission_id=? ORDER BY score DESC,id DESC LIMIT 1",
            (workflow["admission_id"],), one=True,
        )
    med_admin = [dict(r) for r in query_db(
        "SELECT * FROM ehr_med_admin WHERE admission_id=? ORDER BY datetime(scheduled_at) DESC,id DESC LIMIT 8",
        (workflow["admission_id"],),
    )]
    nursing = [dict(r) for r in query_db(
        "SELECT * FROM ehr_nursing_notes WHERE admission_id=? ORDER BY datetime(observed_at) DESC,id DESC LIMIT 5",
        (workflow["admission_id"],),
    )]
    return {
        "admission": dict(admission), "vitals": dict(vitals) if vitals else None, "labs": labs,
        "anomaly": dict(anomaly) if anomaly else None, "med_admin": med_admin, "nursing": nursing,
    }


def _hospital_metrics(detail):
    if not detail or not detail.get("vitals"):
        return []
    v = detail["vitals"]
    specs = [
        ("heart_rate", "Heart rate", "bpm"), ("spo2", "SpO₂", "%"), ("resp_rate", "Respiratory rate", "/min"),
        ("systolic_bp", "Systolic BP", "mmHg"), ("diastolic_bp", "Diastolic BP", "mmHg"),
        ("temperature", "Temperature", "°C"), ("oxygen_lpm", "Oxygen", "L/min"),
    ]
    return [
        {
            "key": key, "label": label, "value": v.get(key), "unit": unit,
            "source": v.get("source") or "hospital EHR", "measured_at": v.get("measured_at"),
            "freshness": _freshness(v.get("measured_at")),
        }
        for key, label, unit in specs
    ]


def _hospital_rule_result(run_id):
    row = query_db(
        "SELECT output_json FROM ai_agent_results WHERE run_id=? AND step_no=3 ORDER BY id DESC LIMIT 1",
        (run_id,), one=True,
    )
    return _loads(row["output_json"] if row else None)


def _triage_result(run_id):
    row = query_db(
        "SELECT execution_mode,confidence,llm_model,output_json FROM ai_agent_results WHERE run_id=? AND step_no=5 ORDER BY id DESC LIMIT 1",
        (run_id,), one=True,
    )
    if not row:
        return None
    data = dict(row)
    data["output"] = _loads(row["output_json"])
    return data


def _workflow_steps(run_id):
    return [dict(row) for row in query_db("SELECT * FROM agentic_steps WHERE run_id=? ORDER BY step_no", (run_id,))]


def _why_hospital_priority(workflow, detail, rule_result):
    anomaly = (rule_result or {}).get("rule") or (detail or {}).get("anomaly") or {}
    inputs = anomaly.get("rule_inputs") or {}
    labels = {
        "temperature": ("Temperature", "°C"), "heart_rate": ("Heart rate", "bpm"),
        "resp_rate": ("Respiratory rate", "/min"), "systolic_bp": ("Systolic BP", "mmHg"),
        "lactate": ("Lactate", "mmol/L"), "wbc": ("White blood cells", "10^9/L"),
        "map": ("MAP", "mmHg"), "omissions_24h": ("Medication omissions in 24h", ""),
        "creatinine_latest": ("Latest creatinine", "µmol/L"), "creatinine_previous": ("Previous creatinine", "µmol/L"),
        "delta": ("Creatinine change", "µmol/L"), "fall_event": ("Inpatient fall", ""),
        "mobility_status": ("Mobility", ""), "spo2": ("SpO₂", "%"), "oxygen_lpm": ("Oxygen", "L/min"),
    }
    factors = []
    for key, value in inputs.items():
        if value is None:
            continue
        label, unit = labels.get(key, (key.replace("_", " ").title(), ""))
        if isinstance(value, bool):
            value_text = "Detected" if value else "Not detected"
        elif isinstance(value, (int, float)):
            value_text = f"{value:g}{(' ' + unit) if unit else ''}"
        else:
            value_text = f"{value}{(' ' + unit) if unit else ''}"
        factors.append({"label": label, "value": value_text})
    if not factors and anomaly.get("evidence"):
        factors = [{"label": "Rule evidence", "value": anomaly["evidence"]}]
    return factors


def build_clinical_evidence(patient_id, run_id=None):
    patient = _patient(patient_id)
    if not patient:
        return None
    p = dict(patient)
    workflow = _workflow_row(patient_id, run_id=run_id, active_only=False) if run_id else _workflow_row(patient_id, active_only=True)
    workflow = dict(workflow) if workflow else None
    priority = priority_snapshot(patient_id)

    # A workflow is allowed to influence CURRENT patient priority only
    # while it is operationally active. Completed/rejected workflows
    # remain available as history but must not override current risk.
    active_workflow_statuses = {
        "running",
        "awaiting_approval",
        "responding",
    }

    workflow_status = (
        (workflow.get("status") or "").lower()
        if workflow else ""
    )

    workflow_is_active = (
        workflow_status in active_workflow_statuses
    )

    if workflow and run_id and workflow_is_active:
        if workflow["module_key"] == "hospital":
            priority = {
                "level": (workflow["severity"] or "low").lower(),
                "score": int(workflow.get("hybrid_score") or workflow.get("rule_score") or 0),
                "reason": workflow.get("summary") or workflow.get("scenario"),
                "source": "Hospital AI workflow", "run_id": workflow["id"], "module_key": "hospital",
                "updated_at": workflow.get("started_at"),
            }
        else:
            priority = {
                "level": (workflow["severity"] or "low").lower(),
                "score": int(workflow.get("triage_score") or workflow.get("sensor_confidence") or 0),
                "reason": workflow.get("summary") or workflow.get("scenario"),
                "source": "Agentic Care workflow", "run_id": workflow["id"], "module_key": workflow.get("module_key"),
                "updated_at": workflow.get("started_at"),
            }

    detail = _hospital_detail(workflow)
    home_vitals = _home_vitals(patient_id)

    # Historical Hospital AI runs remain available for audit/history,
    # but their old inpatient observations must not masquerade as the
    # patient's current Patient 360 measurements.
    inpatient_vitals = (
        _hospital_metrics(detail)
        if workflow_is_active else []
    )

    clinical_data = (
        inpatient_vitals
        if inpatient_vitals else home_vitals
    )
    conditions = _conditions(patient_id)
    medications = _medications(patient_id)
    risks = risk_breakdown(patient_id)

    missing = []
    for metric in clinical_data:
        if metric.get("value") is None:
            missing.append(f"{metric['label']} is not available")
        elif metric.get("freshness", {}).get("state") == "stale":
            missing.append(f"{metric['label']} is stale ({metric['freshness']['label']})")

    rule_result = (
        _hospital_rule_result(workflow["id"])
        if workflow_is_active
        and workflow
        and workflow["module_key"] == "hospital"
        else {}
    )

    triage = (
        _triage_result(workflow["id"])
        if workflow_is_active
        and workflow
        and workflow["module_key"] == "hospital"
        else None
    )

    # Historical workflow steps are still shown for audit/history.
    steps = (
        _workflow_steps(workflow["id"])
        if workflow else []
    )

    why_factors = (
        _why_hospital_priority(
            workflow,
            detail,
            rule_result,
        )
        if workflow_is_active and detail
        else []
    )

    home_overall = query_db(
        "SELECT * FROM risk_scores WHERE patient_id=? AND risk_type='overall' ORDER BY id DESC LIMIT 1",
        (patient_id,), one=True,
    )
    fall_risk = next((r for r in risks if r["risk_type"] == "fall_prediction"), None)
    if not why_factors and home_overall:
        dominant_text = home_overall["explanation"] or "Care intelligence overall risk"
        evidence_text = home_overall["evidence"] or "No additional evidence text stored"
        why_factors = [
            {"label": "Overall rule result", "value": f"{home_overall['score']}/100 · {(home_overall['level'] or 'low').upper()}"},
            {"label": "Dominant risk", "value": dominant_text},
            {"label": "Stored evidence", "value": evidence_text},
        ]

    location = p.get("city") or "Not recorded"

    # Only an active hospital episode becomes the CURRENT location.
    if detail and workflow_is_active:
        a = detail["admission"]
        location = (
            f"{a['ward']} · "
            f"Room {a['room']} · "
            f"Bed {a['bed']}"
        )

    recommendation = priority["reason"]

    if (
        workflow_is_active
        and detail
        and detail.get("anomaly")
    ):
        recommendation = (
            detail["anomaly"].get("recommended_review")
            or recommendation
        )

    return {
        "patient": {
            "id": p["id"], "external_ref": p.get("external_ref"),
            "name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "birth_date": p.get("birth_date"), "age": _age(p.get("birth_date")), "sex": p.get("sex"),
            "location": location,
            "assigned_nurse": p.get("assigned_nurse_name") or (workflow.get("assigned_professional") if workflow else None) or "Not assigned",
            "assigned_clinician": p.get("assigned_clinician_name") or p.get("gp_name") or (detail["admission"].get("attending_physician") if detail else None) or "Not assigned",
            "conditions": conditions,
            "medications": medications,
            "fall_risk": fall_risk,
            "medical_history": [row["display"] for row in conditions],
        },
        "priority": priority,
        "thresholds": RISK_LEVEL_THRESHOLDS,
        "risk_breakdown": risks,
        "home_overall": dict(home_overall) if home_overall else None,
        "clinical_data": clinical_data,
        "home_data": home_vitals,
        "hospital": detail,
        "workflow": workflow,
        "workflow_is_active": workflow_is_active,
        "agent_trail": steps,
        "why_priority": why_factors,
        "recommendation": recommendation,
        "missing_or_stale": missing,
        "freshness_note": f"UI freshness marker: values older than {_STALE_AFTER_HOURS} hours are shown as stale. This is not a clinical threshold.",
        "triage": triage,
        "data_origin": "Synthetic/demo hospital data" if detail else ("Synthetic/demo remote-care data" if p.get("lawful_basis") == "demo_synthetic" else "Recorded CareAI data"),
    }
