"""Patient-dynamic controlled-autonomy workflow for the Care.AI prototype.

The workflow is intentionally conservative:
- Steps 1-6 gather evidence, enrich context and prioritise the event.
- Step 7 is a mandatory human approval gate for high-risk follow-up.
- Steps 8-10 create demo-safe operational actions only after approval.
- No diagnosis, treatment decision, medication change or autonomous emergency
  escalation is executed by this service.

All patient/event values come from the selected patient, the selected simulation
inputs, existing CareAI records, or a generic synthetic sensor profile. There is
no patient-specific workflow branch.
"""

import json
import random
import secrets
from datetime import date, datetime

from services.db import execute_db, query_db


MEDICAL_DISCLAIMER = (
    "This AI assistant provides general information and explanation only. "
    "It is not a medical diagnosis and must not replace advice from a qualified "
    "healthcare professional."
)

AGENT_CATALOG = [
    (
        "Monitoring Agent",
        "Patient data is fragmented",
        "Continuously consolidates signals from devices, wearables, sensors and EHR feeds.",
    ),
    (
        "Patient Twin Agent",
        "Clinicians lack a unified patient context",
        "Maintains a live representation of health, behaviour, environment and care history.",
    ),
    (
        "Deterioration Agent",
        "Clinical decline is recognised too late",
        "Detects abnormal trends and emerging risk patterns and contributes them to triage.",
    ),
    (
        "Fall Intelligence Agent",
        "Traditional alarms generate uncertainty",
        "Verifies fall-related events using movement, posture, location, immobility and vital-sign evidence.",
    ),
    (
        "AI Triage Agent",
        "Every alert competes for attention",
        "Ranks patients by urgency, confidence and potential harm for human review.",
    ),
    (
        "Clinical Copilot",
        "Staff spend time searching and summarising",
        "Produces an evidence-grounded summary and recommended next action.",
    ),
    (
        "Care Coordination Agent",
        "Follow-up is manual and inconsistent",
        "Creates approved tasks, identifies the responsible professional and tracks progress.",
    ),
    (
        "Virtual Care Agent",
        "Avoidable physical visits consume capacity",
        "Initiates a demo teleconsultation workflow after human approval when appropriate.",
    ),
    (
        "Medication Agent",
        "Missed doses are discovered late",
        "Checks adherence context and prepares reminders or escalation suggestions without changing medication plans.",
    ),
    (
        "Family Engagement Agent",
        "Families lack appropriate visibility",
        "Provides consent-based updates only after the human approval gate.",
    ),
    (
        "Outcome Agent",
        "Care pathways do not continuously improve",
        "Measures interventions, outcomes, response times and operational impact.",
    ),
]

# Kept for backwards compatibility with the existing module-routing route. The
# dynamic Agentic Care page itself no longer needs to present unrelated modules.
MODULE_CATALOG = [
    {"key": "general", "name": "General CareAI Assistant", "description": "Existing Care Copilot"},
    {"key": "lung", "name": "Lung Cancer AI", "description": "Existing lung workflow"},
    {"key": "wound", "name": "WoundAI", "description": "Existing wound workflow"},
    {"key": "skin", "name": "Skin Cancer AI", "description": "Existing skin workflow"},
]

EVENT_TYPES = [
    {"key": "fall", "label": "Fall event", "description": "Wearable impact + posture + immobility verification"},
    {"key": "immobility", "label": "Immobility event", "description": "Prolonged inactivity requiring contextual review"},
    {"key": "deterioration", "label": "Deterioration event", "description": "Abnormal trend requiring human clinical review"},
]
EVENT_TYPE_KEYS = {row["key"] for row in EVENT_TYPES}

LOCATION_OPTIONS = ["Bathroom", "Bedroom", "Living room", "Kitchen", "Hallway", "Outdoors", "Unknown location"]

SIMULATION_PROFILES = [
    {
        "key": "high_confidence",
        "label": "High-confidence event",
        "description": "Clear sensor pattern designed to demonstrate urgent agentic prioritisation.",
    },
    {
        "key": "moderate_confidence",
        "label": "Moderate-confidence event",
        "description": "More ambiguous sensor pattern that may rank below critical depending on patient context.",
    },
]
SIMULATION_PROFILE_KEYS = {row["key"] for row in SIMULATION_PROFILES}

TOTAL_STEPS = 10


def _json(value):
    return json.dumps(value or {}, ensure_ascii=False)


def _loads(value):
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _patient_name(patient):
    return f"{patient['first_name']} {patient['last_name']}".strip()


def _patient_age(patient):
    try:
        born = datetime.strptime(patient["birth_date"], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _conditions(patient_id):
    return [
        row["display"]
        for row in query_db(
            "SELECT display FROM conditions WHERE patient_id=? AND active=1 ORDER BY display",
            (patient_id,),
        )
    ]


def _medications(patient_id):
    return [
        f"{row['name']} {row['dose']} ({row['schedule']})"
        for row in query_db(
            "SELECT name,dose,schedule FROM medications WHERE patient_id=? AND active=1 ORDER BY id",
            (patient_id,),
        )
    ]


def _latest_vitals(patient_id):
    rows = query_db(
        """
        SELECT v.kind, v.value, v.unit, v.measured_at
        FROM vitals v
        JOIN (
          SELECT kind, MAX(id) AS max_id
          FROM vitals
          WHERE patient_id=?
          GROUP BY kind
        ) latest ON latest.max_id=v.id
        ORDER BY v.kind
        """,
        (patient_id,),
    )
    return {
        row["kind"]: {
            "value": row["value"],
            "unit": row["unit"],
            "measured_at": row["measured_at"],
        }
        for row in rows
    }


def _format_vitals(vitals):
    wanted = [
        ("spo2", "SpO₂"),
        ("heart_rate", "heart rate"),
        ("resp_rate", "respiratory rate"),
        ("bp_sys", "systolic BP"),
        ("bp_dia", "diastolic BP"),
        ("activity", "activity"),
    ]
    parts = []
    for key, label in wanted:
        item = vitals.get(key)
        if item:
            value = item["value"]
            value_text = f"{value:.1f}" if isinstance(value, (int, float)) else str(value)
            parts.append(f"{label} {value_text} {item['unit']}")
    return ", ".join(parts) if parts else "No recent vital signs recorded"


def _latest_overall_risk(patient_id):
    return query_db(
        """
        SELECT score,level,explanation,evidence,created_at
        FROM risk_scores
        WHERE patient_id=? AND risk_type='overall'
        ORDER BY id DESC LIMIT 1
        """,
        (patient_id,),
        one=True,
    )


def _mobility_context(patient_id):
    fall = query_db(
        """
        SELECT score,level,status,immobility_seconds,location,created_at
        FROM fall_assessments WHERE patient_id=? ORDER BY id DESC LIMIT 1
        """,
        (patient_id,),
        one=True,
    )
    activity = query_db(
        """
        SELECT value,unit,measured_at FROM vitals
        WHERE patient_id=? AND kind='activity'
        ORDER BY id DESC LIMIT 6
        """,
        (patient_id,),
    )
    if fall:
        return f"Latest fall/mobility assessment: {fall['level']} ({fall['score']}/100), status {fall['status']}"
    if activity:
        values = [float(row["value"]) for row in activity if row["value"] is not None]
        if values:
            current = values[0]
            baseline = sum(values[1:]) / len(values[1:]) if len(values) > 1 else current
            delta = current - baseline
            direction = "lower" if delta < -5 else "higher" if delta > 5 else "stable"
            return f"Recent activity is {direction} versus the recent recorded baseline ({current:.0f} current)"
    return "No dedicated mobility assessment is recorded; use current event evidence and available patient context"


def _recent_trend(patient_id):
    risk = _latest_overall_risk(patient_id)
    if risk:
        return f"Latest overall risk: {risk['level']} ({risk['score']}/100). {risk['explanation'] or ''}".strip()
    return "No recent overall-risk trend is recorded"


def _family_status(patient_id):
    # Patient Management contacts are the current notification source of truth.
    # Existing delegated family portal users remain a backwards-compatible fallback.
    family = query_db(
        """
        SELECT id,display_name,relationship,consent_reference,
               'summary,alerts,messages,teleconsult' access_scope,
               notification_consent consent_granted,expires_at,active
        FROM patient_contacts
        WHERE patient_id=? AND active=1 AND authorised_for_updates=1
        ORDER BY notification_consent DESC,id LIMIT 1
        """,
        (patient_id,),
        one=True,
    )
    if not family:
        family = query_db(
            """
            SELECT id,display_name,relationship,consent_reference,access_scope,
                   consent_granted,expires_at,active
            FROM family_users
            WHERE patient_id=? AND active=1
            ORDER BY id LIMIT 1
            """,
            (patient_id,),
            one=True,
        )
    if not family:
        return {"eligible": False, "reason": "no_contact", "message": "No authorised family contact available", "contact": None}

    contact = dict(family)
    if not family["consent_granted"]:
        return {
            "eligible": False,
            "reason": "no_consent",
            "message": "Family notification skipped because consent is not available",
            "contact": contact,
        }
    if family["expires_at"]:
        try:
            expires = datetime.fromisoformat(family["expires_at"])
            if expires <= datetime.now():
                return {
                    "eligible": False,
                    "reason": "consent_expired",
                    "message": "Family notification skipped because delegated consent has expired",
                    "contact": contact,
                }
        except ValueError:
            pass
    scope = (family["access_scope"] or "").lower()
    if "alerts" not in scope and "messages" not in scope:
        return {
            "eligible": False,
            "reason": "scope_not_permitted",
            "message": "Family notification skipped because notification access is not permitted",
            "contact": contact,
        }
    return {
        "eligible": True,
        "reason": "eligible",
        "message": f"Authorised family contact available: {family['display_name']} ({family['relationship']})",
        "contact": contact,
    }


def _actor(user_id):
    if not user_id:
        return None
    return query_db(
        "SELECT id,role,display_name FROM users WHERE id=? AND active=1",
        (user_id,),
        one=True,
    )


def _actor_display(user_id):
    row = _actor(user_id)
    return row["display_name"] if row else "Responsible clinician"


def available_nurses():
    return [
        dict(row)
        for row in query_db(
            "SELECT id,display_name,role FROM users WHERE role='nurse' AND active=1 ORDER BY display_name"
        )
    ]


def _default_nurse():
    return query_db(
        "SELECT id,display_name,role FROM users WHERE role='nurse' AND active=1 ORDER BY id LIMIT 1",
        one=True,
    )


def _resolve_assigned_nurse(user_id=None):
    if user_id:
        row = query_db(
            "SELECT id,display_name,role FROM users WHERE id=? AND role='nurse' AND active=1",
            (user_id,),
            one=True,
        )
        if row:
            return row
    return _default_nurse()


def _new_room_code():
    return "CARE-AGENT-" + secrets.token_hex(4).upper()


def patient_options():
    patients = query_db(
        """
        SELECT id,external_ref,first_name,last_name,birth_date,city,current_status,
               living_setting,gp_name
        FROM patients WHERE active=1 ORDER BY last_name,first_name
        """
    )
    result = []
    for row in patients:
        patient = dict(row)
        patient["age"] = _patient_age(row)
        patient["conditions"] = _conditions(row["id"])
        patient["medications"] = _medications(row["id"])
        patient["vitals"] = _latest_vitals(row["id"])
        patient["vitals_summary"] = _format_vitals(patient["vitals"])
        patient["mobility_context"] = _mobility_context(row["id"])
        patient["recent_trend"] = _recent_trend(row["id"])
        family = _family_status(row["id"])
        patient["family_status"] = family["message"]
        patient["family_eligible"] = family["eligible"]
        result.append(patient)
    return result


def _event_label(event_type):
    return {
        "fall": "Fall-related event",
        "immobility": "Immobility event",
        "deterioration": "Deterioration event",
    }.get(event_type, "Patient event")


def _event_agent(event_type):
    if event_type == "fall":
        return "Sensor-Fusion / Fall Intelligence Agent"
    if event_type == "immobility":
        return "Sensor-Fusion / Deterioration Agent"
    return "Deterioration Agent"


def _simulate_sensor_event(patient_id, event_type, profile, latest_vitals):
    # Generic synthetic values. The seed changes per event, so no patient receives
    # a fixed patient-specific sensor story.
    rng = random.Random(f"{patient_id}:{event_type}:{profile}:{datetime.now().isoformat(timespec='microseconds')}")
    high = profile == "high_confidence"

    if event_type == "fall":
        impact_g = round(rng.uniform(2.4, 3.2) if high else rng.uniform(1.4, 2.3), 1)
        orientation = round(rng.uniform(65, 86) if high else rng.uniform(38, 66))
        immobility = rng.randint(180, 300) if high else rng.randint(45, 170)
        confidence = rng.randint(94, 99) if high else rng.randint(78, 91)
        posture = "person-on-floor pattern" if high else "possible low-posture pattern"
    elif event_type == "immobility":
        impact_g = None
        orientation = round(rng.uniform(4, 22))
        immobility = rng.randint(600, 1200) if high else rng.randint(240, 600)
        confidence = rng.randint(90, 97) if high else rng.randint(75, 89)
        posture = "prolonged stationary posture"
    else:
        impact_g = None
        orientation = None
        immobility = rng.randint(60, 240)
        confidence = rng.randint(88, 96) if high else rng.randint(72, 88)
        posture = "abnormal multi-signal trend"

    sensor_vitals = {}
    for key in ("spo2", "heart_rate", "resp_rate", "bp_sys", "bp_dia", "activity"):
        if key in latest_vitals:
            sensor_vitals[key] = latest_vitals[key]

    return {
        "impact_g": impact_g,
        "orientation_deg": orientation,
        "immobility_seconds": immobility,
        "confidence": confidence,
        "posture": posture,
        "sensor_vitals": sensor_vitals,
    }


def _add_step(
    run_id,
    number,
    agent,
    problem,
    action,
    evidence="Waiting for agent execution.",
    status="pending",
    approval=0,
    meta=None,
):
    execute_db(
        """
        INSERT INTO agentic_steps(
          run_id,step_no,agent_name,problem_text,action_text,evidence_text,
          status,requires_human_approval,metadata_json
        ) VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (run_id, number, agent, problem, action, evidence, status, approval, _json(meta)),
    )


def create_patient_event_run(
    patient_id,
    actor_user_id=None,
    event_type="fall",
    location="Unknown location",
    simulation_profile="high_confidence",
    assigned_user_id=None,
):
    """Create a patient-dynamic staged workflow with no operational action before approval."""
    patient = query_db("SELECT * FROM patients WHERE id=? AND active=1", (patient_id,), one=True)
    if not patient:
        raise RuntimeError("Patient not found")
    if event_type not in EVENT_TYPE_KEYS:
        raise RuntimeError("Unsupported event type")
    if simulation_profile not in SIMULATION_PROFILE_KEYS:
        raise RuntimeError("Unsupported simulation profile")
    if location not in LOCATION_OPTIONS:
        location = "Unknown location"

    name = _patient_name(patient)
    event_dt = datetime.now()
    event_detected_at = event_dt.strftime("%Y-%m-%d %H:%M:%S")
    event_clock = event_dt.strftime("%H:%M")
    conditions = _conditions(patient_id)
    medications = _medications(patient_id)
    vitals = _latest_vitals(patient_id)
    mobility = _mobility_context(patient_id)
    trend = _recent_trend(patient_id)
    family = _family_status(patient_id)
    nurse = _resolve_assigned_nurse(assigned_user_id)
    sensor = _simulate_sensor_event(patient_id, event_type, simulation_profile, vitals)
    label = _event_label(event_type)

    condition_text = ", ".join(conditions) if conditions else "No active conditions recorded"
    medication_text = ", ".join(medications[:3]) if medications else "No active medication recorded"
    family_text = family["message"]
    nurse_text = nurse["display_name"] if nurse else "Responsible nurse"

    run_id = execute_db(
        """
        INSERT INTO agentic_runs(
          patient_id,module_key,scenario,status,severity,created_by,summary
        ) VALUES(?,?,?,?,?,?,?)
        """,
        (
            patient_id,
            "general",
            f"{event_clock} · {name} · {label.lower()}",
            "running",
            "pending",
            actor_user_id,
            f"{name}: synthetic {label.lower()} captured at {event_clock}; controlled agentic review started.",
        ),
    )

    execute_db(
        """
        INSERT INTO agentic_event_context(
          run_id,event_type,event_label,event_detected_at,location,simulation_profile,source,
          impact_g,orientation_change_deg,immobility_seconds,sensor_confidence,sensor_posture,
          sensor_vitals_json,assigned_user_id,mobility_context,recent_trend
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            event_type,
            label,
            event_detected_at,
            location,
            simulation_profile,
            "synthetic_demo",
            sensor["impact_g"],
            sensor["orientation_deg"],
            sensor["immobility_seconds"],
            sensor["confidence"],
            sensor["posture"],
            _json(sensor["sensor_vitals"]),
            nurse["id"] if nurse else None,
            mobility,
            trend,
        ),
    )

    execute_db(
        """
        INSERT INTO agentic_workflow_metrics(
          run_id,event_detected_at,workflow_completion_status,teleconsultation_status,
          outcome_status,operational_impact
        ) VALUES(?,?,?,?,?,?)
        """,
        (
            run_id,
            event_detected_at,
            "running",
            "not_started",
            "pending",
            "Controlled-autonomy workflow in progress",
        ),
    )

    event_word = label.lower()
    verification_action = {
        "fall": "Fuse movement, posture, location, immobility and available vital-sign evidence.",
        "immobility": "Fuse inactivity duration, location, posture and available vital-sign evidence.",
        "deterioration": "Combine available vital signs, risk trends and patient context to verify an emerging deterioration signal.",
    }[event_type]

    steps = [
        (
            1,
            "Monitoring Agent",
            "Patient data is fragmented across home and clinical sources.",
            f"Capture the selected synthetic {event_word} for {name} at {event_clock}.",
            "Waiting for sensor event.",
            0,
            {"source": "synthetic sensor", "event_type": event_type, "event_detected_at": event_detected_at},
        ),
        (
            2,
            "Digital Twin",
            "A sensor signal alone does not establish the person's environment.",
            f"Resolve {name}'s selected event location and environmental context.",
            "Waiting for location context.",
            0,
            {"location": location, "source": "digital_twin_demo"},
        ),
        (
            3,
            _event_agent(event_type),
            "Single-sensor alerts can be uncertain without corroborating evidence.",
            verification_action,
            "Waiting for multisignal verification.",
            0,
            {"event_type": event_type, "synthetic_demo": True},
        ),
        (
            4,
            "Patient Twin Agent",
            "Clinicians need patient-specific context before acting on an event.",
            "Add age, conditions, medication, mobility context, recent trend and current CareAI vitals without changing the clinical record.",
            f"Current context available: {condition_text}; medication: {medication_text}.",
            0,
            {
                "age": _patient_age(patient),
                "ehr_conditions": conditions,
                "medications": medications,
                "mobility_context": mobility,
                "recent_trend": trend,
                "vitals": vitals,
                "supporting_agents": ["Medication Agent", "Deterioration Agent"],
                "read_only": True,
            },
        ),
        (
            5,
            "AI Triage Agent",
            "Every alert competes for attention.",
            "Rank the event by urgency, confidence and potential harm, then update the Command Centre priority overlay.",
            "Waiting for verified event and patient context.",
            0,
            {"human_review_required": True},
        ),
        (
            6,
            "Clinical Copilot",
            "Staff spend time searching and summarising evidence.",
            "Produce a concise evidence-grounded explanation and recommend human review without making a diagnosis.",
            "Waiting for triage result.",
            0,
            {"grounding": "synthetic event + read-only patient context"},
        ),
        (
            7,
            "Responsible Nurse",
            "High-risk follow-up requires human accountability.",
            "Review evidence, add a note, then approve or reject the proposed response.",
            "Human review checkpoint.",
            1,
            {
                "owner_role": "nurse",
                "owner_name": nurse_text,
                "owner_user_id": nurse["id"] if nurse else None,
                "human_in_the_loop": True,
            },
        ),
        (
            8,
            "Care Coordination + Virtual Care Agents",
            "Follow-up is manual and inconsistent and avoidable visits consume capacity.",
            "After approval, create a follow-up task and initiate a demo teleconsultation linked to this patient and event.",
            "Blocked until nurse approval.",
            0,
            {"requires_approval": True, "integration": "demo/mock teleconsultation"},
        ),
        (
            9,
            "Family Engagement Agent",
            "Families need appropriate visibility without bypassing consent.",
            "After approval, verify authorised family access and send or safely skip a calm family update.",
            f"Family context: {family_text}.",
            0,
            {"requires_approval": True, "family_status": family_text},
        ),
        (
            10,
            "Outcome Agent",
            "Care pathways need measurable follow-up.",
            "Record event, triage, review, decision, response, family, teleconsultation and workflow metrics.",
            "Waiting for the approved response workflow.",
            0,
            {"metrics": ["response time", "intervention", "outcome", "operational impact"]},
        ),
    ]

    for number, agent, problem, action, evidence, approval, meta in steps:
        _add_step(
            run_id,
            number,
            agent,
            problem,
            action,
            evidence,
            "active" if number == 1 else "pending",
            approval,
            meta,
        )
    execute_db(
        "UPDATE agentic_steps SET started_at=CURRENT_TIMESTAMP WHERE run_id=? AND step_no=1",
        (run_id,),
    )
    return run_id


# Backwards-compatible name used by the previous generic route. It now delegates
# to the same patient-dynamic implementation and contains no patient exception.
def create_patient_fall_run(patient_id, actor_user_id=None):
    return create_patient_event_run(patient_id, actor_user_id, event_type="fall")


def _finish(run_id, number, evidence, meta=None):
    execute_db(
        """
        UPDATE agentic_steps
        SET status='completed', evidence_text=?, metadata_json=?,
            started_at=COALESCE(started_at,CURRENT_TIMESTAMP),
            completed_at=CURRENT_TIMESTAMP
        WHERE run_id=? AND step_no=?
        """,
        (evidence, _json(meta), run_id, number),
    )


def _activate(run_id, number):
    execute_db(
        "UPDATE agentic_steps SET status='active',started_at=CURRENT_TIMESTAMP WHERE run_id=? AND step_no=?",
        (run_id, number),
    )


def _record_action(run_id, action_type, status, reference_table=None, reference_id=None, message=None):
    existing = query_db(
        "SELECT id FROM agentic_actions WHERE run_id=? AND action_type=? ORDER BY id DESC LIMIT 1",
        (run_id, action_type),
        one=True,
    )
    if existing:
        return existing["id"]
    return execute_db(
        """
        INSERT INTO agentic_actions(run_id,action_type,status,reference_table,reference_id,message,completed_at)
        VALUES(?,?,?,?,?,?,CASE WHEN ? IN ('created','sent','initiated','skipped') THEN CURRENT_TIMESTAMP ELSE NULL END)
        """,
        (run_id, action_type, status, reference_table, reference_id, message, status),
    )


def _run_action(run_id, action_type):
    return query_db(
        "SELECT * FROM agentic_actions WHERE run_id=? AND action_type=? ORDER BY id DESC LIMIT 1",
        (run_id, action_type),
        one=True,
    )


def _event_context(run_id):
    row = query_db(
        """
        SELECT e.*,u.display_name assigned_professional,u.role assigned_role
        FROM agentic_event_context e
        LEFT JOIN users u ON u.id=e.assigned_user_id
        WHERE e.run_id=?
        """,
        (run_id,),
        one=True,
    )
    return dict(row) if row else None


def _legacy_event_context(run_id, run=None, steps=None):
    """Read older runs safely without introducing patient-specific assumptions."""
    run = run or query_db("SELECT * FROM agentic_runs WHERE id=?", (run_id,), one=True)
    steps = steps or query_db("SELECT * FROM agentic_steps WHERE run_id=? ORDER BY step_no", (run_id,))
    step3 = next((row for row in steps if row["step_no"] == 3), None)
    step5 = next((row for row in steps if row["step_no"] == 5), None)
    meta3 = _loads(step3["metadata_json"] if step3 else None)
    meta5 = _loads(step5["metadata_json"] if step5 else None)
    scenario = (run["scenario"] if run else "") or ""
    event_type = "fall" if "fall" in scenario.lower() else "event"
    nurse = _default_nurse()
    metrics = query_db("SELECT * FROM agentic_workflow_metrics WHERE run_id=?", (run_id,), one=True)
    return {
        "run_id": run_id,
        "event_type": event_type,
        "event_label": _event_label(event_type) if event_type in EVENT_TYPE_KEYS else "Patient event",
        "event_detected_at": metrics["event_detected_at"] if metrics else (run["started_at"] if run else None),
        "location": meta3.get("location") or "Location not recorded",
        "simulation_profile": "legacy",
        "source": "legacy_agentic_run",
        "impact_g": meta3.get("impact_g"),
        "orientation_change_deg": meta3.get("orientation_deg"),
        "immobility_seconds": meta3.get("immobility_seconds"),
        "sensor_confidence": meta3.get("fall_confidence") or meta5.get("confidence"),
        "sensor_posture": None,
        "sensor_vitals_json": _json(meta3.get("scenario_vitals") or {}),
        "assigned_user_id": nurse["id"] if nurse else None,
        "assigned_professional": nurse["display_name"] if nurse else "Responsible nurse",
        "assigned_role": "nurse",
        "mobility_context": "Legacy run context",
        "recent_trend": "Legacy run context",
        "triage_score": meta5.get("triage_score") or meta5.get("confidence"),
        "triage_at": step5["completed_at"] if step5 else None,
    }


def _parse_sqlite_ts(value):
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%H:%M",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def _response_seconds(run_id):
    run = query_db("SELECT started_at FROM agentic_runs WHERE id=?", (run_id,), one=True)
    metrics = query_db("SELECT response_started_at FROM agentic_workflow_metrics WHERE run_id=?", (run_id,), one=True)
    start = _parse_sqlite_ts(run["started_at"] if run else None)
    response = _parse_sqlite_ts(metrics["response_started_at"] if metrics else None)
    if not start or not response or start.year == 1900 or response.year == 1900:
        return None
    return max(0, int((response - start).total_seconds()))


def _triage(patient, event):
    patient_id = patient["id"]
    confidence = int(event.get("sensor_confidence") or 0)
    score = confidence
    event_type = event.get("event_type")
    immobility = int(event.get("immobility_seconds") or 0)
    impact = float(event.get("impact_g") or 0)
    orientation = float(event.get("orientation_change_deg") or 0)

    if event_type == "fall":
        if immobility >= 180:
            score += 5
        if impact >= 2.4:
            score += 3
        if orientation >= 60:
            score += 3
    elif event_type == "immobility" and immobility >= 600:
        score += 6

    age = _patient_age(patient)
    if age and age >= 75:
        score += 3

    conditions = _conditions(patient_id)
    if len(conditions) >= 2:
        score += 2

    risk = _latest_overall_risk(patient_id)
    if risk:
        score += {"critical": 7, "high": 5, "medium": 2, "low": 0}.get((risk["level"] or "").lower(), 0)

    vitals = _latest_vitals(patient_id)
    spo2 = vitals.get("spo2", {}).get("value")
    activity = vitals.get("activity", {}).get("value")
    if spo2 is not None:
        if float(spo2) < 92:
            score += 6
        elif float(spo2) < 95:
            score += 3
    if activity is not None and float(activity) < 50:
        score += 3

    score = min(100, max(0, int(round(score))))
    if score >= 94:
        severity = "critical"
    elif score >= 82:
        severity = "high"
    elif score >= 68:
        severity = "medium"
    else:
        severity = "low"
    return severity, score


def _sensor_evidence(event):
    event_type = event["event_type"]
    confidence = event.get("sensor_confidence") or 0
    location = event.get("location") or "unknown location"
    if event_type == "fall":
        return (
            f"Sensor fusion verified a fall-related pattern at {confidence}% confidence in {location}: "
            f"impact {event.get('impact_g') or 'n/a'}g, orientation/posture change {event.get('orientation_change_deg') or 'n/a'}°, "
            f"{event.get('sensor_posture') or 'posture signal'} and {event.get('immobility_seconds') or 0} seconds immobility."
        )
    if event_type == "immobility":
        return (
            f"Sensor fusion identified prolonged immobility at {confidence}% confidence in {location}: "
            f"{event.get('immobility_seconds') or 0} seconds with {event.get('sensor_posture') or 'stationary posture'} evidence."
        )
    return (
        f"Deterioration Agent identified a multi-signal deterioration pattern at {confidence}% confidence in {location}. "
        "The signal is treated as decision support and requires human clinical review."
    )


def advance_run(run_id, actor_user_id=None):
    run = query_db("SELECT * FROM agentic_runs WHERE id=?", (run_id,), one=True)
    if not run:
        raise RuntimeError("Run not found")
    if run["status"] in ("completed", "rejected"):
        return run_state(run_id)

    patient = query_db("SELECT * FROM patients WHERE id=?", (run["patient_id"],), one=True)
    if not patient:
        raise RuntimeError("Patient not found")
    name = _patient_name(patient)
    event = _event_context(run_id)
    if not event:
        event = _legacy_event_context(run_id, run=run)

    step = query_db(
        "SELECT * FROM agentic_steps WHERE run_id=? AND status='active' ORDER BY step_no LIMIT 1",
        (run_id,),
        one=True,
    )
    if not step:
        return run_state(run_id)

    number = step["step_no"]
    label = event.get("event_label") or "Patient event"
    event_time = event.get("event_detected_at") or run["started_at"]
    location = event.get("location") or "Unknown location"
    confidence = int(event.get("sensor_confidence") or 0)

    if number == 1:
        event_type = event.get("event_type")
        if event_type == "fall":
            signal = f"wearable impact {event.get('impact_g') or 'n/a'}g"
        elif event_type == "immobility":
            signal = f"prolonged immobility {event.get('immobility_seconds') or 0}s"
        else:
            signal = "abnormal multi-signal trend"
        _finish(
            run_id,
            1,
            f"Monitoring Agent captured the selected synthetic {label.lower()} for {name} at {event_time}: {signal}.",
            {"event_type": event.get("event_type"), "event_detected_at": event_time, "source": event.get("source")},
        )
        _activate(run_id, 2)

    elif number == 2:
        _finish(
            run_id,
            2,
            f"Digital Twin associated the event with {location} and attached environmental context for {name}.",
            {"location": location, "synthetic_demo": True},
        )
        _activate(run_id, 3)

    elif number == 3:
        evidence = _sensor_evidence(event)
        _finish(
            run_id,
            3,
            evidence,
            {
                "event_type": event.get("event_type"),
                "confidence": confidence,
                "impact_g": event.get("impact_g"),
                "orientation_deg": event.get("orientation_change_deg"),
                "immobility_seconds": event.get("immobility_seconds"),
                "location": location,
                "sensor_posture": event.get("sensor_posture"),
                "sensor_vitals": _loads(event.get("sensor_vitals_json")),
                "synthetic_demo": True,
            },
        )
        _activate(run_id, 4)

    elif number == 4:
        conditions = _conditions(run["patient_id"])
        medications = _medications(run["patient_id"])
        vitals = _latest_vitals(run["patient_id"])
        age = _patient_age(patient)
        mobility = event.get("mobility_context") or _mobility_context(run["patient_id"])
        trend = event.get("recent_trend") or _recent_trend(run["patient_id"])
        evidence = (
            f"Read-only Patient Twin context for {name}: age {age if age is not None else 'not recorded'}; "
            f"conditions = {', '.join(conditions) if conditions else 'none recorded'}; "
            f"medications = {', '.join(medications[:4]) if medications else 'none recorded'}; "
            f"mobility = {mobility}; recent trend = {trend}; current recorded values = {_format_vitals(vitals)}. "
            "No clinical record was changed by the agent workflow."
        )
        _finish(
            run_id,
            4,
            evidence,
            {
                "age": age,
                "ehr_conditions": conditions,
                "medications": medications,
                "mobility_context": mobility,
                "recent_trend": trend,
                "vitals": vitals,
                "supporting_agents": ["Medication Agent", "Deterioration Agent"],
                "read_only": True,
            },
        )
        _activate(run_id, 5)

    elif number == 5:
        severity, triage_score = _triage(patient, event)
        evidence = (
            f"AI Triage ranked the {label.lower()} {severity.upper()} with triage score {triage_score}/100 "
            f"and sensor confidence {confidence}%. Ranking combines event evidence, age, available conditions, "
            "recorded risk context and current vitals. The Command Centre priority overlay is updated for human review; "
            "the normal clinical risk-score table is not overwritten."
        )
        _finish(
            run_id,
            5,
            evidence,
            {
                "severity": severity,
                "triage_score": triage_score,
                "confidence": confidence,
                "human_review_required": severity in ("critical", "high"),
            },
        )
        execute_db(
            "UPDATE agentic_runs SET severity=?,summary=? WHERE id=?",
            (
                severity,
                f"{severity.upper()} · {name} · {label.lower()} · {confidence}% sensor confidence · human review required",
                run_id,
            ),
        )
        if _event_context(run_id):
            execute_db(
                "UPDATE agentic_event_context SET triage_score=?,triage_at=CURRENT_TIMESTAMP WHERE run_id=?",
                (triage_score, run_id),
            )
        _activate(run_id, 6)

    elif number == 6:
        conditions = _conditions(run["patient_id"])
        severity, triage_score = _triage(patient, event)
        context = ", ".join(conditions) if conditions else "available recorded patient context"
        recommended = (
            "Immediate human review" if severity == "critical" else
            "Prompt human review" if severity == "high" else
            "Human review and contextual verification"
        )
        evidence = (
            f"Clinical Copilot summary for {name}: {label.lower()} at {confidence}% sensor confidence in {location}; "
            f"triage {severity.upper()} ({triage_score}/100); patient context includes {context}. "
            f"Recommended next action: {recommended}. Confirm patient status and decide the appropriate follow-up under local protocol. "
            "Care.AI has not made a diagnosis, treatment decision or autonomous emergency-escalation decision."
        )
        _finish(
            run_id,
            6,
            evidence,
            {
                "recommended_action": recommended,
                "clinical_context": context,
                "severity": severity,
                "triage_score": triage_score,
                "no_diagnosis": True,
            },
        )
        nurse = _resolve_assigned_nurse(event.get("assigned_user_id"))
        owner = nurse["display_name"] if nurse else "Responsible nurse"
        execute_db(
            """
            UPDATE agentic_steps
            SET status='awaiting_approval',started_at=CURRENT_TIMESTAMP,evidence_text=?
            WHERE run_id=? AND step_no=7
            """,
            (f"Assigned to {owner}. Review evidence, add an optional note, then approve or reject.", run_id),
        )
        execute_db("UPDATE agentic_runs SET status='awaiting_approval' WHERE id=?", (run_id,))
        execute_db(
            """
            UPDATE agentic_workflow_metrics
            SET nurse_review_at=CURRENT_TIMESTAMP,workflow_completion_status='awaiting_approval',updated_at=CURRENT_TIMESTAMP
            WHERE run_id=?
            """,
            (run_id,),
        )

    elif number == 8:
        approval = query_db(
            "SELECT * FROM agentic_approvals WHERE run_id=? AND decision='approved' ORDER BY id DESC LIMIT 1",
            (run_id,),
            one=True,
        )
        if not approval:
            execute_db("UPDATE agentic_runs SET status='awaiting_approval' WHERE id=?", (run_id,))
            return run_state(run_id)

        owner = _actor_display(approval["reviewer_user_id"])
        task_action = _run_action(run_id, "care_task")
        if not task_action:
            task_id = execute_db(
                """
                INSERT INTO care_tasks(patient_id,task_type,priority,status,assigned_role,rationale,created_by)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    run["patient_id"],
                    "agentic_follow_up",
                    run["severity"] if run["severity"] in ("critical", "high", "medium", "low") else "high",
                    "open",
                    "nurse",
                    f"Agentic workflow #{run_id}: nurse-approved follow-up for {label.lower()}. Verify patient status and use virtual care or escalation according to protocol.",
                    approval["reviewer_user_id"],
                ),
            )
            _record_action(run_id, "care_task", "created", "care_tasks", task_id, "Nurse-approved follow-up task created")
        else:
            task_id = task_action["reference_id"]

        consult_action = _run_action(run_id, "teleconsultation")
        if not consult_action:
            consult_id = execute_db(
                """
                INSERT INTO teleconsultations(
                  patient_id,requested_by_type,requested_by_id,clinician_role,clinician_name,status,reason,room_code
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    run["patient_id"],
                    "agentic_demo",
                    approval["reviewer_user_id"],
                    "nurse",
                    owner,
                    "requested",
                    f"DEMO/MOCK · Nurse-approved teleconsultation for {label.lower()} · workflow #{run_id}",
                    _new_room_code(),
                ),
            )
            _record_action(
                run_id,
                "teleconsultation",
                "initiated",
                "teleconsultations",
                consult_id,
                f"Demo/mock teleconsultation initiated for {name} after nurse approval",
            )
        else:
            consult_id = consult_action["reference_id"]

        existing_care_event = query_db(
            "SELECT id FROM care_events WHERE patient_id=? AND source='agentic_care' AND description LIKE ? ORDER BY id DESC LIMIT 1",
            (run["patient_id"], f"Workflow #{run_id}:%"),
            one=True,
        )
        if not existing_care_event:
            execute_db(
                "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                (
                    run["patient_id"],
                    "agentic_follow_up_started",
                    "agentic_care",
                    f"Workflow #{run_id}: nurse-approved follow-up task and demo teleconsultation initiated for {label.lower()}.",
                ),
            )
        execute_db(
            """
            UPDATE agentic_workflow_metrics
            SET response_started_at=COALESCE(response_started_at,CURRENT_TIMESTAMP),
                teleconsultation_status='initiated_demo',intervention='Nurse-approved follow-up + demo teleconsultation',
                workflow_completion_status='responding',updated_at=CURRENT_TIMESTAMP
            WHERE run_id=?
            """,
            (run_id,),
        )
        _finish(
            run_id,
            8,
            f"Nurse approval verified. Follow-up task #{task_id} created and demo/mock teleconsultation #{consult_id} initiated for {name} and linked to workflow #{run_id}. No emergency escalation was automatically executed.",
            {
                "owner": owner,
                "care_task_id": task_id,
                "teleconsultation_id": consult_id,
                "teleconsultation_is_demo": True,
                "event_type": event.get("event_type"),
                "emergency_escalation": "not_automatically_executed",
            },
        )
        _activate(run_id, 9)

    elif number == 9:
        approval = query_db(
            "SELECT * FROM agentic_approvals WHERE run_id=? AND decision='approved' ORDER BY id DESC LIMIT 1",
            (run_id,),
            one=True,
        )
        if not approval:
            execute_db("UPDATE agentic_runs SET status='awaiting_approval' WHERE id=?", (run_id,))
            return run_state(run_id)

        family = _family_status(run["patient_id"])
        if family["eligible"]:
            contact = family["contact"]
            notification_action = _run_action(run_id, "family_notification")
            message = (
                f"CareAI has detected a {label.lower()} for {name}. A nurse has reviewed the alert and initiated follow-up. "
                "Further updates will be shared if required."
            )
            if not notification_action:
                notification_id = execute_db(
                    """
                    INSERT INTO notifications(patient_id,audience,channel,message,status,source)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (run["patient_id"], "family", "family_portal", message, "sent", "agentic_care"),
                )
                _record_action(
                    run_id,
                    "family_notification",
                    "sent",
                    "notifications",
                    notification_id,
                    f"Sent to authorised family contact {contact['display_name']} under consent {contact['consent_reference']}",
                )
                execute_db(
                    "INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                    (
                        run["patient_id"],
                        "family_update_sent",
                        "agentic_care",
                        f"Agentic workflow #{run_id}: consent-based family update sent to {contact['display_name']} ({contact['relationship']}).",
                    ),
                )
            else:
                notification_id = notification_action["reference_id"]
            execute_db(
                """
                UPDATE agentic_workflow_metrics
                SET family_notification_at=COALESCE(family_notification_at,CURRENT_TIMESTAMP),updated_at=CURRENT_TIMESTAMP
                WHERE run_id=?
                """,
                (run_id,),
            )
            evidence = (
                f"Consent and notification-scope checks passed for {contact['display_name']} ({contact['relationship']}); "
                f"family update #{notification_id} sent after nurse approval. Message: “{message}”"
            )
            meta = {
                "family_notified": True,
                "family_name": contact["display_name"],
                "relationship": contact["relationship"],
                "consent_reference": contact["consent_reference"],
                "notification_id": notification_id,
                "message": message,
            }
        else:
            _record_action(run_id, "family_notification", "skipped", None, None, family["message"])
            evidence = family["message"] + ". No family notification was sent."
            meta = {"family_notified": False, "reason": family["reason"], "message": family["message"]}

        _finish(run_id, 9, evidence, meta)
        _activate(run_id, 10)

    elif number == 10:
        response_seconds = _response_seconds(run_id)
        if response_seconds is None:
            response_seconds = 0
        consult_action = _run_action(run_id, "teleconsultation")
        family_action = _run_action(run_id, "family_notification")
        outcome = query_db("SELECT id FROM care_outcomes WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,), one=True)
        intervention = "Nurse-approved follow-up + demo teleconsultation"
        outcome_text = "Follow-up initiated; clinical outcome remains for human care-team review"
        operational = "Controlled-autonomy workflow completed; no autonomous diagnosis, treatment or emergency escalation"
        if not outcome:
            execute_db(
                """
                INSERT INTO care_outcomes(
                  patient_id,run_id,response_time_seconds,intervention,outcome,
                  avoidable_visit,escalation_type
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    run["patient_id"],
                    run_id,
                    response_seconds,
                    intervention,
                    outcome_text,
                    0,
                    "virtual_care_demo",
                ),
            )
        execute_db(
            """
            UPDATE agentic_workflow_metrics
            SET response_time_seconds=?,intervention=?,outcome_status='follow_up_initiated',
                workflow_completion_status='completed',teleconsultation_status=?,
                operational_impact=?,updated_at=CURRENT_TIMESTAMP
            WHERE run_id=?
            """,
            (
                response_seconds,
                intervention,
                "initiated_demo" if consult_action else "not_started",
                operational,
                run_id,
            ),
        )
        family_result = "sent" if family_action and family_action["status"] == "sent" else "skipped/not sent"
        _finish(
            run_id,
            10,
            f"Outcome metrics recorded for {name}: response workflow started in {response_seconds}s; intervention = {intervention}; family update = {family_result}; workflow completed.",
            {
                "response_time_seconds": response_seconds,
                "intervention": intervention,
                "outcome_status": "follow_up_initiated",
                "family_update_status": family_result,
                "operational_impact": operational,
            },
        )
        execute_db(
            "UPDATE agentic_runs SET status='completed',completed_at=CURRENT_TIMESTAMP,summary=? WHERE id=?",
            (
                f"COMPLETED · {name} · {label.lower()} · nurse-approved follow-up initiated · family consent checked · outcome metrics recorded",
                run_id,
            ),
        )

    return run_state(run_id)


def _assigned_nurse_for_run(run_id):
    event = _event_context(run_id)
    if event and event.get("assigned_user_id"):
        return _actor(event["assigned_user_id"])
    return _default_nurse()


def approve_run(run_id, user_id, note=""):
    run = query_db("SELECT * FROM agentic_runs WHERE id=?", (run_id,), one=True)
    if not run:
        raise RuntimeError("Run not found")
    if run["status"] != "awaiting_approval":
        return run_state(run_id)

    actor = _actor(user_id)
    if not actor or actor["role"] not in ("nurse", "admin"):
        raise PermissionError("Only an authorised nurse or admin can approve this high-risk workflow")

    assigned = _assigned_nurse_for_run(run_id)
    if actor["role"] == "nurse" and assigned and actor["id"] != assigned["id"]:
        raise PermissionError(f"This workflow is assigned to {assigned['display_name']}")

    clean_note = (note or "").strip()[:500]
    existing = query_db("SELECT id FROM agentic_approvals WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,), one=True)
    if not existing:
        execute_db(
            "INSERT INTO agentic_approvals(run_id,reviewer_user_id,decision,note) VALUES(?,?,?,?)",
            (run_id, user_id, "approved", clean_note),
        )
    evidence = f"{actor['display_name']} reviewed the evidence and approved the proposed follow-up."
    if clean_note:
        evidence += f" Reviewer note: {clean_note}"
    evidence += " Diagnosis, treatment, medication changes and emergency escalation remain human-controlled."
    execute_db(
        """
        UPDATE agentic_steps
        SET status='completed',evidence_text=?,
            started_at=COALESCE(started_at,CURRENT_TIMESTAMP),completed_at=CURRENT_TIMESTAMP,
            metadata_json=?
        WHERE run_id=? AND step_no=7
        """,
        (
            evidence,
            _json({"approved_by": actor["display_name"], "decision": "approved", "note": clean_note}),
            run_id,
        ),
    )
    execute_db(
        "UPDATE agentic_runs SET approved_at=CURRENT_TIMESTAMP,approved_by=?,status='responding' WHERE id=?",
        (user_id, run_id),
    )
    execute_db(
        """
        UPDATE agentic_workflow_metrics
        SET approval_at=CURRENT_TIMESTAMP,workflow_completion_status='responding',updated_at=CURRENT_TIMESTAMP
        WHERE run_id=?
        """,
        (run_id,),
    )
    _activate(run_id, 8)
    return run_state(run_id)


def reject_run(run_id, user_id, note=""):
    run = query_db("SELECT * FROM agentic_runs WHERE id=?", (run_id,), one=True)
    if not run:
        raise RuntimeError("Run not found")
    if run["status"] != "awaiting_approval":
        return run_state(run_id)

    actor = _actor(user_id)
    if not actor or actor["role"] not in ("nurse", "admin"):
        raise PermissionError("Only an authorised nurse or admin can reject this high-risk workflow")

    assigned = _assigned_nurse_for_run(run_id)
    if actor["role"] == "nurse" and assigned and actor["id"] != assigned["id"]:
        raise PermissionError(f"This workflow is assigned to {assigned['display_name']}")

    clean_note = (note or "").strip()[:500]
    execute_db(
        "INSERT INTO agentic_approvals(run_id,reviewer_user_id,decision,note) VALUES(?,?,?,?)",
        (run_id, user_id, "rejected", clean_note),
    )
    evidence = f"{actor['display_name']} rejected the proposed automated continuation."
    if clean_note:
        evidence += f" Reviewer note: {clean_note}"
    evidence += " No care task, teleconsultation or family update was created by this workflow."
    execute_db(
        """
        UPDATE agentic_steps
        SET status='rejected',evidence_text=?,completed_at=CURRENT_TIMESTAMP,metadata_json=?
        WHERE run_id=? AND step_no=7
        """,
        (evidence, _json({"decision": "rejected", "note": clean_note}), run_id),
    )
    execute_db(
        """
        UPDATE agentic_steps
        SET status='cancelled',evidence_text='Cancelled because the responsible human reviewer rejected the proposed response.'
        WHERE run_id=? AND step_no IN (8,9,10) AND status='pending'
        """,
        (run_id,),
    )
    patient = query_db("SELECT * FROM patients WHERE id=?", (run["patient_id"],), one=True)
    execute_db(
        "UPDATE agentic_runs SET status='rejected',completed_at=CURRENT_TIMESTAMP,summary=? WHERE id=?",
        (f"REJECTED · {_patient_name(patient)} · human review stopped the workflow", run_id),
    )
    execute_db(
        """
        UPDATE agentic_workflow_metrics
        SET workflow_completion_status='rejected',outcome_status='stopped_by_human',
            operational_impact='Human reviewer stopped the workflow before operational actions',updated_at=CURRENT_TIMESTAMP
        WHERE run_id=?
        """,
        (run_id,),
    )
    return run_state(run_id)


def run_summary(run_id):
    run = query_db(
        """
        SELECT r.*,p.first_name,p.last_name,p.external_ref,p.birth_date,p.city,p.current_status,p.living_setting,p.gp_name
        FROM agentic_runs r
        JOIN patients p ON p.id=r.patient_id
        WHERE r.id=?
        """,
        (run_id,),
        one=True,
    )
    if not run:
        return None, [], None
    steps = query_db("SELECT * FROM agentic_steps WHERE run_id=? ORDER BY step_no", (run_id,))
    outcome = query_db(
        "SELECT * FROM care_outcomes WHERE run_id=? ORDER BY id DESC LIMIT 1",
        (run_id,),
        one=True,
    )
    return run, steps, outcome


def run_state(run_id):
    run, steps, outcome = run_summary(run_id)
    if not run:
        raise RuntimeError("Run not found")
    completed = sum(1 for step in steps if step["status"] == "completed")
    active = next((step for step in steps if step["status"] in ("active", "awaiting_approval")), None)
    approval = query_db(
        """
        SELECT a.*,u.display_name reviewer_name,u.role reviewer_role
        FROM agentic_approvals a JOIN users u ON u.id=a.reviewer_user_id
        WHERE a.run_id=? ORDER BY a.id DESC LIMIT 1
        """,
        (run_id,),
        one=True,
    )
    actions = query_db("SELECT * FROM agentic_actions WHERE run_id=? ORDER BY id", (run_id,))
    metrics_row = query_db("SELECT * FROM agentic_workflow_metrics WHERE run_id=?", (run_id,), one=True)
    event = _event_context(run_id) or _legacy_event_context(run_id, run=run, steps=steps)

    run_dict = dict(run)
    run_dict.update({
        "event_type": event.get("event_type"),
        "event_label": event.get("event_label"),
        "event_detected_at": event.get("event_detected_at"),
        "event_location": event.get("location"),
        "sensor_confidence": event.get("sensor_confidence"),
        "triage_score": event.get("triage_score"),
        "triage_at": event.get("triage_at"),
        "assigned_user_id": event.get("assigned_user_id"),
        "assigned_professional": event.get("assigned_professional") or "Responsible nurse",
        "assigned_role": event.get("assigned_role") or "nurse",
    })

    metrics = dict(metrics_row) if metrics_row else {}
    step5 = next((step for step in steps if step["step_no"] == 5), None)
    metrics["triage_at"] = event.get("triage_at") or (step5["completed_at"] if step5 else None)
    metrics["decision_at"] = approval["created_at"] if approval else None
    metrics["decision"] = approval["decision"] if approval else None

    return {
        "run": run_dict,
        "event": event,
        "completed": completed,
        "total": len(steps),
        "percent": round((completed / max(1, len(steps))) * 100),
        "active_step": active["step_no"] if active else None,
        "steps": [dict(step) for step in steps],
        "outcome": dict(outcome) if outcome else None,
        "approval": dict(approval) if approval else None,
        "actions": [dict(row) for row in actions],
        "metrics": metrics,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def recent_runs(limit=10):
    return query_db(
        """
        SELECT r.*,p.first_name,p.last_name,p.external_ref,p.city,
               e.event_type,e.event_label,e.location,e.sensor_confidence,e.triage_score,
               u.display_name assigned_professional
        FROM agentic_runs r
        JOIN patients p ON p.id=r.patient_id
        LEFT JOIN agentic_event_context e ON e.run_id=r.id
        LEFT JOIN users u ON u.id=e.assigned_user_id
        ORDER BY r.id DESC LIMIT ?
        """,
        (limit,),
    )


def dashboard_agentic_cases(limit=12):
    # Fetch a bounded recent pool, then select one representative case per patient:
    # the highest-priority active workflow when one exists, otherwise the latest run.
    # This keeps Command Centre aligned with the shared clinical priority snapshot.
    rows = query_db(
        """
        SELECT r.id run_id,r.patient_id,r.module_key,r.scenario,r.status,r.severity,r.started_at,r.approved_at,r.completed_at,r.summary,
               p.first_name,p.last_name,p.external_ref,p.city,p.living_setting,
               e.event_type,e.event_label,e.event_detected_at context_event_at,e.location,e.sensor_confidence,e.triage_score,
               h.rule_score,h.ml_score,h.hybrid_score,h.anomaly_type,ha.ward hospital_ward,ha.room hospital_room,ha.bed hospital_bed,
               COALESCE(ue.display_name,uh.display_name) assigned_professional,
               m.event_detected_at,m.nurse_review_at,m.approval_at,m.response_started_at,m.family_notification_at,
               m.response_time_seconds,m.intervention,m.outcome_status,m.workflow_completion_status,
               m.teleconsultation_status,m.operational_impact
        FROM agentic_runs r
        JOIN patients p ON p.id=r.patient_id
        LEFT JOIN agentic_event_context e ON e.run_id=r.id
        LEFT JOIN users ue ON ue.id=e.assigned_user_id
        LEFT JOIN hospital_agentic_context h ON h.run_id=r.id
        LEFT JOIN hospital_admissions ha ON ha.id=h.admission_id
        LEFT JOIN users uh ON uh.id=h.assigned_user_id
        LEFT JOIN agentic_workflow_metrics m ON m.run_id=r.id
        WHERE p.active=1
          AND r.severity IN ('critical','high','medium','low')
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (max(limit * 20, 100),),
    )

    grouped = {}
    active_statuses = {"running", "awaiting_approval", "responding"}
    rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for raw in rows:
        row = dict(raw)
        grouped.setdefault(row["patient_id"], []).append(row)

    selected = []
    for patient_rows in grouped.values():
        active = [row for row in patient_rows if row.get("status") in active_statuses]
        if active:
            chosen = max(
                active,
                key=lambda row: (
                    rank.get((row.get("severity") or "low").lower(), 1),
                    int(row.get("hybrid_score") or row.get("triage_score") or row.get("sensor_confidence") or row.get("rule_score") or 0),
                    int(row.get("run_id") or 0),
                ),
            )
        else:
            chosen = max(patient_rows, key=lambda row: int(row.get("run_id") or 0))
        selected.append(chosen)

    selected.sort(
        key=lambda row: (
            1 if row.get("status") in active_statuses else 0,
            rank.get((row.get("severity") or "low").lower(), 1),
            int(row.get("hybrid_score") or row.get("triage_score") or row.get("sensor_confidence") or row.get("rule_score") or 0),
            int(row.get("run_id") or 0),
        ),
        reverse=True,
    )

    result = []
    for case in selected[:limit]:
        steps = query_db("SELECT step_no,status,metadata_json,completed_at FROM agentic_steps WHERE run_id=?", (case["run_id"],))
        completed = sum(1 for step in steps if step["status"] == "completed")
        step3 = next((step for step in steps if step["step_no"] == 3), None)
        step5 = next((step for step in steps if step["step_no"] == 5), None)
        meta3 = _loads(step3["metadata_json"] if step3 else None)
        meta5 = _loads(step5["metadata_json"] if step5 else None)
        case["progress_completed"] = completed
        case["progress_total"] = len(steps) or TOTAL_STEPS

        if case.get("module_key") == "hospital":
            case["confidence"] = int(case.get("ml_score") or 0)
            case["triage_score"] = int(case.get("hybrid_score") or case.get("rule_score") or 0)
            anomaly = case.get("anomaly_type") or (case.get("scenario") or "").replace("Hospital EHR anomaly: ", "")
            case["alert_type"] = anomaly.replace("_", " ").title() if anomaly else "Hospital clinical finding"
            loc = [case.get("hospital_ward"), f"Room {case['hospital_room']}" if case.get("hospital_room") else None, f"Bed {case['hospital_bed']}" if case.get("hospital_bed") else None]
            case["location"] = " · ".join(part for part in loc if part) or "Hospital location not recorded"
        else:
            case["confidence"] = case.get("sensor_confidence") or meta3.get("confidence") or meta3.get("fall_confidence") or meta5.get("confidence") or 0
            case["triage_score"] = case.get("triage_score") or meta5.get("triage_score") or case["confidence"]
            case["alert_type"] = case.get("event_label") or ("Fall-related event" if "fall" in (case.get("scenario") or "").lower() else "Patient event")
            case["location"] = case.get("location") or meta3.get("location") or "Location not recorded"

        case["event_detected_at"] = case.get("context_event_at") or case.get("event_detected_at") or case.get("started_at")
        case["assigned_nurse"] = case.get("assigned_professional") or "Responsible clinician"
        if case["status"] == "awaiting_approval":
            case["next_action"] = "Human review and approval required"
        elif case["status"] == "responding":
            case["next_action"] = "Complete care coordination, family check and outcome capture"
        elif case["status"] == "completed":
            case["next_action"] = "Review teleconsultation and Outcome Agent metrics"
        elif case["status"] == "rejected":
            case["next_action"] = "Human reviewer stopped the proposed workflow"
        else:
            case["next_action"] = "Agent evidence gathering in progress"
        tele = _run_action(case["run_id"], "teleconsultation")
        family = _run_action(case["run_id"], "family_notification")
        case["teleconsultation_id"] = tele["reference_id"] if tele else None
        case["family_update_status"] = family["status"] if family else "pending"
        result.append(case)
    return result


def outcome_detail(run_id):
    return run_state(run_id)
