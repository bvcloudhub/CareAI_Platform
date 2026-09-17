"""Database-backed homepage metrics and patient overview helpers for Care.AI.

This module never inserts demo records. Optional-table failures are logged and converted to
safe zero/empty results so the landing page remains usable while the database is being set up.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime

from services.db import query_db

log = logging.getLogger(__name__)

VALID_PATIENT_FILTERS = {
    "all",
    "high_critical",
    "medium",
    "low",
    "copd",
    "fall_risk",
    "medication",
}

_FILTER_ALIASES = {
    "high": "high_critical",
    "critical": "high_critical",
    "fall": "fall_risk",
    "fall-risk": "fall_risk",
    "meds": "medication",
}

_PRIORITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _safe_rows(sql: str, args=(), label: str = "query"):
    try:
        return query_db(sql, args)
    except sqlite3.OperationalError as exc:
        log.warning("Dashboard query %s unavailable: %s", label, exc)
        return []


def _safe_one(sql: str, args=(), label: str = "query"):
    try:
        return query_db(sql, args, one=True)
    except sqlite3.OperationalError as exc:
        log.warning("Dashboard query %s unavailable: %s", label, exc)
        return None


def _safe_count(sql: str, args=(), label: str = "count") -> int:
    row = _safe_one(sql, args, label)
    return int(row["c"] or 0) if row and "c" in row.keys() else 0


def _normalise_priority(value: str | None) -> str:
    value = (value or "low").strip().lower()
    if value == "attention":
        return "medium"
    if value not in _PRIORITY_RANK:
        return "low"
    return value


def _highest_priority(*values: str | None) -> str:
    levels = [_normalise_priority(v) for v in values if v]
    return max(levels or ["low"], key=lambda level: _PRIORITY_RANK[level])


def normalise_patient_filter(value: str | None) -> str:
    value = (value or "all").strip().lower()
    value = _FILTER_ALIASES.get(value, value)
    return value if value in VALID_PATIENT_FILTERS else "all"


def patient_overview_rows(filter_key: str = "all", limit: int | None = None) -> list[dict]:
    """Return active patients with a single, consistent priority/tag model.

    The same function is used by the landing Patient Overview and the full Patients page.
    That prevents the two views from disagreeing about who is High/Critical or which tag a
    patient belongs to.
    """
    filter_key = normalise_patient_filter(filter_key)

    rows = _safe_rows(
        """
        SELECT p.id,p.external_ref,p.first_name,p.last_name,p.birth_date,p.city,
               p.living_setting,p.current_status,p.profile_photo_path,p.assigned_nurse_id,
          (SELECT r.level FROM risk_scores r
             WHERE r.patient_id=p.id AND r.risk_type='overall'
             ORDER BY r.id DESC LIMIT 1) risk_level,
          (SELECT r.score FROM risk_scores r
             WHERE r.patient_id=p.id AND r.risk_type='overall'
             ORDER BY r.id DESC LIMIT 1) risk_score,
          EXISTS(SELECT 1 FROM conditions c
                   WHERE c.patient_id=p.id AND c.active=1 AND lower(c.display) LIKE '%copd%') has_copd,
          EXISTS(SELECT 1 FROM conditions c
                   WHERE c.patient_id=p.id AND c.active=1 AND lower(c.display) LIKE '%fall%') has_fall_condition,
          (SELECT r.level FROM risk_scores r
             WHERE r.patient_id=p.id AND r.risk_type IN ('fall_prediction','fall_event')
             ORDER BY r.id DESC LIMIT 1) fall_risk_level,
          (SELECT r.score FROM risk_scores r
             WHERE r.patient_id=p.id AND r.risk_type IN ('fall_prediction','fall_event')
             ORDER BY r.id DESC LIMIT 1) fall_risk_score,
          EXISTS(SELECT 1 FROM alerts a
                   WHERE a.patient_id=p.id AND a.status='open' AND lower(a.alert_type) LIKE '%fall%') has_fall_alert,
          (SELECT r.level FROM risk_scores r
             WHERE r.patient_id=p.id AND r.risk_type='medication'
             ORDER BY r.id DESC LIMIT 1) medication_risk_level,
          (SELECT r.score FROM risk_scores r
             WHERE r.patient_id=p.id AND r.risk_type='medication'
             ORDER BY r.id DESC LIMIT 1) medication_risk_score,
          EXISTS(SELECT 1 FROM alerts a
                   WHERE a.patient_id=p.id AND a.status='open' AND lower(a.alert_type) LIKE '%medication%') has_medication_alert
        FROM patients p
        WHERE p.active=1
        """,
        label="patient_overview",
    )

    # Agentic priority is intentionally queried separately. If an older database does not
    # yet have the agentic tables, the normal patient list still renders correctly.
    agentic_rows = _safe_rows(
        """
        SELECT ar.patient_id,ar.severity,ar.status
        FROM agentic_runs ar
        JOIN (
          SELECT patient_id,MAX(id) max_id
          FROM agentic_runs
          WHERE status!='rejected'
          GROUP BY patient_id
        ) latest ON latest.max_id=ar.id
        """,
        label="latest_agentic_priority",
    )
    agentic_by_patient = {int(r["patient_id"]): dict(r) for r in agentic_rows}

    result: list[dict] = []
    for row in rows:
        item = dict(row)
        agentic = agentic_by_patient.get(int(item["id"]))
        item["agentic_severity"] = agentic.get("severity") if agentic else None
        item["priority"] = _highest_priority(
            item.get("current_status"), item.get("risk_level"), item.get("agentic_severity")
        )
        item["priority_label"] = item["priority"].title()
        item["risk_score"] = int(item.get("risk_score") or 0)

        fall_level = _normalise_priority(item.get("fall_risk_level"))
        fall_score = int(item.get("fall_risk_score") or 0)
        item["has_fall_risk"] = bool(
            item.get("has_fall_condition")
            or item.get("has_fall_alert")
            or fall_level in {"medium", "high", "critical"}
            or fall_score >= 45
        )

        med_level = _normalise_priority(item.get("medication_risk_level"))
        med_score = int(item.get("medication_risk_score") or 0)
        item["has_medication_risk"] = bool(
            item.get("has_medication_alert")
            or med_level in {"medium", "high", "critical"}
            or med_score >= 45
        )
        # Backwards-compatible name used by the landing template before this update.
        item["has_medication"] = item["has_medication_risk"]
        item["has_copd"] = bool(item.get("has_copd"))

        matches = (
            filter_key == "all"
            or (filter_key == "high_critical" and item["priority"] in {"high", "critical"})
            or (filter_key == "medium" and item["priority"] == "medium")
            or (filter_key == "low" and item["priority"] == "low")
            or (filter_key == "copd" and item["has_copd"])
            or (filter_key == "fall_risk" and item["has_fall_risk"])
            or (filter_key == "medication" and item["has_medication_risk"])
        )
        if matches:
            result.append(item)

    result.sort(
        key=lambda p: (
            _PRIORITY_RANK.get(p["priority"], 1),
            int(p.get("risk_score") or 0),
            p.get("last_name", "").lower(),
        ),
        reverse=True,
    )
    if limit is not None:
        return result[: max(0, int(limit))]
    return result


def dashboard_metrics() -> dict:
    """Return the four landing-page statistics from existing database records only."""
    all_patients = patient_overview_rows("all", limit=None)
    active_ids = {int(p["id"]) for p in all_patients}
    high_ids = {
        int(p["id"])
        for p in all_patients
        if p.get("priority") in {"high", "critical"}
    }

    todays_consultations = _safe_count(
        """
        SELECT COUNT(*) c
        FROM teleconsultations
        WHERE date(COALESCE(scheduled_at,created_at))=date('now','localtime')
        """,
        label="todays_consultations",
    )

    open_questions = _safe_count(
        "SELECT COUNT(*) c FROM patient_questions WHERE status='open'",
        label="open_patient_questions",
    )
    queued_notifications = _safe_count(
        """
        SELECT COUNT(*) c
        FROM notifications
        WHERE status='queued' AND lower(COALESCE(audience,'')) IN ('care_team','family','patient')
        """,
        label="queued_notifications",
    )

    return {
        "active_patients": len(active_ids),
        "high_priority": len(high_ids),
        "todays_consultations": todays_consultations,
        "new_messages": open_questions + queued_notifications,
        "new_message_detail": {
            "open_patient_questions": open_questions,
            "queued_notifications": queued_notifications,
            # Current schema has no user-specific read flag for chat_messages.
            "chat_unread_tracking": False,
        },
    }


def landing_patient_overview(limit: int | None = 5, filter_key: str = "all") -> list[dict]:
    """Compatibility wrapper for the landing Patient Overview."""
    return patient_overview_rows(filter_key=filter_key, limit=limit)


def landing_featured_patient(patient_id: int | None = None) -> dict | None:
    """Return database-backed Patient 360 preview data for the homepage."""
    if patient_id is None:
        overview = landing_patient_overview(limit=1)
        patient_id = overview[0]["id"] if overview else None
    if not patient_id:
        return None

    patient = _safe_one(
        "SELECT * FROM patients WHERE id=? AND active=1",
        (patient_id,),
        label="landing_featured_patient",
    )
    if not patient:
        return None

    p = dict(patient)
    vitals_rows = _safe_rows(
        """
        SELECT v.kind,v.value,v.unit
        FROM vitals v
        JOIN (
          SELECT kind,MAX(id) max_id FROM vitals WHERE patient_id=? GROUP BY kind
        ) latest ON latest.max_id=v.id
        WHERE v.patient_id=?
        """,
        (patient_id, patient_id),
        label="landing_featured_vitals",
    )
    vitals = {r["kind"]: {"value": r["value"], "unit": r["unit"]} for r in vitals_rows}

    risk_rows = _safe_rows(
        """
        SELECT r.risk_type,r.score,r.level,r.explanation
        FROM risk_scores r
        JOIN (
          SELECT risk_type,MAX(id) max_id FROM risk_scores WHERE patient_id=? GROUP BY risk_type
        ) latest ON latest.max_id=r.id
        WHERE r.patient_id=? AND r.risk_type!='overall'
        ORDER BY r.score DESC
        LIMIT 3
        """,
        (patient_id, patient_id),
        label="landing_featured_risks",
    )

    conditions = [
        r["display"]
        for r in _safe_rows(
            "SELECT display FROM conditions WHERE patient_id=? AND active=1 ORDER BY display",
            (patient_id,),
            label="landing_featured_conditions",
        )
    ]

    latest_location = _safe_one(
        """
        SELECT location,created_at FROM device_events
        WHERE patient_id=? AND location IS NOT NULL AND trim(location)!=''
        ORDER BY id DESC LIMIT 1
        """,
        (patient_id,),
        label="landing_featured_location",
    )

    birth_date = None
    try:
        birth_date = datetime.strptime(p.get("birth_date") or "", "%Y-%m-%d").date()
    except ValueError:
        pass
    age = None
    if birth_date:
        today = date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

    def value(kind, default=None):
        entry = vitals.get(kind)
        return entry["value"] if entry else default

    bp_sys = value("bp_sys")
    bp_dia = value("bp_dia")
    bp_text = "Not recorded"
    if bp_sys is not None and bp_dia is not None:
        bp_text = f"{bp_sys:.0f}/{bp_dia:.0f}"

    p.update(
        {
            "age": age,
            "conditions": conditions,
            "vitals": vitals,
            "spo2_text": f"{value('spo2'):.1f}%" if value("spo2") is not None else "Not recorded",
            "hr_text": f"{value('heart_rate'):.0f}" if value("heart_rate") is not None else "Not recorded",
            "bp_text": bp_text,
            "resp_text": f"{value('resp_rate'):.0f}" if value("resp_rate") is not None else "Not recorded",
            "risks": [dict(r) for r in risk_rows],
            "latest_location": latest_location["location"] if latest_location else None,
            "latest_location_at": latest_location["created_at"] if latest_location else None,
        }
    )
    return p
