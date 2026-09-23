"""Population analytics built from the live CareAI database.

This module intentionally reads the same patient, risk, condition, event,
wearable, consultation and Agentic Care tables used elsewhere in the platform.
No population-only patient records or hardcoded KPI counts are created.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from services.access_scope import patient_scope_clause
from services.db import query_db
from services.patient_service import patient_age
from services.clinical_evidence_service import priority_snapshot

RISK_RANK = {"critical": 4, "high": 3, "medium": 2, "attention": 2, "low": 1, "stable": 1}
ACTIVE_AGENTIC_STATUSES = {"running", "created", "detected", "triaged", "awaiting_approval", "approved", "responding", "in_progress"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _parse_date(value: str | None, fallback: date) -> date:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return fallback


def normalise_population_filters(args: Any) -> dict[str, Any]:
    today = date.today()
    default_from = today - timedelta(days=30)
    date_to = _parse_date(args.get("date_to"), today)
    date_from = _parse_date(args.get("date_from"), default_from)
    if date_from > date_to:
        date_from, date_to = date_to, date_from

    risk = _text(args.get("risk")).lower() or "all"
    if risk not in {"all", "critical", "high", "high_critical", "medium", "low"}:
        risk = "all"
    status = _text(args.get("status")).lower() or "active"
    if status not in {"active", "archived", "all"}:
        status = "active"
    wearable = _text(args.get("wearable")).lower() or "all"
    if wearable not in {"all", "connected", "not_connected", "recent", "stale"}:
        wearable = "all"
    age_group = _text(args.get("age_group")).lower() or "all"
    if age_group not in {"all", "under_65", "65_74", "75_84", "85_plus"}:
        age_group = "all"
    sex = _text(args.get("sex")).upper() or "all"
    if sex not in {"all", "M", "F", "X", "U"}:
        sex = "all"

    try:
        clinician_id = int(args.get("clinician_id") or 0) or None
    except (TypeError, ValueError):
        clinician_id = None

    cohort = _text(args.get("cohort")).lower()
    valid_cohorts = {
        "", "high_critical", "medium_risk", "low_risk", "fall_risk", "copd", "hypertension", "diabetes",
        "recent_deterioration", "recent_fall", "recent_device", "recent_alerts", "recent_consultations",
        "no_recent_wearable", "open_alerts", "open_agentic",
    }
    if cohort not in valid_cohorts:
        cohort = ""

    return {
        "risk": risk,
        "condition": _text(args.get("condition")),
        "age_group": age_group,
        "sex": sex,
        "location": _text(args.get("location")),
        "clinician_id": clinician_id,
        "wearable": wearable,
        "status": status,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "cohort": cohort,
    }


def _priority(current_status: str | None, risk_level: str | None, agentic_severity: str | None) -> str:
    candidates = [
        (_text(current_status).lower() or "stable"),
        (_text(risk_level).lower() or "low"),
        (_text(agentic_severity).lower() or "low"),
    ]
    mapped = ["medium" if c == "attention" else "low" if c == "stable" else c for c in candidates]
    return max(mapped, key=lambda v: RISK_RANK.get(v, 1))


def _scoped_base_rows(*, user_id: int, role: str, status: str = "active") -> list[dict[str, Any]]:
    scope_sql, scope_args = patient_scope_clause(user_id=user_id, role=role, alias="p")
    if status == "active":
        status_sql = "AND p.active=1"
    elif status == "archived":
        status_sql = "AND p.active=0"
    else:
        status_sql = ""

    rows = query_db(
        f"""
        SELECT p.*,
               n.display_name assigned_nurse_name,
               cuser.display_name assigned_clinician_name,
               (SELECT r.level FROM risk_scores r WHERE r.patient_id=p.id AND r.risk_type='overall' ORDER BY r.id DESC LIMIT 1) risk_level,
               (SELECT r.score FROM risk_scores r WHERE r.patient_id=p.id AND r.risk_type='overall' ORDER BY r.id DESC LIMIT 1) risk_score,
               (SELECT ar.severity
                  FROM agentic_runs ar
                 WHERE ar.patient_id=p.id
                   AND lower(ar.status) IN ('running','awaiting_approval','responding')
                 ORDER BY CASE lower(ar.severity)
                            WHEN 'critical' THEN 4
                            WHEN 'high' THEN 3
                            WHEN 'medium' THEN 2
                            ELSE 1
                          END DESC,
                          ar.id DESC
                 LIMIT 1) agentic_severity,
               (SELECT GROUP_CONCAT(c.display, ', ') FROM conditions c WHERE c.patient_id=p.id AND c.active=1) condition_summary,
               (SELECT MAX(v.measured_at) FROM vitals v WHERE v.patient_id=p.id) latest_vitals_at,
               (SELECT MAX(v.measured_at) FROM vitals v WHERE v.patient_id=p.id AND lower(COALESCE(v.source,'')) NOT IN ('simulator','manual')) latest_external_vital_at,
               (SELECT MAX(m.measured_at) FROM device_adapter_measurements m WHERE m.patient_id=p.id) latest_adapter_at,
               (SELECT MAX(wm.measured_at) FROM wearable_measurements wm WHERE wm.patient_id=p.id) latest_wearable_at,
               (SELECT COUNT(*) FROM wearable_devices wd WHERE wd.patient_id=p.id AND wd.status='active') active_wearable_devices,
               (SELECT a.alert_type FROM alerts a WHERE a.patient_id=p.id ORDER BY a.created_at DESC LIMIT 1) latest_alert_type,
               (SELECT a.created_at FROM alerts a WHERE a.patient_id=p.id ORDER BY a.created_at DESC LIMIT 1) latest_alert_at,
               (SELECT ce.event_type FROM care_events ce WHERE ce.patient_id=p.id ORDER BY ce.created_at DESC LIMIT 1) latest_event_type,
               (SELECT ce.created_at FROM care_events ce WHERE ce.patient_id=p.id ORDER BY ce.created_at DESC LIMIT 1) latest_event_at,
               (SELECT COUNT(*) FROM alerts a WHERE a.patient_id=p.id AND a.status='open') open_alert_count,
               (SELECT COUNT(*) FROM agentic_runs ar WHERE ar.patient_id=p.id AND lower(ar.status) IN ('running','created','detected','triaged','awaiting_approval','approved','responding','in_progress')) open_agentic_count
        FROM patients p
        LEFT JOIN users n ON n.id=p.assigned_nurse_id
        LEFT JOIN users cuser ON cuser.id=p.assigned_clinician_id
        WHERE {scope_sql} {status_sql}
        ORDER BY p.active DESC, p.last_name, p.first_name
        """,
        scope_args,
    )

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["age"] = patient_age(item.get("birth_date"))

        # Keep raw Care Intelligence values available for audit/explanation.
        item["care_risk_level"] = (
            _text(item.get("risk_level")).lower() or "low"
        )
        item["care_risk_score"] = int(item.get("risk_score") or 0)

        # Use exactly the same current-priority resolver as Command Centre
        # and Patient 360. This prevents combinations such as
        # "CRITICAL · Score 30".
        snapshot = priority_snapshot(int(item["id"]))

        item["priority"] = (
            _text(snapshot.get("level")).lower() or "low"
        )
        item["risk_score"] = int(snapshot.get("score") or 0)
        item["priority_source"] = (
            snapshot.get("source") or "Care intelligence"
        )
        item["priority_reason"] = snapshot.get("reason") or ""
        item["priority_run_id"] = snapshot.get("run_id")
        external_times = [item.get("latest_external_vital_at"), item.get("latest_adapter_at"), item.get("latest_wearable_at")]
        external_times = [t for t in external_times if t]
        item["latest_device_at"] = max(external_times) if external_times else None
        item["wearable_connected"] = bool(int(item.get("active_wearable_devices") or 0) or item.get("latest_device_at"))
        item["assigned_professional_id"] = item.get("assigned_nurse_id") or item.get("assigned_clinician_id")
        item["assigned_professional"] = item.get("assigned_nurse_name") or item.get("assigned_clinician_name") or "Unassigned"
        alert_at = item.get("latest_alert_at") or ""
        event_at = item.get("latest_event_at") or ""
        if alert_at and alert_at >= event_at:
            item["latest_activity"] = item.get("latest_alert_type") or "Alert"
            item["latest_activity_at"] = alert_at
        elif event_at:
            item["latest_activity"] = item.get("latest_event_type") or "Care event"
            item["latest_activity_at"] = event_at
        else:
            item["latest_activity"] = "No recent event"
            item["latest_activity_at"] = None
        result.append(item)
    return result


def _ids_with_recent_activity(patient_ids: list[int], *, date_from: str, date_to: str) -> dict[str, set[int]]:
    result = {
        "alerts": set(), "consultations": set(), "agentic": set(),
        "deterioration": set(), "falls": set(), "wearable": set(),
    }
    if not patient_ids:
        return result
    marks = ",".join("?" for _ in patient_ids)
    end = f"{date_to} 23:59:59"
    args = (*patient_ids, f"{date_from} 00:00:00", end)

    for r in query_db(f"SELECT DISTINCT patient_id FROM alerts WHERE patient_id IN ({marks}) AND created_at BETWEEN ? AND ?", args):
        result["alerts"].add(int(r["patient_id"]))
    for r in query_db(f"SELECT DISTINCT patient_id FROM teleconsultations WHERE patient_id IN ({marks}) AND created_at BETWEEN ? AND ?", args):
        result["consultations"].add(int(r["patient_id"]))
    for r in query_db(f"SELECT DISTINCT patient_id FROM agentic_runs WHERE patient_id IN ({marks}) AND started_at BETWEEN ? AND ?", args):
        result["agentic"].add(int(r["patient_id"]))
    for r in query_db(
        f"""SELECT DISTINCT patient_id FROM agentic_runs
            WHERE patient_id IN ({marks}) AND started_at BETWEEN ? AND ?
              AND (lower(scenario) LIKE '%deterior%' OR lower(scenario) LIKE '%declin%' OR lower(scenario) LIKE '%instabil%')""",
        args,
    ):
        result["deterioration"].add(int(r["patient_id"]))
    for r in query_db(
        f"""SELECT DISTINCT patient_id FROM agentic_runs
            WHERE patient_id IN ({marks}) AND started_at BETWEEN ? AND ?
              AND lower(scenario) LIKE '%fall%'""",
        args,
    ):
        result["falls"].add(int(r["patient_id"]))
    for r in query_db(
        f"""SELECT DISTINCT patient_id FROM device_events
            WHERE patient_id IN ({marks}) AND created_at BETWEEN ? AND ?
              AND lower(event_type) LIKE '%fall%'""",
        args,
    ):
        result["falls"].add(int(r["patient_id"]))

    wearable_sql = f"""
        SELECT DISTINCT patient_id FROM (
          SELECT patient_id, measured_at activity_at FROM device_adapter_measurements WHERE patient_id IN ({marks})
          UNION ALL
          SELECT patient_id, measured_at activity_at FROM wearable_measurements WHERE patient_id IN ({marks})
          UNION ALL
          SELECT patient_id, measured_at activity_at FROM vitals
            WHERE patient_id IN ({marks}) AND lower(COALESCE(source,'')) NOT IN ('simulator','manual')
        ) x WHERE datetime(activity_at) BETWEEN datetime(?) AND datetime(?)
    """
    wearable_args = (*patient_ids, *patient_ids, *patient_ids, f"{date_from} 00:00:00", end)
    for r in query_db(wearable_sql, wearable_args):
        result["wearable"].add(int(r["patient_id"]))
    return result


def _matches_age(age: int | None, age_group: str) -> bool:
    if age_group == "all":
        return True
    if age is None:
        return False
    if age_group == "under_65":
        return age < 65
    if age_group == "65_74":
        return 65 <= age <= 74
    if age_group == "75_84":
        return 75 <= age <= 84
    return age >= 85


def _has_condition(item: dict[str, Any], needle: str) -> bool:
    return needle.casefold() in _text(item.get("condition_summary")).casefold()


def _cohort_match(item: dict[str, Any], cohort: str, recent: dict[str, set[int]]) -> bool:
    pid = int(item["id"])
    if not cohort:
        return True
    if cohort == "high_critical":
        return item["priority"] in {"high", "critical"}
    if cohort == "medium_risk":
        return item["priority"] == "medium"
    if cohort == "low_risk":
        return item["priority"] == "low"
    if cohort == "fall_risk":
        return _has_condition(item, "fall risk")
    if cohort == "copd":
        return _has_condition(item, "copd")
    if cohort == "hypertension":
        return _has_condition(item, "hypertension")
    if cohort == "diabetes":
        return _has_condition(item, "diabetes")
    if cohort == "recent_deterioration":
        return pid in recent["deterioration"]
    if cohort == "recent_fall":
        return pid in recent["falls"]
    if cohort == "recent_device":
        return pid in recent["wearable"]
    if cohort == "recent_alerts":
        return pid in recent["alerts"]
    if cohort == "recent_consultations":
        return pid in recent["consultations"]
    if cohort == "open_agentic":
        return int(item.get("open_agentic_count") or 0) > 0
    if cohort == "no_recent_wearable":
        return pid not in recent["wearable"]
    if cohort == "open_alerts":
        return int(item.get("open_alert_count") or 0) > 0
    return True


def population_dashboard(*, user_id: int, role: str, filters: dict[str, Any]) -> dict[str, Any]:
    base_rows = _scoped_base_rows(user_id=user_id, role=role, status=filters["status"])
    ids = [int(p["id"]) for p in base_rows]
    recent = _ids_with_recent_activity(ids, date_from=filters["date_from"], date_to=filters["date_to"])

    filtered: list[dict[str, Any]] = []
    condition_q = filters["condition"].casefold()
    location_q = filters["location"].casefold()
    for item in base_rows:
        pid = int(item["id"])
        if filters["risk"] == "high_critical" and item["priority"] not in {"high", "critical"}:
            continue
        if filters["risk"] in {"critical", "high", "medium", "low"} and item["priority"] != filters["risk"]:
            continue
        if condition_q and condition_q not in _text(item.get("condition_summary")).casefold():
            continue
        if not _matches_age(item.get("age"), filters["age_group"]):
            continue
        if filters["sex"] != "all" and _text(item.get("sex")).upper() != filters["sex"]:
            continue
        if location_q and location_q != _text(item.get("city")).casefold():
            continue
        if filters["clinician_id"] and int(item.get("assigned_nurse_id") or 0) != filters["clinician_id"] and int(item.get("assigned_clinician_id") or 0) != filters["clinician_id"]:
            continue
        if filters["wearable"] == "connected" and not item["wearable_connected"]:
            continue
        if filters["wearable"] == "not_connected" and item["wearable_connected"]:
            continue
        if filters["wearable"] == "recent" and pid not in recent["wearable"]:
            continue
        if filters["wearable"] == "stale" and pid in recent["wearable"]:
            continue
        if not _cohort_match(item, filters["cohort"], recent):
            continue
        filtered.append(item)

    filtered_ids = [int(p["id"]) for p in filtered]
    filtered_set = set(filtered_ids)

    metrics = {
        "total": len(filtered),
        "active": sum(1 for p in filtered if int(p.get("active") or 0) == 1),
        "high_critical": sum(1 for p in filtered if p["priority"] in {"high", "critical"}),
        "medium": sum(1 for p in filtered if p["priority"] == "medium"),
        "low": sum(1 for p in filtered if p["priority"] == "low"),
        "fall_risk": sum(1 for p in filtered if _has_condition(p, "fall risk")),
        "copd": sum(1 for p in filtered if _has_condition(p, "copd")),
        "hypertension": sum(1 for p in filtered if _has_condition(p, "hypertension")),
        "diabetes": sum(1 for p in filtered if _has_condition(p, "diabetes")),
        "wearable_recent": len(filtered_set & recent["wearable"]),
        "recent_alerts": len(filtered_set & recent["alerts"]),
        "recent_consultations": len(filtered_set & recent["consultations"]),
        "open_agentic": sum(1 for p in filtered if int(p.get("open_agentic_count") or 0) > 0),
        "open_alerts": sum(int(p.get("open_alert_count") or 0) for p in filtered),
    }

    cohort_cards = [
        {"key": "high_critical", "label": "High / Critical risk", "value": metrics["high_critical"]},
        {"key": "fall_risk", "label": "Fall risk", "value": metrics["fall_risk"]},
        {"key": "copd", "label": "COPD", "value": metrics["copd"]},
        {"key": "hypertension", "label": "Hypertension", "value": metrics["hypertension"]},
        {"key": "diabetes", "label": "Diabetes", "value": metrics["diabetes"]},
        {"key": "recent_deterioration", "label": "Recent deterioration", "value": len(filtered_set & recent["deterioration"])},
        {"key": "recent_fall", "label": "Recent fall event", "value": len(filtered_set & recent["falls"])},
        {"key": "recent_device", "label": "Recent wearable / device data", "value": len(filtered_set & recent["wearable"])},
        {"key": "no_recent_wearable", "label": "No recent wearable data", "value": len(filtered_set - recent["wearable"])},
        {"key": "open_alerts", "label": "Patients with open alerts", "value": sum(1 for p in filtered if int(p.get("open_alert_count") or 0) > 0)},
        {"key": "open_agentic", "label": "Open Agentic Care workflow", "value": metrics["open_agentic"]},
    ]

    charts = _population_charts(filtered, filtered_ids, filters, recent)
    return {
        "metrics": metrics,
        "patients": filtered,
        "cohorts": cohort_cards,
        "charts": charts,
        "available_total": len(base_rows),
    }


def _population_charts(patients: list[dict[str, Any]], patient_ids: list[int], filters: dict[str, Any], recent: dict[str, set[int]]) -> dict[str, list[dict[str, Any]]]:
    risk_counter = Counter(p["priority"] for p in patients)
    risk_order = ["critical", "high", "medium", "low"]
    risk = [{"label": key.title(), "value": risk_counter.get(key, 0)} for key in risk_order]

    conditions: Counter[str] = Counter()
    for p in patients:
        for condition in [c.strip() for c in _text(p.get("condition_summary")).split(",") if c.strip()]:
            conditions[condition] += 1
    condition = [{"label": k, "value": v} for k, v in conditions.most_common(10)]

    recent_device = len(set(patient_ids) & recent["wearable"])
    wearable = [
        {"label": "Data in selected period", "value": recent_device},
        {"label": "No data in selected period", "value": max(len(patients) - recent_device, 0)},
    ]

    alerts, consultations, agentic_types = [], [], []
    if patient_ids:
        marks = ",".join("?" for _ in patient_ids)
        date_args = (*patient_ids, f"{filters['date_from']} 00:00:00", f"{filters['date_to']} 23:59:59")
        alert_rows = query_db(
            f"SELECT date(created_at) label, COUNT(*) value FROM alerts WHERE patient_id IN ({marks}) AND created_at BETWEEN ? AND ? GROUP BY date(created_at) ORDER BY label",
            date_args,
        )
        consult_rows = query_db(
            f"SELECT date(created_at) label, COUNT(*) value FROM teleconsultations WHERE patient_id IN ({marks}) AND created_at BETWEEN ? AND ? GROUP BY date(created_at) ORDER BY label",
            date_args,
        )
        agentic_rows = query_db(
            f"SELECT scenario label, COUNT(*) value FROM agentic_runs WHERE patient_id IN ({marks}) AND started_at BETWEEN ? AND ? GROUP BY scenario ORDER BY value DESC LIMIT 8",
            date_args,
        )
        alerts = [dict(r) for r in alert_rows]
        consultations = [dict(r) for r in consult_rows]
        agentic_types = [{"label": _text(r["label"]).replace("Hospital EHR anomaly: ", ""), "value": r["value"]} for r in agentic_rows]

    return {
        "risk": risk,
        "conditions": condition,
        "wearable": wearable,
        "alerts": alerts,
        "consultations": consultations,
        "agentic_types": agentic_types,
    }


def population_filter_options(*, user_id: int, role: str) -> dict[str, Any]:
    rows = _scoped_base_rows(user_id=user_id, role=role, status="all")
    conditions = sorted({c.strip() for p in rows for c in _text(p.get("condition_summary")).split(",") if c.strip()}, key=str.casefold)
    locations = sorted({_text(p.get("city")) for p in rows if _text(p.get("city"))}, key=str.casefold)
    scope_sql, scope_args = patient_scope_clause(user_id=user_id, role=role, alias="p")
    clinicians = query_db(
        f"""SELECT DISTINCT u.id,u.display_name,u.role
            FROM users u JOIN patients p ON (p.assigned_nurse_id=u.id OR p.assigned_clinician_id=u.id)
            WHERE {scope_sql} AND u.active=1 ORDER BY u.display_name""",
        scope_args,
    )
    return {"conditions": conditions, "locations": locations, "clinicians": [dict(r) for r in clinicians]}


def population_metrics(*, user_id: int | None = None, role: str | None = None) -> dict[str, int]:
    """Compatibility KPI helper used by Command Centre and older views.

    Without a user context this preserves the historical organisation-level
    metrics. Reports/Population pass the logged-in user and receive scoped data.
    """
    if user_id is not None and role:
        filters = normalise_population_filters({})
        return population_dashboard(user_id=user_id, role=role, filters=filters)["metrics"]

    p = query_db(
        """SELECT COUNT(*) total,
           SUM(CASE WHEN current_status='stable' THEN 1 ELSE 0 END) stable,
           SUM(CASE WHEN current_status='attention' THEN 1 ELSE 0 END) attention,
           SUM(CASE WHEN current_status='high' THEN 1 ELSE 0 END) high,
           SUM(CASE WHEN current_status='critical' THEN 1 ELSE 0 END) critical
           FROM patients WHERE active=1""",
        one=True,
    )
    alerts = query_db(
        """SELECT COUNT(*) c FROM alerts a JOIN patients p ON p.id=a.patient_id
           WHERE a.status='open' AND p.active=1""",
        one=True,
    )["c"]
    tasks = query_db(
        """SELECT COUNT(*) c FROM care_tasks t JOIN patients p ON p.id=t.patient_id
           WHERE t.status='open' AND p.active=1""",
        one=True,
    )["c"]
    interventions = query_db(
        "SELECT COUNT(*) c FROM care_events WHERE event_type IN ('agent_recommendation','human_action') AND created_at>=datetime('now','-30 days')",
        one=True,
    )["c"]
    return {**dict(p), "open_alerts": alerts, "open_tasks": tasks, "interventions_30d": interventions}
