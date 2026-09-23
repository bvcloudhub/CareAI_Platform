"""Database-backed homepage metrics and patient overview helpers for Care.AI.

This module never inserts demo records. Optional-table failures are logged and
converted to safe zero/empty results so the landing page remains usable while
the database is being set up.

Important consistency rule
--------------------------
Any screen using patient_overview_rows() receives one canonical CURRENT
priority value and score.

The current priority is resolved by clinical_evidence_service.priority_snapshot()
and may come from:

1. the latest Care Intelligence overall risk, or
2. a genuinely ACTIVE Agentic / Hospital workflow.

Completed or rejected historical workflows do not become the patient's current
priority.

Historical workflow scores remain available in the workflow history itself.
For example, a completed fall workflow may retain CRITICAL 100 for audit
purposes while the patient's current Care Intelligence state has returned to
LOW 30.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime

from services.db import query_db
from services.clinical_evidence_service import priority_snapshot
from services.risk_engine import latest_vital_records


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


_PRIORITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _safe_rows(sql: str, args=(), label: str = "query"):
    """Execute a query returning rows without breaking optional dashboards."""
    try:
        return query_db(sql, args)
    except sqlite3.OperationalError as exc:
        log.warning(
            "Dashboard query %s unavailable: %s",
            label,
            exc,
        )
        return []


def _safe_one(sql: str, args=(), label: str = "query"):
    """Execute a query returning one row without breaking the dashboard."""
    try:
        return query_db(
            sql,
            args,
            one=True,
        )
    except sqlite3.OperationalError as exc:
        log.warning(
            "Dashboard query %s unavailable: %s",
            label,
            exc,
        )
        return None


def _safe_count(sql: str, args=(), label: str = "count") -> int:
    row = _safe_one(
        sql,
        args,
        label,
    )

    return (
        int(row["c"] or 0)
        if row and "c" in row.keys()
        else 0
    )


def _normalise_priority(value: str | None) -> str:
    """Convert all supported priority labels into one standard vocabulary."""
    value = (
        value or "low"
    ).strip().lower()

    if value == "attention":
        return "medium"

    if value == "stable":
        return "low"

    if value not in _PRIORITY_RANK:
        return "low"

    return value


def _highest_priority(*values: str | None) -> str:
    """Backward-compatible helper retained for callers that may import it."""
    levels = [
        _normalise_priority(value)
        for value in values
        if value
    ]

    return max(
        levels or ["low"],
        key=lambda level: _PRIORITY_RANK[level],
    )


def _fallback_current_priority(item: dict) -> dict:
    """Build a safe current-priority result from current Care Intelligence.

    This is used only if the shared priority resolver cannot be queried because
    an optional workflow table is unavailable.

    The level and score deliberately come from the same Care Intelligence
    record so invalid combinations such as CRITICAL · 30 cannot be produced.
    """
    level = _normalise_priority(
        item.get("risk_level")
    )

    score = int(
        item.get("risk_score") or 0
    )

    return {
        "level": level,
        "score": score,
        "reason": (
            "Current Care Intelligence risk assessment"
        ),
        "source": "Care intelligence",
        "run_id": None,
        "module_key": None,
        "updated_at": None,
    }


def _safe_priority_snapshot(patient_id: int, item: dict) -> dict:
    """Resolve one canonical CURRENT priority for a patient.

    priority_snapshot() is the platform source of truth.

    If an older/incomplete database is missing an optional Agentic or Hospital
    table, the patient list still renders using the current Care Intelligence
    risk rather than failing the whole page.
    """
    try:
        snapshot = priority_snapshot(
            int(patient_id)
        )

        if snapshot:
            level = _normalise_priority(
                snapshot.get("level")
            )

            score = int(
                snapshot.get("score") or 0
            )

            return {
                **snapshot,
                "level": level,
                "score": score,
            }

    except sqlite3.OperationalError as exc:
        log.warning(
            "Current priority unavailable for patient %s: %s",
            patient_id,
            exc,
        )

    return _fallback_current_priority(
        item
    )


def normalise_patient_filter(value: str | None) -> str:
    value = (
        value or "all"
    ).strip().lower()

    value = _FILTER_ALIASES.get(
        value,
        value,
    )

    return (
        value
        if value in VALID_PATIENT_FILTERS
        else "all"
    )


def patient_overview_rows(
    filter_key: str = "all",
    limit: int | None = None,
) -> list[dict]:
    """Return active patients using one consistent current-priority model.

    This function is shared by the homepage Patient Overview and the complete
    Patients page.

    IMPORTANT:
    `patients.current_status` can contain an older operational/database state.
    Therefore the returned view model deliberately replaces `current_status`
    with the resolved CURRENT priority.

    This is only an in-memory dictionary change. It does NOT update the patient
    record in the database.

    As a result, all of these fields describe the same current state:

        priority
        priority_label
        priority_score
        risk_score
        current_status
        current_priority
        current_priority_score

    Historical Agentic workflow severity remains stored in Agentic Care and is
    not erased.
    """
    filter_key = normalise_patient_filter(
        filter_key
    )

    rows = _safe_rows(
        """
        SELECT
            p.id,
            p.external_ref,
            p.first_name,
            p.last_name,
            p.birth_date,
            p.city,
            p.living_setting,
            p.current_status,
            p.profile_photo_path,
            p.assigned_nurse_id,

            (
                SELECT r.level
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type='overall'
                ORDER BY r.id DESC
                LIMIT 1
            ) AS risk_level,

            (
                SELECT r.score
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type='overall'
                ORDER BY r.id DESC
                LIMIT 1
            ) AS risk_score,

            EXISTS(
                SELECT 1
                FROM conditions c
                WHERE c.patient_id=p.id
                  AND c.active=1
                  AND lower(c.display) LIKE '%copd%'
            ) AS has_copd,

            EXISTS(
                SELECT 1
                FROM conditions c
                WHERE c.patient_id=p.id
                  AND c.active=1
                  AND lower(c.display) LIKE '%fall%'
            ) AS has_fall_condition,

            (
                SELECT r.level
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type IN (
                      'fall_prediction',
                      'fall_event'
                  )
                ORDER BY r.id DESC
                LIMIT 1
            ) AS fall_risk_level,

            (
                SELECT r.score
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type IN (
                      'fall_prediction',
                      'fall_event'
                  )
                ORDER BY r.id DESC
                LIMIT 1
            ) AS fall_risk_score,

            EXISTS(
                SELECT 1
                FROM alerts a
                WHERE a.patient_id=p.id
                  AND a.status='open'
                  AND lower(a.alert_type) LIKE '%fall%'
            ) AS has_fall_alert,

            (
                SELECT r.level
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type='medication'
                ORDER BY r.id DESC
                LIMIT 1
            ) AS medication_risk_level,

            (
                SELECT r.score
                FROM risk_scores r
                WHERE r.patient_id=p.id
                  AND r.risk_type='medication'
                ORDER BY r.id DESC
                LIMIT 1
            ) AS medication_risk_score,

            EXISTS(
                SELECT 1
                FROM alerts a
                WHERE a.patient_id=p.id
                  AND a.status='open'
                  AND lower(a.alert_type) LIKE '%medication%'
            ) AS has_medication_alert

        FROM patients p
        WHERE p.active=1
        """,
        label="patient_overview",
    )

    result: list[dict] = []

    for row in rows:
        item = dict(row)

        # ----------------------------------------------------------
        # Preserve raw database/current Care Intelligence information
        # ----------------------------------------------------------

        item["stored_patient_status"] = (
            item.get("current_status")
        )

        item["care_risk_level"] = (
            _normalise_priority(
                item.get("risk_level")
            )
        )

        item["care_risk_score"] = int(
            item.get("risk_score") or 0
        )

        # ----------------------------------------------------------
        # Resolve ONE canonical CURRENT patient priority.
        # ----------------------------------------------------------

        snapshot = _safe_priority_snapshot(
            int(item["id"]),
            item,
        )

        resolved_level = (
            _normalise_priority(
                snapshot.get("level")
            )
        )

        resolved_score = int(
            snapshot.get("score") or 0
        )

        # ----------------------------------------------------------
        # Canonical current-priority fields
        # ----------------------------------------------------------

        item["priority"] = resolved_level

        item["priority_label"] = (
            resolved_level.title()
        )

        item["priority_score"] = (
            resolved_score
        )

        item["current_priority"] = (
            resolved_level
        )

        item["current_priority_label"] = (
            resolved_level.title()
        )

        item["current_priority_score"] = (
            resolved_score
        )

        # ----------------------------------------------------------
        # Backwards compatibility
        # ----------------------------------------------------------
        #
        # Older templates may still display:
        #
        #     patient.current_status
        #
        # instead of:
        #
        #     patient.priority
        #
        # If we leave the original database current_status untouched in
        # this view dictionary, the UI can display:
        #
        #     CRITICAL · 30
        #
        # even when the current Care Intelligence result is LOW · 30.
        #
        # Therefore current_status in THIS RETURNED VIEW MODEL is the
        # canonical current priority.
        #
        # The database is NOT modified.
        # ----------------------------------------------------------

        item["current_status"] = (
            resolved_level
        )

        # Existing templates commonly use risk_score beside the priority.
        # Keep it from exactly the same snapshot.
        item["risk_score"] = (
            resolved_score
        )

        item["priority_source"] = (
            snapshot.get("source")
            or "Care intelligence"
        )

        item["priority_reason"] = (
            snapshot.get("reason")
            or ""
        )

        item["priority_run_id"] = (
            snapshot.get("run_id")
        )

        item["priority_module_key"] = (
            snapshot.get("module_key")
        )

        item["priority_updated_at"] = (
            snapshot.get("updated_at")
        )

        # Backwards-compatible field.
        #
        # It is set only when an ACTIVE workflow currently contributes
        # the operational priority.
        #
        # Completed historical workflows therefore cannot keep this field
        # Critical.
        item["agentic_severity"] = (
            resolved_level
            if snapshot.get("run_id")
            else None
        )

        # ----------------------------------------------------------
        # Existing fall-risk filter logic
        # ----------------------------------------------------------

        fall_level = _normalise_priority(
            item.get("fall_risk_level")
        )

        fall_score = int(
            item.get("fall_risk_score") or 0
        )

        item["has_fall_risk"] = bool(
            item.get("has_fall_condition")
            or item.get("has_fall_alert")
            or fall_level
            in {
                "medium",
                "high",
                "critical",
            }
            or fall_score >= 45
        )

        # ----------------------------------------------------------
        # Existing medication-risk filter logic
        # ----------------------------------------------------------

        med_level = _normalise_priority(
            item.get("medication_risk_level")
        )

        med_score = int(
            item.get("medication_risk_score") or 0
        )

        item["has_medication_risk"] = bool(
            item.get("has_medication_alert")
            or med_level
            in {
                "medium",
                "high",
                "critical",
            }
            or med_score >= 45
        )

        # Backwards-compatible name used by the existing landing template.
        item["has_medication"] = (
            item["has_medication_risk"]
        )

        item["has_copd"] = bool(
            item.get("has_copd")
        )

        # ----------------------------------------------------------
        # Apply current-patient filters
        # ----------------------------------------------------------

        matches = (
            filter_key == "all"

            or (
                filter_key == "high_critical"
                and resolved_level
                in {
                    "high",
                    "critical",
                }
            )

            or (
                filter_key == "medium"
                and resolved_level == "medium"
            )

            or (
                filter_key == "low"
                and resolved_level == "low"
            )

            or (
                filter_key == "copd"
                and item["has_copd"]
            )

            or (
                filter_key == "fall_risk"
                and item["has_fall_risk"]
            )

            or (
                filter_key == "medication"
                and item[
                    "has_medication_risk"
                ]
            )
        )

        if matches:
            result.append(
                item
            )

    # Highest CURRENT priority first.
    result.sort(
        key=lambda patient: (
            _PRIORITY_RANK.get(
                patient.get(
                    "priority",
                    "low",
                ),
                1,
            ),
            int(
                patient.get(
                    "priority_score",
                    patient.get(
                        "risk_score",
                        0,
                    ),
                )
                or 0
            ),
            (
                patient.get(
                    "last_name",
                    "",
                )
                or ""
            ).lower(),
        ),
        reverse=True,
    )

    if limit is not None:
        return result[
            : max(
                0,
                int(limit),
            )
        ]

    return result


def dashboard_metrics() -> dict:
    """Return landing-page statistics from current database records."""
    all_patients = patient_overview_rows(
        "all",
        limit=None,
    )

    active_ids = {
        int(patient["id"])
        for patient in all_patients
    }

    high_ids = {
        int(patient["id"])
        for patient in all_patients
        if patient.get("priority")
        in {
            "high",
            "critical",
        }
    }

    todays_consultations = _safe_count(
        """
        SELECT COUNT(*) c
        FROM teleconsultations
        WHERE date(
            COALESCE(
                scheduled_at,
                created_at
            )
        ) = date(
            'now',
            'localtime'
        )
        """,
        label="todays_consultations",
    )

    open_questions = _safe_count(
        """
        SELECT COUNT(*) c
        FROM patient_questions
        WHERE status='open'
        """,
        label="open_patient_questions",
    )

    queued_notifications = _safe_count(
        """
        SELECT COUNT(*) c
        FROM notifications
        WHERE status='queued'
          AND lower(
              COALESCE(
                  audience,
                  ''
              )
          ) IN (
              'care_team',
              'family',
              'patient'
          )
        """,
        label="queued_notifications",
    )

    return {
        "active_patients": len(
            active_ids
        ),
        "high_priority": len(
            high_ids
        ),
        "todays_consultations": (
            todays_consultations
        ),
        "new_messages": (
            open_questions
            + queued_notifications
        ),
        "new_message_detail": {
            "open_patient_questions": (
                open_questions
            ),
            "queued_notifications": (
                queued_notifications
            ),

            # Current schema has no user-specific read flag
            # for chat_messages.
            "chat_unread_tracking": False,
        },
    }


def landing_patient_overview(
    limit: int | None = 5,
    filter_key: str = "all",
) -> list[dict]:
    """Compatibility wrapper for the landing Patient Overview."""
    return patient_overview_rows(
        filter_key=filter_key,
        limit=limit,
    )


def landing_featured_patient(
    patient_id: int | None = None,
) -> dict | None:
    """Return database-backed Patient 360 preview data for the homepage."""
    if patient_id is None:
        overview = landing_patient_overview(
            limit=1
        )

        patient_id = (
            overview[0]["id"]
            if overview
            else None
        )

    if not patient_id:
        return None

    patient = _safe_one(
        """
        SELECT *
        FROM patients
        WHERE id=?
          AND active=1
        """,
        (patient_id,),
        label="landing_featured_patient",
    )

    if not patient:
        return None

    p = dict(
        patient
    )

    # --------------------------------------------------------------
    # Use the shared latest-vitals resolver.
    # --------------------------------------------------------------

    vital_records = latest_vital_records(
        patient_id
    )

    vitals = {
        kind: {
            "value": row["value"],
            "unit": row.get("unit"),
            "source": row.get("source"),
            "measured_at": row.get(
                "measured_at"
            ),
        }
        for kind, row
        in vital_records.items()
        if row
    }

    # --------------------------------------------------------------
    # Current individual risk breakdown
    # --------------------------------------------------------------

    risk_rows = _safe_rows(
        """
        SELECT
            r.risk_type,
            r.score,
            r.level,
            r.explanation
        FROM risk_scores r
        JOIN (
            SELECT
                risk_type,
                MAX(id) max_id
            FROM risk_scores
            WHERE patient_id=?
            GROUP BY risk_type
        ) latest
          ON latest.max_id=r.id
        WHERE r.patient_id=?
          AND r.risk_type!='overall'
        ORDER BY r.score DESC
        LIMIT 3
        """,
        (
            patient_id,
            patient_id,
        ),
        label="landing_featured_risks",
    )

    conditions = [
        row["display"]
        for row in _safe_rows(
            """
            SELECT display
            FROM conditions
            WHERE patient_id=?
              AND active=1
            ORDER BY display
            """,
            (patient_id,),
            label=(
                "landing_featured_conditions"
            ),
        )
    ]

    latest_location = _safe_one(
        """
        SELECT
            location,
            created_at
        FROM device_events
        WHERE patient_id=?
          AND location IS NOT NULL
          AND trim(location)!=''
        ORDER BY id DESC
        LIMIT 1
        """,
        (patient_id,),
        label="landing_featured_location",
    )

    birth_date = None

    try:
        birth_date = datetime.strptime(
            p.get("birth_date") or "",
            "%Y-%m-%d",
        ).date()

    except ValueError:
        pass

    age = None

    if birth_date:
        today = date.today()

        age = (
            today.year
            - birth_date.year
            - (
                (
                    today.month,
                    today.day,
                )
                < (
                    birth_date.month,
                    birth_date.day,
                )
            )
        )

    def value(
        kind,
        default=None,
    ):
        entry = vitals.get(
            kind
        )

        return (
            entry["value"]
            if entry
            else default
        )

    bp_sys = value(
        "bp_sys"
    )

    bp_dia = value(
        "bp_dia"
    )

    bp_text = (
        "Not recorded"
    )

    if (
        bp_sys is not None
        and bp_dia is not None
    ):
        bp_text = (
            f"{bp_sys:.0f}/"
            f"{bp_dia:.0f}"
        )

    # --------------------------------------------------------------
    # Also expose the exact same current priority on the featured
    # homepage patient.
    # --------------------------------------------------------------

    priority = _safe_priority_snapshot(
        int(patient_id),
        {
            "risk_level": None,
            "risk_score": 0,
        },
    )

    current_level = _normalise_priority(
        priority.get("level")
    )

    current_score = int(
        priority.get("score") or 0
    )

    p.update(
        {
            "age": age,

            "conditions": (
                conditions
            ),

            "vitals": (
                vitals
            ),

            "spo2_text": (
                f"{value('spo2'):.1f}%"
                if value("spo2")
                is not None
                else "Not recorded"
            ),

            "hr_text": (
                f"{value('heart_rate'):.0f}"
                if value("heart_rate")
                is not None
                else "Not recorded"
            ),

            "bp_text": (
                bp_text
            ),

            "resp_text": (
                f"{value('resp_rate'):.0f}"
                if value("resp_rate")
                is not None
                else "Not recorded"
            ),

            "risks": [
                dict(row)
                for row in risk_rows
            ],

            "latest_location": (
                latest_location["location"]
                if latest_location
                else None
            ),

            "latest_location_at": (
                latest_location[
                    "created_at"
                ]
                if latest_location
                else None
            ),

            # Current canonical priority.
            "priority": (
                current_level
            ),

            "priority_label": (
                current_level.title()
            ),

            "priority_score": (
                current_score
            ),

            "risk_score": (
                current_score
            ),

            "current_priority": (
                current_level
            ),

            "current_priority_score": (
                current_score
            ),

            "priority_source": (
                priority.get("source")
                or "Care intelligence"
            ),

            "priority_reason": (
                priority.get("reason")
                or ""
            ),

            "priority_run_id": (
                priority.get("run_id")
            ),
        }
    )

    return p