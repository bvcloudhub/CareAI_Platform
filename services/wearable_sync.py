"""CareAI wearable normalization, authentication and idempotent ingestion.

CareAI remains the source of truth. Wearable clients are paired to exactly one
patient by a random token stored only as a SHA-256 hash in the database. Incoming
vendor payloads are normalized through wearable_mappings.json and then written to
both the wearable audit store and the existing ``vitals`` table used by CareAI's
risk, Monitoring Agent, Patient Twin and Command Centre logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import secrets
from typing import Any

from services.db import get_conn, query_db

_MAPPING_FILE = Path(__file__).with_name("wearable_mappings.json")
with _MAPPING_FILE.open("r", encoding="utf-8") as handle:
    PROVIDER_MAPPINGS = json.load(handle)

PROVIDERS = {
    key: {
        "key": key,
        "label": value["label"],
        "sdk": value["sdk"],
        "icon": value.get("icon", "⌚"),
    }
    for key, value in PROVIDER_MAPPINGS.items()
}

_METRIC_RANGES = {
    "heart_rate": (20.0, 250.0),
    "spo2": (50.0, 100.0),
    "resp_rate": (4.0, 80.0),
    "bp_sys": (50.0, 300.0),
    "bp_dia": (30.0, 200.0),
    "activity": (0.0, 100.0),
    "sleep": (0.0, 100.0),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalise_timestamp(value: Any) -> str:
    if not value:
        return _utc_now()
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("Invalid measurement timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _steps_to_activity_score(steps: float) -> float:
    # Preserve the source project's demo convention: 5,000 steps ~= score 80.
    return round(min(100.0, (float(steps) / 5000.0) * 80.0), 2)


def _sleep_hours_to_score(hours: float) -> float:
    # Preserve the source project's demo convention: 8h ~= score 90.
    return round(max(0.0, min(100.0, (float(hours) / 8.0) * 90.0)), 2)


def _transform_value(transform_name: str | None, value: float) -> tuple[float, float | None]:
    if transform_name == "steps_to_activity":
        return _steps_to_activity_score(value), float(value)
    if transform_name == "sleep_to_score":
        return _sleep_hours_to_score(value), float(value)
    return float(value), None


def _fallback_source_record_id(provider: str, vendor_type: str, measured_at: str, value: Any) -> str:
    raw = f"{provider}|{vendor_type}|{measured_at}|{value}".encode("utf-8")
    return "fallback-" + hashlib.sha256(raw).hexdigest()


def _validate_metric(metric: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"Invalid {metric} value")
    bounds = _METRIC_RANGES.get(metric)
    if bounds and not (bounds[0] <= value <= bounds[1]):
        raise ValueError(f"{metric} value outside accepted range")


def create_pairing_token(patient_ref: str, provider: str = "apple_watch", account_label: str | None = None) -> dict[str, Any]:
    """Create a one-time-display token tied to a CareAI patient.

    Plaintext tokens are never stored. The returned token must be copied into the
    companion app and should be treated as a credential.
    """
    if provider not in PROVIDERS:
        raise ValueError("Unknown wearable provider")
    patient = query_db(
        "SELECT id,external_ref,first_name,last_name,active,consent_monitoring FROM patients WHERE external_ref=?",
        (patient_ref,),
        one=True,
    )
    if not patient or not patient["active"]:
        raise ValueError("Active patient not found")
    if not patient["consent_monitoring"]:
        raise ValueError("Patient monitoring consent is not active")

    token = secrets.token_urlsafe(32)
    now = _utc_now()
    conn = get_conn()
    try:
        cursor = conn.execute(
            """
            INSERT INTO wearable_devices(
              patient_id,provider,token_hash,account_label,status,connected_at
            ) VALUES(?,?,?,?,?,?)
            """,
            (
                patient["id"],
                provider,
                _token_hash(token),
                account_label or f"{PROVIDERS[provider]['label']} companion",
                "active",
                now,
            ),
        )
        conn.commit()
        device_row_id = cursor.lastrowid
    finally:
        conn.close()
    return {
        "wearable_device_id": device_row_id,
        "patient_id": patient["id"],
        "patient_ref": patient["external_ref"],
        "patient_name": f"{patient['first_name']} {patient['last_name']}",
        "provider": provider,
        "token": token,
    }


def revoke_device(wearable_device_id: int) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE wearable_devices SET status='revoked' WHERE id=?",
            (wearable_device_id,),
        )
        conn.commit()
    finally:
        conn.close()


def authenticate_device(token: str, provider: str, installation_id: str, device_id: str | None = None, device_name: str | None = None):
    if not token:
        raise PermissionError("Missing wearable API token")
    if provider not in PROVIDERS:
        raise PermissionError("Unsupported wearable provider")
    if not installation_id or len(installation_id) > 200:
        raise PermissionError("Missing installation identifier")

    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT wd.*,p.external_ref,p.first_name,p.last_name,p.active,p.consent_monitoring
            FROM wearable_devices wd
            JOIN patients p ON p.id=wd.patient_id
            WHERE wd.token_hash=? AND wd.provider=? AND wd.status='active'
            """,
            (_token_hash(token), provider),
        ).fetchone()
        if not row:
            raise PermissionError("Invalid or revoked wearable credential")
        if not row["active"]:
            raise PermissionError("Patient is inactive")
        if not row["consent_monitoring"]:
            raise PermissionError("Patient monitoring consent is not active")

        bound_installation = row["installation_id"]
        if bound_installation and bound_installation != installation_id:
            raise PermissionError("Wearable credential is already bound to another installation")

        now = _utc_now()
        try:
            conn.execute(
                """
                UPDATE wearable_devices
                SET installation_id=COALESCE(installation_id,?),
                    device_id=COALESCE(?,device_id),
                    device_name=COALESCE(?,device_name),
                    last_seen_at=?
                WHERE id=?
                """,
                (installation_id, device_id, device_name, now, row["id"]),
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise PermissionError("Installation is already paired to another wearable credential") from exc

        return conn.execute(
            """
            SELECT wd.*,p.external_ref,p.first_name,p.last_name,p.active,p.consent_monitoring
            FROM wearable_devices wd JOIN patients p ON p.id=wd.patient_id
            WHERE wd.id=?
            """,
            (row["id"],),
        ).fetchone()
    finally:
        conn.close()


def get_patient_devices(patient_id: int) -> list[dict[str, Any]]:
    rows = query_db(
        """
        SELECT id,provider,installation_id,device_id,device_name,account_label,status,
               connected_at,last_seen_at,last_synced_at,last_sync_summary
        FROM wearable_devices WHERE patient_id=? ORDER BY connected_at DESC,id DESC
        """,
        (patient_id,),
    )
    output = []
    for row in rows:
        item = dict(row)
        item["provider_label"] = PROVIDERS.get(row["provider"], {}).get("label", row["provider"])
        output.append(item)
    return output


def get_latest_vitals_meta(patient_id: int) -> dict[str, dict[str, Any] | None]:
    kinds = ["spo2", "resp_rate", "heart_rate", "bp_sys", "bp_dia", "activity", "temperature", "sleep"]
    output: dict[str, dict[str, Any] | None] = {}
    for kind in kinds:
        row = query_db(
            """
            SELECT value,unit,source,measured_at FROM vitals
            WHERE patient_id=? AND kind=? ORDER BY measured_at DESC,id DESC LIMIT 1
            """,
            (patient_id, kind),
            one=True,
        )
        output[kind] = dict(row) if row else None
    return output


def get_patient_wearable_status(patient_id: int) -> dict[str, Any]:
    latest = query_db(
        """
        SELECT measured_at,source,device_name,provider
        FROM wearable_measurements
        WHERE patient_id=? ORDER BY measured_at DESC,id DESC LIMIT 1
        """,
        (patient_id,),
        one=True,
    )
    count = query_db(
        "SELECT COUNT(*) AS c FROM wearable_measurements WHERE patient_id=?",
        (patient_id,),
        one=True,
    )
    return {
        "has_data": bool(latest),
        "latest": dict(latest) if latest else None,
        "measurement_count": count["c"] if count else 0,
    }


def _normalise_vendor_payload(patient_id: int, provider: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    config = PROVIDER_MAPPINGS.get(provider)
    if not config or not isinstance(payload, dict):
        raise ValueError("Unsupported provider payload")
    rows = payload.get(config["records_key"])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing or empty {config['records_key']} list")

    source = str(payload.get("source") or config["label"])[:100]
    device_name = str(payload.get("device") or config["label"])[:150]
    records: list[dict[str, Any]] = []
    rejected: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            rejected.append("non-object sample")
            continue
        vendor_type = row.get(config["type_key"])
        mapping = config["metrics"].get(vendor_type)
        if not mapping or row.get("value") is None:
            rejected.append(str(vendor_type or "unknown"))
            continue
        try:
            raw_numeric = float(row["value"])
            metric, canonical_unit = mapping
            transform_name = config.get("transform", {}).get(vendor_type)
            normalized_value, raw_value = _transform_value(transform_name, raw_numeric)
            _validate_metric(metric, normalized_value)
            measured_at = _normalise_timestamp(
                row.get("measured_at") or row.get("timestamp") or row.get("startDate") or row.get("date")
            )
            source_record_id = str(
                row.get("source_record_id")
                or row.get("record_id")
                or row.get("uuid")
                or _fallback_source_record_id(provider, str(vendor_type), measured_at, raw_numeric)
            )[:255]
            records.append(
                {
                    "patient_id": patient_id,
                    "provider": provider,
                    "source_record_id": source_record_id,
                    "metric": metric,
                    "value": round(float(normalized_value), 2),
                    "unit": canonical_unit,
                    "raw_value": round(float(raw_value), 4) if raw_value is not None else None,
                    "raw_unit": str(row.get("unit"))[:40] if row.get("unit") is not None else None,
                    "source": source,
                    "device_name": device_name,
                    "measured_at": measured_at,
                }
            )
        except (TypeError, ValueError):
            rejected.append(str(vendor_type or "unknown"))

    if not records:
        raise ValueError("No recognised valid wearable measurements supplied")
    return records


def _same_measurement(existing, record: dict[str, Any]) -> bool:
    return (
        existing["metric"] == record["metric"]
        and abs(float(existing["value"]) - float(record["value"])) < 0.0001
        and existing["unit"] == record["unit"]
        and existing["measured_at"] == record["measured_at"]
    )


def _upsert_measurement(conn, wearable_device_id: int, record: dict[str, Any]) -> str:
    existing = conn.execute(
        """
        SELECT * FROM wearable_measurements
        WHERE patient_id=? AND provider=? AND source_record_id=?
        """,
        (record["patient_id"], record["provider"], record["source_record_id"]),
    ).fetchone()

    vital_source = "apple_healthkit" if record["provider"] == "apple_watch" else record["provider"]
    if existing:
        if _same_measurement(existing, record):
            return "duplicate"
        vital_id = existing["vital_id"]
        if vital_id:
            conn.execute(
                """
                UPDATE vitals SET kind=?,value=?,unit=?,source=?,measured_at=?
                WHERE id=? AND patient_id=?
                """,
                (
                    record["metric"], record["value"], record["unit"], vital_source,
                    record["measured_at"], vital_id, record["patient_id"],
                ),
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO vitals(patient_id,kind,value,unit,source,measured_at)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    record["patient_id"], record["metric"], record["value"], record["unit"],
                    vital_source, record["measured_at"],
                ),
            )
            vital_id = cursor.lastrowid
        conn.execute(
            """
            UPDATE wearable_measurements
            SET wearable_device_id=?,metric=?,value=?,unit=?,raw_value=?,raw_unit=?,source=?,
                device_name=?,measured_at=?,received_at=?,vital_id=?
            WHERE id=?
            """,
            (
                wearable_device_id, record["metric"], record["value"], record["unit"],
                record["raw_value"], record["raw_unit"], record["source"], record["device_name"],
                record["measured_at"], _utc_now(), vital_id, existing["id"],
            ),
        )
        return "updated"

    cursor = conn.execute(
        """
        INSERT INTO vitals(patient_id,kind,value,unit,source,measured_at)
        VALUES(?,?,?,?,?,?)
        """,
        (
            record["patient_id"], record["metric"], record["value"], record["unit"],
            vital_source, record["measured_at"],
        ),
    )
    vital_id = cursor.lastrowid
    conn.execute(
        """
        INSERT INTO wearable_measurements(
          patient_id,wearable_device_id,provider,source_record_id,metric,value,unit,
          raw_value,raw_unit,source,device_name,measured_at,received_at,vital_id
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            record["patient_id"], wearable_device_id, record["provider"], record["source_record_id"],
            record["metric"], record["value"], record["unit"], record["raw_value"], record["raw_unit"],
            record["source"], record["device_name"], record["measured_at"], _utc_now(), vital_id,
        ),
    )
    return "inserted"


def ingest_apple_healthkit(authenticated_device, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON payload")
    patient_id = int(authenticated_device["patient_id"])
    records = _normalise_vendor_payload(patient_id, "apple_watch", payload)

    counts = {"inserted": 0, "updated": 0, "duplicate": 0}
    conn = get_conn()
    try:
        conn.execute("BEGIN")
        for record in records:
            outcome = _upsert_measurement(conn, authenticated_device["id"], record)
            counts[outcome] += 1

        latest_at = max(record["measured_at"] for record in records)
        summary = (
            f"{counts['inserted']} new, {counts['updated']} updated, "
            f"{counts['duplicate']} duplicate"
        )
        now = _utc_now()
        conn.execute(
            """
            UPDATE wearable_devices
            SET last_seen_at=?,last_synced_at=?,last_sync_summary=?,
                device_name=COALESCE(?,device_name),device_id=COALESCE(?,device_id)
            WHERE id=?
            """,
            (
                now, now, summary, payload.get("device"), payload.get("device_id"), authenticated_device["id"],
            ),
        )
        conn.execute(
            """
            INSERT INTO wearable_sync_state(
              wearable_device_id,last_request_at,last_success_at,last_measurement_at,last_error
            ) VALUES(?,?,?,?,NULL)
            ON CONFLICT(wearable_device_id) DO UPDATE SET
              last_request_at=excluded.last_request_at,
              last_success_at=excluded.last_success_at,
              last_measurement_at=excluded.last_measurement_at,
              last_error=NULL
            """,
            (authenticated_device["id"], now, now, latest_at),
        )
        conn.execute(
            """
            INSERT INTO care_events(patient_id,event_type,source,description)
            VALUES(?,?,?,?)
            """,
            (
                patient_id,
                "device_sync",
                "apple_healthkit",
                f"Apple HealthKit wearable sync completed: {summary}.",
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "patient_id": patient_id,
        "patient_ref": authenticated_device["external_ref"],
        "patient_name": f"{authenticated_device['first_name']} {authenticated_device['last_name']}",
        "provider": "apple_watch",
        "source": "apple_healthkit",
        "device": payload.get("device") or authenticated_device["device_name"] or "Apple device",
        "received": len(records),
        "inserted": counts["inserted"],
        "updated": counts["updated"],
        "duplicates": counts["duplicate"],
        "last_measurement_at": max(record["measured_at"] for record in records),
        "synced_at": _utc_now(),
    }


def record_sync_error(wearable_device_id: int | None, message: str) -> None:
    if not wearable_device_id:
        return
    conn = get_conn()
    try:
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO wearable_sync_state(wearable_device_id,last_request_at,last_error)
            VALUES(?,?,?)
            ON CONFLICT(wearable_device_id) DO UPDATE SET
              last_request_at=excluded.last_request_at,last_error=excluded.last_error
            """,
            (wearable_device_id, now, str(message)[:500]),
        )
        conn.commit()
    finally:
        conn.close()
