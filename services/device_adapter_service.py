"""JSON device-adapter framework for CareAI medical-device pilots.

The adapter layer deliberately separates device/vendor ingestion from CareAI's
normalised clinical storage. Today the source can be the built-in JSON
simulator or a manually uploaded JSON document. A future vendor/BLE/API client
can call ``register_payload`` with ``source_mode='api'`` and then the same
``process_record`` path writes normalised values into CareAI's existing
``vitals`` / ``device_events`` tables.

This module does not replace or modify the Apple HealthKit wearable pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import uuid
from typing import Any

from services.db import get_conn, query_db
from services.risk_engine import compute_all_risks

SCHEMA_VERSION = "1.0"
MAX_JSON_BYTES = 1024 * 1024
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas" / "device_adapters"
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "device_adapters"

ADAPTERS: dict[str, dict[str, Any]] = {
    "generic_ble_glucometer": {
        "name": "Generic BLE glucometer",
        "short_name": "Glucometer",
        "category": "Medical devices",
        "source_label": "Generic BLE glucose adapter",
        "default_device_identifier": "GLUCO-SIM-001",
        "schema_file": "generic_ble_glucometer.schema.json",
        "example_file": "generic_ble_glucometer.example.json",
        "summary": "Validates glucose measurements and stores them as patient-linked CareAI vitals.",
    },
    "generic_ecg_patch": {
        "name": "Generic ECG patch",
        "short_name": "ECG patch",
        "category": "Medical devices",
        "source_label": "Generic ECG patch adapter",
        "default_device_identifier": "ECG-SIM-001",
        "schema_file": "generic_ecg_patch.schema.json",
        "example_file": "generic_ecg_patch.example.json",
        "summary": "Validates a single-lead ECG recording with waveform samples, heart rate and ECG intervals, while preserving any source-provided rhythm label without inventing a diagnosis.",
    },
    "omron_connect": {
        "name": "OMRON connect",
        "short_name": "OMRON connect",
        "category": "Medical devices",
        "source_label": "OMRON connect adapter",
        "default_device_identifier": "OMRON-SIM-001",
        "schema_file": "omron_connect.schema.json",
        "example_file": "omron_connect.example.json",
        "summary": "Normalises systolic/diastolic blood pressure and heart rate into CareAI vitals.",
    },
    "generic_ble_pulse_oximeter": {
        "name": "Generic BLE pulse oximeter",
        "short_name": "Pulse oximeter",
        "category": "Medical devices",
        "source_label": "Generic BLE pulse oximeter adapter",
        "default_device_identifier": "PULSEOX-SIM-001",
        "schema_file": "generic_ble_pulse_oximeter.schema.json",
        "example_file": "generic_ble_pulse_oximeter.example.json",
        "summary": "Normalises oxygen saturation and pulse rate into the existing CareAI SpO₂ and heart-rate vitals.",
    },
    "withings": {
        "name": "Withings",
        "short_name": "Withings",
        "category": "Medical devices",
        "source_label": "Withings adapter",
        "default_device_identifier": "WITHINGS-SIM-001",
        "schema_file": "withings.schema.json",
        "example_file": "withings.example.json",
        "summary": "Normalises supported Withings blood pressure, heart rate, weight and body-fat measurements into patient-linked CareAI vitals.",
    },
}

INTEGRATION_NAME_TO_KEY = {cfg["name"]: key for key, cfg in ADAPTERS.items()}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalise_timestamp(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return _utc_now()
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("Measurement timestamp must be a valid ISO-8601 date/time") from exc
    if parsed.tzinfo is None:
        # Simulator forms do not carry a browser timezone. Treat a timezone-less
        # value consistently as UTC rather than depending on the server locale.
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _payload_hash(payload: dict[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be a finite number")
    return number


def _range(value: float, low: float, high: float, label: str) -> None:
    if value < low or value > high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}")


def _optional_measurement(measurements: dict[str, Any], key: str, label: str, expected_unit: str, low: float, high: float):
    item = measurements.get(key)
    if item is None:
        return None
    value, unit = _measurement(measurements, key, label)
    if unit != expected_unit:
        raise ValueError(f"{label} unit must be {expected_unit}")
    _range(value, low, high, label)
    return value


def _synthetic_ecg_waveform(heart_rate: float, sampling_rate_hz: int, duration_seconds: float,
                            pr_interval_ms: float, qrs_duration_ms: float, qt_interval_ms: float) -> list[float]:
    """Create a clearly simulated single-lead P-QRS-T waveform in mV.

    This is demo data only. It is not a diagnostic ECG generator and does not
    infer arrhythmias or disease. The shape is intentionally generic so the
    adapter/UI can be exercised before a real patch or vendor API is available.
    """
    rr_seconds = 60.0 / heart_rate
    sample_count = int(round(sampling_rate_hz * duration_seconds))
    pr_seconds = pr_interval_ms / 1000.0
    qrs_seconds = qrs_duration_ms / 1000.0
    qt_seconds = qt_interval_ms / 1000.0

    def g(x: float, center: float, width: float, amplitude: float) -> float:
        width = max(width, 0.004)
        return amplitude * math.exp(-0.5 * ((x - center) / width) ** 2)

    samples: list[float] = []
    for i in range(sample_count):
        t = i / sampling_rate_hz
        phase = t % rr_seconds
        r_center = min(rr_seconds * 0.42, max(0.22, pr_seconds + 0.12))
        p_center = max(0.05, r_center - max(0.10, pr_seconds * 0.72))
        q_center = r_center - qrs_seconds * 0.20
        s_center = r_center + qrs_seconds * 0.24
        t_center = min(rr_seconds - 0.06, r_center + max(0.18, qt_seconds * 0.58))
        value = (
            g(phase, p_center, 0.030, 0.12)
            + g(phase, q_center, max(0.008, qrs_seconds * 0.08), -0.16)
            + g(phase, r_center, max(0.009, qrs_seconds * 0.07), 1.05)
            + g(phase, s_center, max(0.010, qrs_seconds * 0.09), -0.28)
            + g(phase, t_center, 0.055, 0.30)
        )
        # Small baseline wander and sensor noise make the demo trace look like a
        # sampled patch signal without implying a clinical morphology.
        value += 0.025 * math.sin(2 * math.pi * 0.28 * t)
        value += random.uniform(-0.010, 0.010)
        samples.append(round(value, 4))
    return samples


def _measurement(measurements: dict[str, Any], key: str, label: str) -> tuple[float, str]:
    item = measurements.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{label} measurement is required")
    if "value" not in item or "unit" not in item:
        raise ValueError(f"{label} must contain value and unit")
    return _float(item.get("value"), label), str(item.get("unit") or "").strip()


def adapter_config(adapter_key: str) -> dict[str, Any]:
    cfg = ADAPTERS.get(str(adapter_key or "").strip())
    if not cfg:
        raise KeyError("Unknown device adapter")
    return {"key": adapter_key, **cfg}


def adapter_cards() -> dict[str, str]:
    return dict(INTEGRATION_NAME_TO_KEY)


def patient_options():
    return query_db(
        """SELECT id,external_ref,first_name,last_name,birth_date,current_status
           FROM patients WHERE active=1 ORDER BY last_name,first_name,external_ref"""
    )


def _patient_by_ref(patient_ref: str):
    ref = str(patient_ref or "").strip()
    if not ref:
        raise ValueError("Patient ID is required")
    row = query_db(
        "SELECT id,external_ref,first_name,last_name,active FROM patients WHERE external_ref=?",
        (ref,),
        one=True,
    )
    if not row:
        raise ValueError("Patient ID does not exist in CareAI")
    if not int(row["active"]):
        raise ValueError("Archived patients cannot receive new device measurements")
    return row


def schema_for(adapter_key: str) -> dict[str, Any]:
    cfg = adapter_config(adapter_key)
    path = SCHEMA_DIR / cfg["schema_file"]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def example_for(adapter_key: str) -> dict[str, Any]:
    cfg = adapter_config(adapter_key)
    path = EXAMPLE_DIR / cfg["example_file"]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_common(adapter_key: str, payload: dict[str, Any]):
    if not isinstance(payload, dict):
        raise ValueError("Uploaded content must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("device_type") != adapter_key:
        expected = adapter_config(adapter_key)["name"]
        supplied = payload.get("device_type") or "missing"
        raise ValueError(f"Wrong adapter: {expected} cannot process device_type '{supplied}'")

    record_id = str(payload.get("record_id") or "").strip()
    if not record_id or len(record_id) > 160:
        raise ValueError("record_id is required and must be 160 characters or fewer")

    device_identifier = str(payload.get("device_identifier") or "").strip()
    if not device_identifier or len(device_identifier) > 160:
        raise ValueError("device_identifier is required and must be 160 characters or fewer")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    source_type = str(source.get("type") or "").strip().lower()
    if source_type not in {"simulated", "vendor_api", "ble_gateway", "manual_json", "api"}:
        raise ValueError("source.type must identify a supported ingestion source")
    if not str(source.get("name") or "").strip():
        raise ValueError("source.name is required")

    measured_at = _normalise_timestamp(payload.get("measurement_timestamp"))
    patient = _patient_by_ref(str(payload.get("patient_id") or ""))
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("measurements must be a JSON object")
    return patient, record_id, device_identifier, measured_at, measurements


def validate_payload(adapter_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a device payload and return the normalised CareAI measurements.

    Returned items use ``storage='vital'`` for numeric signals and
    ``storage='device_event'`` for source-provided textual ECG rhythm labels.
    No clinical interpretation is inferred by this adapter.
    """
    patient, record_id, device_identifier, measured_at, measurements = _validate_common(adapter_key, payload)
    cfg = adapter_config(adapter_key)
    normalised: list[dict[str, Any]] = []

    if adapter_key == "generic_ble_glucometer":
        value, unit = _measurement(measurements, "glucose", "Glucose")
        if unit not in {"mmol/L", "mg/dL"}:
            raise ValueError("Glucose unit must be mmol/L or mg/dL")
        if unit == "mmol/L":
            _range(value, 1.0, 40.0, "Glucose")
        else:
            _range(value, 18.0, 720.0, "Glucose")
        normalised.append({
            "storage": "vital", "metric": "glucose", "value": value, "unit": unit,
            "measured_at": measured_at,
        })
        context = payload.get("context") or {}
        if context and not isinstance(context, dict):
            raise ValueError("context must be an object when supplied")
        meal_context = str(context.get("meal_context") or "").strip() if isinstance(context, dict) else ""
        if meal_context and meal_context not in {"fasting", "before_meal", "after_meal", "random", "unknown"}:
            raise ValueError("context.meal_context is not recognised")

    elif adapter_key == "generic_ecg_patch":
        value, unit = _measurement(measurements, "heart_rate", "Heart rate")
        if unit != "bpm":
            raise ValueError("ECG heart rate unit must be bpm")
        _range(value, 20.0, 250.0, "Heart rate")
        normalised.append({
            "storage": "vital", "metric": "heart_rate", "value": value, "unit": "bpm",
            "measured_at": measured_at,
        })

        ecg_metric_specs = (
            ("rr_interval", "RR interval", "ms", 150.0, 4000.0),
            ("pr_interval", "PR interval", "ms", 60.0, 500.0),
            ("qrs_duration", "QRS duration", "ms", 30.0, 300.0),
            ("qt_interval", "QT interval", "ms", 150.0, 800.0),
            ("qtc", "QTc", "ms", 150.0, 800.0),
        )
        ecg_metrics: dict[str, float] = {}
        for metric_key, label, expected_unit, low, high in ecg_metric_specs:
            metric_value = _optional_measurement(measurements, metric_key, label, expected_unit, low, high)
            if metric_value is not None:
                ecg_metrics[metric_key] = metric_value
                normalised.append({
                    "storage": "ecg_metric", "metric": metric_key, "value": metric_value,
                    "unit": expected_unit, "measured_at": measured_at,
                })

        rhythm_text = ""
        rhythm = measurements.get("rhythm")
        if rhythm is not None:
            if not isinstance(rhythm, dict):
                raise ValueError("rhythm must contain value and unit when supplied")
            rhythm_text = str(rhythm.get("value") or "").strip()
            rhythm_unit = str(rhythm.get("unit") or "").strip()
            if rhythm_text:
                if len(rhythm_text) > 160:
                    raise ValueError("Rhythm value must be 160 characters or fewer")
                if rhythm_unit != "label":
                    raise ValueError("Rhythm unit must be 'label'")
                normalised.append({
                    "storage": "device_event", "metric": "ecg_rhythm", "text_value": rhythm_text,
                    "unit": "label", "measured_at": measured_at,
                })

        ecg_recording = None
        recording = payload.get("recording")
        if recording is not None:
            if not isinstance(recording, dict):
                raise ValueError("recording must be an object when supplied")
            lead_name = str(recording.get("lead") or "").strip()
            if not lead_name or len(lead_name) > 50:
                raise ValueError("ECG recording lead is required and must be 50 characters or fewer")
            sampling_rate = int(_float(recording.get("sampling_rate_hz"), "ECG sampling rate"))
            _range(float(sampling_rate), 100.0, 1000.0, "ECG sampling rate")
            duration_seconds = _float(recording.get("duration_seconds"), "ECG duration")
            _range(duration_seconds, 1.0, 60.0, "ECG duration")
            amplitude_unit = str(recording.get("amplitude_unit") or "").strip()
            if amplitude_unit != "mV":
                raise ValueError("ECG waveform amplitude_unit must be mV")
            samples = recording.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError("ECG recording samples must be a non-empty array")
            expected_count = int(round(sampling_rate * duration_seconds))
            tolerance = max(2, int(expected_count * 0.01))
            if abs(len(samples) - expected_count) > tolerance:
                raise ValueError(f"ECG waveform sample count must be approximately {expected_count} for the selected sampling rate and duration")
            if len(samples) > 60000:
                raise ValueError("ECG waveform contains too many samples")
            clean_samples: list[float] = []
            for index, sample in enumerate(samples):
                sample_value = _float(sample, f"ECG sample {index + 1}")
                _range(sample_value, -20.0, 20.0, f"ECG sample {index + 1}")
                clean_samples.append(round(sample_value, 6))
            ecg_recording = {
                "lead": lead_name,
                "sampling_rate_hz": sampling_rate,
                "duration_seconds": duration_seconds,
                "amplitude_unit": "mV",
                "samples": clean_samples,
                "heart_rate_bpm": value,
                "rr_interval_ms": ecg_metrics.get("rr_interval"),
                "pr_interval_ms": ecg_metrics.get("pr_interval"),
                "qrs_duration_ms": ecg_metrics.get("qrs_duration"),
                "qt_interval_ms": ecg_metrics.get("qt_interval"),
                "qtc_ms": ecg_metrics.get("qtc"),
                "rhythm_label": rhythm_text or None,
            }

    elif adapter_key == "omron_connect":
        systolic, sys_unit = _measurement(measurements, "systolic_bp", "Systolic BP")
        diastolic, dia_unit = _measurement(measurements, "diastolic_bp", "Diastolic BP")
        heart_rate, hr_unit = _measurement(measurements, "heart_rate", "Heart rate")
        if sys_unit != "mmHg" or dia_unit != "mmHg":
            raise ValueError("OMRON blood pressure units must be mmHg")
        if hr_unit != "bpm":
            raise ValueError("OMRON heart rate unit must be bpm")
        _range(systolic, 50.0, 300.0, "Systolic BP")
        _range(diastolic, 30.0, 200.0, "Diastolic BP")
        _range(heart_rate, 20.0, 250.0, "Heart rate")
        if diastolic >= systolic:
            raise ValueError("Diastolic BP must be lower than systolic BP")
        normalised.extend([
            {"storage": "vital", "metric": "bp_sys", "value": systolic, "unit": "mmHg", "measured_at": measured_at},
            {"storage": "vital", "metric": "bp_dia", "value": diastolic, "unit": "mmHg", "measured_at": measured_at},
            {"storage": "vital", "metric": "heart_rate", "value": heart_rate, "unit": "bpm", "measured_at": measured_at},
        ])

    elif adapter_key == "generic_ble_pulse_oximeter":
        spo2, spo2_unit = _measurement(measurements, "spo2", "SpO₂")
        pulse_rate, pulse_unit = _measurement(measurements, "pulse_rate", "Pulse rate")
        if spo2_unit != "%":
            raise ValueError("Pulse oximeter SpO₂ unit must be %")
        if pulse_unit != "bpm":
            raise ValueError("Pulse oximeter pulse-rate unit must be bpm")
        _range(spo2, 50.0, 100.0, "SpO₂")
        _range(pulse_rate, 20.0, 250.0, "Pulse rate")
        normalised.extend([
            {"storage": "vital", "metric": "spo2", "value": spo2, "unit": "%", "measured_at": measured_at},
            {"storage": "vital", "metric": "heart_rate", "value": pulse_rate, "unit": "bpm", "measured_at": measured_at},
        ])

    elif adapter_key == "withings":
        has_sys = "systolic_bp" in measurements
        has_dia = "diastolic_bp" in measurements
        if has_sys != has_dia:
            raise ValueError("Withings systolic and diastolic BP must be supplied together")
        if has_sys:
            systolic, sys_unit = _measurement(measurements, "systolic_bp", "Systolic BP")
            diastolic, dia_unit = _measurement(measurements, "diastolic_bp", "Diastolic BP")
            if sys_unit != "mmHg" or dia_unit != "mmHg":
                raise ValueError("Withings blood pressure units must be mmHg")
            _range(systolic, 50.0, 300.0, "Systolic BP")
            _range(diastolic, 30.0, 200.0, "Diastolic BP")
            if diastolic >= systolic:
                raise ValueError("Diastolic BP must be lower than systolic BP")
            normalised.extend([
                {"storage": "vital", "metric": "bp_sys", "value": systolic, "unit": "mmHg", "measured_at": measured_at},
                {"storage": "vital", "metric": "bp_dia", "value": diastolic, "unit": "mmHg", "measured_at": measured_at},
            ])

        if "heart_rate" in measurements:
            heart_rate, hr_unit = _measurement(measurements, "heart_rate", "Heart rate")
            if hr_unit != "bpm":
                raise ValueError("Withings heart rate unit must be bpm")
            _range(heart_rate, 20.0, 250.0, "Heart rate")
            normalised.append({
                "storage": "vital", "metric": "heart_rate", "value": heart_rate, "unit": "bpm",
                "measured_at": measured_at,
            })

        if "weight" in measurements:
            weight, weight_unit = _measurement(measurements, "weight", "Weight")
            if weight_unit != "kg":
                raise ValueError("Withings weight unit must be kg")
            _range(weight, 2.0, 350.0, "Weight")
            normalised.append({
                "storage": "vital", "metric": "weight", "value": weight, "unit": "kg",
                "measured_at": measured_at,
            })

        if "body_fat" in measurements:
            body_fat, fat_unit = _measurement(measurements, "body_fat", "Body fat")
            if fat_unit != "%":
                raise ValueError("Withings body-fat unit must be %")
            _range(body_fat, 1.0, 75.0, "Body fat")
            normalised.append({
                "storage": "vital", "metric": "body_fat", "value": body_fat, "unit": "%",
                "measured_at": measured_at,
            })
    else:  # pragma: no cover - adapter_config already rejects this
        raise KeyError("Unknown device adapter")

    if not normalised:
        raise ValueError("No supported measurements were found in the JSON")

    result = {
        "adapter_key": adapter_key,
        "adapter_name": cfg["name"],
        "patient_id": int(patient["id"]),
        "patient_ref": patient["external_ref"],
        "patient_name": f"{patient['first_name']} {patient['last_name']}",
        "record_id": record_id,
        "device_identifier": device_identifier,
        "measured_at": measured_at,
        "measurements": normalised,
    }
    if adapter_key == "generic_ecg_patch":
        result["ecg_recording"] = ecg_recording
    return result


def _sample_values(adapter_key: str) -> dict[str, Any]:
    if adapter_key == "generic_ble_glucometer":
        return {"glucose": round(random.uniform(4.3, 8.8), 1), "glucose_unit": "mmol/L", "meal_context": "random"}
    if adapter_key == "generic_ecg_patch":
        heart_rate = random.randint(58, 96)
        rr = round(60000.0 / heart_rate)
        qtc = random.randint(395, 430)
        qt = round(qtc * math.sqrt(rr / 1000.0))
        return {
            "heart_rate": heart_rate, "rhythm": "", "ecg_lead": "Lead I",
            "ecg_sampling_rate": 250, "ecg_duration": 10,
            "rr_interval": rr, "pr_interval": random.randint(145, 185),
            "qrs_duration": random.randint(82, 104), "qt_interval": qt, "qtc": qtc,
        }
    if adapter_key == "omron_connect":
        systolic = random.randint(112, 152)
        return {"systolic_bp": systolic, "diastolic_bp": random.randint(68, min(94, systolic - 20)), "heart_rate": random.randint(58, 92)}
    if adapter_key == "generic_ble_pulse_oximeter":
        return {"spo2": random.randint(94, 99), "heart_rate": random.randint(58, 96)}
    if adapter_key == "withings":
        systolic = random.randint(112, 148)
        return {
            "systolic_bp": systolic,
            "diastolic_bp": random.randint(68, min(92, systolic - 20)),
            "heart_rate": random.randint(58, 92),
            "weight": round(random.uniform(58.0, 92.0), 1),
            "body_fat": round(random.uniform(16.0, 34.0), 1),
        }
    raise KeyError("Unknown device adapter")


def sample_values(adapter_key: str) -> dict[str, Any]:
    adapter_config(adapter_key)
    return _sample_values(adapter_key)


def build_simulated_payload(adapter_key: str, form: dict[str, Any]) -> dict[str, Any]:
    cfg = adapter_config(adapter_key)
    patient_ref = str(form.get("patient_id") or "").strip()
    _patient_by_ref(patient_ref)
    measured_at = _normalise_timestamp(form.get("measurement_timestamp"))
    device_identifier = str(form.get("device_identifier") or cfg["default_device_identifier"]).strip()[:160]
    if not device_identifier:
        raise ValueError("Device identifier is required")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"sim-{uuid.uuid4().hex}",
        "patient_id": patient_ref,
        "device_type": adapter_key,
        "device_identifier": device_identifier,
        "source": {"type": "simulated", "name": "CareAI Device JSON Generator"},
        "measurement_timestamp": measured_at,
        "measurements": {},
    }

    if adapter_key == "generic_ble_glucometer":
        payload["measurements"]["glucose"] = {
            "value": _float(form.get("glucose"), "Glucose"),
            "unit": str(form.get("glucose_unit") or "mmol/L").strip(),
        }
        meal_context = str(form.get("meal_context") or "").strip()
        if meal_context:
            payload["context"] = {"meal_context": meal_context}
    elif adapter_key == "generic_ecg_patch":
        heart_rate = _float(form.get("ecg_heart_rate"), "Heart rate")
        sampling_rate = int(_float(form.get("ecg_sampling_rate") or 250, "ECG sampling rate"))
        duration_seconds = _float(form.get("ecg_duration") or 10, "ECG duration")
        lead_name = str(form.get("ecg_lead") or "Lead I").strip()[:50]

        rr_raw = form.get("rr_interval")
        rr_interval = _float(rr_raw, "RR interval") if rr_raw is not None and str(rr_raw).strip() else round(60000.0 / heart_rate)
        pr_raw = form.get("pr_interval")
        pr_interval = _float(pr_raw, "PR interval") if pr_raw is not None and str(pr_raw).strip() else 160.0
        qrs_raw = form.get("qrs_duration")
        qrs_duration = _float(qrs_raw, "QRS duration") if qrs_raw is not None and str(qrs_raw).strip() else 90.0
        qtc_raw = form.get("qtc")
        qtc = _float(qtc_raw, "QTc") if qtc_raw is not None and str(qtc_raw).strip() else 410.0
        qt_raw = form.get("qt_interval")
        qt_interval = _float(qt_raw, "QT interval") if qt_raw is not None and str(qt_raw).strip() else round(qtc * math.sqrt(rr_interval / 1000.0), 1)

        payload["measurements"].update({
            "heart_rate": {"value": heart_rate, "unit": "bpm"},
            "rr_interval": {"value": rr_interval, "unit": "ms"},
            "pr_interval": {"value": pr_interval, "unit": "ms"},
            "qrs_duration": {"value": qrs_duration, "unit": "ms"},
            "qt_interval": {"value": qt_interval, "unit": "ms"},
            "qtc": {"value": qtc, "unit": "ms"},
        })
        rhythm = str(form.get("rhythm") or "").strip()
        if rhythm:
            payload["measurements"]["rhythm"] = {"value": rhythm, "unit": "label"}
        payload["recording"] = {
            "lead": lead_name,
            "sampling_rate_hz": sampling_rate,
            "duration_seconds": duration_seconds,
            "amplitude_unit": "mV",
            "samples": _synthetic_ecg_waveform(
                heart_rate, sampling_rate, duration_seconds, pr_interval, qrs_duration, qt_interval
            ),
        }
    elif adapter_key == "omron_connect":
        payload["measurements"] = {
            "systolic_bp": {"value": _float(form.get("systolic_bp"), "Systolic BP"), "unit": "mmHg"},
            "diastolic_bp": {"value": _float(form.get("diastolic_bp"), "Diastolic BP"), "unit": "mmHg"},
            "heart_rate": {"value": _float(form.get("omron_heart_rate"), "Heart rate"), "unit": "bpm"},
        }
    elif adapter_key == "generic_ble_pulse_oximeter":
        payload["measurements"] = {
            "spo2": {"value": _float(form.get("pulse_spo2"), "SpO₂"), "unit": "%"},
            "pulse_rate": {"value": _float(form.get("pulse_heart_rate"), "Pulse rate"), "unit": "bpm"},
        }
    elif adapter_key == "withings":
        withings_measurements: dict[str, Any] = {}
        optional_fields = (
            ("withings_systolic_bp", "systolic_bp", "Systolic BP", "mmHg"),
            ("withings_diastolic_bp", "diastolic_bp", "Diastolic BP", "mmHg"),
            ("withings_heart_rate", "heart_rate", "Heart rate", "bpm"),
            ("withings_weight", "weight", "Weight", "kg"),
            ("withings_body_fat", "body_fat", "Body fat", "%"),
        )
        for form_key, metric_key, label, unit in optional_fields:
            raw_value = form.get(form_key)
            if raw_value is not None and str(raw_value).strip() != "":
                withings_measurements[metric_key] = {"value": _float(raw_value, label), "unit": unit}
        payload["measurements"] = withings_measurements

    # Run the same validation path used for uploaded/vendor payloads before it is persisted.
    validate_payload(adapter_key, payload)
    return payload


def register_payload(
    adapter_key: str,
    payload: dict[str, Any],
    *,
    source_mode: str,
    actor_user_id: int | None,
    original_filename: str | None = None,
) -> dict[str, Any]:
    """Validate and register a payload without yet writing clinical measurements."""
    adapter_config(adapter_key)
    normalised = validate_payload(adapter_key, payload)
    digest = _payload_hash(payload)
    payload_text = json.dumps(payload, indent=2, ensure_ascii=False)
    filename = (Path(original_filename).name if original_filename else None)
    filename = filename[:180] if filename else None

    conn = get_conn()
    try:
        existing = conn.execute(
            "SELECT * FROM device_adapter_records WHERE adapter_key=? AND payload_hash=?",
            (adapter_key, digest),
        ).fetchone()
        if existing:
            return {"record_id": existing["id"], "duplicate": True, "record": dict(existing), "normalised": normalised}

        same_source = conn.execute(
            """SELECT * FROM device_adapter_records
               WHERE adapter_key=? AND patient_id=? AND source_identifier=?
               ORDER BY id DESC LIMIT 1""",
            (adapter_key, normalised["patient_id"], normalised["record_id"]),
        ).fetchone()
        if same_source:
            return {"record_id": same_source["id"], "duplicate": True, "record": dict(same_source), "normalised": normalised}

        cur = conn.execute(
            """INSERT INTO device_adapter_records(
                   adapter_key,patient_id,source_mode,source_identifier,device_identifier,
                   original_filename,payload_json,payload_hash,validation_status,processing_status,
                   generated_by,generated_at,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,'generated',?,?,?)""",
            (
                adapter_key, normalised["patient_id"], source_mode, normalised["record_id"],
                normalised["device_identifier"], filename, payload_text, digest, "valid",
                actor_user_id, _utc_now(), _utc_now(),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM device_adapter_records WHERE id=?", (cur.lastrowid,)).fetchone()
        return {"record_id": cur.lastrowid, "duplicate": False, "record": dict(row), "normalised": normalised}
    finally:
        conn.close()


def record_failed_upload(
    adapter_key: str,
    *,
    actor_user_id: int | None,
    original_filename: str | None,
    error_message: str,
    raw_text: str | None = None,
) -> int:
    """Record failed adapter attempts without writing any clinical data."""
    adapter_config(adapter_key)
    filename = Path(original_filename or "invalid.json").name[:180]
    digest = sha256((raw_text or f"{filename}:{error_message}:{uuid.uuid4().hex}").encode("utf-8", errors="replace")).hexdigest()
    safe_payload = None
    if raw_text:
        try:
            parsed = json.loads(raw_text)
            safe_payload = json.dumps(parsed, indent=2, ensure_ascii=False) if isinstance(parsed, dict) else None
        except Exception:
            safe_payload = None
    conn = get_conn()
    try:
        # Failed uploads use the same payload-hash uniqueness rule as valid
        # records. Re-uploading the same invalid/wrong-device JSON must therefore
        # be idempotent: reuse the existing failure instead of raising an
        # IntegrityError from the unique (adapter_key, payload_hash) index.
        existing = conn.execute(
            "SELECT id FROM device_adapter_records WHERE adapter_key=? AND payload_hash=?",
            (adapter_key, digest),
        ).fetchone()
        if existing:
            return int(existing["id"])

        now = _utc_now()
        conn.execute(
            """INSERT OR IGNORE INTO device_adapter_records(
                   adapter_key,patient_id,source_mode,source_identifier,device_identifier,
                   original_filename,payload_json,payload_hash,validation_status,processing_status,
                   error_message,generated_by,generated_at,created_at
               ) VALUES(?,NULL,'upload',NULL,NULL,?,?,?,'invalid','failed',?,?,?,?)""",
            (adapter_key, filename, safe_payload, digest, str(error_message)[:600], actor_user_id, now, now),
        )
        conn.commit()

        # SELECT after INSERT OR IGNORE also covers two requests arriving at
        # nearly the same time: whichever request inserted first owns the row,
        # and both callers receive the same stable record id.
        saved = conn.execute(
            "SELECT id FROM device_adapter_records WHERE adapter_key=? AND payload_hash=?",
            (adapter_key, digest),
        ).fetchone()
        if not saved:
            raise RuntimeError("Could not record failed device JSON upload")
        return int(saved["id"])
    finally:
        conn.close()


def create_generated_record(adapter_key: str, form: dict[str, Any], actor_user_id: int | None) -> dict[str, Any]:
    payload = build_simulated_payload(adapter_key, form)
    result = register_payload(
        adapter_key, payload, source_mode="simulator", actor_user_id=actor_user_id,
        original_filename=f"{adapter_key}_{payload['patient_id']}_{payload['record_id'][-8:]}.json",
    )
    result["payload"] = payload
    return result


def get_record(record_db_id: int):
    return query_db(
        """SELECT r.*,p.external_ref,p.first_name,p.last_name
           FROM device_adapter_records r
           LEFT JOIN patients p ON p.id=r.patient_id
           WHERE r.id=?""",
        (record_db_id,), one=True,
    )


def records_for_adapter(adapter_key: str, limit: int = 100):
    adapter_config(adapter_key)
    return query_db(
        """SELECT r.*,p.external_ref,p.first_name,p.last_name
           FROM device_adapter_records r
           LEFT JOIN patients p ON p.id=r.patient_id
           WHERE r.adapter_key=? ORDER BY r.id DESC LIMIT ?""",
        (adapter_key, int(limit)),
    )


def preview_record(adapter_key: str, record_db_id: int) -> dict[str, Any]:
    row = get_record(record_db_id)
    if not row or row["adapter_key"] != adapter_key:
        raise LookupError("Device JSON record not found")
    payload = None
    normalised = None
    if row["payload_json"]:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            payload = None
    if payload and row["validation_status"] == "valid":
        normalised = validate_payload(adapter_key, payload)
    return {
        "record": row,
        "payload": payload,
        "payload_pretty": json.dumps(payload, indent=2, ensure_ascii=False) if payload else "",
        "normalised": normalised,
    }


def _duplicate_measurement(conn, patient_id: int, adapter_key: str, item: dict[str, Any]) -> bool:
    if item["storage"] in {"vital", "ecg_metric"}:
        row = conn.execute(
            """SELECT 1 FROM device_adapter_measurements
               WHERE patient_id=? AND adapter_key=? AND metric=? AND measured_at=?
                 AND value_real=? AND unit=? LIMIT 1""",
            (patient_id, adapter_key, item["metric"], item["measured_at"], item["value"], item["unit"]),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT 1 FROM device_adapter_measurements
               WHERE patient_id=? AND adapter_key=? AND metric=? AND measured_at=?
                 AND value_text=? LIMIT 1""",
            (patient_id, adapter_key, item["metric"], item["measured_at"], item.get("text_value")),
        ).fetchone()
    return bool(row)


def process_record(adapter_key: str, record_db_id: int, actor_user_id: int | None) -> dict[str, Any]:
    """Process one registered record atomically into existing CareAI stores."""
    adapter_config(adapter_key)
    row = get_record(record_db_id)
    if not row or row["adapter_key"] != adapter_key:
        raise LookupError("Device JSON record not found")
    if row["processing_status"] == "processed":
        return {"already_processed": True, "record_id": record_db_id, "patient_id": row["patient_id"]}
    if row["validation_status"] != "valid" or not row["payload_json"]:
        raise ValueError(row["error_message"] or "This JSON record is not valid and cannot be processed")

    payload = json.loads(row["payload_json"])
    normalised = validate_payload(adapter_key, payload)
    patient_id = normalised["patient_id"]
    source = f"device_adapter:{adapter_key}"

    conn = get_conn()
    try:
        current = conn.execute("SELECT processing_status FROM device_adapter_records WHERE id=?", (record_db_id,)).fetchone()
        if current and current["processing_status"] == "processed":
            return {"already_processed": True, "record_id": record_db_id, "patient_id": patient_id}

        duplicates = [m["metric"] for m in normalised["measurements"] if _duplicate_measurement(conn, patient_id, adapter_key, m)]
        if duplicates:
            message = "Duplicate measurement rejected: " + ", ".join(sorted(set(duplicates)))
            conn.execute(
                "UPDATE device_adapter_records SET processing_status='failed',error_message=? WHERE id=?",
                (message, record_db_id),
            )
            conn.commit()
            raise ValueError(message)

        ecg_recording_id = None
        ecg_recording = normalised.get("ecg_recording")
        if adapter_key == "generic_ecg_patch" and ecg_recording:
            existing_ecg = conn.execute(
                "SELECT id FROM ecg_recordings WHERE device_adapter_record_id=?",
                (record_db_id,),
            ).fetchone()
            if existing_ecg:
                ecg_recording_id = existing_ecg["id"]
            else:
                cur = conn.execute(
                    """INSERT INTO ecg_recordings(
                           patient_id,device_adapter_record_id,device_identifier,lead_name,sampling_rate_hz,
                           duration_seconds,amplitude_unit,waveform_json,heart_rate_bpm,rr_interval_ms,
                           pr_interval_ms,qrs_duration_ms,qt_interval_ms,qtc_ms,rhythm_label,measured_at,source
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        patient_id, record_db_id, normalised["device_identifier"], ecg_recording["lead"],
                        ecg_recording["sampling_rate_hz"], ecg_recording["duration_seconds"],
                        ecg_recording["amplitude_unit"], json.dumps(ecg_recording["samples"], separators=(",", ":")),
                        ecg_recording["heart_rate_bpm"], ecg_recording.get("rr_interval_ms"),
                        ecg_recording.get("pr_interval_ms"), ecg_recording.get("qrs_duration_ms"),
                        ecg_recording.get("qt_interval_ms"), ecg_recording.get("qtc_ms"),
                        ecg_recording.get("rhythm_label"), normalised["measured_at"], source,
                    ),
                )
                ecg_recording_id = cur.lastrowid

        stored = []
        for item in normalised["measurements"]:
            vital_id = None
            device_event_id = None
            if item["storage"] == "vital":
                cur = conn.execute(
                    "INSERT INTO vitals(patient_id,kind,value,unit,source,measured_at) VALUES(?,?,?,?,?,?)",
                    (patient_id, item["metric"], item["value"], item["unit"], source, item["measured_at"]),
                )
                vital_id = cur.lastrowid
                value_real = item["value"]
                value_text = None
            elif item["storage"] == "device_event":
                cur = conn.execute(
                    """INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (patient_id, adapter_key, item["metric"], item.get("text_value"), "info", None, item["measured_at"]),
                )
                device_event_id = cur.lastrowid
                value_real = None
                value_text = item.get("text_value")
            else:  # ECG interval metrics live in ecg_recordings, not the generic vitals table.
                value_real = item["value"]
                value_text = None

            conn.execute(
                """INSERT INTO device_adapter_measurements(
                       record_id,patient_id,adapter_key,metric,value_real,value_text,unit,measured_at,vital_id,device_event_id
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    record_db_id, patient_id, adapter_key, item["metric"], value_real, value_text,
                    item["unit"], item["measured_at"], vital_id, device_event_id,
                ),
            )
            stored.append({**item, "vital_id": vital_id, "device_event_id": device_event_id, "ecg_recording_id": ecg_recording_id})

        conn.execute(
            """UPDATE device_adapter_records
               SET processing_status='processed',processed_by=?,processed_at=?,error_message=NULL
               WHERE id=?""",
            (actor_user_id, _utc_now(), record_db_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Reuse the current risk engine. OMRON/Withings BP/HR, ECG HR and pulse-
    # oximeter SpO₂/pulse can influence existing CareAI risk logic. Glucose,
    # weight and body-fat values are stored/displayed without inventing new
    # clinical risk models that CareAI does not currently have.
    compute_all_risks(patient_id)
    return {
        "already_processed": False,
        "record_id": record_db_id,
        "patient_id": patient_id,
        "patient_ref": normalised["patient_ref"],
        "patient_name": normalised["patient_name"],
        "stored": stored,
    }


def record_payload(record_db_id: int) -> dict[str, Any]:
    row = get_record(record_db_id)
    if not row or not row["payload_json"]:
        raise LookupError("JSON payload not found")
    return json.loads(row["payload_json"])


def ecg_recordings_for_patient(patient_id: int, limit: int = 8) -> list[dict[str, Any]]:
    rows = query_db(
        """SELECT * FROM ecg_recordings WHERE patient_id=? ORDER BY measured_at DESC,id DESC LIMIT ?""",
        (patient_id, int(limit)),
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["waveform_samples"] = json.loads(item.get("waveform_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            item["waveform_samples"] = []
        output.append(item)
    return output


def default_generator_values(adapter_key: str) -> dict[str, Any]:
    cfg = adapter_config(adapter_key)
    sample = sample_values(adapter_key)
    values = {
        "device_identifier": cfg["default_device_identifier"],
        "measurement_timestamp": _utc_now()[:16],
        **sample,
    }
    if adapter_key == "generic_ecg_patch":
        values["ecg_heart_rate"] = values.pop("heart_rate")
    elif adapter_key == "omron_connect":
        values["omron_heart_rate"] = values.pop("heart_rate")
    elif adapter_key == "generic_ble_pulse_oximeter":
        values["pulse_spo2"] = values.pop("spo2")
        values["pulse_heart_rate"] = values.pop("heart_rate")
    elif adapter_key == "withings":
        values["withings_systolic_bp"] = values.pop("systolic_bp")
        values["withings_diastolic_bp"] = values.pop("diastolic_bp")
        values["withings_heart_rate"] = values.pop("heart_rate")
        values["withings_weight"] = values.pop("weight")
        values["withings_body_fat"] = values.pop("body_fat")
    return values
