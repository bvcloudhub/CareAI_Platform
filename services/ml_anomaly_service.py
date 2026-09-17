"""Interpretable ML anomaly signal for the synthetic Hospital AI demo.

Isolation Forest is an additional ranking signal only. It never diagnoses a
condition, lowers deterministic safety rules, or bypasses human approval.
"""

from __future__ import annotations

from services.db import query_db

FEATURES = ["heart_rate", "spo2", "resp_rate", "systolic_bp", "diastolic_bp", "temperature", "oxygen_lpm"]
LABELS = {
    "heart_rate": "heart rate",
    "spo2": "SpO₂",
    "resp_rate": "respiratory rate",
    "systolic_bp": "systolic BP",
    "diastolic_bp": "diastolic BP",
    "temperature": "temperature",
    "oxygen_lpm": "oxygen flow",
}


def _vector(row):
    return [float(row[f] if row[f] is not None else 0.0) for f in FEATURES]


def ml_anomaly_signal(admission_id: int) -> dict:
    latest = query_db(
        "SELECT * FROM ehr_vitals WHERE admission_id=? ORDER BY measured_at DESC,id DESC LIMIT 1",
        (admission_id,), one=True,
    )
    if not latest:
        return {
            "available": False, "score": 0, "label": "unavailable", "features": [],
            "explanation": "No hospital vital-sign data is available.", "method": "not_run",
        }

    # Lazy imports keep the whole CareAI application runnable even if the optional
    # hospital ML dependency has not yet been installed.
    try:
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {
            "available": False, "score": 0, "label": "dependency_unavailable", "features": [],
            "explanation": f"Optional ML dependency unavailable ({type(exc).__name__}); deterministic rules remain active.",
            "method": "deterministic_fallback",
        }

    # Use the available synthetic inpatient history as the reference population.
    # The initial demo seed intentionally has a stable early baseline, so training
    # only on observations older than eight hours can result in zero variance.
    # Including all historical observations except the current row gives the
    # Isolation Forest enough variation to provide a meaningful *supporting*
    # anomaly signal while deterministic rules remain the clinical safety floor.
    baseline = query_db(
        """SELECT v.* FROM ehr_vitals v
           JOIN hospital_admissions a ON a.id=v.admission_id
           WHERE a.status='admitted' AND NOT (v.admission_id=? AND v.id=?)
           ORDER BY v.measured_at""",
        (admission_id, latest["id"]),
    )
    if len(baseline) < 12:
        return {
            "available": False, "score": 0, "label": "insufficient_baseline", "features": [],
            "explanation": "Not enough historical inpatient data for ML scoring; deterministic rules remain active.",
            "method": "deterministic_fallback",
        }

    X = np.asarray([_vector(row) for row in baseline], dtype=float)
    x = np.asarray([_vector(latest)], dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    xs = scaler.transform(x)
    model = IsolationForest(n_estimators=180, contamination=0.12, random_state=42)
    model.fit(Xs)
    baseline_scores = model.score_samples(Xs)
    latest_score = float(model.score_samples(xs)[0])

    # Lower IsolationForest score means more unusual. Convert it to a readable
    # percentile-like score for the demo.
    percentile = float(np.mean(baseline_scores > latest_score) * 100.0)
    ml_score = int(round(max(0.0, min(100.0, percentile))))

    z = xs[0]
    top_idx = np.argsort(np.abs(z))[::-1][:3]
    feature_signals = []
    for index in top_idx:
        key = FEATURES[int(index)]
        feature_signals.append({
            "feature": key,
            "label": LABELS[key],
            "value": float(latest[key] or 0),
            "z_score": round(float(z[index]), 2),
        })

    label = (
        "highly_unusual" if ml_score >= 85 else
        "unusual" if ml_score >= 65 else
        "mild" if ml_score >= 45 else
        "within_baseline"
    )
    explanation = ", ".join(f"{item['label']} z={item['z_score']}" for item in feature_signals)
    return {
        "available": True,
        "score": ml_score,
        "label": label,
        "raw_model_score": round(latest_score, 4),
        "features": feature_signals,
        "explanation": explanation,
        "method": "IsolationForest + StandardScaler on older synthetic inpatient vitals",
    }
