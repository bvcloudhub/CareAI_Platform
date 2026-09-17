"""
Care.AI Copilot for clinicians: answers questions about one patient from the
CareAI record.

Hybrid by design:
- Record lookup first. Questions the record answers exactly — status, 7-day
  trends, alerts, conditions, latest readings, medication, falls, open tasks,
  GP, deterioration risk, score explanations and a handover note — come
  straight from the database: instant, exact and checkable.
- Anything else goes to OpenAI, grounded in a pseudonymised snapshot of the
  same record, when the Hospital AI switch is on (CAREAI_USE_OPENAI=1 with
  OPENAI_API_KEY and OPENAI_MODEL). Without it, a record overview is returned.
- A question describing an acute event gets the escalation message first.

It replaces a keyword matcher (copilot_service.clinician_answer) under which 7
of 12 routine questions fell through to one generic paragraph, "how has his
oxygen changed" returned only the latest value although a week of readings is
stored, and a patient without vitals crashed the respiratory branch.

The snapshot sent to OpenAI carries no name, birth date, city, GP or contacts;
the patient is identified only by the synthetic record reference, and names
known to CareAI are redacted from free-text notes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from services.db import query_db
from services.risk_engine import latest_risks

AI_TIMEOUT_SECONDS = 25
TREND_DAYS = 7
MAX_ANSWER_CHARS = 2500

SOURCE_LABELS = {
    "record": "From the record",
    "openai": "AI answer · grounded in the record",
    "safety": "Safety rule",
}

VITALS = {  # kind: (label, unit, format)
    "spo2": ("SpO₂", "%", "{:.1f}"),
    "heart_rate": ("heart rate", " bpm", "{:.0f}"),
    "resp_rate": ("respiratory rate", "/min", "{:.0f}"),
    "bp_sys": ("systolic BP", " mmHg", "{:.0f}"),
    "bp_dia": ("diastolic BP", " mmHg", "{:.0f}"),
    "temperature": ("temperature", " °C", "{:.1f}"),
    "activity": ("activity score", "", "{:.0f}"),
    "sleep": ("sleep score", "", "{:.0f}"),
}
NOTABLE_CHANGE = {"spo2": 2.0, "heart_rate": 10, "resp_rate": 3, "bp_sys": 15, "temperature": 0.8, "activity": 15}

VITAL_TERMS = {
    "spo2": ["oxygen", "spo2", "saturation", "sats", "o2"],
    "heart_rate": ["heart rate", "pulse", "hr"],
    "resp_rate": ["respiratory rate", "breathing rate", "resp rate", "rr"],
    "bp": ["blood pressure", "bp", "systolic", "diastolic"],
    "temperature": ["temperature", "temp", "fever"],
    "activity": ["activity", "mobility", "steps"],
    "sleep": ["sleep"],
}
CONDITION_TERMS = {  # substring of conditions.display: words a clinician uses
    "diabetes": ["diabetes", "diabetic"],
    "copd": ["copd", "emphysema", "chronic bronchitis"],
    "heart failure": ["heart failure", "chf"],
    "hypertension": ["hypertension", "high blood pressure"],
    "atrial": ["atrial fibrillation", "afib", "af"],
    "frailty": ["frailty", "frail"],
    "cognitive": ["dementia", "cognitive impairment", "memory problems"],
    "osteoporosis": ["osteoporosis"],
}
RISK_TERMS = {
    "copd": ["copd", "respiratory", "breathing"],
    "fall_prediction": ["fall", "falls", "falling"],
    "medication": ["medication", "adherence"],
    "cardiac": ["cardiac", "heart"],
    "infection": ["infection", "sepsis"],
    "frailty": ["frailty"],
}
RISK_NAMES = {"overall": "Overall", "copd": "COPD / respiratory", "fall_prediction": "Fall-risk prediction",
              "fall_event": "Active fall event", "medication": "Medication adherence", "cardiac": "Cardiac",
              "infection": "Infection", "frailty": "Frailty"}

# Present-tense acute events only: "has he had chest pain?" is a clinician's
# question about history, not an emergency report.
URGENT = [r"\b(is|am|are) having chest pain\b", r"\bcan'?t breathe\b", r"\bcannot breathe\b", r"\bunconscious\b",
          r"\bnot responding\b", r"\bunresponsive\b", r"\bhaving a stroke\b", r"\bface is drooping\b",
          r"\bsevere bleeding\b", r"\bhaving a seizure\b", r"\bhas collapsed\b"]

_AI_INSTRUCTIONS = """You are Care.AI Copilot, decision support for nurses and GPs in a SYNTHETIC remote-care demo for older adults.
Answer the clinician's question using ONLY the patient record JSON supplied. Earlier conversation is context only.
- If the record does not contain the answer, say "Not recorded in Care.AI" and suggest what to check.
- Quote the values you rely on, with units and timestamps where the record gives them.
- Separate what the record shows from your interpretation. Do not diagnose, prescribe or give doses.
- "protocols" are demo organisational protocols: cite one by id only if it is relevant; they are not real guidance.
- A qualified clinician remains responsible for every decision.
- At most 150 words, plain text, short paragraphs or "-" bullets. No markdown headings or tables."""


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def _dt(value) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _when(value) -> str:
    moment = value if isinstance(value, datetime) else _dt(value)
    return moment.strftime("%d %b %H:%M") if moment else "time not recorded"


def _age(birth_date) -> Optional[int]:
    born = _dt(birth_date)
    if not born:
        return None
    today = datetime.now().date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _trend(points) -> Optional[dict]:
    """First vs latest reading across the TREND_DAYS before the latest one."""
    if len(points) < 2:
        return None
    newest = points[0][1]
    window = [pt for pt in points if (newest - pt[1]).total_seconds() <= TREND_DAYS * 86400]
    if len(window) < 2:
        return None
    values = [v for v, _ in window]
    return {"first": window[-1][0], "first_at": window[-1][1], "last": window[0][0], "last_at": window[0][1],
            "min": min(values), "max": max(values), "n": len(window)}


def record(patient_id: int) -> Optional[dict]:
    patient = query_db("SELECT * FROM patients WHERE id=?", (patient_id,), one=True)
    if not patient:
        return None
    vitals = {}
    for kind in VITALS:
        # The latest reading plus the TREND_DAYS before it, however dense the feed.
        rows = query_db("""SELECT value, measured_at FROM vitals WHERE patient_id=? AND kind=?
                             AND replace(measured_at, 'T', ' ') >= (
                               SELECT datetime(MAX(replace(measured_at, 'T', ' ')), ?)
                               FROM vitals WHERE patient_id=? AND kind=?)
                           ORDER BY measured_at DESC""",
                        (patient_id, kind, f"-{TREND_DAYS} days", patient_id, kind))
        points = [(r["value"], _dt(r["measured_at"])) for r in rows if _dt(r["measured_at"])]
        vitals[kind] = {"latest": points[0] if points else None, "trend": _trend(points)}
    missed = query_db("""SELECT COUNT(*) n FROM medication_events WHERE patient_id=? AND status='missed'
                         AND scheduled_at >= datetime('now', '-7 days')""", (patient_id,), one=True)
    fall = query_db("SELECT * FROM fall_assessments WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,), one=True)
    return {
        "patient": dict(patient),
        "age": _age(patient["birth_date"]),
        "vitals": vitals,
        "risks": {r["risk_type"]: dict(r) for r in latest_risks(patient_id)},
        "conditions": [dict(r) for r in query_db(
            "SELECT code, display FROM conditions WHERE patient_id=? AND active=1", (patient_id,))],
        "medications": [dict(r) for r in query_db(
            "SELECT name, dose, schedule FROM medications WHERE patient_id=? AND active=1", (patient_id,))],
        "missed_doses_7d": missed["n"] if missed else 0,
        "tasks": [dict(r) for r in query_db("""SELECT task_type, priority, assigned_role, rationale, created_at
            FROM care_tasks WHERE patient_id=? AND status='open'
            ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
                     created_at DESC""", (patient_id,))],
        "alerts": [dict(r) for r in query_db("""SELECT alert_type, severity, message, created_at FROM alerts
            WHERE patient_id=? AND status='open' ORDER BY created_at DESC LIMIT 10""", (patient_id,))],
        "events": [dict(r) for r in query_db("""SELECT event_type, source, description, created_at FROM care_events
            WHERE patient_id=? ORDER BY created_at DESC LIMIT 10""", (patient_id,))],
        "fall": dict(fall) if fall else None,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").lower()).strip()


def _has(text: str, term: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text) is not None


def _any(text: str, terms) -> bool:
    return any(_has(text, t) for t in terms)


def _value(kind: str, value) -> str:
    label, unit, spec = VITALS[kind]
    return "not recorded" if value is None else spec.format(value) + unit


def _name(rec) -> str:
    return rec["patient"]["first_name"]


def _explicit_vitals(q: str) -> List[str]:
    kinds: List[str] = []
    for key, terms in VITAL_TERMS.items():
        if _any(q, terms):
            kinds += ["bp_sys", "bp_dia"] if key == "bp" else [key]
    return kinds


def _mentioned_vitals(q: str) -> List[str]:
    """Named vitals, plus SpO2 and respiratory rate when breathing is mentioned."""
    kinds = _explicit_vitals(q)
    if _any(q, ["breathing", "breath", "breathless"]):
        kinds += [k for k in ("spo2", "resp_rate") if k not in kinds]
    return kinds


def _latest(rec, kind: str) -> str:
    point = rec["vitals"][kind]["latest"]
    return f"{_value(kind, point[0])} ({_when(point[1])})" if point else "not recorded"


def _latest_line(rec) -> str:
    sys_pt, dia_pt = rec["vitals"]["bp_sys"]["latest"], rec["vitals"]["bp_dia"]["latest"]
    bp = f"BP {sys_pt[0]:.0f}/{dia_pt[0]:.0f} mmHg" if sys_pt and dia_pt else "BP not recorded"
    parts = [f"SpO₂ {_latest(rec, 'spo2')}", f"heart rate {_value('heart_rate', (rec['vitals']['heart_rate']['latest'] or [None])[0])}",
             f"respiratory rate {_value('resp_rate', (rec['vitals']['resp_rate']['latest'] or [None])[0])}", bp,
             f"temperature {_value('temperature', (rec['vitals']['temperature']['latest'] or [None])[0])}"]
    return ", ".join(parts)


def _trend_line(kind: str, trend: dict) -> str:
    label, unit, spec = VITALS[kind]
    change = float(spec.format(trend["last"])) - float(spec.format(trend["first"]))  # as displayed
    direction = "no net change" if abs(change) < 1e-9 else f"{'up' if change > 0 else 'down'} {spec.format(abs(change))}{unit}"
    return (f"{label}: {_value(kind, trend['first'])} ({_when(trend['first_at'])}) → "
            f"{_value(kind, trend['last'])} ({_when(trend['last_at'])}), {direction}; "
            f"range {_value(kind, trend['min'])}–{_value(kind, trend['max'])} over {trend['n']} readings")


def _notable_changes(rec) -> List[str]:
    out = []
    for kind, threshold in NOTABLE_CHANGE.items():
        trend = rec["vitals"][kind]["trend"]
        if trend and abs(trend["last"] - trend["first"]) >= threshold:
            out.append(f"{VITALS[kind][0]} {_value(kind, trend['first'])} → {_value(kind, trend['last'])}")
    return out


def _top_risks(rec, n: int) -> List[str]:
    risks = [r for t, r in rec["risks"].items() if t not in ("overall", "fall_event") and r["score"] > 0]
    risks.sort(key=lambda r: r["score"], reverse=True)
    return [f"{RISK_NAMES.get(r['risk_type'], r['risk_type'])} {r['score']} ({r['level']})" for r in risks[:n]]


def _concern(overall) -> str:
    text = overall["explanation"] or ""
    return text if text.lower().startswith("dominant") else f"Dominant concern: {text}"


def _tasks_line(rec, limit: int = 3) -> str:
    tasks = rec["tasks"][:limit]
    if not tasks:
        return ""
    return "Open tasks: " + "; ".join(
        f"{t['task_type'].replace('_', ' ')} ({t['priority']}, {t['assigned_role'] or 'unassigned'}, {_when(t['created_at'])})"
        for t in tasks) + "."


# ---------------------------------------------------------------------------
# Record-lookup answers
# ---------------------------------------------------------------------------

def _summary(q, rec) -> str:
    p = rec["patient"]
    overall = rec["risks"].get("overall")
    lines = [f"{p['first_name']} {p['last_name']} ({p['external_ref']}, {rec['age'] or '?'}y): risk status {p['current_status']}."]
    if overall:
        lines.append(f"{_concern(overall)} (evidence: {overall['evidence']}).")
    lines.append(f"Latest readings: {_latest_line(rec)}.")
    changes = _notable_changes(rec)
    if changes:
        lines.append(f"Notable {TREND_DAYS}-day changes: {'; '.join(changes)}.")
    lines.append(_tasks_line(rec) or "No open tasks.")
    lines.append(f"{len(rec['alerts'])} open alert(s)." if rec["alerts"] else "No open alerts.")
    return "\n".join(lines)


def _handover(q, rec) -> str:
    p = rec["patient"]
    overall = rec["risks"].get("overall")
    meds = ", ".join(f"{m['name']} {m['dose']} {m['schedule']}" for m in rec["medications"]) or "none recorded"
    lines = [
        f"Handover — {p['first_name']} {p['last_name']} ({p['external_ref']}), {rec['age'] or '?'}y, {p.get('living_setting') or 'setting not recorded'}",
        f"S: Risk status {p['current_status']}." + (f" {_concern(overall)}." if overall else ""),
        f"B: {', '.join(c['display'] for c in rec['conditions']) or 'No active conditions recorded'}. Medication: {meds}.",
        f"A: {_latest_line(rec)}.",
    ]
    changes = _notable_changes(rec)
    if changes:
        lines.append(f"   {TREND_DAYS}-day changes: {'; '.join(changes)}.")
    top = _top_risks(rec, 3)
    if top:
        lines.append(f"   Highest risks: {', '.join(top)}.")
    lines.append(f"R: {_tasks_line(rec) or 'No open tasks.'} "
                 f"{str(len(rec['alerts'])) + ' open alert(s).' if rec['alerts'] else 'No open alerts.'} "
                 "Every action needs human clinical review.")
    return "\n".join(lines)


def _alerts(q, rec) -> str:
    if not rec["alerts"]:
        return f"No open alerts for {_name(rec)}."
    lines = [f"{len(rec['alerts'])} open alert(s) for {_name(rec)}:"]
    lines += [f"- {a['severity']}: {(a['alert_type'] or 'alert').replace('_', ' ')} — {a['message']} ({_when(a['created_at'])})"
              for a in rec["alerts"]]
    return "\n".join(lines)


def _conditions(q, rec) -> str:
    recorded = rec["conditions"]
    all_line = ", ".join(f"{c['display']} ({c['code']})" for c in recorded) or "none recorded"
    asked = [key for key, terms in CONDITION_TERMS.items() if _any(q, terms)]
    if not asked:
        return f"Active conditions for {_name(rec)}: {all_line}."
    parts = []
    for key in asked:
        hit = next((c for c in recorded if key in c["display"].lower()), None)
        label = CONDITION_TERMS[key][0]
        parts.append(f"Yes — {hit['display']} ({hit['code']}) is recorded." if hit
                     else f"No {label} is recorded among the active conditions.")
    return " ".join(parts) + f" All active conditions: {all_line}."


def _gp(q, rec) -> str:
    p = rec["patient"]
    return (f"GP: {p.get('gp_name') or 'not recorded'}. Emergency contact: {p.get('emergency_contact_name') or 'not recorded'}"
            f"{' (' + p['emergency_contact_relation'] + ')' if p.get('emergency_contact_relation') else ''}. "
            f"Living setting: {p.get('living_setting') or 'not recorded'}, {p.get('city') or 'city not recorded'}.")


def _admission(q, rec) -> str:
    overall = rec["risks"].get("overall")
    lines = [f"Overall deterioration risk: {overall['score']}/100 ({overall['level']}) — {overall['explanation']}."
             if overall else "No overall risk score is recorded."]
    top = _top_risks(rec, 3)
    if top:
        lines.append(f"Highest component risks: {', '.join(top)}.")
    urgent_tasks = [t for t in rec["tasks"] if t["priority"] in ("critical", "high")]
    if urgent_tasks:
        lines.append("Open high-priority tasks: " + "; ".join(
            f"{t['task_type'].replace('_', ' ')} ({t['priority']})" for t in urgent_tasks[:3]) + ".")
    changes = _notable_changes(rec)
    if changes:
        lines.append(f"Notable {TREND_DAYS}-day changes: {'; '.join(changes)}.")
    lines.append("Care.AI does not predict admission directly: these are rule-based deterioration signals for clinician review.")
    return "\n".join(lines)


def _explain(q, rec) -> str:
    targets = [rt for rt, terms in RISK_TERMS.items() if _any(q, terms)] or ["overall"]
    lines = []
    for rt in targets:
        risk = rec["risks"].get(rt)
        if not risk:
            lines.append(f"{RISK_NAMES.get(rt, rt)}: no score recorded yet.")
            continue
        lines.append(f"{RISK_NAMES.get(rt, rt)} {risk['score']}/100 ({risk['level']}): {risk['explanation']}. "
                     f"Evidence: {risk['evidence']}. Scored {_when(risk['created_at'])} by {risk['model_version']}.")
        if rt == "fall_prediction":
            event = rec["risks"].get("fall_event")
            if event and event["score"] > 0:
                lines.append(f"Active fall event: {event['score']}% confidence — {event['evidence']}.")
            else:
                lines.append("There is no active fall event; the prediction reflects longer-term mobility, frailty and sleep.")
    lines.append("Bands: low <45, medium 45–69, high 70–84, critical 85+. Rule-based scores for clinician review.")
    return "\n".join(lines)


def _trend_answer(q, rec) -> str:
    kinds = _mentioned_vitals(q)
    specific = bool(kinds)
    kinds = kinds or ["spo2", "heart_rate", "resp_rate", "bp_sys", "temperature", "activity"]
    lines = [f"{_name(rec)} — {TREND_DAYS}-day trend, counted back from the latest reading:"]
    for kind in kinds:
        trend = rec["vitals"][kind]["trend"]
        lines.append(f"- {_trend_line(kind, trend)}" if trend else f"- {VITALS[kind][0]}: not enough readings for a trend")
    if not specific and rec["events"]:
        lines.append("Recent care events: " + "; ".join(
            f"{e['event_type'].replace('_', ' ')} ({_when(e['created_at'])})" for e in rec["events"][:3]) + ".")
    return "\n".join(lines)


def _reading(q, rec) -> str:
    kinds = _mentioned_vitals(q)
    if "bp_sys" in kinds and "bp_dia" in kinds:
        sys_pt, dia_pt = rec["vitals"]["bp_sys"]["latest"], rec["vitals"]["bp_dia"]["latest"]
        kinds = [k for k in kinds if k not in ("bp_sys", "bp_dia")]
        bp = (f"Latest blood pressure: {sys_pt[0]:.0f}/{dia_pt[0]:.0f} mmHg ({_when(sys_pt[1])})."
              if sys_pt and dia_pt else "No blood pressure reading is recorded.")
    else:
        bp = None
    lines = [f"Latest {VITALS[k][0]}: {_latest(rec, k)}." for k in kinds]
    return " ".join(([bp] if bp else []) + lines)


def _falls(q, rec) -> str:
    return _explain("fall", rec)


def _respiratory(q, rec) -> str:
    lines = [f"Latest: SpO₂ {_latest(rec, 'spo2')}, respiratory rate {_latest(rec, 'resp_rate')}, "
             f"heart rate {_latest(rec, 'heart_rate')}."]
    for kind in ("spo2", "resp_rate"):
        trend = rec["vitals"][kind]["trend"]
        if trend:
            lines.append(f"{TREND_DAYS}-day {_trend_line(kind, trend)}.")
    copd = rec["risks"].get("copd")
    if copd:
        lines.append(f"COPD / respiratory risk {copd['score']}/100 ({copd['level']}): {copd['evidence']}.")
    return "\n".join(lines)


def _medication(q, rec) -> str:
    meds = ", ".join(f"{m['name']} {m['dose']} {m['schedule']}" for m in rec["medications"]) or "none recorded"
    lines = [f"Active medication: {meds}.",
             f"Missed doses in the last 7 days: {rec['missed_doses_7d']}."]
    risk = rec["risks"].get("medication")
    if risk:
        lines.append(f"Medication-adherence risk {risk['score']}/100 ({risk['level']}): {risk['evidence']}.")
    return "\n".join(lines)


def _tasks(q, rec) -> str:
    if not rec["tasks"]:
        top = _top_risks(rec, 1)
        return ("No open care tasks." + (f" The highest current risk is {top[0]}; consider running the Care.AI agents "
                                         "to generate a human-review workflow." if top else ""))
    lines = [f"{len(rec['tasks'])} open task(s), most urgent first:"]
    lines += [f"- {t['task_type'].replace('_', ' ')} — {t['priority']}, {t['assigned_role'] or 'unassigned'}, "
              f"{_when(t['created_at'])}. {t['rationale'] or ''}".rstrip() for t in rec["tasks"][:5]]
    lines.append("A qualified clinician remains responsible for the decision.")
    return "\n".join(lines)


_EXPLAIN_WORDS = ["explain", "why", "reason", "reasons", "based on", "evidence", "how was"]
_TREND_WORDS = ["trend", "trends", "changed", "change", "changes", "over the last", "past week", "this week",
                "last week", "7 days", "seven days", "since", "compare", "compared", "worse", "better",
                "improving", "improved", "deteriorating", "deteriorated"]
_READING_WORDS = ["last", "latest", "current", "currently", "now", "reading", "readings", "value", "when was", "level",
                  "what is", "what's", "what was", "show", "ok", "okay", "normal", "high", "low"]
_RISK_OF = ["at risk", "risk of"]
# Interpretation rather than lookup: these go to the grounded AI first when it is on.
_INTERPRETIVE = ["could", "cause", "caused", "causing", "linked", "link", "related", "consistent", "interpret",
                 "mean", "means", "worry", "worried", "concerning", "significant", "likely", "correlate",
                 "correlated", "differential", "suggest", "suggests", "explain why"]


def _names_risk(q: str) -> bool:
    return any(_any(q, t) for t in RISK_TERMS.values())

# Order matters: the first match wins, so specific intents precede broad ones.
INTENTS = [
    ("handover", lambda q, r: _any(q, ["handover", "hand-over", "handoff", "hand-off", "sbar", "shift note"]), _handover),
    ("alerts", lambda q, r: _any(q, ["alert", "alerts", "alarm", "alarms", "warning", "warnings"]), _alerts),
    ("conditions", lambda q, r: (any(_any(q, t) for t in CONDITION_TERMS.values())
                                 and _any(q, ["have", "has", "history", "diagnosed", "diagnosis", "known", "suffer", "got"])
                                 and not _any(q, ["risk", "score", "scores"]))
                                or _any(q, ["conditions", "diagnoses", "problem list", "comorbidities"]), _conditions),
    ("gp", lambda q, r: _any(q, ["gp", "general practitioner", "family doctor", "emergency contact", "next of kin",
                                 "contact person"]) or (_has(q, "who") and _has(q, "doctor")), _gp),
    ("admission_risk", lambda q, r: _any(q, ["admission", "admitted", "hospitalisation", "hospitalization", "hospital",
                                             "how sick", "how unwell"])
                                    or (_any(q, _RISK_OF) and not _names_risk(q)), _admission),
    ("explain", lambda q, r: (_any(q, _EXPLAIN_WORDS) or _any(q, _RISK_OF))
                             and (_names_risk(q) or _any(q, ["risk", "score"])), _explain),
    ("trend", lambda q, r: _any(q, _TREND_WORDS), _trend_answer),
    ("reading", lambda q, r: bool(_explicit_vitals(q)) and _any(q, _READING_WORDS), _reading),
    ("falls", lambda q, r: _any(q, ["fall", "falls", "fell", "fallen"]), _falls),
    ("respiratory", lambda q, r: _any(q, ["copd", "breath", "breathing", "breathless", "respiratory", "oxygen",
                                          "spo2", "saturation", "wheeze", "wheezing", "inhaler"]), _respiratory),
    ("medication", lambda q, r: _any(q, ["medication", "medications", "medicine", "medicines", "meds", "drug", "drugs",
                                         "dose", "doses", "tablet", "tablets", "pill", "pills", "adherence"]), _medication),
    ("tasks", lambda q, r: _any(q, ["task", "tasks", "next", "action", "actions", "should", "recommend", "to do",
                                    "todo", "plan", "follow up", "follow-up", "priority"]), _tasks),
    # A bare vital ("and his pulse?") is a request for the latest reading.
    ("reading", lambda q, r: bool(_explicit_vitals(q)), _reading),
    ("summary", lambda q, r: _any(q, ["summary", "summarise", "summarize", "status", "overview", "how is", "how's",
                                      "doing", "update", "brief"]), _summary),
]


# ---------------------------------------------------------------------------
# Grounded AI answers
# ---------------------------------------------------------------------------

def copilot_status() -> Dict[str, object]:
    try:
        from services.openai_clinical_ai import ai_status
        status = ai_status()
        return {"ai": bool(status["configured"]), "model": status["model"]}
    except Exception:
        return {"ai": False, "model": "Not configured"}


def _known_names() -> List[str]:
    names = [r["display_name"] for r in query_db("SELECT display_name FROM users")]
    names += [r["display_name"] for r in query_db("SELECT display_name FROM family_users")]
    for r in query_db("SELECT first_name, last_name, gp_name, emergency_contact_name FROM patients"):
        names += [r["first_name"], r["last_name"], f"{r['first_name']} {r['last_name']}", r["gp_name"] or "",
                  r["emergency_contact_name"] or ""]
    # Longest first, so "Pieter Bakker" is replaced before "Pieter".
    return sorted({n for n in names if n and len(n) > 2}, key=len, reverse=True)


def _redact(text, names: List[str]) -> str:
    text = str(text or "")
    for name in names:
        text = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", "[name]", text)
    return text


def snapshot(rec) -> dict:
    """The record as sent to OpenAI: no name, birth date, city, GP or contacts."""
    names = _known_names()
    p = rec["patient"]
    vitals = {}
    for kind, data in rec["vitals"].items():
        shown = lambda v, spec=VITALS[kind][2]: float(spec.format(v))      # the precision the page displays
        entry = {}
        if data["latest"]:
            entry["latest"] = {"value": shown(data["latest"][0]), "unit": VITALS[kind][1].strip(),
                               "measured_at": str(data["latest"][1])}
        if data["trend"]:
            t = data["trend"]
            entry[f"trend_{TREND_DAYS}_days"] = {"first": shown(t["first"]), "first_at": str(t["first_at"]),
                                                 "last": shown(t["last"]), "last_at": str(t["last_at"]),
                                                 "min": shown(t["min"]), "max": shown(t["max"]), "readings": t["n"]}
        if entry:
            vitals[kind] = entry
    fall = rec["fall"] or {}
    return {
        "record_ref": p["external_ref"], "age": rec["age"], "sex": p.get("sex"),
        "living_setting": p.get("living_setting"), "risk_status": p.get("current_status"),
        "conditions": [f"{c['display']} ({c['code']})" for c in rec["conditions"]],
        "medications": [f"{m['name']} {m['dose']} {m['schedule']}" for m in rec["medications"]],
        "missed_doses_last_7_days": rec["missed_doses_7d"],
        "vitals": vitals,
        "risk_scores": {t: {k: r[k] for k in ("score", "level", "explanation", "evidence", "model_version", "created_at")}
                        for t, r in rec["risks"].items()},
        "open_tasks": [{"task": t["task_type"], "priority": t["priority"], "role": t["assigned_role"],
                        "rationale": _redact(t["rationale"], names), "created_at": t["created_at"]} for t in rec["tasks"]],
        "open_alerts": [{"type": a["alert_type"], "severity": a["severity"], "message": _redact(a["message"], names),
                         "created_at": a["created_at"]} for a in rec["alerts"]],
        "recent_care_events": [{"type": e["event_type"], "source": e["source"], "description": _redact(e["description"], names),
                                "created_at": e["created_at"]} for e in rec["events"]],
        "latest_fall_assessment": {k: fall.get(k) for k in ("assessment_type", "score", "level", "status", "impact_g",
                                                            "immobility_seconds", "location", "created_at")} if fall else None,
    }


def _protocols(question: str, rec) -> List[dict]:
    """Up to two demo protocols relevant to the question and the dominant concern."""
    try:
        from services.rag_service import retrieve_protocols
        overall = rec["risks"].get("overall")
        query = f"{question} {overall['explanation'] if overall else ''}"
        return [{"id": d.get("id"), "title": d.get("title"), "content": str(d.get("content", ""))[:700]}
                for d in retrieve_protocols(query, top_k=2) if (d.get("retrieval_score") or 0) > 0]
    except Exception:
        return []


def _ai_answer(rec, question: str, history: List[dict]) -> Optional[str]:
    if not copilot_status()["ai"]:
        return None
    try:
        from services.openai_clinical_ai import _client, _configured_model
        client = _client()
        if client is None:
            return None
        names = _known_names()
        conversation = [{"role": m["sender_type"], "text": _redact(m["content"], names)[:600]}
                        for m in (dict(row) for row in (history or [])[-6:])]
        payload = {"record": snapshot(rec), "protocols": _protocols(question, rec),
                   "conversation": conversation, "question": _redact(question, names)}
        response = client.with_options(timeout=AI_TIMEOUT_SECONDS).responses.create(
            model=_configured_model(), instructions=_AI_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False, default=str))
        text = (response.output_text or "").strip()
        return text[:MAX_ANSWER_CHARS] or None
    except Exception as exc:                                   # the record answer is always the fallback
        print(f"Copilot AI answer failed: {type(exc).__name__}: {str(exc)[:200]}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _result(text: str, source: str, safety: str = "normal", intent: Optional[str] = None) -> dict:
    return {"answer": text, "safety": safety, "source": source, "source_label": SOURCE_LABELS[source], "intent": intent}


def answer(patient_id: int, question: str, history: Optional[List[dict]] = None) -> dict:
    rec = record(patient_id)
    if rec is None:
        return _result("Patient not found.", "record")
    q = _norm(question)
    if any(re.search(pattern, q) for pattern in URGENT):
        return _result("This describes a potentially urgent situation. Follow the organisation's emergency protocol and "
                       "arrange immediate human clinical assessment — Care.AI is not a substitute for emergency services.",
                       "safety", safety="urgent", intent="urgent")
    interpretive = _any(q, _INTERPRETIVE)
    ai = _ai_answer(rec, question, history or []) if interpretive else None
    if ai:
        return _result(ai, "openai", intent="ai")
    for name, matches, handler in INTENTS:
        if matches(q, rec):
            return _result(handler(q, rec), "record", intent=name)
    ai = None if interpretive else _ai_answer(rec, question, history or [])   # one attempt per question
    if ai:
        return _result(ai, "openai", intent="ai")
    return _result(_summary(q, rec) + "\nRecord lookup covers status, trends, alerts, conditions, readings, medication, "
                   "falls, tasks, GP, deterioration risk, score explanations and handover notes.", "record", intent="overview")
