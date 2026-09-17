"""Optional OpenAI decision-support layer for the synthetic Hospital AI demo.

Deterministic rules are the safety floor. ML is an anomaly signal. LLM output is
supportive explanation only and cannot diagnose, prescribe, lower rule severity,
or bypass the mandatory human approval gate.
"""

from __future__ import annotations

import json
import os
import re

try:
    from openai import OpenAI
except Exception:  # Optional dependency; deterministic workflow must still run.
    OpenAI = None


def _configured_model():
    return (os.getenv("OPENAI_MODEL") or "").strip()


def ai_status() -> dict:
    enabled = os.getenv("CAREAI_USE_OPENAI", "0") == "1"
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    model = _configured_model()
    configured = enabled and has_key and bool(model) and OpenAI is not None
    if configured:
        reason = "ready"
    elif not enabled:
        reason = "OpenAI disabled; rules + ML + local RAG are active"
    elif not model:
        reason = "OPENAI_MODEL is not configured"
    else:
        reason = "OPENAI_API_KEY/client is not configured"
    return {
        "enabled": enabled,
        "configured": configured,
        "model": model or "Not configured",
        "rag_mode": os.getenv("CAREAI_RAG_MODE", "tfidf"),
        "reason": reason,
    }


def _client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if ai_status()["configured"] else None


def _json_from_text(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _protocol_text(protocols):
    return "\n\n".join(f"[{p.get('id')}] {p.get('title')}\n{p.get('content')}" for p in protocols)


def triage_agent(ehr_context: dict, rule_signal: dict, ml_signal: dict, protocols: list[dict]) -> dict:
    fallback = {
        "mode": "deterministic_fallback",
        "priority_suggestion": rule_signal.get("severity", "high"),
        "confidence": 0.0,
        "clinical_change": rule_signal.get("headline", "Clinical anomaly"),
        "contextual_risk_factors": [],
        "contradictions_or_data_quality": ["LLM not configured; deterministic rules + ML used."],
        "why_priority": rule_signal.get("evidence", ""),
        "recommended_human_review": rule_signal.get("recommendation", "Human clinical review required."),
        "citations": [p.get("id") for p in protocols],
    }
    client = _client()
    if not client:
        return fallback

    instructions = """You are Care.AI AI Triage in a SYNTHETIC hospital demonstration.
You are decision support, not the responsible clinician. Analyse only supplied evidence.
Do not diagnose, prescribe, change medication or oxygen, or claim certainty. Do not invent
missing facts. Deterministic rule severity is a safety floor and must not be downgraded.
Human nurse/physician review is mandatory. Return ONLY valid JSON with keys:
priority_suggestion, confidence, clinical_change, contextual_risk_factors,
contradictions_or_data_quality, why_priority, recommended_human_review, citations.
confidence is 0.0-1.0 and citations may only use supplied protocol IDs."""
    payload = {
        "ehr_context": ehr_context,
        "deterministic_rule_signal": rule_signal,
        "ml_anomaly_signal": ml_signal,
        "retrieved_demo_protocols": _protocol_text(protocols),
    }
    try:
        response = client.responses.create(
            model=_configured_model(),
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False, default=str),
        )
        data = _json_from_text(response.output_text)
        data["mode"] = "openai_llm"
        data["model"] = _configured_model()
        return data
    except Exception as exc:
        fallback["contradictions_or_data_quality"].append(
            f"OpenAI call failed; safe fallback used: {type(exc).__name__}"
        )
        return fallback


def copilot_agent(ehr_context: dict, rule_signal: dict, ml_signal: dict, triage: dict, protocols: list[dict]) -> dict:
    fallback = {
        "mode": "deterministic_fallback",
        "one_line_summary": f"{rule_signal.get('headline','Clinical anomaly')}: {rule_signal.get('evidence','')}",
        "what_changed": [rule_signal.get("evidence", "")],
        "relevant_context": [],
        "evidence_for_escalation": [rule_signal.get("evidence", "")],
        "recommended_next_human_action": rule_signal.get("recommendation", "Clinical review required."),
        "questions_for_clinician": [],
        "data_gaps": ["LLM not configured; deterministic summary shown."],
        "citations": [p.get("id") for p in protocols],
    }
    client = _client()
    if not client:
        return fallback

    instructions = """You are the Care.AI Clinical Copilot for a SYNTHETIC inpatient demo.
Create a concise evidence-grounded handoff for a nurse or physician using only supplied EHR,
rule, ML, triage, and demo protocol context. Do not diagnose, prescribe, recommend a specific
drug/dose, or autonomously order treatment. Separate observed data from interpretation.
Human review is mandatory. Return ONLY valid JSON with keys: one_line_summary, what_changed,
relevant_context, evidence_for_escalation, recommended_next_human_action,
questions_for_clinician, data_gaps, citations."""
    payload = {
        "ehr_context": ehr_context,
        "deterministic_rule_signal": rule_signal,
        "ml_anomaly_signal": ml_signal,
        "triage_agent_output": triage,
        "retrieved_demo_protocols": _protocol_text(protocols),
    }
    try:
        response = client.responses.create(
            model=_configured_model(),
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False, default=str),
        )
        data = _json_from_text(response.output_text)
        data["mode"] = "openai_llm_rag"
        data["model"] = _configured_model()
        return data
    except Exception as exc:
        fallback["data_gaps"].append(f"OpenAI call failed; safe fallback used: {type(exc).__name__}")
        return fallback
