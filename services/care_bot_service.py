"""
Care.AI care chatbot — guided symptom triage with an OpenAI interview layer.

Modelled on MediCare AI (Lung_Cancer_V3 services/clinical_chat.py): a symptom
phrase opens a short series of questions, and the answers produce an
assessment with an urgency level, self-care advice and an offer to involve
the care team.

Two kinds of interview:
- Rule-based flows for 11 common symptoms (care_bot_flows*.py): fixed yes/no
  questions, warning signs first, deterministic urgency.
- An OpenAI interview (care_bot_ai.py) for any other health issue: the model
  picks each next question from the answers so far, then writes the result.
When OpenAI is on it also writes the analysis at the end of a rule-based flow
and reads free-text replies to its yes/no questions, and it answers general
health questions ("what is a normal blood pressure?") directly.

Safety, whatever the model says:
- Every message is screened for emergency and crisis phrases BEFORE any flow
  or AI call; a hit abandons the interview and tells the person to call 112.
- The warning answer to a warning-sign question ends the interview at once,
  at the level that question sets; for an emergency or a self-harm question
  the model is not consulted again. "Don't know" to a warning sign raises the
  result to at least same-day review.
- The final level is the most urgent of the rules and the model: AI can raise
  a level, never lower it. Without AI, or when a call fails, the rules decide.
- Progress is explicit server-side state, and replies are structured data the
  widget renders as text — never HTML. MediCare re-derived progress from old
  message text and inserted raw <button> markup with innerHTML.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Dict, List, Optional

from services import care_bot_ai
from services.care_bot_content import (
    ANSWER_LABELS, CRISIS_PHRASES, EMERGENCY_PHRASES, LANGS, LEVELS, LIKELIHOOD, MESSAGES,
    NO_WORDS, UNKNOWN_WORDS, YES_WORDS, pick,
)
from services.care_bot_flows import FLOWS, PRIORITY
from services.db import execute_db, query_db

MAX_MESSAGE_CHARS = 1000
EARLIER_MESSAGES = 3          # the person's earlier messages the AI sees as context

UI = {
    "title": {"en": "Care assistant", "nl": "Zorgassistent", "de": "Pflegeassistent"},
    "subtitle": {"en": "Symptom guidance · not a diagnosis", "nl": "Hulp bij klachten · geen diagnose",
                 "de": "Orientierung bei Beschwerden · keine Diagnose"},
    "open": {"en": "Open the care assistant", "nl": "Zorgassistent openen", "de": "Pflegeassistent öffnen"},
    "close": {"en": "Close", "nl": "Sluiten", "de": "Schließen"},
    "placeholder": {"en": "Describe what's bothering you…", "nl": "Beschrijf wat u dwarszit…",
                    "de": "Beschreiben Sie Ihre Beschwerden…"},
    "send": {"en": "Send", "nl": "Verstuur", "de": "Senden"},
    "restart": {"en": "Start over", "nl": "Opnieuw beginnen", "de": "Neu beginnen"},
    "handoff": {"en": "Send summary to my care team", "nl": "Stuur samenvatting naar mijn zorgteam",
                "de": "Zusammenfassung an mein Pflegeteam senden"},
    "thinking": {"en": "Thinking…", "nl": "Even denken…", "de": "Einen Moment…"},
    "error": {"en": "Something went wrong. Please try again.", "nl": "Er ging iets mis. Probeer het opnieuw.",
              "de": "Etwas ist schiefgelaufen. Bitte versuchen Sie es erneut."},
    "emergency_banner": {"en": "Emergency? Call 112", "nl": "Spoed? Bel 112", "de": "Notfall? 112 anrufen"},
}


def ui_strings(lang: str) -> Dict[str, object]:
    """Widget chrome for one language, including the quick-reply labels."""
    strings: Dict[str, object] = {key: pick(texts, lang) for key, texts in UI.items()}
    strings["answers"] = [pick(ANSWER_LABELS[k], lang) for k in ("yes", "no", "unknown")]
    return strings


# ---------------------------------------------------------------------------
# Text understanding
# ---------------------------------------------------------------------------

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "´": "'"})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").translate(_APOSTROPHES).lower()).strip()


def _find(text: str, phrase: str) -> Optional[int]:
    """Start index of `phrase` as a whole word/phrase in `text`, else None."""
    match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
    return match.start() if match else None


def _has_any(text: str, phrases) -> bool:
    return any(_find(text, p) is not None for p in phrases)


def screen(text: str) -> Optional[str]:
    """'crisis', 'emergency' or None. Checks every language, whatever the UI
    language, because people type in the language they think in."""
    t = _norm(text)
    if any(_has_any(t, phrases) for phrases in CRISIS_PHRASES.values()):
        return "crisis"
    if any(_has_any(t, phrases) for phrases in EMERGENCY_PHRASES.values()):
        return "emergency"
    return None


def detect_flow(text: str) -> Optional[str]:
    t = _norm(text)
    hits = {name for name, flow in FLOWS.items()
            if any(_has_any(t, keywords) for keywords in flow["keywords"].values())}
    return next((name for name in PRIORITY if name in hits), None)


def parse_answer(text: str) -> Optional[str]:
    """'yes', 'no', 'unknown' or None. When both appear, the first one wins:
    "yes, but not at night" is a yes, "no, not really" is a no."""
    t = _norm(text).strip(" .!?,")
    if t in ("yes", "no", "unknown"):
        return t
    if _has_any(t, UNKNOWN_WORDS):
        return "unknown"
    yes_at = min((i for i in (_find(t, w) for w in YES_WORDS) if i is not None), default=None)
    no_at = min((i for i in (_find(t, w) for w in NO_WORDS) if i is not None), default=None)
    if yes_at is None and no_at is None:
        return None
    if no_at is None:
        return "yes"
    if yes_at is None:
        return "no"
    return "yes" if yes_at < no_at else "no"


_BARE_ANSWERS = set(YES_WORDS + NO_WORDS + UNKNOWN_WORDS) | {
    _norm(label) for labels in ANSWER_LABELS.values() for label in labels.values()}


def _own_words(text: str) -> Optional[str]:
    """The reply when it says more than a bare yes / no / don't know - the AI
    analysis gets "yes, 38.6 this morning", not just "yes"."""
    return None if _norm(text).strip(" .!?,") in _BARE_ANSWERS else text.strip()


_QUESTION_WORDS = {"what", "what's", "whats", "why", "how", "when", "which", "who", "is", "are", "can", "could",
                   "should", "does", "do", "wat", "waarom", "hoe", "wanneer", "welke", "wie", "kan", "mag", "moet",
                   "zijn", "warum", "wann", "welche", "welcher", "welches", "ist", "sind", "kann", "darf", "soll",
                   "sollte"}
_FIRST_PERSON = {"i", "i'm", "im", "i've", "ive", "i'd", "me", "my", "mine", "we", "our", "us",
                 "ik", "mij", "mijn", "wij", "ons", "onze", "ich", "mir", "mich", "mein", "meine", "meinen",
                 "meinem", "meiner", "wir", "uns", "unser", "unsere"}


def _general_question(text: str) -> bool:
    """A question about health in general ("what temperature counts as a fever?")
    rather than the person's own problem ("my knee hurts"). With AI on these are
    answered directly instead of opening a symptom checklist on a keyword."""
    words = re.findall(r"[\w']+", _norm(text))
    return bool(words) and words[0] in _QUESTION_WORDS and not _FIRST_PERSON.intersection(words)


# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------

def _load(session_id: Optional[str]) -> Optional[dict]:
    if not session_id:
        return None
    row = query_db("SELECT * FROM care_bot_sessions WHERE id=?", (session_id,), one=True)
    if not row:
        return None
    session = dict(row)
    session["state"] = json.loads(session.get("state_json") or "{}")
    return session


def _log(session_id: str, role: str, text: str, payload: Optional[dict] = None) -> None:
    execute_db("INSERT INTO care_bot_messages(session_id, role, content, payload_json) VALUES(?,?,?,?)",
               (session_id, role, (text or "")[:4000],
                json.dumps(payload, ensure_ascii=False) if payload else None))


def _save_state(session_id: str, state: dict, lang: str) -> None:
    execute_db("UPDATE care_bot_sessions SET state_json=?, lang=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
               (json.dumps(state, ensure_ascii=False), lang, session_id))


def _create(lang: str, patient_id=None, family_user_id=None, user_id=None) -> dict:
    session_id = uuid.uuid4().hex
    execute_db("""INSERT INTO care_bot_sessions(id, patient_id, family_user_id, user_id, lang, state_json)
                  VALUES(?,?,?,?,?,?)""", (session_id, patient_id, family_user_id, user_id, lang, "{}"))
    greeting_key = "greeting_ai" if care_bot_ai.available() else "greeting"
    greeting = {"kind": "text", "text": pick(MESSAGES[greeting_key], lang)}
    _log(session_id, "bot", greeting["text"], greeting)
    return _load(session_id)


def open_session(session_id: Optional[str], lang: str, *, patient_id=None, family_user_id=None,
                 user_id=None) -> dict:
    """Resume a session or start one. A session is bound to at most one patient:
    context for a different patient starts a fresh chat instead of mixing them."""
    session = _load(session_id)
    if session is not None:
        conflict = ((patient_id and session["patient_id"] and session["patient_id"] != patient_id)
                    or (family_user_id and session["family_user_id"]
                        and session["family_user_id"] != family_user_id))
        if conflict:
            session = None
        elif (patient_id and not session["patient_id"]) or (family_user_id and not session["family_user_id"]):
            execute_db("""UPDATE care_bot_sessions SET patient_id=COALESCE(patient_id, ?),
                          family_user_id=COALESCE(family_user_id, ?) WHERE id=?""",
                       (patient_id, family_user_id, session["id"]))
            session = _load(session["id"])
    if session is None:
        session = _create(lang, patient_id, family_user_id, user_id)
    return session


def restart(session_id: Optional[str], lang: str, **context) -> dict:
    """Start over in a fresh session (same patient binding); the old transcript is kept."""
    old = _load(session_id)
    patient_id = context.get("patient_id") or (old or {}).get("patient_id")
    family_user_id = context.get("family_user_id") or (old or {}).get("family_user_id")
    return _create(lang, patient_id, family_user_id, context.get("user_id"))


def transcript(session_id: str, limit: int = 60) -> List[dict]:
    rows = query_db("""SELECT role, content, payload_json FROM care_bot_messages
                       WHERE session_id=? ORDER BY id DESC LIMIT ?""", (session_id, limit))
    return [{"role": r["role"], "text": r["content"],
             "reply": json.loads(r["payload_json"]) if r["payload_json"] else None}
            for r in reversed(rows)]


def _earlier_messages(session_id: str) -> List[str]:
    """The person's last few messages before the one being handled."""
    rows = query_db("""SELECT content FROM care_bot_messages WHERE session_id=? AND role='user'
                       ORDER BY id DESC LIMIT ?""", (session_id, EARLIER_MESSAGES + 1))
    return [r["content"] for r in reversed(rows[1:])]


# ---------------------------------------------------------------------------
# Levels and the result card
# ---------------------------------------------------------------------------

def level_for(flow_name: str, answers: Dict[str, str]) -> str:
    """Most urgent level any rule reaches — never the first rule that matches."""
    flow = FLOWS[flow_name]
    level = flow["default"]

    def raise_to(candidate: str) -> None:
        nonlocal level
        if LEVELS[candidate]["rank"] > LEVELS[level]["rank"]:
            level = candidate

    for rule in flow["rules"]:
        ids = rule.get("any") or rule.get("all") or []
        check = any if "any" in rule else all
        if ids and check(answers.get(i) == "yes" for i in ids):
            raise_to(rule["level"])
    # Not knowing whether a warning sign is present is not reassurance.
    if any(q["stop_on_yes"] and answers.get(q["id"]) == "unknown" for q in flow["questions"]):
        raise_to("urgent")
    return level


def _higher(*levels: Optional[str]) -> str:
    """The most urgent of the given levels."""
    return max((lvl for lvl in levels if lvl in LEVELS), key=lambda lvl: LEVELS[lvl]["rank"], default="self_care")


def _assessment_reply(*, topic: str, level: str, answers: List[dict], lang: str,
                      assessment: Optional[dict] = None, advice: Optional[List[str]] = None,
                      stopped_early: bool = False, notice: Optional[str] = None) -> dict:
    """The result card for both interview kinds. In an emergency only the action
    matters: no explanations, possible causes or self-care tips to read first."""
    emergency = level == "emergency"
    found = assessment or {}
    title = pick(MESSAGES["summary_title"], lang).format(topic=topic)
    reply = {
        "kind": "summary",
        "title": title,
        "level": level,
        "level_label": pick(LEVELS[level]["label"], lang),
        "action": pick(LEVELS[level]["action"], lang),
        "stopped_early": pick(MESSAGES["stopped_early"], lang) if stopped_early else None,
        "analysis_title": pick(MESSAGES["analysis_title"], lang),
        "analysis": None if emergency else (found.get("analysis") or None),
        "advice_title": pick(MESSAGES["what_you_can_do"], lang),
        "advice": [] if emergency else (found.get("advice") or list(advice or [])),
        "watch_title": pick(MESSAGES["watch_title"], lang),
        "watch_for": [] if emergency else found.get("watch_for", []),
        "causes_title": pick(MESSAGES["causes_title"], lang),
        "causes": [] if emergency else [{**cause, "likelihood_label": pick(LIKELIHOOD[cause["likelihood"]], lang)}
                                        for cause in found.get("possible_causes", [])],
        "answers_title": pick(MESSAGES["your_answers"], lang),
        "answers": answers,
        "notice": notice,
        "disclaimer": pick(MESSAGES["disclaimer"], lang),
    }
    lines = [title, f"{reply['level_label']}: {reply['action']}"]
    if reply["analysis"]:
        lines.append(reply["analysis"])
    lines += [f"- {a['question']}: {a['answer']}" for a in answers]
    reply["text"] = "\n".join(lines)
    return reply


def _alarm_reply(kind: str, lang: str) -> dict:
    """'emergency' or 'crisis': call 112 now - and, for a crisis, the helplines."""
    return {"kind": kind, "level": "emergency", "text": pick(MESSAGES[kind], lang),
            "level_label": pick(LEVELS["emergency"]["label"], lang)}


def _ai_text(message: str, lang: str) -> dict:
    return {"kind": "ai", "text": message, "notice": pick(MESSAGES["ai_notice"], lang)}


def _fallback(lang: str) -> dict:
    return {"kind": "text", "text": pick(MESSAGES["fallback"], lang)}


# ---------------------------------------------------------------------------
# Rule-based flows
# ---------------------------------------------------------------------------

def _question_reply(flow_name: str, step: int, lang: str, intro: Optional[str] = None) -> dict:
    flow = FLOWS[flow_name]
    question = flow["questions"][step]
    return {
        "kind": "question",
        "flow": flow_name,
        "title": pick(flow["title"], lang),
        "counter": pick(MESSAGES["question_counter"], lang).format(i=step + 1, n=len(flow["questions"])),
        "intro": intro,
        "text": pick(question["text"], lang),
        "quick_replies": [pick(ANSWER_LABELS[k], lang) for k in ("yes", "no", "unknown")],
    }


def _start_flow(state: dict, flow_name: str, lang: str, opening: str, earlier: List[str]) -> dict:
    flow = FLOWS[flow_name]
    state.clear()
    state.update({"flow": flow_name, "step": 0, "answers": {}, "notes": {}, "finished": False,
                  "opening": opening, "earlier": earlier})
    intro = pick(MESSAGES["flow_intro"], lang).format(n=len(flow["questions"]), topic=pick(flow["topic"], lang))
    return _question_reply(flow_name, 0, lang, intro=intro)


def _flow_answer(session: dict, state: dict, text: str, lang: str) -> dict:
    flow_name = state["flow"]
    flow = FLOWS[flow_name]
    step = state.get("step", 0)
    question = flow["questions"][step]
    answer = parse_answer(text)
    if answer is None:
        other = detect_flow(text)
        if other and other != flow_name:
            return _start_flow(state, other, lang, text, _earlier_messages(session["id"]))
        answer = care_bot_ai.classify_answer(pick(question["text"], lang), text)
        if answer is None:
            return _question_reply(flow_name, step, lang, intro=pick(MESSAGES["reprompt"], lang))
    state["answers"][question["id"]] = answer
    own = _own_words(text)
    if own:
        state.setdefault("notes", {})[question["id"]] = own
    if question["stop_on_yes"] and answer == "yes":
        return _finish_flow(session, state, lang, stopped_early=True)
    step += 1
    if step >= len(flow["questions"]):
        return _finish_flow(session, state, lang, stopped_early=False)
    state["step"] = step
    return _question_reply(flow_name, step, lang)


def _finish_flow(session: dict, state: dict, lang: str, stopped_early: bool) -> dict:
    flow = FLOWS[state["flow"]]
    answers = state["answers"]
    rule_level = level_for(state["flow"], answers)
    assessment, notice = None, None
    if rule_level != "emergency" and care_bot_ai.available():    # an emergency must not wait for a model
        notes = state.get("notes", {})
        checklist = [{"question": pick(q["text"], lang), "answer": answers[q["id"]],
                      **({"own_words": notes[q["id"]]} if q["id"] in notes else {})}
                     for q in flow["questions"] if q["id"] in answers]
        assessment = care_bot_ai.analyse_checklist(
            lang=lang, topic=flow["title"]["en"].lower(), opening=state.get("opening", ""), checklist=checklist,
            profile=care_bot_ai.patient_profile(session.get("patient_id")), rule_level=rule_level,
            earlier=state.get("earlier", []))
        notice = pick(MESSAGES["ai_analysis_notice" if assessment else "ai_unavailable"], lang)
    level = _higher(rule_level, assessment["level"] if assessment else None)
    shown = [{"question": pick(q["short"], lang), "answer": pick(ANSWER_LABELS[answers[q["id"]]], lang)}
             for q in flow["questions"] if q["id"] in answers]
    reply = _assessment_reply(topic=pick(flow["title"], lang), level=level, answers=shown, lang=lang,
                              assessment=assessment, advice=pick(flow["advice"], lang),
                              stopped_early=stopped_early, notice=notice)
    reply["flow"] = state["flow"]
    reply["handoff_available"] = bool(session.get("patient_id")) and level != "emergency"
    state["finished"] = True
    state["summary"] = {"flow": state["flow"], "level": level, "rule_level": rule_level, "answers": dict(answers),
                        "stopped_early": stopped_early, "clinician_note": (assessment or {}).get("clinician_note", "")}
    return reply


# ---------------------------------------------------------------------------
# AI interview
# ---------------------------------------------------------------------------

def _start_interview(session: dict, state: dict, text: str, lang: str) -> Optional[dict]:
    earlier = _earlier_messages(session["id"])
    turn = care_bot_ai.next_turn(lang=lang, opening=text, turns=[],
                                 profile=care_bot_ai.patient_profile(session.get("patient_id")),
                                 minimum_level="self_care", must_conclude=False, earlier=earlier)
    if turn is None:
        return None
    if turn["action"] == "answer":
        return _ai_text(turn["message"], lang)
    state.clear()
    topic = turn["topic"] or text[:40]
    state.update({"mode": "ai", "topic": topic[:1].upper() + topic[1:], "opening": text, "earlier": earlier,
                  "turns": [], "pending": None, "floor": "self_care", "finished": False})
    if turn["action"] == "result":
        return _finish_interview(session, state, lang, turn["assessment"])
    return _ask(state, turn, lang)


def _ask(state: dict, turn: dict, lang: str) -> dict:
    first = not state["turns"]
    # Warning-sign questions always get the fixed yes / no / don't know replies,
    # so a tap is read by the same rules as the flows.
    options = ([pick(ANSWER_LABELS[k], lang) for k in ("yes", "no", "unknown")]
               if turn["red_flag"] else turn["options"])
    state["pending"] = {key: turn[key] for key in ("question", "label", "red_flag", "warning_answer",
                                                   "warning_level")}
    return {
        "kind": "question",
        "mode": "ai",
        "title": state["topic"],
        "counter": pick(MESSAGES["ai_counter"], lang).format(i=len(state["turns"]) + 1),
        "lead": (turn["message"] or None) if first else None,
        "intro": pick(MESSAGES["ai_intro"], lang) if first else None,
        "text": turn["question"],
        "quick_replies": options,
        "notice": pick(MESSAGES["ai_question_notice"], lang) if first else None,
    }


def _interview_answer(session: dict, state: dict, text: str, lang: str) -> dict:
    pending = state.get("pending")
    if not pending:                      # state left by an interrupted request: begin again
        state.clear()
        return _start_interview(session, state, text, lang) or _fallback(lang)
    state["pending"] = None
    state["turns"].append({"question": pending["question"], "label": pending["label"], "answer": text})
    if pending["red_flag"]:
        verdict = parse_answer(text) or care_bot_ai.classify_answer(pending["question"], text)
        if verdict == pending.get("warning_answer", "yes"):
            if pending.get("warning_level") == "crisis":
                state.clear()
                return _alarm_reply("crisis", lang)
            state["floor"] = _higher(state["floor"], pending.get("warning_level") or "urgent")
            if state["floor"] == "emergency":
                return _finish_interview(session, state, lang, None, stopped_early=True)
            return _interview_step(session, state, lang, must_conclude=True, stopped_early=True)
        if verdict == "unknown":
            state["floor"] = _higher(state["floor"], "urgent")
    return _interview_step(session, state, lang,
                           must_conclude=len(state["turns"]) >= care_bot_ai.MAX_QUESTIONS)


def _interview_step(session: dict, state: dict, lang: str, *, must_conclude: bool,
                    stopped_early: bool = False) -> dict:
    turn = care_bot_ai.next_turn(
        lang=lang, opening=state["opening"],
        turns=[{"question": t["question"], "answer": t["answer"]} for t in state["turns"]],
        profile=care_bot_ai.patient_profile(session.get("patient_id")), minimum_level=state["floor"],
        must_conclude=must_conclude, earlier=state.get("earlier", []))
    if turn is None:
        return _finish_interview(session, state, lang, None, stopped_early)
    if turn["action"] == "ask":
        return _ask(state, turn, lang)
    if turn["action"] == "answer":
        state.clear()
        return _ai_text(turn["message"], lang)
    return _finish_interview(session, state, lang, turn["assessment"], stopped_early)


def _finish_interview(session: dict, state: dict, lang: str, assessment: Optional[dict],
                      stopped_early: bool = False) -> dict:
    """Without an assessment - an emergency answer, or the AI failing - the level
    is the floor, and at least "book a GP appointment" when the AI could not
    finish: an unfinished assessment is no reassurance."""
    floor = state.get("floor", "self_care")
    if assessment:
        level, notice = _higher(floor, assessment["level"]), pick(MESSAGES["ai_analysis_notice"], lang)
    elif floor == "emergency":
        level, notice = "emergency", None
    else:
        level, notice = _higher(floor, "soon"), pick(MESSAGES["ai_unavailable"], lang)
    answers = [{"question": t["label"], "answer": t["answer"]} for t in state["turns"]]
    reply = _assessment_reply(topic=state["topic"], level=level, answers=answers, lang=lang,
                              assessment=assessment, stopped_early=stopped_early, notice=notice)
    reply["mode"] = "ai"
    reply["handoff_available"] = bool(session.get("patient_id")) and level != "emergency"
    state.update({"finished": True, "pending": None})
    state["summary"] = {"mode": "ai", "topic": state["topic"], "level": level, "answers": answers,
                        "stopped_early": stopped_early, "clinician_note": (assessment or {}).get("clinician_note", "")}
    return reply


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _next_reply(session: dict, state: dict, text: str, lang: str) -> dict:
    alarm = screen(text)
    if alarm:
        state.clear()
        return _alarm_reply(alarm, lang)

    if state.get("mode") == "ai" and not state.get("finished"):
        return _interview_answer(session, state, text, lang)
    if state.get("flow") and not state.get("finished"):
        return _flow_answer(session, state, text, lang)

    ai = care_bot_ai.available()
    new_flow = None if ai and _general_question(text) else detect_flow(text)
    if new_flow:
        return _start_flow(state, new_flow, lang, text, _earlier_messages(session["id"]))
    reply = _start_interview(session, state, text, lang) if ai else None
    return reply or _fallback(lang)


# ---------------------------------------------------------------------------
# Public API used by the routes
# ---------------------------------------------------------------------------

def ai_available() -> bool:
    return care_bot_ai.available()


def handle_message(session_id: Optional[str], text: str, lang: str, **context) -> dict:
    lang = lang if lang in LANGS else "en"
    text = (text or "").strip()[:MAX_MESSAGE_CHARS]
    if not text:
        raise ValueError("Type a message first.")
    session = open_session(session_id, lang, **context)
    state = session["state"]
    _log(session["id"], "user", text)
    reply = _next_reply(session, state, text, lang)
    _save_state(session["id"], state, lang)
    _log(session["id"], "bot", reply.get("text", ""), reply)
    return {"session_id": session["id"], "reply": reply}


def handoff(session_id: Optional[str], lang: str, *, family_name: Optional[str] = None) -> dict:
    """Send the finished assessment to the linked patient's nurse inbox, once.
    The note is written in English, like every other care event in CareAI; the
    answers of an AI interview stay in the language the person used."""
    lang = lang if lang in LANGS else "en"
    session = _load(session_id)
    if session is None:
        raise ValueError("No chat session.")
    if not session["patient_id"]:
        raise PermissionError("No patient is linked to this chat.")
    state = session["state"]
    summary = state.get("summary")
    if not summary:
        raise ValueError("Finish the questions first.")

    patient = query_db("SELECT first_name FROM patients WHERE id=?", (session["patient_id"],), one=True)
    done_text = pick(MESSAGES["handoff_done"], lang).format(name=patient["first_name"] if patient else "")
    if state.get("handoff_done"):
        return {"ok": True, "already": True, "text": done_text}

    who = f"Family member {family_name}" if family_name else "Patient"
    if summary.get("mode") == "ai":
        what = f"described {summary['topic']} and answered {len(summary['answers'])} AI-selected question(s)"
        answers = "; ".join(f"{a['question']}: {a['answer']}" for a in summary["answers"])
    else:
        flow = FLOWS[summary["flow"]]
        what = f"completed the {flow['title']['en'].lower()} check"
        answers = "; ".join(f"{q['short']['en']}: {ANSWER_LABELS[a]['en']}" for q in flow["questions"]
                            if (a := summary["answers"].get(q["id"])))
    note = f"[Care.AI care assistant] {who} {what}. Result: {LEVELS[summary['level']]['label']['en']}."
    if summary.get("clinician_note"):
        note += f" AI summary: {summary['clinician_note']}"
    note += f" Answers: {answers}."
    execute_db("INSERT INTO patient_questions(patient_id, question, status, assigned_role) VALUES(?,?,?,?)",
               (session["patient_id"], note[:1000], "open", "nurse"))
    execute_db("INSERT INTO care_events(patient_id, event_type, source, description) VALUES(?,?,?,?)",
               (session["patient_id"], "care_bot_handoff", "care_bot", note[:250]))
    state["handoff_done"] = True
    _save_state(session["id"], state, lang)
    _log(session["id"], "bot", done_text, {"kind": "handoff", "text": done_text})
    return {"ok": True, "already": False, "text": done_text}
