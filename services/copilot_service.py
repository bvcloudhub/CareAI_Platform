import re
from datetime import datetime
from services.db import query_db, execute_db
from services.risk_engine import get_latest_vitals, latest_risks

URGENT_PATTERNS = [
    r"\bchest pain\b", r"\bcan't breathe\b", r"\bcannot breathe\b",
    r"\bsevere shortness of breath\b", r"\bunconscious\b", r"\bnot responding\b",
    r"\bstroke\b", r"\bface droop\b", r"\bsevere bleeding\b"
]

def patient_context(patient_id):
    p = query_db("SELECT * FROM patients WHERE id=?", (patient_id,), one=True)
    if not p:
        return None
    conditions = [dict(r) for r in query_db("SELECT code,display FROM conditions WHERE patient_id=? AND active=1",(patient_id,))]
    meds = [dict(r) for r in query_db("SELECT name,dose,schedule FROM medications WHERE patient_id=? AND active=1",(patient_id,))]
    vitals = get_latest_vitals(patient_id)
    risks = [dict(r) for r in latest_risks(patient_id)]
    tasks = [dict(r) for r in query_db("""SELECT task_type,priority,status,rationale,created_at FROM care_tasks
                                         WHERE patient_id=? ORDER BY created_at DESC LIMIT 8""",(patient_id,))]
    events = [dict(r) for r in query_db("""SELECT event_type,source,description,created_at FROM care_events
                                          WHERE patient_id=? ORDER BY created_at DESC LIMIT 12""",(patient_id,))]
    fall = query_db("SELECT * FROM fall_assessments WHERE patient_id=? ORDER BY id DESC LIMIT 1",(patient_id,),one=True)
    return {
        "patient": dict(p),
        "conditions": conditions,
        "medications": meds,
        "vitals": vitals,
        "risks": risks,
        "tasks": tasks,
        "events": events,
        "fall": dict(fall) if fall else None,
    }

def _risk_map(ctx):
    return {r["risk_type"]: r for r in ctx["risks"]}

def _reading(value, spec, unit=""):
    return "not recorded" if value is None else format(value, spec) + unit

def _urgent_message(text):
    t=text.lower()
    return any(re.search(p,t) for p in URGENT_PATTERNS)

def clinician_answer(patient_id, question, history=None):
    """Clinician Copilot answer: record lookup first, grounded OpenAI answer for
    everything else. See services/clinical_copilot.py."""
    from services.clinical_copilot import answer
    return answer(patient_id, question, history=history)

def patient_answer(patient_id, question):
    """
    Patient-facing answer: simpler language, no diagnostic certainty, encourages clinician interaction.
    """
    ctx = patient_context(patient_id)
    if not ctx:
        return {"answer":"I couldn't find your care profile.","safety":"normal"}

    p=ctx["patient"]; v=ctx["vitals"]; rm=_risk_map(ctx)
    q=question.lower().strip()

    if _urgent_message(question):
        return {
          "answer":"Your message may describe something urgent. Please use your emergency contact plan or emergency services now, and do not wait for this chat.",
          "safety":"urgent"
        }

    if any(k in q for k in ["how am i","how am i doing","my health","status","today"]):
        overall=rm.get("overall")
        msg=f"Hi {p['first_name']}. Your latest monitored values include oxygen {_reading(v.get('spo2'), '.1f', '%')}, heart rate {_reading(v.get('heart_rate'), '.0f', ' bpm')} and activity score {_reading(v.get('activity'), '.0f')}."
        if overall and overall["level"] in ("high","critical"):
            msg += " Care.AI has flagged something that should be reviewed by your care team."
        else:
            msg += " There is no new urgent alert in this demo."
        msg += " I can explain your readings, help you send a question to your nurse, or show your care plan."
        return {"answer":msg,"safety":"normal"}

    if "oxygen" in q or "spo2" in q or "breathing" in q:
        return {"answer":f"Your latest oxygen reading in this demo is {_reading(v.get('spo2'), '.1f', '%')}. I can explain what Care.AI has noticed, but I can't diagnose you. If you feel breathless or unwell, contact your care team; if symptoms are severe, use your emergency plan.","safety":"normal"}

    if "fall" in q:
        event=rm.get("fall_event")
        if event and event["score"]>0:
            return {"answer":"Care.AI detected a possible fall event from your connected sensors and sent it into the care workflow. A human care professional should review the event and decide what happens next.","safety":"normal"}
        return {"answer":"Care.AI does not currently show an active fall event. It may still track longer-term fall risk from mobility and other patterns.","safety":"normal"}

    if "medication" in q or "medicine" in q:
        meds=", ".join(m["name"] for m in ctx["medications"]) or "no active medicines recorded"
        return {"answer":f"Your recorded medicines include: {meds}. I can remind you what is recorded, but medication changes should be discussed with your nurse, GP or pharmacist.","safety":"normal"}

    if any(k in q for k in ["nurse","doctor","gp","message","contact"]):
        return {"answer":"You can send this question to your care team from the 'Ask nurse/GP' box below. Care.AI will place it in the clinician inbox for follow-up.","safety":"normal"}

    return {"answer":"I can help explain your monitored readings, falls, medication reminders and care tasks in simple language. For diagnosis or treatment decisions, your nurse or GP should review your situation.","safety":"normal"}

def get_or_create_thread(patient_id, thread_type, user_id=None):
    row=query_db("""SELECT * FROM chat_threads WHERE patient_id=? AND thread_type=?
                    ORDER BY id DESC LIMIT 1""",(patient_id,thread_type),one=True)
    if row:
        return row["id"]
    return execute_db("INSERT INTO chat_threads(patient_id,thread_type,created_by_user_id) VALUES(?,?,?)",
                      (patient_id,thread_type,user_id))

def save_message(thread_id,sender_type,content,sender_id=None,safety_label="normal",source=None):
    execute_db("""INSERT INTO chat_messages(thread_id,sender_type,sender_id,content,safety_label,source)
                  VALUES(?,?,?,?,?,?)""",(thread_id,sender_type,sender_id,content,safety_label,source))
    execute_db("UPDATE chat_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=?",(thread_id,))

def thread_messages(thread_id,limit=30):
    rows=query_db("""SELECT * FROM chat_messages WHERE thread_id=? ORDER BY id DESC LIMIT ?""",(thread_id,limit))
    return list(reversed([dict(r) for r in rows]))
