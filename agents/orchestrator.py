from services.db import query_db,execute_db
from agents.monitoring_agent import inspect
from agents.risk_agent import assess
from agents.care_agent import recommend
from agents.notification_agent import create_notification

def orchestrate_patient(pid,actor_user_id=None):
    monitoring=inspect(pid)
    risks=assess(pid)
    recommendation=recommend(risks)
    level=recommendation["level"]

    existing=query_db("""SELECT 1 FROM care_tasks WHERE patient_id=? AND status='open'
                         AND created_at>=datetime('now','-6 hours')""",(pid,),one=True)
    actions=[]

    if level in ("high","critical") and not existing:
        priority="critical" if level=="critical" else "high"
        execute_db("""INSERT INTO care_tasks(patient_id,task_type,priority,status,assigned_role,rationale,created_by)
                      VALUES(?,?,?,?,?,?,?)""",
                   (pid,recommendation["task_type"],priority,"open","nurse",recommendation["rationale"],actor_user_id))
        execute_db("""INSERT INTO alerts(patient_id,alert_type,severity,status,message)
                      VALUES(?,?,?,?,?)""",
                   (pid,recommendation["risk"],priority,"open",recommendation["rationale"]))
        create_notification(pid,"care_team","app",recommendation["rationale"],"care_agent")
        actions += ["care_task_created","care_team_notified"]
    elif level=="medium":
        create_notification(pid,"care_team","app","Patient trend requires review during next care round.","care_agent")
        actions.append("monitoring_notification")
    else:
        actions.append("continue_monitoring")

    execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
               (pid,"agent_recommendation","agentic_workflow",
                f"{recommendation['risk'].title()} / {level}: {recommendation['rationale']}"))
    return {"monitoring":monitoring,"risks":risks,"recommendation":recommendation,"actions":actions}

def summarize_patient(pid):
    p=query_db("SELECT first_name,last_name,current_status FROM patients WHERE id=?",(pid,),one=True)
    r=query_db("""SELECT * FROM risk_scores WHERE patient_id=? AND risk_type='overall'
                  ORDER BY created_at DESC LIMIT 1""",(pid,),one=True)
    if not r:
        return "No risk assessment yet."
    return f"{p['first_name']} is currently {p['current_status']}. {r['explanation']}. Evidence: {r['evidence']}."
