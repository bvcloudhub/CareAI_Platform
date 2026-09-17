from services.db import query_db

def inspect(pid):
    rows=query_db("SELECT kind,value,unit,measured_at FROM vitals WHERE patient_id=? ORDER BY measured_at DESC LIMIT 40",(pid,))
    latest={}
    for r in rows:
        latest.setdefault(r["kind"],dict(r))
    events=query_db("SELECT * FROM device_events WHERE patient_id=? ORDER BY created_at DESC LIMIT 5",(pid,))
    missed=query_db("""SELECT COUNT(*) c FROM medication_events WHERE patient_id=? AND status='missed'
                       AND scheduled_at>=datetime('now','-3 days')""",(pid,),one=True)["c"]
    return {"latest":latest,"events":[dict(e) for e in events],"missed_medications":missed}
