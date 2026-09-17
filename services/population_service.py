from services.db import query_db

def population_metrics():
    p=query_db("""SELECT COUNT(*) total,
      SUM(CASE WHEN current_status='stable' THEN 1 ELSE 0 END) stable,
      SUM(CASE WHEN current_status='attention' THEN 1 ELSE 0 END) attention,
      SUM(CASE WHEN current_status='high' THEN 1 ELSE 0 END) high,
      SUM(CASE WHEN current_status='critical' THEN 1 ELSE 0 END) critical
      FROM patients WHERE active=1""",one=True)
    alerts=query_db("SELECT COUNT(*) c FROM alerts WHERE status='open'",one=True)["c"]
    tasks=query_db("SELECT COUNT(*) c FROM care_tasks WHERE status='open'",one=True)["c"]
    interventions=query_db("SELECT COUNT(*) c FROM care_events WHERE event_type IN ('agent_recommendation','human_action') AND created_at>=datetime('now','-30 days')",one=True)["c"]
    return {**dict(p),"open_alerts":alerts,"open_tasks":tasks,"interventions_30d":interventions}
