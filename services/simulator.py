from services.db import query_db,execute_db
from datetime import datetime
import random

BASELINE={"spo2":96.0,"resp_rate":17.0,"heart_rate":74.0,"bp_sys":134.0,"bp_dia":82.0,"activity":72.0,"temperature":36.6,"sleep":72.0}
UNITS={"spo2":"%","resp_rate":"/min","heart_rate":"bpm","bp_sys":"mmHg","bp_dia":"mmHg","activity":"score","temperature":"°C","sleep":"score"}

def insert(pid,kind,val,unit=None,source="simulator"):
    execute_db("INSERT INTO vitals(patient_id,kind,value,unit,source,measured_at) VALUES(?,?,?,?,?,?)",
               (pid,kind,round(float(val),2),unit or UNITS[kind],source,datetime.now().isoformat(timespec="seconds")))

def reset_scenario_state(pid):
    # Reset active demo physiology so scenarios don't contaminate one another.
    for k,v in BASELINE.items():
        insert(pid,k,v,UNITS[k],"scenario_reset")
    execute_db("UPDATE alerts SET status='closed' WHERE patient_id=? AND source IS NULL",(pid,)) if False else None
    # clear active fall detection by adding an explicit cleared assessment
    execute_db("""INSERT INTO fall_assessments(patient_id,assessment_type,score,level,status,evidence)
                  VALUES(?,?,?,?,?,?)""",(pid,"detection",0,"low","cleared","scenario reset"))
    execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
               (pid,"scenario_reset","simulator","Scenario state reset to healthy synthetic baseline."))

def last(pid,kind,default):
    r=query_db("SELECT value FROM vitals WHERE patient_id=? AND kind=? ORDER BY measured_at DESC LIMIT 1",(pid,kind),one=True)
    return r["value"] if r else default

def tick_all_patients():
    for p in query_db("SELECT id FROM patients WHERE active=1"):
        pid=p["id"]
        for kind,default,delta in [
            ("spo2",96,.25),("resp_rate",17,.4),("heart_rate",74,2.5),
            ("bp_sys",132,3.5),("bp_dia",78,2.5),("activity",72,3.5),
            ("temperature",36.6,.08),("sleep",72,3.0)
        ]:
            insert(pid,kind,last(pid,kind,default)+random.uniform(-delta,delta),UNITS[kind],"monitoring_sim")

def run_named_scenario(pid,scenario):
    reset_scenario_state(pid)
    if scenario=="copd":
        for k,v in [("spo2",91),("resp_rate",24),("heart_rate",98),("activity",34)]:
            insert(pid,k,v,UNITS[k],"copd_scenario")
        execute_db("""INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location)
                      VALUES(?,?,?,?,?,?)""",(pid,"microphone","cough_increase","frequent","medium","Living room"))
        desc="COPD scenario: progressive respiratory deterioration injected after baseline reset."
    elif scenario=="fall":
        # A pure fall scenario keeps oxygenation near baseline and changes fall-specific signals.
        for k,v in [("spo2",95),("resp_rate",20),("heart_rate",94),("bp_sys",142),("bp_dia",88),("activity",18)]:
            insert(pid,k,v,UNITS[k],"fall_scenario")
        execute_db("""INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location)
                      VALUES(?,?,?,?,?,?)""",(pid,"wearable_imu","impact","2.8g","high","Bathroom"))
        execute_db("""INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location)
                      VALUES(?,?,?,?,?,?)""",(pid,"wearable_imu","orientation_change","76 degrees","high","Bathroom"))
        execute_db("""INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location)
                      VALUES(?,?,?,?,?,?)""",(pid,"edge_camera","posture","on_floor","high","Bathroom"))
        execute_db("""INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location)
                      VALUES(?,?,?,?,?,?)""",(pid,"motion_sensor","no_movement","4 minutes","high","Bathroom"))
        execute_db("""INSERT INTO fall_assessments(patient_id,assessment_type,score,level,status,impact_g,orientation_change_deg,
                      camera_posture,immobility_seconds,location,evidence)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                   (pid,"detection",94,"critical","active",2.8,76,"on_floor",240,"Bathroom",
                    "IMU impact + rapid orientation change + person-on-floor + 4 min immobility"))
        desc="Fall scenario: 2.8g impact, 76° orientation shift, person-on-floor, 4 min immobility."
    elif scenario=="medication":
        med=query_db("SELECT id FROM medications WHERE patient_id=? AND active=1 LIMIT 1",(pid,),one=True)
        if med:
            now=datetime.now().isoformat(timespec="seconds")
            for _ in range(3):
                execute_db("INSERT INTO medication_events(patient_id,medication_id,scheduled_at,status) VALUES(?,?,?,?)",
                           (pid,med["id"],now,"missed"))
        desc="Medication scenario: repeated synthetic missed-dose pattern."
    elif scenario=="recovery":
        desc="Recovery/reset scenario: signals restored to synthetic baseline."
    else:
        return {"ok":False}
    execute_db("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",(pid,"simulation",scenario,desc))
    return {"ok":True,"scenario":scenario}
