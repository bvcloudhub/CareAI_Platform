from datetime import datetime,timedelta
import random

PATIENTS=[
("NL-CR-001","Jan","de Vries","1948-02-11","M","Rotterdam","home","Dr. Eva van Dijk",[("J44","COPD"),("I10","Hypertension")],[("Tiotropium","18 mcg","daily"),("Amlodipine","5 mg","daily")]),
("NL-CR-002","Maria","Jansen","1943-07-19","F","The Hague","home care","Dr. Noor Bakker",[("I50","Heart failure"),("R29.6","Fall risk")],[("Furosemide","20 mg","daily"),("Bisoprolol","2.5 mg","daily")]),
("NL-CR-003","Pieter","Bakker","1952-11-02","M","Eindhoven","home","Dr. Lars Smit",[("E11","Type 2 diabetes"),("I10","Hypertension")],[("Metformin","500 mg","twice daily"),("Lisinopril","10 mg","daily")]),
("NL-CR-004","Anna","Smit","1945-05-26","F","Utrecht","assisted living","Dr. Sara Visser",[("R54","Frailty"),("I10","Hypertension")],[("Amlodipine","5 mg","daily")]),
("NL-CR-005","Karin","Visser","1950-09-08","F","Amsterdam","home","Dr. Eva van Dijk",[("J44","COPD")],[("Budesonide/Formoterol","160/4.5","twice daily")]),
("NL-CR-006","Willem","Meijer","1947-01-14","M","Leiden","home care","Dr. Noor Bakker",[("I48","Atrial fibrillation")],[("Apixaban","5 mg","twice daily")]),
("NL-CR-007","Elise","de Boer","1942-12-30","F","Groningen","assisted living","Dr. Lars Smit",[("M81","Osteoporosis"),("R29.6","Fall risk")],[("Calcium/Vit D","1 tab","daily")]),
("NL-CR-008","Henk","Mulder","1951-06-04","M","Tilburg","home","Dr. Sara Visser",[("I10","Hypertension")],[("Losartan","50 mg","daily")]),
("NL-CR-009","Ria","Vos","1941-03-21","F","Delft","home care","Dr. Noor Bakker",[("G30","Cognitive impairment"),("R54","Frailty")],[("Donepezil","5 mg","daily")]),
("NL-CR-010","Joost","Kuiper","1949-08-29","M","Breda","home","Dr. Lars Smit",[("E11","Type 2 diabetes")],[("Metformin","500 mg","daily")]),
("NL-CR-011","Mieke","Bos","1946-10-13","F","Amersfoort","assisted living","Dr. Sara Visser",[("I50","Heart failure")],[("Furosemide","20 mg","daily")]),
("NL-CR-012","Theo","Dekker","1953-04-07","M","Zwolle","home","Dr. Eva van Dijk",[("J44","COPD"),("E11","Type 2 diabetes")],[("Tiotropium","18 mcg","daily"),("Metformin","500 mg","daily")]),
]

def add_vital(conn,pid,kind,value,unit,when,source="simulator"):
    conn.execute("INSERT INTO vitals(patient_id,kind,value,unit,source,measured_at) VALUES(?,?,?,?,?,?)",
                 (pid,kind,round(float(value),2),unit,source,when.isoformat(timespec="seconds")))

def seed_database(conn):
    now=datetime.now()
    random.seed(24)
    for idx,rec in enumerate(PATIENTS,1):
        ext,fn,ln,dob,sex,city,setting,gp,conditions,meds=rec
        status="stable"
        cur=conn.execute("""INSERT INTO patients(
            external_ref,first_name,last_name,birth_date,sex,city,living_setting,gp_name,
            emergency_contact_name,emergency_contact_relation,consent_monitoring,consent_updated_at,
            lawful_basis,purpose_code,data_residency,retention_until,current_status
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ext,fn,ln,dob,sex,city,setting,gp,"Demo Family Contact","family",1,now.isoformat(timespec="seconds"),
         "demo_synthetic","remote_care_demo","EU",(now+timedelta(days=365)).date().isoformat(),status))
        pid=cur.lastrowid
        for code,display in conditions:
            conn.execute("INSERT INTO conditions(patient_id,code,display,onset_date) VALUES(?,?,?,?)",(pid,code,display,"2020-01-01"))
        med_ids=[]
        for name,dose,schedule in meds:
            m=conn.execute("INSERT INTO medications(patient_id,name,dose,schedule) VALUES(?,?,?,?)",(pid,name,dose,schedule)).lastrowid
            med_ids.append(m)

        # 4-hour observations across 7 days
        for h in range(168,0,-4):
            t=now-timedelta(hours=h)
            spo2=96.5; rr=16.5; hr=72; sys=132; dia=78; act=78; temp=36.6; sleep=78
            # Built-in deterioration examples for investor demo
            if ext in ("NL-CR-001","NL-CR-005","NL-CR-012") and h<48:
                prog=(48-h)/48
                spo2-=prog*4.8; rr+=prog*5.5; hr+=prog*16; act-=prog*32; sleep-=prog*18
            if ext in ("NL-CR-002","NL-CR-011") and h<32:
                prog=(32-h)/32
                hr+=prog*12; sys+=prog*15; act-=prog*20
            if ext in ("NL-CR-007","NL-CR-009") and h<24:
                act-=15
            noise=lambda s: random.uniform(-s,s)
            add_vital(conn,pid,"spo2",spo2+noise(.5),"%",t)
            add_vital(conn,pid,"resp_rate",rr+noise(.8),"/min",t)
            add_vital(conn,pid,"heart_rate",hr+noise(3),"bpm",t)
            add_vital(conn,pid,"bp_sys",sys+noise(6),"mmHg",t)
            add_vital(conn,pid,"bp_dia",dia+noise(4),"mmHg",t)
            add_vital(conn,pid,"activity",max(10,act+noise(5)),"score",t)
            add_vital(conn,pid,"temperature",temp+noise(.15),"°C",t)
            add_vital(conn,pid,"sleep",max(20,sleep+noise(7)),"score",t)

        for mid in med_ids:
            for d in range(6,-1,-1):
                when=now-timedelta(days=d,hours=random.choice([0,1,2]))
                status="taken"
                if idx in (1,4,9) and d in (0,2):
                    status="missed"
                conn.execute("INSERT INTO medication_events(patient_id,medication_id,scheduled_at,status) VALUES(?,?,?,?)",
                             (pid,mid,when.isoformat(timespec="seconds"),status))

        # home sensor events
        conn.execute("INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location) VALUES(?,?,?,?,?,?)",
                     (pid,"motion_sensor","room_activity","normal","info","Living room"))
        conn.execute("INSERT INTO device_events(patient_id,device_type,event_type,value,severity,location) VALUES(?,?,?,?,?,?)",
                     (pid,"bed_sensor","sleep_presence","present","info","Bedroom"))
        conn.execute("INSERT INTO care_events(patient_id,event_type,source,description) VALUES(?,?,?,?)",
                     (pid,"enrolment","platform","Synthetic person enrolled in Care.AI EU demo"))
