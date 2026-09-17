from services.db import query_db,execute_db

def _latest(pid,kind,default=None):
    r=query_db("SELECT value FROM vitals WHERE patient_id=? AND kind=? ORDER BY measured_at DESC LIMIT 1",(pid,kind),one=True)
    return r["value"] if r else default

def _condition(pid,text):
    return bool(query_db("SELECT 1 FROM conditions WHERE patient_id=? AND active=1 AND lower(display) LIKE ?",(pid,f"%{text.lower()}%"),one=True))

def _missed_meds(pid,days=3):
    r=query_db("""SELECT COUNT(*) c FROM medication_events WHERE patient_id=? AND status='missed'
                  AND scheduled_at >= datetime('now',?)""",(pid,f"-{days} days"),one=True)
    return r["c"] if r else 0

def classify(score):
    if score>=85:return "critical"
    if score>=70:return "high"
    if score>=45:return "medium"
    return "low"

def save(pid,risk,score,explanation,evidence):
    level=classify(score)
    execute_db("""INSERT INTO risk_scores(patient_id,risk_type,score,level,explanation,evidence)
                  VALUES(?,?,?,?,?,?)""",(pid,risk,int(score),level,explanation,evidence))
    return {"risk":risk,"score":int(score),"level":level,"explanation":explanation,"evidence":evidence}

def get_latest_vitals(pid):
    kinds=["spo2","resp_rate","heart_rate","bp_sys","bp_dia","activity","temperature","sleep"]
    return {k:_latest(pid,k) for k in kinds}

def latest_risks(pid):
    # one current card per risk type
    rows=query_db("""
      SELECT r.* FROM risk_scores r
      JOIN (
        SELECT risk_type, MAX(id) max_id
        FROM risk_scores WHERE patient_id=? GROUP BY risk_type
      ) x ON r.id=x.max_id
      WHERE r.patient_id=?
      ORDER BY CASE r.risk_type
        WHEN 'overall' THEN 0 WHEN 'fall_prediction' THEN 1 WHEN 'fall_event' THEN 2
        WHEN 'copd' THEN 3 WHEN 'medication' THEN 4 WHEN 'cardiac' THEN 5
        WHEN 'infection' THEN 6 WHEN 'frailty' THEN 7 ELSE 8 END
    """,(pid,pid))
    return rows

def _fall_prediction(pid,act,sleep):
    s=12; why=[]
    if _condition(pid,"fall"): s+=22; why.append("previous/known fall risk")
    if _condition(pid,"frailty"): s+=18; why.append("frailty")
    if act<55: s+=18; why.append("reduced mobility")
    if act<40: s+=12; why.append("marked activity decline")
    if sleep<50: s+=8; why.append("poor sleep")
    # recent prior fall is a predictor too, but weighted below active event
    recent=query_db("SELECT 1 FROM device_events WHERE patient_id=? AND event_type='fall' AND created_at < datetime('now','-2 minutes') AND created_at>=datetime('now','-30 days')",(pid,),one=True)
    if recent: s+=18; why.append("recent fall history")
    return min(s,100), why

def _fall_detection(pid):
    fa=query_db("""SELECT * FROM fall_assessments WHERE patient_id=? AND assessment_type='detection'
                   ORDER BY id DESC LIMIT 1""",(pid,),one=True)
    if not fa:
        return 0, [], None
    # only treat recent active fall event as detection
    active=query_db("""SELECT 1 FROM fall_assessments WHERE id=? AND created_at>=datetime('now','-30 minutes')""",(fa["id"],),one=True)
    if not active or fa["status"]=="cleared":
        return 0, [], fa
    why=[]
    if (fa["impact_g"] or 0)>=2.0: why.append(f"{fa['impact_g']:.1f}g impact")
    if (fa["orientation_change_deg"] or 0)>=45: why.append(f"{fa['orientation_change_deg']:.0f}° orientation change")
    if fa["camera_posture"]=="on_floor": why.append("person detected near floor")
    if (fa["immobility_seconds"] or 0)>=60: why.append(f"no movement for {int(fa['immobility_seconds']/60)} min")
    return fa["score"], why, fa

def compute_all_risks(pid):
    v=get_latest_vitals(pid)
    spo2=v["spo2"] or 97; rr=v["resp_rate"] or 16; hr=v["heart_rate"] or 70
    act=v["activity"] or 75; temp=v["temperature"] or 36.6; bps=v["bp_sys"] or 130
    sleep=v["sleep"] or 75
    missed=_missed_meds(pid)
    results=[]

    # COPD / respiratory
    s=8; why=[]
    if _condition(pid,"copd"): s+=22; why.append("known COPD")
    if spo2<93: s+=32; why.append(f"SpO₂ {spo2:.1f}%")
    elif spo2<95: s+=14; why.append(f"SpO₂ trending low ({spo2:.1f}%)")
    if rr>21: s+=22; why.append(f"respiratory rate {rr:.0f}/min")
    if hr>92: s+=10; why.append(f"heart rate {hr:.0f} bpm")
    if act<50: s+=14; why.append("activity reduced")
    results.append(save(pid,"copd",min(s,100),"Respiratory deterioration pattern detected" if s>=45 else "Respiratory pattern stable",", ".join(why) or "baseline signals stable"))

    # fall risk prediction
    fs,fwhy=_fall_prediction(pid,act,sleep)
    results.append(save(pid,"fall_prediction",fs,"Increasing fall risk" if fs>=45 else "Fall risk currently controlled",", ".join(fwhy) or "mobility and sleep near baseline"))

    # active fall event detection confidence
    ds,dwhy,fa=_fall_detection(pid)
    if ds>0:
        results.append(save(pid,"fall_event",ds,"Probable fall detected" if ds>=70 else "Possible fall event",", ".join(dwhy) or "insufficient evidence"))
    else:
        results.append(save(pid,"fall_event",0,"No active fall event","no recent sensor-fusion event"))

    # medication
    s=min(100,10+missed*28); why=[f"{missed} missed dose(s) in 3 days"] if missed else []
    results.append(save(pid,"medication",s,"Medication adherence risk" if s>=45 else "Medication adherence acceptable",", ".join(why) or "no recent missed doses"))

    # cardiac
    s=12; why=[]
    if _condition(pid,"heart failure"): s+=28; why.append("heart failure history")
    if _condition(pid,"atrial"): s+=18; why.append("atrial fibrillation history")
    if hr>100: s+=22; why.append(f"heart rate {hr:.0f}")
    if bps>160: s+=20; why.append(f"systolic BP {bps:.0f}")
    if act<45: s+=10; why.append("reduced activity")
    results.append(save(pid,"cardiac",min(s,100),"Cardiovascular deterioration signal" if s>=45 else "Cardiac signals stable",", ".join(why) or "baseline stable"))

    # infection
    s=8; why=[]
    if temp>37.8:s+=38;why.append(f"temperature {temp:.1f}°C")
    if rr>22:s+=20;why.append("respiratory rate elevated")
    if hr>96:s+=14;why.append("heart rate elevated")
    if act<45:s+=12;why.append("activity reduced")
    results.append(save(pid,"infection",min(s,100),"Possible infection / deterioration pattern" if s>=45 else "No strong infection signal",", ".join(why) or "no strong indicators"))

    # frailty
    s=10; why=[]
    if _condition(pid,"frailty"):s+=30;why.append("known frailty")
    if act<50:s+=24;why.append("low activity")
    if sleep<45:s+=12;why.append("poor sleep")
    results.append(save(pid,"frailty",min(s,100),"Frailty / functional decline risk" if s>=45 else "Functional status stable",", ".join(why) or "stable function"))

    # Overall: active severe event gets event priority; otherwise highest chronic deterioration risk.
    event = next(r for r in results if r["risk"]=="fall_event")
    chronic=[r for r in results if r["risk"] not in ("fall_event","fall_prediction")]
    if event["score"]>=70:
        dominant=event
    else:
        dominant=max(chronic,key=lambda x:x["score"])
    overall=dominant["score"]
    o=save(pid,"overall",overall,f"Dominant risk: {dominant['risk'].replace('_',' ').title()} — {dominant['explanation']}",dominant["evidence"])
    status="stable"
    if o["level"]=="medium": status="attention"
    elif o["level"] in ("high","critical"): status=o["level"]
    execute_db("UPDATE patients SET current_status=? WHERE id=?",(status,pid))
    return results+[o]
