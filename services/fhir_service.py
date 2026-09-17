from services.db import query_db
from datetime import datetime

def patient_bundle(pid):
    p=query_db("SELECT * FROM patients WHERE id=?",(pid,),one=True)
    if not p:return {"resourceType":"Bundle","type":"collection","entry":[]}
    entry=[{"resource":{
        "resourceType":"Patient","id":p["external_ref"],
        "name":[{"family":p["last_name"],"given":[p["first_name"]]}],
        "birthDate":p["birth_date"],"gender":"male" if p["sex"]=="M" else "female",
        "address":[{"city":p["city"],"country":p["country"]}]
    }}]
    for c in query_db("SELECT * FROM conditions WHERE patient_id=? AND active=1",(pid,)):
        entry.append({"resource":{
            "resourceType":"Condition","clinicalStatus":{"text":"active"},
            "code":{"text":c["display"]},"subject":{"reference":f"Patient/{p['external_ref']}"}
        }})
    for v in query_db("SELECT * FROM vitals WHERE patient_id=? ORDER BY measured_at DESC LIMIT 16",(pid,)):
        entry.append({"resource":{
            "resourceType":"Observation","status":"final","code":{"text":v["kind"]},
            "subject":{"reference":f"Patient/{p['external_ref']}"},
            "effectiveDateTime":v["measured_at"],
            "valueQuantity":{"value":v["value"],"unit":v["unit"]}
        }})
    return {"resourceType":"Bundle","type":"collection","timestamp":datetime.utcnow().isoformat()+"Z",
            "meta":{"tag":[{"code":"SYNTHETIC-DEMO"}]},"entry":entry}
