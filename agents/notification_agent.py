from services.db import execute_db
def create_notification(pid,audience,channel,message,source="agent"):
    return execute_db("""INSERT INTO notifications(patient_id,audience,channel,message,status,source)
                         VALUES(?,?,?,?,?,?)""",(pid,audience,channel,message,"queued",source))
