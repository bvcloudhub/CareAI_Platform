from services.risk_engine import compute_all_risks
def assess(pid):
    return compute_all_risks(pid)
