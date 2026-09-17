def recommend(risks):
    dominant=max([r for r in risks if r["risk"]!="overall"],key=lambda x:x["score"])
    level=dominant["level"]; risk=dominant["risk"]
    mapping={
      "copd":("nurse_review","Repeat SpO₂ and symptom check; consider teleconsultation if persistent."),
      "fall_event":("urgent_welfare_check","Contact patient/caregiver and verify mobility/injury status."),
      "medication":("medication_check","Confirm adherence, barriers and medication supply."),
      "cardiac":("gp_review","Clinical review of symptoms, pulse and blood pressure."),
      "infection":("nurse_review","Check temperature, symptoms and hydration; escalate per local protocol."),
      "fall_prediction":("falls_prevention_review","Review gait, mobility, medication and environmental fall hazards."),
      "frailty":("home_care_review","Review mobility, nutrition, sleep and home support.")
    }
    task,rationale=mapping.get(risk,("nurse_review","Human clinical review recommended."))
    if level=="critical":
        task="urgent_clinical_review"
        rationale="Urgent human clinical review. " + rationale
    return {"risk":risk,"level":level,"task_type":task,"rationale":rationale}
