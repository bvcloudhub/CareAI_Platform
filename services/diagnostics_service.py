import hashlib
import uuid
from datetime import datetime

# Only Skin AI is still a rule-based demo. Lung and Wound AI run real models
# (lung_imaging_service, wound_imaging_service).
MODULES = {
  "skin": {
    "label":"Skin AI",
    "classes":[
      ("Low-risk benign pattern",82,"Routine monitoring; clinician review if changing or symptomatic."),
      ("Atypical skin lesion pattern",76,"Dermatology review recommended."),
      ("Inflammatory skin pattern",79,"Clinical review if persistent, spreading or symptomatic.")
    ]
  }
}

def analyse_demo(module, filename, content_bytes):
    cfg=MODULES.get(module)
    if not cfg:
        return None
    digest=hashlib.sha256((filename or "").encode()+content_bytes[:4096]).hexdigest()
    idx=int(digest[:4],16)%len(cfg["classes"])
    label,base_conf,recommendation=cfg["classes"][idx]
    confidence=min(94, base_conf + int(digest[4:6],16)%9)
    return {
      "run_id":uuid.uuid4().hex[:10],
      "kind":"demo",
      "module":module,
      "module_label":cfg["label"],
      "modality":"Demo image",
      "created_at":datetime.now().isoformat(sep=" ",timespec="seconds"),
      "source_filename":filename,
      "result":label,
      "confidence":confidence,
      "recommendation":recommendation,
      "recommendations":[recommendation],
      "model_version":"Care.AI Demo Vision v0.1",
      "disclaimer":"Synthetic demonstration output only; not validated for diagnosis or treatment."
    }
