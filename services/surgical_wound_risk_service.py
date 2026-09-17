"""
Surgical-wound infection-risk image screening for Care.AI Wound AI.

Ported from WoundAI services/surgical_ensemble_service.py and
services/surgical_ensemble_router_service.py. The two model classes are copied
from the training scripts (train_surgical_v9_biomedclip_cv.py and
train_surgical_v8a_multitask_fullimage_cv.py) rather than imported: importing
those scripts drags in scikit-learn, PyYAML and the whole training loop.

The ensemble, as the source deploys it:
  0.37 x V9  — BiomedCLIP ViT-B/16; 5 folds stored as small deltas on the
               public base weights
  0.63 x V8A — DenseNet121 multitask; 5 full folds
  score >= 0.26 -> elevated infection-risk image category. Scores between
  0.20 and 0.32, or components on opposite sides of 0.26, are flagged for
  review.

What the score is not: a calibrated probability of surgical-site infection.
It was evaluated only on development cross-fitted folds (sensitivity 0.80,
specificity 0.75, n=549). The source never ran it on its held-out test split,
and the manifest says so. As in the source, no Grad-CAM is produced — a map
from one component would not explain the ensemble's decision.

The V9 base (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) is
fetched from HuggingFace by open_clip on first use and cached locally.

One deliberate difference: the source applies the V9 fold deltas with
load_state_dict(strict=False) and ignores the result. strict=False silently
skips any key it does not recognise, so under an open_clip build that renames
a parameter the base weights would quietly stay in place. Deltas are validated
against the live model here, and a mismatch fails loudly instead.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from services.checkpoint_loading import load_checkpoint

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("CAREAI_MODEL_DIR", BASE_DIR / "models"))
MANIFEST_FILE = "surgical_wound_risk_ensemble.json"

POSITIVE_CLASS = "surgical_elevated_risk"
NEGATIVE_CLASS = "surgical_low_risk"
DISPLAY_NAMES = {
    POSITIVE_CLASS: "Elevated infection-risk image category",
    NEGATIVE_CLASS: "Low infection-risk image category",
}

# Output order of the V9 head (train_surgical_v9_biomedclip_cv.py CLASSES).
V9_CLASSES = ("surgical_elevated_risk", "surgical_low_risk")
V9_POSITIVE_INDEX = V9_CLASSES.index(POSITIVE_CLASS)

# From train_surgical_v8a_multitask_fullimage_cv.py. Every head is rebuilt at
# its trained size so the fold state_dicts load strictly, though only
# infection_risk is read: the auxiliary heads (healing status, closure method,
# exudate, erythema, oedema) were training signals, unvalidated as outputs, and
# the source never surfaces them either.
V8A_TASK_LABELS = {
    "healing_status": ["Healed", "Not Healed"],
    "closure_method": ["Invisible", "Sutures", "Staples", "Adhesives", "Uncertain"],
    "exudate_type": ["Non-existent", "Serous", "Sanguineous", "Purulent", "Seropurulent", "Uncertain"],
    "erythema": ["Non-existent", "Existent", "Uncertain"],
    "edema": ["Non-existent", "Existent", "Uncertain"],
    "infection_risk": ["surgical_elevated_risk", "surgical_low_risk"],
}
V8A_POSITIVE_INDEX = V8A_TASK_LABELS["infection_risk"].index(POSITIVE_CLASS)

BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_LOAD_LOCK = threading.Lock()
# Fold weights are swapped in and out of shared modules; two requests must
# never interleave that.
_PREDICT_LOCK = threading.Lock()
_LOADED: Optional[Dict[str, object]] = None
_MODEL_CLASSES = None


def _read_manifest() -> Optional[Dict[str, object]]:
    path = MODEL_DIR / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _biomedclip_cached() -> Optional[bool]:
    try:
        from huggingface_hub import constants
    except Exception:
        return None
    snapshots = Path(constants.HF_HUB_CACHE) / ("models--" + BIOMEDCLIP_REPO.replace("/", "--")) / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def runtime_status() -> Dict[str, object]:
    status: Dict[str, object] = {"available": False, "reason": None, "versions": {}, "base_cached": None}
    try:
        import open_clip
        import torch
        status["versions"] = {"torch": torch.__version__, "open_clip": open_clip.__version__}
    except Exception as exc:
        status["reason"] = f"open_clip / PyTorch unavailable ({exc.__class__.__name__})."
        return status
    try:
        import transformers
        status["versions"]["transformers"] = transformers.__version__
    except Exception as exc:
        status["reason"] = f"transformers unavailable ({exc.__class__.__name__}); BiomedCLIP's text tower needs it."
        return status

    manifest = _read_manifest()
    if manifest is None:
        status["reason"] = f"Ensemble manifest not found: {MANIFEST_FILE}"
        return status
    missing = [manifest[k] for k in ("v9_pack", "v8a_pack") if not (MODEL_DIR / manifest[k]).exists()]
    if missing:
        status["reason"] = "Ensemble weight file(s) not found: " + ", ".join(missing)
        return status

    status["base_cached"] = _biomedclip_cached()
    status["available"] = True
    return status


def _model_classes():
    """The two architectures, copied from the training scripts. Attribute names
    must stay identical: they are the prefixes of every stored weight key."""
    global _MODEL_CLASSES
    if _MODEL_CLASSES is not None:
        return _MODEL_CLASSES

    import torch
    import torch.nn as nn
    from torchvision import models

    class BioMedCLIPBinaryClassifier(nn.Module):
        def __init__(self, clip_model, embedding_dim, dropout=0.20):
            super().__init__()
            self.clip_model = clip_model
            self.dropout = nn.Dropout(float(dropout))
            self.classifier = nn.Linear(int(embedding_dim), len(V9_CLASSES))

        def forward(self, images):
            features = self.clip_model.encode_image(images, normalize=False)
            if features.ndim > 2:
                features = features.flatten(1)
            return self.classifier(self.dropout(features))

    class DenseNet121MultiTask(nn.Module):
        def __init__(self, dropout=0.20):
            super().__init__()
            base = models.densenet121(weights=None)
            feature_dim = base.classifier.in_features
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(float(dropout))
            self.heads = nn.ModuleDict({
                task: nn.Linear(feature_dim, len(classes)) for task, classes in V8A_TASK_LABELS.items()
            })

        def forward(self, x):
            features = torch.relu(self.features(x))
            features = self.dropout(self.pool(features).flatten(1))
            return {task: head(features) for task, head in self.heads.items()}

    _MODEL_CLASSES = (BioMedCLIPBinaryClassifier, DenseNet121MultiTask)
    return _MODEL_CLASSES


def _load() -> Dict[str, object]:
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    with _LOAD_LOCK:
        if _LOADED is not None:
            return _LOADED

        import open_clip
        import torch
        from torchvision import transforms

        manifest = _read_manifest()
        if manifest is None:
            raise FileNotFoundError(f"Ensemble manifest not found: {MODEL_DIR / MANIFEST_FILE}")
        v9_path, v8a_path = MODEL_DIR / manifest["v9_pack"], MODEL_DIR / manifest["v8a_pack"]
        for path in (v9_path, v8a_path):
            if not path.exists():
                raise FileNotFoundError(f"Ensemble weights not found: {path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        V9Model, V8AModel = _model_classes()

        v9_pack, v9_mode = load_checkpoint(v9_path)
        clip_model, _, preprocess = open_clip.create_model_and_transforms(v9_pack["base_model_id"])
        v9 = V9Model(clip_model, int(v9_pack["embedding_dim"]), float(v9_pack["dropout"])).to(device).eval()

        live = v9.state_dict()
        mutable = set()
        for fold in v9_pack["folds"]:
            delta = fold["state_delta"]
            unknown = [k for k in delta if k not in live]
            misshapen = [k for k in delta if k in live and tuple(live[k].shape) != tuple(delta[k].shape)]
            if unknown or misshapen:
                example = (unknown or misshapen)[:2]
                raise RuntimeError(
                    f"V9 fold {fold['fold']} does not match this open_clip build: "
                    f"{len(unknown)} unknown and {len(misshapen)} mis-shaped keys (e.g. {example}).")
            mutable.update(delta.keys())
        # Pristine copies of everything a fold overwrites, restored before each
        # fold so no fold inherits another's weights.
        v9_base = {key: live[key].detach().cpu().clone() for key in mutable}

        v8a_pack, v8a_mode = load_checkpoint(v8a_path)
        v8a = V8AModel(dropout=float(v8a_pack["dropout"])).to(device).eval()
        v8a_size = int(v8a_pack["image_size"])

        _LOADED = {
            "device": device,
            "manifest": manifest,
            "v9": v9,
            "v9_preprocess": preprocess,
            "v9_folds": v9_pack["folds"],
            "v9_base": v9_base,
            "v9_load_mode": v9_mode,
            "v8a": v8a,
            "v8a_folds": v8a_pack["folds"],
            "v8a_transform": transforms.Compose([     # src.transforms.build_eval_transform
                transforms.Resize((v8a_size, v8a_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            "v8a_size": v8a_size,
            "v8a_load_mode": v8a_mode,
        }
    return _LOADED


def _fold_scores(image) -> tuple:
    import torch

    state = _load()
    device = state["device"]
    v9, v8a = state["v9"], state["v8a"]

    with _PREDICT_LOCK, torch.inference_mode():
        v9_input = state["v9_preprocess"](image).unsqueeze(0).to(device)
        v9_scores: List[float] = []
        for fold in state["v9_folds"]:
            v9.load_state_dict(state["v9_base"], strict=False)
            v9.load_state_dict(fold["state_delta"], strict=False)   # keys validated in _load()
            v9.eval()
            logits = v9(v9_input)
            v9_scores.append(float(torch.softmax(logits.float(), dim=1)[0, V9_POSITIVE_INDEX].item()))
        v9.load_state_dict(state["v9_base"], strict=False)

        v8a_input = state["v8a_transform"](image).unsqueeze(0).to(device)
        v8a_scores: List[float] = []
        for fold in state["v8a_folds"]:
            v8a.load_state_dict(fold["state_dict"], strict=True)
            v8a.eval()
            logits = v8a(v8a_input)["infection_risk"]
            v8a_scores.append(float(torch.softmax(logits.float(), dim=1)[0, V8A_POSITIVE_INDEX].item()))

    return v9_scores, v8a_scores


def screen(image_path: Path, trigger: str) -> Dict[str, object]:
    """Score one surgical-wound photo. `trigger` records why it ran:
    'clinician' (known surgical wound) or 'classifier' (automatic follow-up)."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")

    manifest = _load()["manifest"]
    v9_scores, v8a_scores = _fold_scores(image)
    v9_score, v8a_score = float(np.mean(v9_scores)), float(np.mean(v8a_scores))

    w9, w8a = float(manifest["v9_weight"]), float(manifest["v8a_weight"])
    threshold = float(manifest["decision_threshold"])
    band = manifest.get("engineering_uncertainty_band") or {"lower": 0.20, "upper": 0.32}
    lower, upper = float(band["lower"]), float(band["upper"])

    score = w9 * v9_score + w8a * v8a_score
    category = POSITIVE_CLASS if score >= threshold else NEGATIVE_CLASS
    in_band = lower <= score <= upper
    disagreement = (v9_score >= threshold) != (v8a_score >= threshold)

    reasons = []
    if in_band:
        reasons.append("the ensemble score is inside the engineering review band")
    if disagreement:
        reasons.append("the BiomedCLIP and DenseNet components fall on opposite sides of the threshold")

    return {
        "trigger": trigger,
        "category": category,
        "category_display": DISPLAY_NAMES[category],
        "elevated": category == POSITIVE_CLASS,
        "ensemble_score": score,
        "ensemble_score_pct": round(score * 100, 1),
        "decision_threshold": threshold,
        "uncertainty_band": {"lower": lower, "upper": upper},
        "v9_score": v9_score,
        "v8a_score": v8a_score,
        "v9_fold_scores": v9_scores,
        "v8a_fold_scores": v8a_scores,
        "weights": {"v9": w9, "v8a": w8a},
        "engineering_uncertainty": in_band,
        "component_disagreement": disagreement,
        "clinical_review_recommended": category == POSITIVE_CLASS or in_band or disagreement,
        "review_reasons": reasons,
        "model_version": manifest["deployment_version"],
        "deployment_status": manifest["status"],
        "evaluation": manifest.get("evaluation") or {},
        "score_interpretation": "AI model score; not a calibrated clinical probability.",
        "clinical_note": manifest["clinical_note"],
    }


def technical_details() -> Dict[str, str]:
    state = _load()
    manifest = state["manifest"]
    band = manifest.get("engineering_uncertainty_band") or {"lower": 0.20, "upper": 0.32}
    evaluation = manifest.get("evaluation") or {}
    details = {
        "Infection-risk ensemble": (
            f"{manifest['v9_weight']:.2f} × V9 BiomedCLIP (5 folds) + "
            f"{manifest['v8a_weight']:.2f} × V8A DenseNet121 (5 folds)"),
        "Risk operating point": (
            f"score ≥ {manifest['decision_threshold']:.2f} → elevated · "
            f"review band {float(band['lower']):.2f}–{float(band['upper']):.2f}"),
        "Risk model version": str(manifest["deployment_version"]),
        "Risk weights": f"{manifest['v9_pack']}, {manifest['v8a_pack']} + {BIOMEDCLIP_REPO}",
        "Risk checkpoint loading": f"V9 {state['v9_load_mode']}, V8A {state['v8a_load_mode']}",
    }
    if evaluation:
        details["Risk evaluation"] = (
            f"development out-of-fold only (n={evaluation.get('samples')}): "
            f"sensitivity {evaluation.get('sensitivity', 0) * 100:.0f}%, "
            f"specificity {evaluation.get('specificity', 0) * 100:.0f}%, "
            f"ROC-AUC {evaluation.get('roc_auc', 0):.2f} — never run on a held-out test set")
    return details
