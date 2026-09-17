"""
PyTorch inference for the Care.AI Wound AI module.

Ported from WoundAI — consolidating services/wound_service.py,
services/wound_gate.py, services/image_quality.py and
wound_model/src/models.py, and orchestrating the boundary and surgical
infection-risk models the way the source's analyze() route does.

Two routes, as in the source:

  Unknown wound type (source "All Types")
    1. Image quality — only severe blur, darkness or overexposure blocks;
                       milder problems continue with a visible warning.
    2. Wound gate    — EfficientNet-B0, wound vs non-wound: supported
                       (>= 0.70), uncertain (continue, warn), unsupported
                       (< 0.10, stop before classification).
    3. Wound type    — EfficientNet-B3 over diabetic / pressure / surgical /
                       venous ulcer with validation-fitted temperature
                       scaling; uncertain when the top class is under 0.65 or
                       the leading two are within 0.15.
    4. Grad-CAM      — final feature block, predicted class.
    5. Follow-up     — a confident surgical-wound call also runs the
                       infection-risk screen (surgical_wound_risk_service).

  Known surgical wound (source "Surgical Wound")
    Quality, then the gate with its lower surgical reject limit (0.02 —
    incisions look unlike the open chronic wounds most gate training images
    show), then the infection-risk screen as the primary result. No wound-type
    model and no Grad-CAM, exactly as the source routes it.

  Both routes finish with boundary segmentation (wound_segmentation_service).
  As in the source, a segmentation or follow-up failure never discards a
  valid primary result.

Thresholds are the source's deployed values; its .env matches config.py for
every setting used here.

Deliberate differences from the source:
- Checkpoints are tried with weights_only=True first (checkpoint_loading).
- Grad-CAM holds a lock: the source's per-request hooks sit on a shared model,
  so two concurrent requests could read each other's activations.
- Grad-CAM is a plain overlay, not a matplotlib figure with English captions
  burnt into the pixels, so the UI can caption it in NL/DE.
- Images are EXIF-transposed before inference, so phone photos are analysed
  upright and overlays line up with the upload as the browser shows it.
- The automatic surgical follow-up is new: the source ran the ensemble only
  when a clinician pre-selected "Surgical Wound".

What the models cannot do, per the source's README_DEMO: no infection
diagnosis and no healing-stage, necrosis, slough, dehiscence, burn or trauma
assessment; the four wound types are the only wound-type outputs.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from services import surgical_wound_risk_service as surgical_risk
from services import wound_segmentation_service as segmentation
from services.checkpoint_loading import load_checkpoint

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("CAREAI_MODEL_DIR", BASE_DIR / "models"))
UPLOAD_DIR = Path(os.getenv("CAREAI_UPLOAD_DIR", BASE_DIR / "uploads" / "diagnostics"))
CAM_DIR = BASE_DIR / "static" / "cam"

CLASSIFIER_FILE = "wound_type_efficientnet_b3.pt"
CALIBRATION_FILE = "wound_type_calibration.json"
CLASSIFIER_METRICS_FILE = "wound_type_test_metrics.json"
GATE_FILE = "wound_gate_efficientnet_b0.pt"
GATE_METRICS_FILE = "wound_gate_test_metrics.json"
REQUIRED_FILES = (CLASSIFIER_FILE, CALIBRATION_FILE, GATE_FILE)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024          # source MAX_UPLOAD_MB=8

WOUND_CONTEXTS = ("unknown", "surgical")
GATE_THRESHOLD = 0.70
GATE_HARD_REJECT_THRESHOLD = 0.10
GATE_HARD_REJECT_THRESHOLD_SURGICAL = 0.02
CONFIDENCE_THRESHOLD = 0.65
AMBIGUITY_MARGIN_THRESHOLD = 0.15

QUALITY_THRESHOLDS = {
    "blur": 45.0,
    "dark": 35.0,
    "bright": 225.0,
    "overexposed_fraction": 0.55,
    "underexposed_fraction": 0.70,
    "severe_blur": 8.0,
    "severe_dark": 18.0,
    "severe_bright": 242.0,
    "severe_overexposed_fraction": 0.90,
    "severe_underexposed_fraction": 0.90,
}

DISPLAY_NAMES = {
    "diabetic_ulcer": "Diabetic ulcer",
    "pressure_ulcer": "Pressure ulcer",
    "surgical_wound": "Surgical wound",
    "venous_ulcer": "Venous ulcer",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_LOAD_LOCK = threading.Lock()
_GRADCAM_LOCK = threading.Lock()
_LOADED: Optional[Dict[str, object]] = None


class WoundAnalysisRejected(ValueError):
    """The image was screened out before analysis. `stage` says where."""

    def __init__(self, message: str, stage: str, details: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


def display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Runtime availability
# ---------------------------------------------------------------------------

def runtime_status() -> Dict[str, object]:
    """Whether the core pipeline can run, plus the state of the two extras."""
    status: Dict[str, object] = {
        "available": False, "reason": None, "versions": {}, "missing": [],
        "segmentation_available": segmentation.available(),
        "surgical_risk": surgical_risk.runtime_status(),
    }
    try:
        import torch
        import torchvision
        status["versions"] = {"torch": torch.__version__, "torchvision": torchvision.__version__}
    except Exception as exc:
        status["reason"] = f"PyTorch unavailable ({exc.__class__.__name__}). Install torch and torchvision."
        return status

    try:
        import cv2
        status["versions"]["opencv"] = cv2.__version__
    except Exception as exc:
        status["reason"] = f"OpenCV unavailable ({exc.__class__.__name__}); it runs the image-quality checks."
        return status

    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).exists()]
    if missing:
        status["missing"] = missing
        status["reason"] = "Model file(s) not found: " + ", ".join(missing)
        return status

    status["available"] = True
    return status


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _eval_transform(size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _build_efficientnet(name: str, num_classes: int):
    """Rebuild the torchvision backbone the checkpoint was trained on.

    Only the two architectures these checkpoints use are supported; the
    source's factory also covered DenseNet, ResNet, ConvNeXt, ViT and Swin.
    """
    import torch.nn as nn
    from torchvision import models

    if name not in ("efficientnet_b0", "efficientnet_b3"):
        raise ValueError(f"Unsupported wound model architecture: {name}")
    model = getattr(models, name)(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, model.features[-1]


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load() -> Dict[str, object]:
    """Load the gate and classifier once. Built locally and published in one
    assignment so a concurrent request never sees a half-initialised state."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    with _LOAD_LOCK:
        if _LOADED is not None:
            return _LOADED

        import torch

        for name in REQUIRED_FILES:
            if not (MODEL_DIR / name).exists():
                raise FileNotFoundError(f"Model file missing: {MODEL_DIR / name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint, classifier_mode = load_checkpoint(MODEL_DIR / CLASSIFIER_FILE)
        classes = list(checkpoint["classes"])
        architecture = str(checkpoint["model"])
        classifier, target_layer = _build_efficientnet(architecture, len(classes))
        classifier.load_state_dict(checkpoint["state_dict"])
        classifier.to(device).eval()
        image_size = int(checkpoint.get("image_size", 300))

        calibration = {
            "method": "none",
            "temperature": 1.0,
            "confidence_threshold": float(checkpoint.get("confidence_threshold", CONFIDENCE_THRESHOLD)),
            "ambiguity_margin_threshold": AMBIGUITY_MARGIN_THRESHOLD,
            "validation_samples": None,
        }
        fitted = _read_json(MODEL_DIR / CALIBRATION_FILE)
        if fitted:
            calibration.update(
                method=fitted.get("method", "temperature_scaling"),
                temperature=max(float(fitted.get("temperature", 1.0)), 1e-6),
                confidence_threshold=float(fitted.get("confidence_threshold", calibration["confidence_threshold"])),
                ambiguity_margin_threshold=float(
                    fitted.get("ambiguity_margin_threshold", calibration["ambiguity_margin_threshold"])),
                validation_samples=fitted.get("validation_samples"),
            )

        gate_checkpoint, gate_mode = load_checkpoint(MODEL_DIR / GATE_FILE)
        gate_classes = list(gate_checkpoint.get("classes", ["non_wound", "wound"]))
        if not {"wound", "non_wound"} <= set(gate_classes):
            raise RuntimeError("Wound gate checkpoint must contain classes 'non_wound' and 'wound'.")
        gate, _ = _build_efficientnet("efficientnet_b0", len(gate_classes))
        gate.load_state_dict(gate_checkpoint["state_dict"])
        gate.to(device).eval()
        gate_size = int(gate_checkpoint.get("image_size", 224))

        _LOADED = {
            "device": device,
            "classifier": classifier,
            "target_layer": target_layer,
            "classes": classes,
            "architecture": architecture,
            "image_size": image_size,
            "transform": _eval_transform(image_size),
            "calibration": calibration,
            "classifier_load_mode": classifier_mode,
            "gate": gate,
            "gate_classes": gate_classes,
            "gate_size": gate_size,
            "gate_transform": _eval_transform(gate_size),
            "gate_threshold": float(gate_checkpoint.get("wound_threshold", GATE_THRESHOLD)),
            "gate_load_mode": gate_mode,
            "heldout": _read_json(MODEL_DIR / CLASSIFIER_METRICS_FILE),
            "gate_heldout": _read_json(MODEL_DIR / GATE_METRICS_FILE),
        }
    return _LOADED


def _open_rgb(image_path: Path):
    from PIL import Image, ImageOps
    with Image.open(image_path) as handle:
        return ImageOps.exif_transpose(handle).convert("RGB")


# ---------------------------------------------------------------------------
# Stage 1 — image quality
# ---------------------------------------------------------------------------

def _local_focus_score(gray) -> float:
    """75th-percentile sharpness over a 3x3 grid.

    A global blur number can be low for valid wound photos with lots of smooth
    skin or background; looking at local regions avoids those false rejections.
    """
    import cv2
    import numpy as np

    height, width = gray.shape[:2]
    scores = []
    for row in range(3):
        for col in range(3):
            tile = gray[row * height // 3:(row + 1) * height // 3,
                        col * width // 3:(col + 1) * width // 3]
            if tile.size:
                scores.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()))
    return float(np.percentile(scores, 75)) if scores else 0.0


def check_quality(image_path: Path) -> Dict[str, object]:
    """Return passed / warning / rejected, with the metrics behind it."""
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    t = QUALITY_THRESHOLDS
    result: Dict[str, object] = {
        "valid": False, "status": "rejected", "warnings": [], "blocking_reasons": [], "metrics": {},
    }

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError):
        result["blocking_reasons"].append("The uploaded file is not a valid or readable image.")
        return result

    # imdecode rather than imread: imread cannot open non-ASCII Windows paths.
    decoded = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        result["blocking_reasons"].append("The image could not be read properly. Please upload another image.")
        return result

    gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    local_focus = _local_focus_score(gray)
    overexposed = float(np.mean(gray >= 245))
    underexposed = float(np.mean(gray <= 15))

    result["metrics"] = {
        "width": width,
        "height": height,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "blur_score": round(blur_score, 1),
        "local_focus_score": round(local_focus, 1),
        "overexposed_fraction": round(overexposed, 4),
        "underexposed_fraction": round(underexposed, 4),
    }

    # Hard blocks: only conditions where the wound genuinely cannot be seen.
    if brightness < t["severe_dark"] or underexposed > t["severe_underexposed_fraction"]:
        result["blocking_reasons"].append(
            "The image is severely dark and the wound cannot be assessed reliably. "
            "Please take another photo in better lighting.")
    if brightness > t["severe_bright"] or overexposed > t["severe_overexposed_fraction"]:
        result["blocking_reasons"].append(
            "The image is severely overexposed and wound detail is not visible. "
            "Please take another photo without strong flash or direct light.")
    # Both global and local sharpness must be very low: smooth skin around a
    # clean incision otherwise reads as blur.
    if blur_score < t["severe_blur"] and local_focus < max(t["severe_blur"] * 1.5, 12.0):
        result["blocking_reasons"].append(
            "The image is severely blurred and the wound is not clear enough to analyse. "
            "Please upload a more focused photo.")
    if result["blocking_reasons"]:
        return result

    if brightness < t["dark"] or underexposed > t["underexposed_fraction"]:
        result["warnings"].append(
            "The image is darker than preferred; a better-lit photo may give a more reliable result.")
    if brightness > t["bright"] or overexposed > t["overexposed_fraction"]:
        result["warnings"].append(
            "The image is brighter or more exposed than preferred; avoiding strong flash may help.")
    if blur_score < t["blur"]:
        result["warnings"].append(
            "The image is less sharp than preferred; a more focused photo may give a more reliable result.")

    result["valid"] = True
    result["status"] = "warning" if result["warnings"] else "passed"
    return result


# ---------------------------------------------------------------------------
# Stage 2 — wound gate
# ---------------------------------------------------------------------------

def screen_wound(image_path: Path, hard_reject_threshold: Optional[float] = None) -> Dict[str, object]:
    import torch

    state = _load()
    threshold = state["gate_threshold"]
    reject_below = GATE_HARD_REJECT_THRESHOLD if hard_reject_threshold is None else float(hard_reject_threshold)
    if not 0 <= reject_below < threshold:
        raise ValueError("Gate hard-reject threshold must be >= 0 and below the gate threshold.")

    tensor = state["gate_transform"](_open_rgb(image_path)).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        probabilities = torch.softmax(state["gate"](tensor), dim=1)[0]

    by_class = {name: float(probabilities[i].item()) for i, name in enumerate(state["gate_classes"])}
    wound_probability = by_class["wound"]

    if wound_probability >= threshold:
        gate_status, warning = "supported", None
    elif wound_probability >= reject_below:
        gate_status = "uncertain"
        warning = ("The wound-screening score is below the preferred threshold but not low enough "
                   "to reject the image. Analysis continued with a domain warning.")
    else:
        gate_status, warning = "unsupported", None

    return {
        "status": gate_status,
        "can_analyse": gate_status != "unsupported",
        "wound_probability": wound_probability,
        "non_wound_probability": by_class["non_wound"],
        "threshold": threshold,
        "hard_reject_threshold": reject_below,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Stage 3 — wound type
# ---------------------------------------------------------------------------

def classify(image_path: Path) -> Dict[str, object]:
    import torch

    state = _load()
    calibration = state["calibration"]
    classes: List[str] = state["classes"]

    tensor = state["transform"](_open_rgb(image_path)).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        logits = state["classifier"](tensor)
        raw = torch.softmax(logits, dim=1)[0]
        calibrated = torch.softmax(logits / calibration["temperature"], dim=1)[0]

    top_values, top_indices = torch.topk(calibrated, k=min(2, len(classes)))
    predicted = classes[int(top_indices[0].item())]
    confidence = float(top_values[0].item())
    runner_up = classes[int(top_indices[1].item())] if len(top_values) > 1 else None
    runner_up_confidence = float(top_values[1].item()) if len(top_values) > 1 else 0.0
    margin = confidence - runner_up_confidence

    low_confidence = confidence < calibration["confidence_threshold"]
    ambiguous = margin < calibration["ambiguity_margin_threshold"]
    if low_confidence and ambiguous:
        reason = "Top probability is below the confidence threshold and the two leading classes are too close."
    elif low_confidence:
        reason = "Top probability is below the confidence threshold."
    elif ambiguous:
        reason = "The two leading wound classes are too close."
    else:
        reason = None

    return {
        "predicted_class": predicted,
        "confidence": confidence,
        "runner_up_class": runner_up,
        "runner_up_confidence": runner_up_confidence,
        "top2_margin": margin,
        "probabilities": {name: float(calibrated[i].item()) for i, name in enumerate(classes)},
        "raw_probabilities": {name: float(raw[i].item()) for i, name in enumerate(classes)},
        "low_confidence": low_confidence,
        "ambiguous": ambiguous,
        "uncertain": low_confidence or ambiguous,
        "uncertainty_reason": reason,
    }


# ---------------------------------------------------------------------------
# Stage 4 — Grad-CAM
# ---------------------------------------------------------------------------

def _blend_heatmap(base_rgb, heat):
    """Blue-to-red overlay that fades in with the heat, so cold regions keep
    the wound readable instead of tinting the whole photo."""
    import numpy as np

    colour = np.zeros(base_rgb.shape, dtype="float32")
    colour[..., 0] = np.clip(heat * 255.0 * 1.5, 0, 255)
    colour[..., 1] = np.clip((1.0 - np.abs(heat - 0.5) * 2.0) * 255.0, 0, 255)
    colour[..., 2] = np.clip((1.0 - heat) * 255.0, 0, 255)
    alpha = (0.15 + 0.55 * heat)[..., None]
    return np.clip(base_rgb * (1.0 - alpha) + colour * alpha, 0, 255).astype("uint8")


def gradcam_overlay(image_path: Path, out_path: Path) -> Optional[Path]:
    """Grad-CAM for the predicted wound type. None if it cannot be produced —
    explainability must never take the prediction down with it."""
    try:
        import numpy as np
        import torch
        from PIL import Image

        state = _load()
        model = state["classifier"]
        original = _open_rgb(image_path)
        tensor = state["transform"](original).unsqueeze(0).to(state["device"])

        with _GRADCAM_LOCK:
            captured: Dict[str, object] = {}
            handle = state["target_layer"].register_forward_hook(
                lambda module, inputs, output: captured.__setitem__("activation", output))
            try:
                model.zero_grad(set_to_none=True)
                logits = model(tensor)                    # gradients needed: no no_grad()
                # Temperature scaling does not change the argmax, so this is the
                # same class the calibrated prediction reported.
                index = int(logits.argmax(dim=1).item())
                activation = captured.get("activation")
                if activation is None:
                    return None
                gradients = torch.autograd.grad(logits[0, index], activation)[0]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1)).squeeze(0).detach()
                cam -= cam.min()
                peak = cam.max()
                if peak > 0:
                    cam /= peak
                heat_small = cam.cpu().numpy()
            finally:
                handle.remove()
                model.zero_grad(set_to_none=True)

        width, height = original.size
        heat = np.asarray(
            Image.fromarray((heat_small * 255).astype("uint8")).resize((width, height), Image.BILINEAR),
            dtype="float32") / 255.0
        overlay = _blend_heatmap(np.asarray(original, dtype="float32"), heat)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out_path)
        return out_path
    except Exception as exc:                                   # pragma: no cover
        print(f"Wound Grad-CAM failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Clinical framing
# ---------------------------------------------------------------------------

def _heldout_for(class_name: str) -> Dict[str, object]:
    metrics = _load()["heldout"]
    per_class = metrics.get("per_class") or {}
    stats = per_class.get(class_name) or {}
    total = sum(int(v.get("support", 0)) for v in per_class.values())
    return {
        "precision": stats.get("precision"),
        "sensitivity": stats.get("recall_sensitivity"),
        "support": stats.get("support"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "test_images": total or None,
    }


def limitations(used_classifier: bool, used_risk: bool, boundary: Optional[Dict[str, object]]) -> List[str]:
    notes = [
        "Nothing here diagnoses infection, and the models do not assess healing stage, "
        "necrosis, slough, dehiscence, burns or traumatic wounds.",
    ]
    if used_classifier:
        notes.insert(0, "Wound type is limited to four classes: diabetic, pressure, surgical and venous ulcer.")
        per_class = _load()["heldout"].get("per_class") or {}
        if per_class:
            weakest, stats = min(per_class.items(), key=lambda item: item[1].get("recall_sensitivity", 1.0))
            notes.append(
                f"Weakest class: it finds {stats['recall_sensitivity'] * 100:.0f}% of "
                f"{display_name(weakest).lower()}s in held-out testing, so a different result "
                "does not rule one out.")
    if used_risk:
        notes.append(
            "The infection-risk score is an AI model score, not a calibrated probability of "
            "surgical-site infection. It was evaluated only on development folds, never on a "
            "held-out test set.")
    if boundary and boundary.get("detected"):
        notes.append(
            "Measurements are in pixels with no physical scale, so they compare only between "
            "photos taken at the same distance and zoom.")
    return notes


def _recommendations(prediction, quality, gate, risk, boundary) -> List[str]:
    # Workflow guidance only. Like the source, this gives no treatment advice
    # keyed to the predicted wound type.
    items = []
    if prediction is not None:
        if prediction["uncertain"]:
            items.append("Wound type uncertain — confirm clinically before choosing a care pathway.")
        else:
            items.append("AI-assisted suggestion — a clinician must confirm the wound type.")
    if risk is not None:
        if risk["elevated"]:
            items.append("Elevated infection-risk image category — arrange prompt clinical review of the wound.")
        elif risk["clinical_review_recommended"]:
            items.append("Borderline or conflicting infection-risk scores — clinical review of the wound recommended.")
        else:
            items.append("Low infection-risk image category — this does not exclude infection; "
                         "continue routine post-operative wound checks.")
    if gate["status"] == "uncertain":
        items.append("The photo may not show a supported wound clearly; check framing and retake if needed.")
    if quality["status"] == "warning":
        items.append("Retake in even light, in focus, with the wound filling most of the frame.")
    if boundary and boundary.get("detected"):
        items.append("Compare wound size over time only between photos taken at the same distance and zoom.")
    items.append("Record the result in the care plan only after clinician review.")
    return items


def _classifier_technical() -> Dict[str, str]:
    state = _load()
    calibration = state["calibration"]
    heldout = state["heldout"]
    details = {
        "Architecture": f"{state['architecture'].replace('_', '-').title()} (torchvision)",
        "Classes": ", ".join(display_name(c) for c in state["classes"]),
        "Input size": f"{state['image_size']}x{state['image_size']}",
        "Calibration": (
            f"{calibration['method'].replace('_', ' ')}, T={calibration['temperature']:.3f}"
            + (f" (n={calibration['validation_samples']} validation)" if calibration["validation_samples"] else "")),
        "Decision rule": (
            f"top-1; uncertain if < {calibration['confidence_threshold']:.2f} "
            f"or top-2 margin < {calibration['ambiguity_margin_threshold']:.2f}"),
        "Weights": CLASSIFIER_FILE,
        "Checkpoint loading": f"classifier {state['classifier_load_mode']}",
    }
    if heldout:
        total = sum(int(v.get("support", 0)) for v in (heldout.get("per_class") or {}).values())
        details["Held-out test"] = (
            f"accuracy {heldout.get('accuracy', 0) * 100:.1f}% · macro-F1 {heldout.get('macro_f1', 0):.2f}"
            + (f" (n={total})" if total else ""))
    return details


def _gate_technical(gate: Dict[str, object]) -> Dict[str, str]:
    state = _load()
    held = state["gate_heldout"]
    details = {
        "Wound gate": (
            f"EfficientNet-B0 · supported ≥ {gate['threshold']:.2f} · "
            f"reject < {gate['hard_reject_threshold']:.2f} · {GATE_FILE} ({state['gate_load_mode']})"),
    }
    if held:
        details["Gate held-out test"] = (
            f"accuracy {held.get('accuracy', 0) * 100:.1f}% · "
            f"non-wound specificity {held.get('non_wound_specificity', 0) * 100:.1f}%")
    return details


# ---------------------------------------------------------------------------
# Entry point used by the routes
# ---------------------------------------------------------------------------

def analyse_upload(file_storage, wound_context: str = "unknown") -> Dict[str, object]:
    """Run the full pipeline on an upload and return what the template needs.

    wound_context is 'unknown' (classify the wound type) or 'surgical' (a known
    surgical wound: screen its infection risk). Raises WoundAnalysisRejected (a
    ValueError) when the quality check or the gate stops the image, ValueError
    for other fixable input problems, and RuntimeError when the models cannot run.
    """
    if wound_context not in WOUND_CONTEXTS:
        raise ValueError(f"Unknown wound context '{wound_context}'.")
    if file_storage is None or not file_storage.filename:
        raise ValueError("Choose a wound photo to analyse.")

    suffix = Path(file_storage.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'. Use JPG, PNG or WEBP.")

    payload_bytes = file_storage.read()
    if not payload_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(payload_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")

    run_id = uuid.uuid4().hex[:10]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / f"{run_id}_wound{suffix}"
    saved_path.write_bytes(payload_bytes)

    surgical_context = wound_context == "surgical"
    reject_below = GATE_HARD_REJECT_THRESHOLD_SURGICAL if surgical_context else GATE_HARD_REJECT_THRESHOLD

    try:
        quality = check_quality(saved_path)
        if not quality["valid"]:
            raise WoundAnalysisRejected(" ".join(quality["blocking_reasons"]), "quality", quality)

        gate = screen_wound(saved_path, reject_below)
        if not gate["can_analyse"]:
            raise WoundAnalysisRejected(
                "The uploaded image does not appear to contain a supported wound "
                f"(wound-screening score {gate['wound_probability'] * 100:.1f}%). "
                "Please upload a photo where the wound or surgical incision is clearly visible.",
                "gate", gate)

        prediction = None if surgical_context else classify(saved_path)
        risk = surgical_risk.screen(saved_path, trigger="clinician") if surgical_context else None
    except ValueError:                               # includes WoundAnalysisRejected
        # Like the source, do not keep images that were screened out.
        saved_path.unlink(missing_ok=True)
        raise
    except FileNotFoundError as exc:
        saved_path.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc
    except ImportError as exc:
        saved_path.unlink(missing_ok=True)
        raise RuntimeError(f"Wound model runtime unavailable: {exc}") from exc
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        if surgical_context:
            raise RuntimeError(f"Surgical infection-risk screen unavailable: {exc}") from exc
        raise

    cam_url = None
    if prediction is not None:
        cam_name = f"{run_id}_wound_cam.png"
        if gradcam_overlay(saved_path, CAM_DIR / cam_name):
            cam_url = f"/static/cam/{cam_name}"

    # A confident surgical call gets the infection-risk screen as a follow-up.
    risk_note = None
    if prediction is not None and prediction["predicted_class"] == "surgical_wound":
        if prediction["uncertain"]:
            risk_note = ("The wound type is uncertain, so the surgical infection-risk screen was not run "
                         "automatically. For a known surgical wound, re-run with that context selected.")
        else:
            try:
                risk = surgical_risk.screen(saved_path, trigger="classifier")
            except Exception as exc:
                risk_note = f"Surgical infection-risk screen unavailable: {exc}"

    boundary, boundary_note = None, None
    if segmentation.available():
        try:
            boundary_name = f"{run_id}_wound_boundary.png"
            boundary = segmentation.segment(saved_path, CAM_DIR / boundary_name)
            boundary["overlay_url"] = f"/static/cam/{boundary_name}" if boundary.pop("overlay_path") else None
            if not boundary["detected"]:
                boundary_note = "No wound boundary was detected automatically, so no measurement is available."
        except Exception as exc:
            boundary = None
            boundary_note = f"Automatic boundary measurement unavailable: {exc}"
    else:
        boundary_note = "The boundary segmentation model is not installed."

    technical: Dict[str, str] = {}
    if prediction is not None:
        technical.update(_classifier_technical())
    technical.update(_gate_technical(gate))
    if risk is not None:
        technical.update(surgical_risk.technical_details())
    if boundary is not None:
        held = segmentation.heldout_summary()
        technical["Segmentation"] = (
            f"DeepLabV3-ResNet50 · {boundary['model_version']} · mask threshold {boundary['threshold']:.2f}")
        if held:
            technical["Segmentation held-out test"] = (
                f"Dice {held.get('dice', 0):.2f} · IoU {held.get('iou', 0):.2f}")
    technical["Device"] = str(_load()["device"])

    if prediction is not None:
        predicted = prediction["predicted_class"]
        if prediction["uncertain"]:
            finding = (f"Most consistent with {display_name(predicted).lower()}, but not confidently: "
                       f"{prediction['uncertainty_reason'][0].lower()}{prediction['uncertainty_reason'][1:]}")
            review_status = "Wound type uncertain — clinician confirmation required"
        else:
            finding = f"Features most consistent with {display_name(predicted).lower()}."
            review_status = "AI-assisted result — medical review required"
        module_label = "Wound AI — wound type"
    else:
        finding = (f"{risk['category_display']} (AI ensemble score {risk['ensemble_score']:.2f}; "
                   f"operating threshold {risk['decision_threshold']:.2f}).")
        review_status = "AI-assisted surgical risk screening — clinical review required"
        module_label = "Wound AI — surgical infection-risk screen"

    result: Dict[str, object] = {
        "run_id": run_id,
        "module": "wound",
        "mode": "surgical_screen" if surgical_context else "wound_type",
        "wound_context": wound_context,
        "module_label": module_label,
        "modality": "Wound photograph",
        "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "source_filename": file_storage.filename,
        "finding": finding,
        "review_status": review_status,
        "recommendations": _recommendations(prediction, quality, gate, risk, boundary),
        "quality": quality,
        "gate": gate,
        "warnings": list(quality["warnings"]) + ([gate["warning"]] if gate["warning"] else []),
        "surgical_risk": risk,
        "surgical_risk_note": risk_note,
        "boundary": boundary,
        "boundary_note": boundary_note,
        "limitations": limitations(prediction is not None, risk is not None, boundary),
        "technical": technical,
        "upload_url": f"/diagnostics-image/{saved_path.name}",
        "cam_url": cam_url,
        "disclaimer": (
            "Research prototype. Not a medical device and not validated for diagnosis. "
            "A qualified clinician must review every output before any decision."),
    }

    if prediction is not None:
        ordered = sorted(prediction["probabilities"].items(), key=lambda item: item[1], reverse=True)
        result.update({
            "prediction": predicted,
            "prediction_display": display_name(predicted),
            "probability": prediction["confidence"],
            "confidence_pct": round(prediction["confidence"] * 100, 1),
            "probabilities": [
                {"label": display_name(name), "value": value, "pct": round(value * 100, 1)}
                for name, value in ordered
            ],
            "uncertain": prediction["uncertain"],
            "uncertainty_reason": prediction["uncertainty_reason"],
            "low_confidence": prediction["uncertain"],
            "heldout": _heldout_for(predicted),
        })
    else:
        result.update({
            "uncertain": risk["clinical_review_recommended"],
            "uncertainty_reason": "; ".join(risk["review_reasons"]) or None,
            "low_confidence": False,
        })
    return result
