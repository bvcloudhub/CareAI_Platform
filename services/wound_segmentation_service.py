"""
Wound boundary segmentation and measurement for Care.AI Wound AI.

Ported from WoundAI services/segmentation_service.py, plus the two pieces of
services/measurement_service.py the automatic path uses (measure_mask and
save_overlay). Not ported: the manual polygon-annotation tools, and the
"annotated review image", which burns English measurement text into the pixels
— CareAI shows the numbers in the page, where they can be translated.

Measurements are in pixels. The source converts to centimetres only when a
same-plane reference scale is supplied, and CareAI has no scale input, so no
physical units are produced. Pixel area is comparable only between photos
taken at the same distance and zoom.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from services.checkpoint_loading import load_checkpoint

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("CAREAI_MODEL_DIR", BASE_DIR / "models"))
SEGMENTATION_FILE = "wound_segmentation_deeplabv3.pt"
SEGMENTATION_METRICS_FILE = "wound_segmentation_test_metrics.json"

DEFAULT_THRESHOLD = 0.50                  # source SEGMENTATION_THRESHOLD
DEFAULT_IMAGE_SIZE = 384

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_LOCK = threading.Lock()
_LOADED: Optional[Dict[str, object]] = None


def available() -> bool:
    return (MODEL_DIR / SEGMENTATION_FILE).exists()


def _load() -> Dict[str, object]:
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    with _LOCK:
        if _LOADED is not None:
            return _LOADED

        import torch
        from torchvision import transforms
        from torchvision.models.segmentation import deeplabv3_resnet50

        path = MODEL_DIR / SEGMENTATION_FILE
        if not path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {path}")

        checkpoint, load_mode = load_checkpoint(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()

        size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
        metrics_path = MODEL_DIR / SEGMENTATION_METRICS_FILE
        heldout = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

        _LOADED = {
            "device": device,
            "model": model,
            "image_size": size,
            "threshold": float(checkpoint.get("threshold", DEFAULT_THRESHOLD)),
            "model_version": checkpoint.get("model_version", "deeplabv3_resnet50-wound-v1"),
            "load_mode": load_mode,
            "transform": transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            "heldout": heldout,
        }
    return _LOADED


def _largest_contour(mask255):
    import cv2
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def measure_mask(mask) -> Dict[str, float]:
    """Pixel geometry of a binary mask: area, rotated-box length and width,
    perimeter, and the share of the frame the wound occupies."""
    import cv2
    import numpy as np

    mask255 = (mask > 0).astype(np.uint8) * 255
    height, width = mask255.shape[:2]
    area = float(cv2.countNonZero(mask255))
    length = breadth = perimeter = 0.0

    contour = _largest_contour(mask255)
    if contour is not None and len(contour) >= 3:
        side_a, side_b = cv2.minAreaRect(contour)[1]
        length, breadth = float(max(side_a, side_b)), float(min(side_a, side_b))
        perimeter = float(cv2.arcLength(contour, True))

    return {
        "area_px": area,
        "length_px": length,
        "width_px": breadth,
        "perimeter_px": perimeter,
        "coverage_pct": (area / (width * height) * 100.0) if width and height else 0.0,
    }


def _save_overlay(image_bgr, mask, out_path: Path) -> None:
    """Translucent teal fill with a visible boundary, as in the source."""
    import cv2
    import numpy as np

    binary = (mask > 0).astype(np.uint8)
    overlay = image_bgr.copy()
    fill = np.array([160, 170, 30], dtype=np.float32)           # BGR teal
    region = overlay[binary == 1].astype(np.float32)
    if region.size:
        overlay[binary == 1] = np.clip(0.72 * region + 0.28 * fill, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # The source's fixed 2 px line disappears on full-resolution phone photos.
    thickness = max(2, int(round(max(overlay.shape[:2]) / 400)))
    cv2.drawContours(overlay, contours, -1, (160, 190, 20), thickness, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise ValueError("Could not encode the segmentation overlay.")
    out_path.write_bytes(encoded.tobytes())      # non-ASCII-path safe, unlike imwrite


def segment(image_path: Path, overlay_path: Path) -> Dict[str, object]:
    """Predict the wound boundary, keep the largest region, and measure it."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    state = _load()
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = image.size

    tensor = state["transform"](image).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        logits = state["model"](tensor)["out"]
        probability = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (probability >= state["threshold"]).astype(np.uint8)

    # Conservative clean-up, then keep only the largest connected region.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    largest = _largest_contour(mask * 255)
    if largest is not None:
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 1, thickness=-1)
        mask = clean

    detected = int(np.count_nonzero(mask)) > 0
    metrics = measure_mask(mask)

    saved_overlay = None
    if detected:
        _save_overlay(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), mask, overlay_path)
        saved_overlay = overlay_path

    return {
        "status": "boundary_detected" if detected else "boundary_not_detected",
        "detected": detected,
        "image_width_px": width,
        "image_height_px": height,
        "max_wound_probability": float(np.max(probability)) if probability.size else 0.0,
        "threshold": state["threshold"],
        "model_version": state["model_version"],
        "overlay_path": saved_overlay,
        "scale_provided": False,
        **metrics,
    }


def heldout_summary() -> Dict[str, float]:
    return dict(_load()["heldout"]) if available() else {}
