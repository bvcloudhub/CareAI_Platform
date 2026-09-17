"""
Keras inference for the Care.AI lung imaging modules.

Ported from Lung_Cancer_V3/third_party/lung_ai_flask — consolidating
utils/prediction.py, utils/explain.py, xray/service.py and histology/service.py
into one registry-driven module.

Three models, three different jobs. The source shipped the pneumonia model under
the filename "lung_cancer_resnet50.h5"; its own config, xray_classes.json and
train_xray.py all agree the classes are NORMAL/PNEUMONIA, so here it is named
and surfaced as pneumonia screening and kept out of the cancer modules.

Two corrections to the source behaviour, both verified against its training
scripts and the saved graphs:

1. Preprocessing. The source fed the X-ray model images scaled to [0,1]
   (app.py: `scale01 = (model_key == "xray")`), but that graph starts with a
   Rescaling(1/255) layer, so inputs were divided by 255 twice. All three models
   take RAW 0-255: CT and X-ray were trained on `image_dataset_from_directory`
   output with no external scaling, and the histology graph carries EfficientNet's
   own Rescaling/Normalization internally.

2. Confidence. The source rescaled every probability into a fixed 0.85-0.95 band
   before display, so a genuine 51% prediction was shown as 85%. This module
   reports the model's actual probability.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("CAREAI_MODEL_DIR", BASE_DIR / "models"))
UPLOAD_DIR = Path(os.getenv("CAREAI_UPLOAD_DIR", BASE_DIR / "uploads" / "diagnostics"))
CAM_DIR = BASE_DIR / "static" / "cam"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024

_LOAD_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, object] = {}
_BACKEND: Dict[str, str] = {}      # key -> "keras3" | "tf_keras" | "tf_keras(patched)"


@dataclass(frozen=True)
class ImagingModel:
    """One imaging model: what it reads, what it can say, and how to feed it."""

    key: str
    label: str
    modality: str
    model_file: str
    target: Tuple[int, int]                       # (height, width)
    class_names: Tuple[str, ...]
    architecture: str
    input_note: str
    class_file: Optional[str] = None              # overrides class_names when present
    positive_classes: Tuple[str, ...] = ()        # classes that warrant escalation
    findings: Dict[str, str] = field(default_factory=dict)
    accepts: str = "JPG, PNG or BMP"

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / self.model_file

    def resolved_class_names(self) -> List[str]:
        if self.class_file:
            path = MODEL_DIR / self.class_file
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "class_names" in data:
                    return list(data["class_names"])
                return list(data)
        return list(self.class_names)


REGISTRY: Dict[str, ImagingModel] = {
    "ct": ImagingModel(
        key="ct",
        label="Lung CT — malignancy screening",
        modality="Chest CT slice",
        model_file="lung_ct_malignancy_densenet.h5",
        target=(256, 256),
        class_names=("Malignant", "Normal"),
        architecture="DenseNet201 (transfer-learned)",
        input_note="raw 0-255 RGB; trained without external rescaling",
        positive_classes=("MALIGNANT",),
        findings={
            "MALIGNANT": "Imaging features suggestive of malignancy.",
            "NORMAL": "No malignant features identified by the model.",
        },
    ),
    "histology": ImagingModel(
        key="histology",
        label="Lung histopathology — subtype",
        modality="Histopathology slide",
        model_file="lung_histology_subtype.h5",
        target=(224, 224),
        class_names=("ADENOCARCINOMA", "BENIGN", "SQUAMOUS_CELL_CARCINOMA"),
        class_file="lung_histology_classes.json",
        architecture="EfficientNetB0 (transfer-learned)",
        input_note="raw 0-255 RGB; graph rescales and normalises internally",
        positive_classes=("ADENOCARCINOMA", "SQUAMOUS_CELL_CARCINOMA"),
        findings={
            "ADENOCARCINOMA": "Glandular pattern consistent with adenocarcinoma.",
            "SQUAMOUS_CELL_CARCINOMA": "Pattern consistent with squamous cell carcinoma.",
            "BENIGN": "No malignant histologic pattern identified by the model.",
        },
    ),
    "xray": ImagingModel(
        key="xray",
        label="Chest X-ray — pneumonia screening",
        modality="Chest radiograph",
        model_file="chest_xray_pneumonia_resnet50.h5",
        target=(224, 224),
        class_names=("NORMAL", "PNEUMONIA"),
        class_file="chest_xray_pneumonia_classes.json",
        architecture="ResNet50 (transfer-learned)",
        input_note="raw 0-255 RGB; graph carries Rescaling(1/255)",
        positive_classes=("PNEUMONIA",),
        findings={
            "PNEUMONIA": "Opacity pattern consistent with pneumonia.",
            "NORMAL": "No pneumonia pattern identified by the model.",
        },
    ),
}

# Modules the Lung Cancer AI tab exposes. The X-ray model is a pneumonia
# classifier and is surfaced separately rather than as cancer detection.
CANCER_MODULES = ("ct", "histology")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_keras3(path: Path):
    import keras
    return keras.saving.load_model(str(path), compile=False)


def _load_tf_keras(path: Path):
    import tf_keras
    return tf_keras.models.load_model(str(path), compile=False)


def _load_tf_keras_lambda_patched(path: Path):
    """Load a Keras 2 .h5 whose Lambda holds Python-version-locked bytecode.

    The histology graph wraps `efficientnet.preprocess_input` in a Lambda. Keras
    marshals that function's bytecode into the file, and bytecode from the
    training interpreter cannot be unmarshalled by a different Python version
    ("bad marshal data"). In current Keras that function is a documented no-op —
    EfficientNet does its scaling in the Rescaling/Normalization layers that
    follow — so substituting a linear Activation reproduces the graph exactly.
    The layer holds no weights, so the weight topology is unchanged.
    """
    import h5py
    import tf_keras

    with h5py.File(path, "r") as handle:
        raw_config = handle.attrs.get("model_config")
    if raw_config is None:
        raise ValueError(f"{path.name} has no model_config to patch.")
    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")

    config = json.loads(raw_config)
    replaced = 0
    for layer in config.get("config", {}).get("layers", []):
        if layer.get("class_name") == "Lambda":
            layer["class_name"] = "Activation"
            layer["config"] = {
                "name": layer["config"].get("name"),
                "trainable": False,
                "dtype": "float32",
                "activation": "linear",
            }
            replaced += 1
    if not replaced:
        raise ValueError(f"{path.name} has no Lambda layer to patch.")

    model = tf_keras.models.model_from_json(json.dumps(config))
    model.load_weights(str(path))
    return model


_LOADERS = (
    ("keras3", _load_keras3),
    ("tf_keras", _load_tf_keras),
    ("tf_keras(patched)", _load_tf_keras_lambda_patched),
)


def load_model(key: str):
    """Load and cache one model. Loading is slow, so it happens on first use."""
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    spec = REGISTRY[key]
    if not spec.model_path.exists():
        raise FileNotFoundError(f"Model file missing: {spec.model_path}")

    with _LOAD_LOCK:
        if key in _MODEL_CACHE:                                # another thread won
            return _MODEL_CACHE[key]

        errors = []
        for backend, loader in _LOADERS:
            try:
                model = loader(spec.model_path)
            except Exception as exc:
                errors.append(f"{backend}: {type(exc).__name__}: {str(exc)[:180]}")
                continue
            _MODEL_CACHE[key] = model
            _BACKEND[key] = backend
            return model

        raise RuntimeError(f"Could not load {spec.model_file}. Tried — " + " | ".join(errors))


def runtime_status() -> Dict[str, object]:
    """Describe whether inference can actually run, for the UI to show."""
    status: Dict[str, object] = {"available": False, "reason": None, "versions": {}, "models": {}}
    try:
        import tensorflow as tf
        import keras
        status["versions"] = {"tensorflow": tf.__version__, "keras": keras.__version__}
    except Exception as exc:
        status["reason"] = (
            f"TensorFlow/Keras unavailable ({exc.__class__.__name__}). "
            "Needs CPython 3.10-3.13 with tensorflow installed."
        )
        for key, spec in REGISTRY.items():
            status["models"][key] = {"present": spec.model_path.exists(), "loaded": False}
        return status

    missing = []
    for key, spec in REGISTRY.items():
        present = spec.model_path.exists()
        status["models"][key] = {
            "present": present,
            "loaded": key in _MODEL_CACHE,
            "backend": _BACKEND.get(key),
        }
        if not present:
            missing.append(spec.model_file)

    if missing:
        status["reason"] = "Model file(s) not found: " + ", ".join(missing)
    else:
        status["available"] = True
    return status


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _softmax(vector):
    import numpy as np
    shifted = vector - np.max(vector)
    exponentiated = np.exp(shifted)
    return exponentiated / (np.sum(exponentiated) + 1e-8)


def _prepare_batch(image_path: Path, target: Tuple[int, int]):
    """Resize to the model's input and return an NHWC batch of RAW 0-255 RGB.

    All three graphs expect unscaled pixel values — see the module docstring.
    """
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as handle:
        image = handle.convert("RGB").resize((target[1], target[0]), Image.BILINEAR)
        array = np.asarray(image, dtype="float32")
    return np.expand_dims(array, axis=0)


def predict(key: str, image_path: Path) -> Tuple[str, List[float], List[str]]:
    """Run one image through one model. Returns (label, probabilities, classes)."""
    import numpy as np

    model = load_model(key)
    spec = REGISTRY[key]
    classes = spec.resolved_class_names()

    batch = _prepare_batch(image_path, spec.target)
    raw = model.predict(batch, verbose=0)
    vector = np.asarray(raw[0] if isinstance(raw, (list, tuple)) else raw).reshape(-1)

    if not (np.all(vector >= 0) and abs(float(vector.sum()) - 1.0) < 1e-3):
        vector = _softmax(vector)
    vector = vector.astype("float64")
    vector = vector / (vector.sum() + 1e-12)

    if len(vector) != len(classes):
        raise ValueError(
            f"{spec.model_file} returned {len(vector)} outputs but {len(classes)} class "
            f"names are configured ({classes})."
        )

    return classes[int(np.argmax(vector))], [float(p) for p in vector], classes


def gradcam_overlay(key: str, image_path: Path, out_path: Path) -> Optional[Path]:
    """Blend a Grad-CAM saliency map over the image. Returns None if unavailable.

    Explainability is a nice-to-have here: a failure must never take down the
    prediction, so every step is guarded.
    """
    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image

        model = load_model(key)
        spec = REGISTRY[key]
        batch = _prepare_batch(image_path, spec.target)

        # Build the gradient model with whichever library loaded this model —
        # mixing Keras 3 and tf_keras objects raises at graph construction.
        if _BACKEND.get(key, "").startswith("tf_keras"):
            import tf_keras as backend_keras
        else:
            import keras as backend_keras

        conv_layer = None
        for layer in reversed(model.layers):
            try:
                shape = layer.output.shape
            except Exception:
                shape = getattr(layer, "output_shape", None)
            if shape is not None and len(shape) == 4:
                conv_layer = layer
                break
        if conv_layer is None:
            return None

        # `model.output` is a list on some Keras 3 functional models, which would
        # nest the gradient model's outputs and break the tensor slice below.
        final_output = model.outputs[0] if getattr(model, "outputs", None) else model.output
        conv_output = conv_layer.output
        if isinstance(conv_output, (list, tuple)):
            conv_output = conv_output[0]

        grad_model = backend_keras.Model(model.inputs, [conv_output, final_output])
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(batch, training=False)
            if isinstance(conv_out, (list, tuple)):
                conv_out = conv_out[0]
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            top = tf.argmax(predictions[0])
            score = predictions[:, top]

        grads = tape.gradient(score, conv_out)
        if grads is None:
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

        with Image.open(image_path) as handle:
            base = handle.convert("RGB")
            width, height = base.size
            base_array = np.asarray(base, dtype="float32")

        heat = np.asarray(
            Image.fromarray((heatmap * 255).astype("uint8")).resize((width, height), Image.BILINEAR),
            dtype="float32",
        ) / 255.0

        # Blue (cold) through green to red (hot), mirroring the source's JET ramp.
        colour = np.zeros((height, width, 3), dtype="float32")
        colour[..., 0] = np.clip(heat * 255.0 * 1.5, 0, 255)
        colour[..., 1] = np.clip((1.0 - np.abs(heat - 0.5) * 2.0) * 255.0, 0, 255)
        colour[..., 2] = np.clip((1.0 - heat) * 255.0, 0, 255)

        # Fade the overlay in with the heat itself. A flat blend (as in the
        # source) tints the entire slice blue and buries the anatomy the
        # clinician is trying to read against the saliency map.
        alpha = (0.15 + 0.55 * heat)[..., None]
        overlay = np.clip(base_array * (1.0 - alpha) + colour * alpha, 0, 255).astype("uint8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out_path)
        return out_path
    except Exception as exc:                                   # pragma: no cover
        print(f"Grad-CAM failed for {key}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Clinical framing
# ---------------------------------------------------------------------------

def _risk_level(spec: ImagingModel, label: str, probability: float) -> str:
    if label.upper() not in {c.upper() for c in spec.positive_classes}:
        return "low"
    if probability >= 0.80:
        return "critical"
    if probability >= 0.60:
        return "high"
    return "medium"


def _recommendations(spec: ImagingModel, risk: str) -> List[str]:
    if spec.key == "histology":
        specialist, extra = "pathologist", "Consider additional sections or immunohistochemistry."
    elif spec.key == "ct":
        specialist, extra = "radiologist", "Consider targeted follow-up imaging or tissue sampling."
    else:
        specialist, extra = "clinician", "Correlate with symptoms, observations and inflammatory markers."

    if risk in ("high", "critical"):
        return [f"Urgent {specialist} review.", extra,
                "Correlate with clinical history and recent observations."]
    if risk == "medium":
        return [f"{specialist.title()} review advised.",
                "Consider short-interval follow-up per local protocol.", "Correlate clinically."]
    return ["No immediate action indicated by the model alone.",
            "Continue the standard clinical workflow."]


def technical_details(key: str) -> Dict[str, str]:
    spec = REGISTRY[key]
    details = {
        "Modality": spec.modality,
        "Architecture": spec.architecture,
        "Classes": ", ".join(spec.resolved_class_names()),
        "Input size": f"{spec.target[0]}x{spec.target[1]}",
        "Preprocessing": spec.input_note,
        "Weights": spec.model_file,
    }
    model = _MODEL_CACHE.get(key)
    if model is not None:
        try:
            details["Parameters"] = f"{model.count_params():,}"
            details["Loaded via"] = _BACKEND.get(key, "unknown")
        except Exception:
            pass
    return details


# ---------------------------------------------------------------------------
# Entry point used by the routes
# ---------------------------------------------------------------------------

def analyse_upload(key: str, file_storage) -> Dict[str, object]:
    """Save an upload, run the model, and return everything the template needs.

    Raises ValueError for anything the user can fix (wrong file type, too big)
    and RuntimeError when the model itself cannot run.
    """
    if key not in REGISTRY:
        raise ValueError(f"Unknown imaging module: {key}")
    spec = REGISTRY[key]

    if file_storage is None or not file_storage.filename:
        raise ValueError("Choose an image file to analyse.")

    suffix = Path(file_storage.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'. Use JPG, PNG or BMP.")

    payload_bytes = file_storage.read()
    if not payload_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(payload_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")

    run_id = uuid.uuid4().hex[:10]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / f"{run_id}_{key}{suffix}"
    saved_path.write_bytes(payload_bytes)

    # Reject anything that is not a decodable image before touching the model.
    try:
        from PIL import Image
        with Image.open(saved_path) as probe:
            probe.verify()
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise ValueError("That file could not be read as an image.") from exc

    try:
        label, probabilities, classes = predict(key, saved_path)
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    except ImportError as exc:
        raise RuntimeError(f"Model runtime unavailable: {exc}") from exc

    top_probability = max(probabilities)
    risk = _risk_level(spec, label, top_probability)

    cam_name = f"{run_id}_{key}_cam.png"
    cam_path = gradcam_overlay(key, saved_path, CAM_DIR / cam_name)

    return {
        "run_id": run_id,
        "module": key,
        "module_label": spec.label,
        "modality": spec.modality,
        "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "source_filename": file_storage.filename,
        "prediction": label,
        "prediction_display": label.replace("_", " ").title(),
        "probability": top_probability,
        "confidence_pct": round(top_probability * 100, 1),
        "probabilities": [
            {"label": name.replace("_", " ").title(), "value": value, "pct": round(value * 100, 1)}
            for name, value in zip(classes, probabilities)
        ],
        "risk": risk,
        "finding": spec.findings.get(label.upper(), "Model output recorded."),
        "recommendations": _recommendations(spec, risk),
        "technical": technical_details(key),
        "upload_url": f"/diagnostics-image/{saved_path.name}",
        "cam_url": f"/static/cam/{cam_name}" if cam_path else None,
        "low_confidence": top_probability < 0.60,
        "disclaimer": (
            "Research prototype. Not a medical device and not validated for diagnosis. "
            "A qualified clinician must review every output before any decision."
        ),
    }
