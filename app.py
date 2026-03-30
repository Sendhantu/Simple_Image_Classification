import logging
import os
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)          # MobileNetV2 expects 224×224
MAX_FILE_BYTES = 5 * 1024 * 1024    # 5 MB
ALLOWED_EXT    = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIMES  = {"image/jpeg", "image/png", "image/webp"}
TOP_K          = 3                   # how many predictions to return

# ── Load pretrained model ─────────────────────────────────────────
# Downloads weights automatically on first run (~14 MB, cached after)
try:
    model = MobileNetV2(weights="imagenet")
    model.trainable = False
    logger.info("MobileNetV2 loaded — ready for inference")
except Exception as exc:
    logger.critical("Failed to load MobileNetV2: %s", exc)
    model = None


# ── Helpers ───────────────────────────────────────────────────────
def validate_file(file) -> str | None:
    if not file or file.filename == "":
        return "No file selected."
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXT)}"
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_BYTES:
        return f"File too large ({size // 1024} KB). Max allowed is 5 MB."
    return None


def preprocess(file) -> np.ndarray:
    """Read → RGB → 224×224 → MobileNetV2 normalisation → (1, 224, 224, 3)."""
    raw = file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)          # scales to [-1, 1] for MobileNetV2
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", error="Model not available.")

    file = request.files.get("file")

    err = validate_file(file)
    if err:
        logger.warning("Validation failed: %s", err)
        return render_template("index.html", error=err)

    try:
        img_array = preprocess(file)
        logger.info("Preprocessed '%s' → shape %s", file.filename, img_array.shape)

        preds = model.predict(img_array, verbose=0)

        # decode_predictions returns: [[(class_id, label, score), ...]]
        top = decode_predictions(preds, top=TOP_K)[0]

        # Primary prediction
        _, label, score = top[0]
        prediction  = label.replace("_", " ").title()
        confidence  = round(float(score) * 100, 1)

        # All top-K as a list of dicts for the template
        alternatives = [
            {
                "label":      lbl.replace("_", " ").title(),
                "confidence": round(float(sc) * 100, 1),
            }
            for _, lbl, sc in top
        ]

        logger.info(
            "Prediction: %s (%.1f%%) | File: '%s'",
            prediction, confidence, file.filename,
        )

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            alternatives=alternatives,
        )

    except Image.UnidentifiedImageError:
        logger.warning("Invalid image file: '%s'", file.filename)
        return render_template("index.html", error="Could not read the image. Please upload a valid file.")
    except Exception:
        logger.exception("Unexpected prediction error for '%s'", file.filename)
        return render_template("index.html", error="Prediction failed. Please try again.")


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model":        "MobileNetV2 (ImageNet)",
        "model_loaded": model is not None,
        "img_size":     IMG_SIZE,
    })


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") != "production"
    logger.info("Starting on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)