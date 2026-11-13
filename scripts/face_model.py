"""
scripts/face_model.py
----------------------
Inference helper for the Face Recognition model.
Mirrors the same pattern as scripts/product_model.py so the CLI app
can call it in a consistent way.

Usage (standalone):
    python scripts/face_model.py --image Images/Laura/neutral.jpeg

From another script:
    from scripts.face_model import identify_face
    result = identify_face("Images/Laura/neutral.jpeg")
"""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import joblib

ROOT           = Path(__file__).resolve().parents[1]
MODEL_PATH     = ROOT / "models" / "face_model.pkl"

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

IMAGE_SIZE = (128, 128)


class UnknownFaceError(ValueError):
    """Raised when the face is not in the known-member set → ACCESS DENIED."""


@lru_cache(maxsize=1)
def _load():
    bundle = joblib.load(MODEL_PATH)
    return bundle


def _extract_histogram_features(image_path: str) -> np.ndarray:
    """Extract the same 24-bin RGB histogram used to build image_features.csv."""
    if not CV2_OK:
        raise ImportError("opencv-python is required. Run: pip install opencv-python")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(img[:, :, ch], bins=8, range=(0, 256))
        feats.extend(hist / hist.sum())
    return np.array(feats).reshape(1, -1)


def identify_face(image_path: str):
    """
    Identify who is in the image.

    Returns a dict:
        {
          "member":     "Laura",
          "class_idx":  0,
          "authorized": True,
          "confidence": 0.87      # probability of top class (if available)
        }

    Raises UnknownFaceError if the predicted class is not in the label map
    (shouldn't happen with a trained model, but kept for safety).
    """
    bundle   = _load()
    model    = bundle["model"]
    labels   = bundle["face_labels"]     # {0: "Laura", 1: "Esther", …}
    feat_cols = bundle.get("feature_cols", [])

    feats = _extract_histogram_features(image_path)

    pred   = int(model.predict(feats)[0])
    name   = labels.get(pred, f"class_{pred}")

    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)[0]
        conf  = round(float(proba[pred]), 3)

    if pred not in labels:
        raise UnknownFaceError(f"Face class {pred} not in label map → ACCESS DENIED")

    return {
        "member":     name,
        "class_idx":  pred,
        "authorized": True,
        "confidence": conf,
    }


def main():
    parser = argparse.ArgumentParser(description="Identify a face using the trained model.")
    parser.add_argument("--image", required=True, help="Path to .jpg / .jpeg / .png image.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run: python scripts/train_face_model.py")

    try:
        result = identify_face(args.image)
    except UnknownFaceError as e:
        raise SystemExit(f"ACCESS DENIED: {e}")

    print(f"Member    : {result['member']}")
    print(f"Class idx : {result['class_idx']}")
    if result["confidence"] is not None:
        print(f"Confidence: {result['confidence']:.1%}")
    print("Status    : AUTHORISED ✅")


if __name__ == "__main__":
    main()

