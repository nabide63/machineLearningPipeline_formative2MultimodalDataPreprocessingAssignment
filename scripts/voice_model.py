"""
scripts/voice_model.py
-----------------------
Inference helper for the Voiceprint Verification model.
Mirrors the same pattern as scripts/product_model.py.

Usage (standalone):
    python scripts/voice_model.py --audio "data/audio/Laura 1.mp4"

From another script:
    from scripts.voice_model import verify_voice
    result = verify_voice("data/audio/Laura 1.mp4")
"""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import joblib

ROOT       = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "voice_model.pkl"

sys.path.insert(0, str(ROOT))
import audio_features as af


class VoiceDeniedError(ValueError):
    """Raised when the voice is not verified → ACCESS DENIED."""


@lru_cache(maxsize=1)
def _load():
    bundle = joblib.load(MODEL_PATH)
    return bundle


def verify_voice(audio_path: str):
    """
    Verify whether the voice in the audio file is an approved team member.

    Returns a dict:
        {
          "approved":    True,
          "label":       "approved",
          "confidence":  0.91
        }

    Raises VoiceDeniedError if not approved.
    """
    bundle       = _load()
    model        = bundle["model"]
    labels       = bundle["voice_labels"]        # {0: "approved", 1: "denied"}
    approved_idx = bundle["approved_class_idx"]
    feat_cols    = bundle.get("feature_cols", [])

    y        = af.load_clean(audio_path)
    feat_dict = af.extract_features(y)

    if feat_cols:
        feat_vec = np.array([feat_dict.get(c, 0.0) for c in feat_cols]).reshape(1, -1)
    else:
        feat_vec = np.array(list(feat_dict.values())).reshape(1, -1)

    pred  = int(model.predict(feat_vec)[0])
    label = labels.get(pred, f"class_{pred}")

    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_vec)[0]
        conf  = round(float(proba[pred]), 3)

    approved = pred == approved_idx
    if not approved:
        raise VoiceDeniedError(f"Voice classified as '{label}' → ACCESS DENIED")

    return {
        "approved":   True,
        "label":      label,
        "confidence": conf,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify a voice sample using the trained model.")
    parser.add_argument("--audio", required=True,
                        help="Path to audio file (.wav / .ogg / .mp4 / .aac).")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run: python scripts/train_voice_model.py")

    try:
        result = verify_voice(args.audio)
        print(f"Status    : APPROVED ✅")
        print(f"Label     : {result['label']}")
        if result["confidence"] is not None:
            print(f"Confidence: {result['confidence']:.1%}")
    except VoiceDeniedError as e:
        raise SystemExit(f"ACCESS DENIED: {e}")


if __name__ == "__main__":
    main()

