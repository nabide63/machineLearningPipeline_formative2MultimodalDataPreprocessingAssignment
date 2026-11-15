"""
main.py
--------
Multimodal User Identity & Product Recommendation System
Command-Line Interface

USAGE:
    python main.py

FLOW:
    1. Enter customer ID
    2. Enter path to face image → Face Recognition (is this a known member?)
       → NOT recognised: ACCESS DENIED
    3. Enter path to voice audio → Voiceprint Verification
       → NOT verified: ACCESS DENIED
    4. Product Recommendation displayed for that customer

BEFORE RUNNING:
    1. Train the models first:
         python scripts/train_face_model.py
         python scripts/train_voice_model.py
    2. Install dependencies:
         pip install -r requirements.txt
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import joblib

# ── audio pipeline (pure numpy, no numba/librosa needed) ──────────────────────
import audio_pipeline_numpy as af

# ── face preprocessing (opencv) ───────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

IMAGE_SIZE = (128, 128)
FACE_CONF_THRESHOLD  = 0.60   # members score 0.92+, unseen faces score ~0.3-0.4
VOICE_CONF_THRESHOLD = 0.60   # min P(approved) to accept a voice

def preprocess_face_image(image_path: str) -> np.ndarray:
    if not CV2_OK:
        print("  [WARNING] opencv not installed — using zero features (install opencv-python).")
        return np.zeros((1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3))
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    # Extract RGB histogram features (8 bins per channel) — matches image_features.csv
    features = {}
    for ch_idx, ch_name in enumerate(["R", "G", "B"]):
        hist, _ = np.histogram(img[:, :, ch_idx], bins=8, range=(0, 256))
        hist = hist / hist.sum()
        for b, v in enumerate(hist):
            features[f"{ch_name}_bin_{b}"] = v
    return np.array(list(features.values())).reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_DIR         = ROOT / "models"
FACE_MODEL_PATH    = MODELS_DIR / "face_model.pkl"
VOICE_MODEL_PATH   = MODELS_DIR / "voice_model.pkl"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_model_numpy.joblib"   # pure-numpy version
FEATURE_STORE_PATH = ROOT / "data" / "customer_feature_store.csv"
VOICE_SCALER = None   # set in load_models() when the voice bundle has scaler stats


class _DummyFace:
    """Fallback when face_model.pkl is missing — always approves as 'Laura'."""
    classes_ = ["Laura"]
    def predict(self, X): print("  [DUMMY] Face model not found."); return [0]
    def predict_proba(self, X): return [[1.0]]

class _DummyVoice:
    """Fallback when voice_model.pkl is missing — always approves."""
    def predict(self, X): print("  [DUMMY] Voice model not found."); return [0]

def load_models():
    print("\nLoading models …")

    if FACE_MODEL_PATH.exists():
        face_bundle = joblib.load(FACE_MODEL_PATH)
        face_model  = face_bundle["model"]
        face_labels = face_bundle["face_labels"]          # {0: "Laura", 1: "Esther", …}
        face_feats  = face_bundle["feature_cols"]
        print(f"  [OK] Face model  ({face_bundle['model_name']})")
    else:
        print(f"  [WARN] {FACE_MODEL_PATH} not found — run scripts/train_face_model.py")
        face_model, face_labels, face_feats = _DummyFace(), {0: "Laura"}, []

    if VOICE_MODEL_PATH.exists():
        voice_bundle  = joblib.load(VOICE_MODEL_PATH)
        voice_model   = voice_bundle["model"]
        voice_labels  = voice_bundle["voice_labels"]      # {0: "approved", 1: "denied"}
        approved_idx  = voice_bundle["approved_class_idx"]
        voice_feats   = voice_bundle["feature_cols"]
        global VOICE_SCALER
        if "scaler_mean" in voice_bundle:
            VOICE_SCALER = (np.asarray(voice_bundle["scaler_mean"]), np.asarray(voice_bundle["scaler_std"]))
        print(f"  [OK] Voice model ({voice_bundle['model_name']})")
    else:
        print(f"  [WARN] {VOICE_MODEL_PATH} not found — run scripts/train_voice_model.py")
        voice_model  = _DummyVoice()
        voice_labels = {0: "approved", 1: "denied"}
        approved_idx = 0
        voice_feats  = []

    if PRODUCT_MODEL_PATH.exists():
        product_bundle = joblib.load(PRODUCT_MODEL_PATH)
        print(f"  [OK] Product model ({product_bundle['model_name']})")
    else:
        print(f"  [WARN] {PRODUCT_MODEL_PATH} not found — run train_product_model.py")
        product_bundle = None

    return (face_model, face_labels, face_feats,
            voice_model, voice_labels, approved_idx, voice_feats,
            product_bundle)


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   🔐  Multimodal Identity & Product Recommendation       ║
║        System  –  Team Simulation                        ║
╚══════════════════════════════════════════════════════════╝
"""

def _bar(ch="─", n=57): return ch * n

def print_step(step, title):
    print(f"\n{_bar()}\n  STEP {step}: {title}\n{_bar()}")

def print_denied(reason):
    print(f"""
╔══════════════════════════════════════════╗
║  🚫  ACCESS DENIED                       ║
║  Reason: {reason:<32}║
╚══════════════════════════════════════════╝""")

def print_welcome(user):
    print(f"""
╔══════════════════════════════════════════╗
║  ✅  IDENTITY VERIFIED                   ║
║  Welcome, {user:<31}║
╚══════════════════════════════════════════╝""")

def print_recommendation(customer_id, recs, source):
    W = 42
    line = lambda t: print("║" + f"  {t}".ljust(W) + "║")
    print("\n╔" + "═" * W + "╗")
    line("PRODUCT RECOMMENDATION")
    line(f"Customer ID: {customer_id}")
    line(f"Source: {source[:30]}")
    for rank, (cat, prob) in enumerate(recs, 1):
        line(f"{rank}. {cat:<20} {prob:.1%}")
    print("╚" + "═" * W + "╝")

def get_file(prompt, exts):
    while True:
        p = input(f"\n  {prompt}: ").strip()
        if not p:
            print("  [!] No path entered — try again.")
            continue
        if not os.path.exists(p):
            print(f"  [!] File not found: {p}")
            continue
        if exts and not any(p.lower().endswith(e) for e in exts):
            print(f"  [!] Expected {exts}, got '{Path(p).suffix}'")
            continue
        return p

def get_int(prompt):
    while True:
        v = input(f"\n  {prompt}: ").strip()
        if v.isdigit():
            return int(v)
        print("  [!] Please enter a number.")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def step_face(face_model, face_labels, image_path):
    print(f"\n  🔍 Processing image: {image_path}")
    feats = preprocess_face_image(image_path)
    proba = np.asarray(face_model.predict_proba(feats)[0], dtype=float)
    best  = int(np.argmax(proba))
    conf  = float(proba[best])
    pred  = int(face_model.classes_[best]) if hasattr(face_model, "classes_") else best
    name  = face_labels.get(pred, f"Unknown (class {pred})")
    # A closed-set classifier ALWAYS names someone, so require high confidence
    authorized = (pred in face_labels) and (conf >= FACE_CONF_THRESHOLD)
    print(f"  Closest match: {name}  (confidence {conf:.0%}, threshold {FACE_CONF_THRESHOLD:.0%})")
    return authorized, name, feats


def step_voice(voice_model, approved_idx, voice_feats, audio_path):
    print(f"\n  🎙️  Processing audio: {audio_path}")
    y = af.load_clean(audio_path)
    feat_dict = af.extract_features(y)
    if voice_feats:
        feat_vec = np.array([feat_dict.get(c, 0.0) for c in voice_feats]).reshape(1, -1)
    else:
        feat_vec = np.array(list(feat_dict.values())).reshape(1, -1)
    if VOICE_SCALER is not None:
        mu, sd = VOICE_SCALER
        feat_vec = (feat_vec - mu) / sd
    proba   = np.asarray(voice_model.predict_proba(feat_vec)[0], dtype=float)
    classes = list(voice_model.classes_)
    p_appr  = float(proba[classes.index(approved_idx)]) if approved_idx in classes else 0.0
    approved = p_appr >= VOICE_CONF_THRESHOLD
    print(f"  P(approved voice) = {p_appr:.0%}  (threshold {VOICE_CONF_THRESHOLD:.0%})")
    return approved, feat_vec


def _load_store_row(customer_id):
    """Read one customer's row from the feature store with the csv module (no pandas)."""
    import csv
    with open(FEATURE_STORE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if int(float(row["customer_id"])) == customer_id:
                    return row
            except (ValueError, KeyError):
                continue
    raise KeyError(f"Customer ID {customer_id} not in feature store")


def _to_float(value, default):
    try:
        x = float(value)
        return default if x != x else x      # NaN -> default
    except (TypeError, ValueError):
        return default


def step_product(product_bundle, customer_id, top_k=3):
    if product_bundle is None:
        return [("Electronics", 0.4), ("Clothing", 0.35), ("Books", 0.25)], "fallback (no model)"

    model       = product_bundle["model"]
    classes     = product_bundle["classes"]
    feat_cols   = product_bundle["feature_cols"]
    col_medians = product_bundle["col_medians"]
    fallback    = product_bundle["fallback_distribution"]

    row = _load_store_row(customer_id)        # KeyError -> "Unknown customer ID"

    try:
        last_cat = str(row.get("last_category", "None"))
        values = []
        for col in feat_cols:
            if col.startswith("last_cat_"):
                values.append(float(last_cat == col[len("last_cat_"):]))
            elif col == "has_social_profile":
                values.append(1.0 if str(row.get(col, "")).strip().lower() in ("true", "1", "1.0") else 0.0)
            else:
                values.append(_to_float(row.get(col), col_medians.get(col, 0.0)))
        feat_vec = np.array(values, dtype=float).reshape(1, -1)
        proba    = model.predict_proba(feat_vec)[0]
        source   = product_bundle["model_name"]
    except Exception as e:
        proba   = list(fallback.values())
        classes = list(fallback.keys())
        source  = f"popularity fallback ({e})"

    pairs = sorted(zip(classes, proba), key=lambda x: -x[1])[:top_k]
    return [(str(cat), round(float(pr), 3)) for cat, pr in pairs], source


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_transaction(face_model, face_labels, face_feats,
                          voice_model, approved_idx, voice_feats,
                          product_bundle):
    """Authorized full transaction simulation."""

    # Step 0: customer ID
    print_step(0, "CUSTOMER ID")
    customer_id = get_int("Enter your Customer ID (e.g. 125)")

    # Step 1: face
    print_step(1, "FACE RECOGNITION")
    image_path = get_file("Path to face image (.jpg / .jpeg / .png)",
                          [".jpg", ".jpeg", ".png"])
    try:
        authorized, user_name, _ = step_face(face_model, face_labels, image_path)
    except Exception as e:
        print(f"  [ERROR] {e}")
        print_denied("Face processing error")
        return

    if not authorized:
        print_denied("Face not recognised")
        return
    print_welcome(user_name)

    # Step 2: voice
    print_step(2, "VOICE VERIFICATION")
    print("  Say one of: 'Yes, approve'  |  'Confirm transaction'")
    audio_path = get_file("Path to voice recording (.wav / .ogg / .mp4 / .aac)",
                          [".wav", ".ogg", ".mp4", ".aac", ".m4a", ".flac"])
    try:
        voice_ok, _ = step_voice(voice_model, approved_idx, voice_feats, audio_path)
    except Exception as e:
        print(f"  [ERROR] {e}")
        print_denied("Voice processing error")
        return

    if not voice_ok:
        print_denied("Voice not verified")
        return
    print("\n  ✅  Voice approved!")

    # Step 3: product recommendation
    print_step(3, "PRODUCT RECOMMENDATION")
    try:
        recs, source = step_product(product_bundle, customer_id)
        print_recommendation(customer_id, recs, source)
    except KeyError as e:
        print(f"  [ERROR] {e}")
        print_denied("Unknown customer ID")
    except Exception as e:
        print(f"  [ERROR] Product recommendation failed: {e}")


def demo_unauthorized_face(face_model, face_labels):
    """Demo: unknown face → access denied."""
    print_step("DEMO", "UNAUTHORIZED FACE ATTEMPT")
    image_path = get_file("Path to an UNAUTHORISED face image (.jpg / .jpeg / .png)",
                          [".jpg", ".jpeg", ".png"])
    try:
        authorized, user_name, _ = step_face(face_model, face_labels, image_path)
    except Exception as e:
        print(f"  [ERROR] {e}")
        print_denied("Face processing error")
        return

    if not authorized:
        print_denied("Face not recognised")
    else:
        print(f"\n  [NOTE] This face was recognised as: {user_name}")
        print("         Try a different image that the model hasn't seen.")


def demo_unauthorized_voice(voice_model, approved_idx, voice_feats):
    """Demo: foreign voice → access denied."""
    print_step("DEMO", "UNAUTHORIZED VOICE ATTEMPT")
    audio_path = get_file("Path to an UNAUTHORISED voice sample (.wav / .ogg / .mp4 / .aac)",
                          [".wav", ".ogg", ".mp4", ".aac", ".m4a", ".flac"])
    try:
        voice_ok, _ = step_voice(voice_model, approved_idx, voice_feats, audio_path)
    except Exception as e:
        print(f"  [ERROR] {e}")
        print_denied("Voice processing error")
        return

    if not voice_ok:
        print_denied("Voice not verified")
    else:
        print("  [NOTE] This voice was actually approved. Use a different audio sample.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MENU
# ═══════════════════════════════════════════════════════════════════════════════

MENU = """
┌──────────────────────────────────────────┐
│              MAIN MENU                   │
├──────────────────────────────────────────┤
│  1. Run Full Transaction (Authorised)    │
│  2. Demo: Unauthorised Face Attempt      │
│  3. Demo: Unauthorised Voice Attempt     │
│  4. Exit                                 │
└──────────────────────────────────────────┘"""

def main():
    print(BANNER)
    (face_model, face_labels, face_feats,
     voice_model, voice_labels, approved_idx, voice_feats,
     product_bundle) = load_models()

    while True:
        print(MENU)
        choice = input("\n  Enter choice (1–4): ").strip()

        if choice == "1":
            run_full_transaction(face_model, face_labels, face_feats,
                                  voice_model, approved_idx, voice_feats,
                                  product_bundle)
        elif choice == "2":
            demo_unauthorized_face(face_model, face_labels)
        elif choice == "3":
            demo_unauthorized_voice(voice_model, approved_idx, voice_feats)
        elif choice == "4":
            print("\n  Goodbye! 👋\n")
            break
        else:
            print("  [!] Invalid choice — enter 1, 2, 3, or 4.")

        input("\n  Press Enter to return to the menu …")


if __name__ == "__main__":
    main()

