"""
scripts/train_voice_model.py
-----------------------------
Builds audio_features.csv from data/audio/ recordings,
trains the Voiceprint Verification Model, and saves it as
models/voice_model.pkl.

Members: Esther, Laura, Rwema, Yusuf (Nabide)
Phrases:
    1 → "Yes, approve"
    2 → "Confirm transaction"

Usage:
    python scripts/train_voice_model.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[1]
AUDIO_DIR  = ROOT / "data" / "audio"
WAV_DIR    = ROOT / "data" / "audio_wav"
AUG_DIR    = ROOT / "data" / "audio_augmented"
FEAT_CSV   = ROOT / "audio_features.csv"
MODEL_PATH = ROOT / "models" / "voice_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

# Add repo root to path so audio_features.py can be imported
sys.path.insert(0, str(ROOT))
import audio_features as af

RANDOM_STATE = 42
APPROVED_LABEL   = "approved"   # class name for approved voice
DENIED_LABEL     = "denied"     # class name for rejected voice

# ── Step 1: Build audio_features.csv ─────────────────────────────────────────
print("=" * 60)
print("STEP 1: Extracting audio features")
print("=" * 60)

audio_files = af.list_audio_files(AUDIO_DIR)
if not audio_files:
    raise FileNotFoundError(f"No audio files found in {AUDIO_DIR}. "
                            "Make sure data/audio/ contains the team recordings.")

print(f"Found {len(audio_files)} recording(s): {[p.name for p in audio_files]}")

audio_df, _ = af.build_audio_dataset(AUDIO_DIR, WAV_DIR, AUG_DIR)
audio_df.to_csv(FEAT_CSV, index=False)
print(f"Saved {FEAT_CSV.name}  shape={audio_df.shape}")
print("Members:", audio_df["member"].value_counts().to_dict())

# ── Step 2: Prepare labels ────────────────────────────────────────────────────
# Every row in audio_df is from a known/approved team member → label "approved"
# We also create synthetic "denied" rows using the same features with added noise
#  (simulates an outsider; the model learns the boundary between known voices and noise)
print("\n" + "=" * 60)
print("STEP 2: Preparing training data with approved + denied classes")
print("=" * 60)

meta_cols = ["member", "sample_id", "phrase", "source_file", "augmentation", "is_augmented"]
feat_cols  = [c for c in audio_df.columns if c not in meta_cols]

approved_X = audio_df[feat_cols].values
approved_y = np.array([APPROVED_LABEL] * len(approved_X))

# Generate "denied" examples: shuffle + noise so they look like strangers
rng = np.random.default_rng(RANDOM_STATE)
denied_X = approved_X.copy()
# Add heavy noise so features drift far from known speakers
noise_scale = approved_X.std(axis=0) * 2.5
denied_X    = denied_X + rng.normal(0, noise_scale, denied_X.shape)
# Also randomly shuffle rows so they don't track any real member
rng.shuffle(denied_X)
denied_y = np.array([DENIED_LABEL] * len(denied_X))

X = np.vstack([approved_X, denied_X])
y_raw = np.concatenate([approved_y, denied_y])

le = LabelEncoder()
y  = le.fit_transform(y_raw)   # 0 = approved / 1 = denied  (alphabetical)
approved_class_idx = int(np.where(le.classes_ == APPROVED_LABEL)[0][0])

print(f"Classes: {list(le.classes_)}  (approved class index = {approved_class_idx})")
print(f"Training rows: {len(X)}  |  Features: {X.shape[1]}")

# ── Step 3: Train + evaluate ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Model training and evaluation")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                               C=1.0, solver="lbfgs"),
}

best_name, best_score, best_model = None, -1, None
for name, clf in candidates.items():
    cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  {name:<25} CV Accuracy = {cv.mean():.4f}  (±{cv.std():.4f})")
    if cv.mean() > best_score:
        best_score, best_name, best_model = cv.mean(), name, clf

print(f"\nBest model: {best_name}")

best_model.fit(X_train, y_train)
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average="weighted")
loss = log_loss(y_test, y_proba)

print(f"\n── Test-set metrics ───────────────────────────────────────────")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  Log Loss : {loss:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Step 4: Refit on ALL data and save ───────────────────────────────────────
print("=" * 60)
print("STEP 4: Saving model")
print("=" * 60)

best_model.fit(X, y)

bundle = {
    "model":               best_model,
    "label_encoder":       le,
    "voice_labels":        {int(i): str(c) for i, c in enumerate(le.classes_)},
    "approved_class_idx":  approved_class_idx,
    "feature_cols":        feat_cols,
    "model_name":          best_name,
    "test_metrics":        {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "log_loss": round(loss, 4)},
}
joblib.dump(bundle, MODEL_PATH)
print(f"Saved voice model → {MODEL_PATH}")
print(f"audio_features.csv → {FEAT_CSV}")

