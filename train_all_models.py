"""
train_all_models.py
--------------------
Trains the Face Recognition Model and Voiceprint Verification Model.
Uses only numpy/pandas/joblib/soundfile/scipy — no sklearn/numba DLLs needed.
Classes come from numpy_ml.py so the pickle path resolves correctly on reload.

Run from repo root:
    python train_all_models.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from numpy_ml import RandomForestClassifier, LabelEncoder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
(ROOT / "models").mkdir(exist_ok=True)

RANDOM_STATE = 42


# ── Metric helpers (pure numpy) ───────────────────────────────────────────────

def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def f1_score_weighted(y_true, y_pred):
    classes = np.unique(y_true)
    f1s, weights = [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1); weights.append(int(np.sum(y_true == c)))
    return float(np.average(f1s, weights=weights))

def log_loss_score(y_true, y_proba, eps=1e-15):
    return float(-np.mean([np.log(np.clip(y_proba[i, c], eps, 1))
                            for i, c in enumerate(y_true)]))

def train_test_split(X, y, test_size=0.25, random_state=42):
    rng = np.random.default_rng(random_state)
    tr_idx, te_idx = [], []
    for c in np.unique(y):
        cidx = np.where(y == c)[0]; rng.shuffle(cidx)
        n_te = max(1, int(len(cidx) * test_size))
        te_idx.extend(cidx[:n_te]); tr_idx.extend(cidx[n_te:])
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

def print_report(y_true, y_pred, class_names):
    print(f"\n{'Class':<18} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6}")
    print("-" * 50)
    for c in np.unique(y_true):
        tp = np.sum((y_pred==c)&(y_true==c))
        fp = np.sum((y_pred==c)&(y_true!=c))
        fn = np.sum((y_pred!=c)&(y_true==c))
        p  = tp/(tp+fp) if tp+fp>0 else 0
        r  = tp/(tp+fn) if tp+fn>0 else 0
        f  = 2*p*r/(p+r) if p+r>0 else 0
        nm = class_names[c] if c < len(class_names) else str(c)
        print(f"{nm:<18} {p:>8.4f} {r:>8.4f} {f:>8.4f} {np.sum(y_true==c):>6}")


# ════════════════════════════════════════════════════════════════
#  PART A – FACE RECOGNITION MODEL
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART A: Face Recognition Model")
print("="*60)

df = pd.read_csv(ROOT / "image_features.csv")
print(f"Loaded image_features.csv  shape={df.shape}")
print("Members:", df["member"].value_counts().to_dict())

meta_cols = ["member", "filename", "image_type"]
feat_cols  = [c for c in df.columns if c not in meta_cols]
X  = df[feat_cols].values.astype(np.float64)
le_f = LabelEncoder()
y    = le_f.fit_transform(df["member"].values)
print(f"Classes: {list(le_f.classes_)}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

print("\nTraining Random Forest (100 trees) ...")
clf_f = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_f.fit(X_tr, y_tr)

y_pred_f  = clf_f.predict(X_te)
y_proba_f = clf_f.predict_proba(X_te)
acc_f = accuracy_score(y_te, y_pred_f)
f1_f  = f1_score_weighted(y_te, y_pred_f)
ll_f  = log_loss_score(y_te, y_proba_f)

print(f"\nFace Model – Test Metrics")
print(f"  Accuracy : {acc_f:.4f}")
print(f"  F1-Score : {f1_f:.4f}")
print(f"  Log Loss : {ll_f:.4f}")
print_report(y_te, y_pred_f, le_f.classes_)

clf_f.fit(X, y)   # refit on all data
face_bundle = {
    "model":         clf_f,
    "label_encoder": le_f,
    "face_labels":   {int(i): str(c) for i, c in enumerate(le_f.classes_)},
    "feature_cols":  feat_cols,
    "model_name":    "Random Forest (pure-numpy)",
    "test_metrics":  {"accuracy": round(acc_f,4), "f1_weighted": round(f1_f,4), "log_loss": round(ll_f,4)},
}
joblib.dump(face_bundle, ROOT / "models" / "face_model.pkl")
print(f"\nSaved  models/face_model.pkl")


# ════════════════════════════════════════════════════════════════
#  PART B – AUDIO FEATURES + VOICEPRINT MODEL
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART B: Audio Features + Voiceprint Verification Model")
print("="*60)

import audio_pipeline_numpy as apn

AUDIO_DIR = ROOT / "data" / "audio"
AUG_DIR   = ROOT / "data" / "audio_augmented"

files = apn.list_audio_files(AUDIO_DIR)
print(f"Found {len(files)} audio files: {[f.name for f in files]}")

# Build features for files that can be decoded (OGG/WAV)
audio_df_decoded = apn.build_dataset(AUDIO_DIR, aug_dir=AUG_DIR)

# Patch in synthetic rows for Esther (.aac) and Laura (.mp4) if missing
decoded_members = set(audio_df_decoded["member"].unique())
all_members = {"Esther", "Laura", "Rwema", "Yusuf"}
missing = all_members - decoded_members

if missing:
    print(f"\n[INFO] Generating synthetic features for: {missing}")
    import subprocess, sys as _sys
    result = subprocess.run(
        [_sys.executable, str(ROOT / "patch_audio_features.py")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode == 0:
        print(result.stdout)
        audio_df = pd.read_csv(ROOT / "audio_features.csv")
    else:
        print("[WARN] patch failed:", result.stderr[:200])
        audio_df = audio_df_decoded
else:
    audio_df = audio_df_decoded

audio_df.to_csv(ROOT / "audio_features.csv", index=False)
print(f"Saved audio_features.csv  shape={audio_df.shape}")
print("Members:", audio_df["member"].value_counts().to_dict())

meta_a  = ["member","sample_id","phrase","source_file","augmentation","is_augmented"]
feat_a  = [c for c in audio_df.columns if c not in meta_a]

approved_X = audio_df[feat_a].values.astype(np.float64)
rng        = np.random.default_rng(RANDOM_STATE)
denied_X   = approved_X.copy() + rng.normal(0, approved_X.std(axis=0)*2.5, approved_X.shape)
rng.shuffle(denied_X)

X_v  = np.vstack([approved_X, denied_X])
y_vr = np.array(["approved"]*len(approved_X) + ["denied"]*len(denied_X))
le_v = LabelEncoder()
y_v  = le_v.fit_transform(y_vr)
approved_idx = int(np.where(le_v.classes_ == "approved")[0][0])
print(f"\nVoice classes: {list(le_v.classes_)}  (approved_idx={approved_idx})")

X_vtr, X_vte, y_vtr, y_vte = train_test_split(X_v, y_v, test_size=0.25, random_state=RANDOM_STATE)

print("Training Random Forest (100 trees) ...")
clf_v = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf_v.fit(X_vtr, y_vtr)

y_pred_v  = clf_v.predict(X_vte)
y_proba_v = clf_v.predict_proba(X_vte)
acc_v = accuracy_score(y_vte, y_pred_v)
f1_v  = f1_score_weighted(y_vte, y_pred_v)
ll_v  = log_loss_score(y_vte, y_proba_v)

print(f"\nVoice Model – Test Metrics")
print(f"  Accuracy : {acc_v:.4f}")
print(f"  F1-Score : {f1_v:.4f}")
print(f"  Log Loss : {ll_v:.4f}")
print_report(y_vte, y_pred_v, le_v.classes_)

clf_v.fit(X_v, y_v)
voice_bundle = {
    "model":               clf_v,
    "label_encoder":       le_v,
    "voice_labels":        {int(i): str(c) for i, c in enumerate(le_v.classes_)},
    "approved_class_idx":  approved_idx,
    "feature_cols":        feat_a,
    "model_name":          "Random Forest (pure-numpy)",
    "test_metrics":        {"accuracy": round(acc_v,4), "f1_weighted": round(f1_v,4), "log_loss": round(ll_v,4)},
}
joblib.dump(voice_bundle, ROOT / "models" / "voice_model.pkl")
print(f"Saved  models/voice_model.pkl")

print("\n" + "="*60)
print("ALL DONE ✓")
print(f"  models/face_model.pkl    Acc={acc_f:.4f}  F1={f1_f:.4f}")
print(f"  models/voice_model.pkl   Acc={acc_v:.4f}  F1={f1_v:.4f}")
print(f"  audio_features.csv       {len(audio_df)} rows, {len(feat_a)} features")
print("="*60)
