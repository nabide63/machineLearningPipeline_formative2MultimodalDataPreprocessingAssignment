"""
retrain_voice_real.py
---------------------
Retrain the voiceprint model from REAL recordings only (no synthetic rows).
  approved = every file in data/audio/          (team members)
  denied   = every file in data/unauthorized/   (strangers)
Each recording is augmented (pitch up/down, time stretch, noise) so the model
sees several versions of every voice. Needs ffmpeg for .aac / .mp4 files.

Usage:  python retrain_voice_real.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import audio_pipeline_numpy as af
from numpy_ml import RandomForestClassifier

APPROVED_DIR = ROOT / "data" / "audio"
DENIED_DIR   = ROOT / "data" / "unauthorized"

rows = []
for folder, approved in [(APPROVED_DIR, 1), (DENIED_DIR, 0)]:
    for p in af.list_audio_files(folder):
        y = af.load_clean(str(p))
        for aug in af.AUGMENTATIONS:
            rows.append({"source": p.name, "approved": approved,
                         **af.extract_features(af.augment(y, aug))})
        print(f"  loaded {folder.name}/{p.name}")

df = pd.DataFrame(rows)
feat_cols = [c for c in df.columns if c not in ("source", "approved")]
X, y = df[feat_cols].values.astype(float), df["approved"].values

# balance classes by repeating the (fewer) denied rows
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
idx = np.concatenate([pos] + [neg] * max(1, len(pos) // len(neg)))
Xb, yb = X[idx], y[idx]
mu, sd = Xb.mean(0), Xb.std(0) + 1e-9

model = RandomForestClassifier(n_estimators=150, random_state=42).fit((Xb - mu) / sd, yb)
pred = model.predict((X - mu) / sd)
print(f"\nTraining accuracy: {(pred == y).mean():.3f}  "
      f"({len(pos)} approved rows, {len(neg)} denied rows)")

joblib.dump({
    "model": model, "voice_labels": {0: "denied", 1: "approved"},
    "approved_class_idx": 1, "feature_cols": feat_cols,
    "scaler_mean": mu, "scaler_std": sd,
    "model_name": "Random Forest (pure-numpy, real strangers)",
    "test_metrics": {"train_accuracy": round(float((pred == y).mean()), 4)},
}, ROOT / "models" / "voice_model.pkl")
print("Saved models/voice_model.pkl")