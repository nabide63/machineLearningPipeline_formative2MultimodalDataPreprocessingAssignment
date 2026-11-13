"""
scripts/train_face_model.py
----------------------------
Trains the Face Recognition Model from image_features.csv
and saves it as models/face_model.pkl.

Members: Laura, Esther, Rwema, Yusuf (Nabide)

Usage:
    python scripts/train_face_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[1]
FEAT_CSV   = ROOT / "image_features.csv"
MODEL_PATH = ROOT / "models" / "face_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ── Load features ─────────────────────────────────────────────────────────────
df = pd.read_csv(FEAT_CSV)
print(f"Loaded {len(df)} rows from {FEAT_CSV.name}")
print("Members:", df["member"].value_counts().to_dict())

# Feature columns: all RGB histogram bins
meta_cols = ["member", "filename", "image_type"]
feat_cols  = [c for c in df.columns if c not in meta_cols]

X = df[feat_cols].values
le = LabelEncoder()
y = le.fit_transform(df["member"])

print(f"Classes: {list(le.classes_)}")
print(f"Feature dimensions: {X.shape}")

# ── Train / test split (stratified) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ── Model comparison ──────────────────────────────────────────────────────────
candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None,
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                              C=1.0, solver="lbfgs", multi_class="multinomial"),
}

print("\n── Cross-validation (5-fold) ─────────────────────────────────────")
best_name, best_score, best_model = None, -1, None
for name, clf in candidates.items():
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    mean_cv   = cv_scores.mean()
    print(f"  {name:<25} CV Accuracy = {mean_cv:.4f}  (±{cv_scores.std():.4f})")
    if mean_cv > best_score:
        best_score, best_name, best_model = mean_cv, name, clf

print(f"\nBest model: {best_name}")

# ── Final evaluation on hold-out test set ────────────────────────────────────
best_model.fit(X_train, y_train)
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average="weighted")
loss = log_loss(y_test, y_proba) if y_proba is not None else float("nan")

print(f"\n── Test-set metrics ({best_name}) ──────────────────────────────────")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  Log Loss : {loss:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Refit on ALL data before saving ──────────────────────────────────────────
best_model.fit(X, y)

# Build label map  {class_index: member_name}
face_labels = {int(i): str(name) for i, name in enumerate(le.classes_)}
print("Label map:", face_labels)

bundle = {
    "model":        best_model,
    "label_encoder": le,
    "face_labels":  face_labels,
    "feature_cols": feat_cols,
    "model_name":   best_name,
    "test_metrics": {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "log_loss": round(loss, 4)},
}
joblib.dump(bundle, MODEL_PATH)
print(f"\nSaved face model → {MODEL_PATH}")

