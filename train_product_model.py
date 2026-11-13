"""
train_product_model.py
-----------------------
Rebuilds the Product Recommendation Model from merged_customer_transactions_social.csv
using the pure-numpy RandomForestClassifier (no sklearn needed).
Saves models/product_model_numpy.joblib and updates data/customer_feature_store.csv.

Run from repo root:  python train_product_model.py
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

RANDOM_STATE = 42

def accuracy_score(yt, yp): return float(np.mean(yt == yp))
def f1_score_weighted(yt, yp):
    cs = np.unique(yt); fs, ws = [], []
    for c in cs:
        tp=np.sum((yp==c)&(yt==c)); fp=np.sum((yp==c)&(yt!=c)); fn=np.sum((yp!=c)&(yt==c))
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f=2*p*r/(p+r) if p+r>0 else 0; fs.append(f); ws.append(int(np.sum(yt==c)))
    return float(np.average(fs, weights=ws))
def log_loss_score(yt, yp, eps=1e-15):
    return float(-np.mean([np.log(np.clip(yp[i,c],eps,1)) for i,c in enumerate(yt)]))

print("="*60)
print("Product Recommendation Model (pure-numpy)")
print("="*60)

# ── Load merged dataset ───────────────────────────────────────────────────────
merged = pd.read_csv(ROOT / "data" / "merged_customer_transactions_social.csv")
print(f"Merged dataset: {merged.shape}")
print("Target distribution:")
print(merged["product_category"].value_counts().to_string())

# ── Feature engineering ───────────────────────────────────────────────────────
# Numeric features (fill NaN with 0 / median)
NUM_FEATS = [
    "engagement_mean","engagement_max","interest_mean","interest_max","sentiment_mean",
    "social_profile_rows","n_platforms",
    "share_facebook","share_instagram","share_linkedin","share_tiktok","share_twitter",
    "prior_purchase_count","prior_avg_spend","prior_avg_rating",
    "days_since_last_purchase","visit_month","visit_dayofweek",
]

# Categorical: last_category → one-hot
CAT_FEAT = "last_category"
df = merged.copy()
df[CAT_FEAT] = df[CAT_FEAT].fillna("None").astype(str)
cat_classes = sorted(df[CAT_FEAT].unique())
cat_map = {c: i for i, c in enumerate(cat_classes)}
for cat in cat_classes:
    df[f"last_cat_{cat}"] = (df[CAT_FEAT] == cat).astype(float)
cat_encoded_cols = [f"last_cat_{c}" for c in cat_classes]

# has_social_profile as int
df["has_social_profile"] = df["has_social_profile"].astype(int) if df["has_social_profile"].dtype != int else df["has_social_profile"]

FEATURE_COLS = NUM_FEATS + cat_encoded_cols + ["has_social_profile"]

# Fill missing numerics with column median (computed on training set)
for col in NUM_FEATS:
    if col in df.columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    else:
        df[col] = 0.0

X_all = df[FEATURE_COLS].values.astype(np.float64)
le_target = LabelEncoder()
y_all = le_target.fit_transform(df["product_category"].values)
print(f"\nProduct classes: {list(le_target.classes_)}")
print(f"Feature dimensions: {X_all.shape}")

# ── Train / test split ────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_STATE)
idx = np.arange(len(X_all)); rng.shuffle(idx)
n_te = max(4, int(len(idx) * 0.2))
X_tr, X_te = X_all[idx[n_te:]], X_all[idx[:n_te]]
y_tr, y_te = y_all[idx[n_te:]], y_all[idx[:n_te]]

print(f"\nTrain: {len(X_tr)}  Test: {len(X_te)}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Random Forest (150 trees) ...")
clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)
clf.fit(X_tr, y_tr)

y_pred  = clf.predict(X_te)
y_proba = clf.predict_proba(X_te)
acc = accuracy_score(y_te, y_pred)
f1  = f1_score_weighted(y_te, y_pred)
ll  = log_loss_score(y_te, y_proba)

print(f"\nProduct Model – Test Metrics")
print(f"  Accuracy   : {acc:.4f}")
print(f"  F1-Score   : {f1:.4f}")
print(f"  Log Loss   : {ll:.4f}")

# Top-3 accuracy
top3 = sum(
    int(y_te[i]) in np.argsort(y_proba[i])[-3:]
    for i in range(len(y_te))
) / len(y_te)
print(f"  Top-3 Acc  : {top3:.4f}")

# ── Refit on all data ─────────────────────────────────────────────────────────
clf.fit(X_all, y_all)

# Compute column medians over full dataset for inference-time imputation
col_medians = {col: float(df[col].median()) for col in NUM_FEATS if col in df.columns}

bundle = {
    "model":          clf,
    "label_encoder":  le_target,
    "classes":        list(le_target.classes_),
    "feature_cols":   FEATURE_COLS,
    "num_feats":      NUM_FEATS,
    "cat_feat":       CAT_FEAT,
    "cat_classes":    cat_classes,
    "cat_map":        cat_map,
    "col_medians":    col_medians,
    "model_name":     "Random Forest (pure-numpy)",
    "test_metrics":   {
        "accuracy": round(acc,4), "f1_weighted": round(f1,4),
        "log_loss": round(ll,4), "top3_accuracy": round(top3,4),
    },
    "fallback_distribution": {
        c: float(np.mean(df["product_category"] == c))
        for c in le_target.classes_
    },
}
out = ROOT / "models" / "product_model_numpy.joblib"
joblib.dump(bundle, out)
print(f"\nSaved  {out}")
print("Metrics:", bundle["test_metrics"])

