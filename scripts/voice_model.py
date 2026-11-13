"""
scripts/voice_model.py
----------------------
Task 4 - Voiceprint Verification Model.

Trains a speaker classifier on audio_features.csv, evaluates it WITHOUT leakage
(leave-one-recording-out, grouped by `source_file`), adds a voiceprint-distance check
for rejecting unknown voices, and saves everything to models/voice_model.joblib.

Run from the repo root:
    python scripts/voice_model.py
    python scripts/voice_model.py --unauthorized-dir data/unauthorized

In the CLI app:
    from scripts.voice_model import load_bundle, verify_voice
    bundle = load_bundle()
    result = verify_voice("my_voice.wav", claimed_member="Esther", bundle=bundle)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             log_loss)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import audio_features as af  # noqa: E402

CSV_PATH = ROOT / "audio_features.csv"
MODEL_PATH = ROOT / "models" / "voice_model.joblib"
REPORT_DIR = ROOT / "reports"

META_COLS = ["member", "sample_id", "phrase", "source_file", "augmentation", "is_augmented"]
# Not speaker-specific: duration/energy depend on the phrase and recording volume,
# so keeping them would let the model "recognise" the phrase or the mic instead of the voice.
EXCLUDE_FEATURES = ["duration_s", "energy"]
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(csv_path: Path | str = CSV_PATH) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in META_COLS and c not in EXCLUDE_FEATURES]
    per_member = df.groupby("member")["source_file"].nunique()
    if (per_member < 2).any():
        bad = ", ".join(per_member[per_member < 2].index)
        raise ValueError(f"Need >= 2 recordings per member for leave-one-recording-out; too few for: {bad}")
    return df, feature_cols


def features_from_file(path: str | Path) -> dict:
    """Any audio file (ogg/aac/mp4/wav...) -> feature dict, same preprocessing as training."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = af.convert_to_wav(path, tmp)
        return af.extract_features_from_file(wav)


# --------------------------------------------------------------------------- #
# Models + evaluation
# --------------------------------------------------------------------------- #
def make_model(name: str):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    raise ValueError(name)


MODEL_NAMES = ["logistic_regression", "random_forest"]


def _metrics(y_true, proba, classes, mask=None) -> dict:
    y_true = np.asarray(y_true)
    if mask is not None:
        y_true, proba = y_true[mask], proba[mask]
    pred = classes[proba.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
        "n": int(len(y_true)),
    }


def cross_validate(df: pd.DataFrame, feature_cols: list[str], model_name: str) -> dict:
    """Leave-one-recording-out CV: every fold holds out ALL versions (original + augmented)
    of one recording, so augmented copies never leak between train and test."""
    X, y = df[feature_cols].values, df["member"].values
    groups = df["source_file"].values
    classes = np.array(sorted(set(y)))
    proba = cross_val_predict(make_model(model_name), X, y, groups=groups,
                              cv=LeaveOneGroupOut(), method="predict_proba")
    orig = ~df["is_augmented"].values
    return {
        "model": model_name,
        "all_rows": _metrics(y, proba, classes),
        "originals_only": _metrics(y, proba, classes, mask=orig),
        "proba": proba,
        "classes": classes,
    }


def compare_models(df, feature_cols) -> tuple[pd.DataFrame, dict]:
    results = {n: cross_validate(df, feature_cols, n) for n in MODEL_NAMES}
    table = pd.DataFrame([{
        "model": n,
        "accuracy": r["all_rows"]["accuracy"],
        "f1_macro": r["all_rows"]["f1_macro"],
        "log_loss": r["all_rows"]["log_loss"],
        "accuracy_originals_only": r["originals_only"]["accuracy"],
    } for n, r in results.items()]).sort_values(["f1_macro", "log_loss"], ascending=[False, True])
    return table.reset_index(drop=True), results


def plot_confusion(df, result, path: Path | None = None):
    y = df["member"].values
    pred = result["classes"][result["proba"].argmax(axis=1)]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=result["classes"], ax=ax,
                                            cmap="Blues", colorbar=False)
    ax.set_title(f"Voice model ({result['model']})\nleave-one-recording-out")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Rejecting unknown voices (voiceprint distance)
# --------------------------------------------------------------------------- #
# A closed-set classifier always picks its closest member, and its confidence is a poor
# stranger detector. So we also compare the voice to each member's enrolled "voiceprint"
# (mean of their standardised features) and reject anything too far from it.
MIN_CONFIDENCE = 0.5


def fit_template(df: pd.DataFrame, feature_cols: list[str]):
    scaler = StandardScaler().fit(df[feature_cols].values)
    Z = scaler.transform(df[feature_cols].values)
    centroids = {m: Z[(df["member"] == m).values].mean(axis=0) for m in sorted(df["member"].unique())}
    return scaler, centroids


def _dist(z: np.ndarray, centroid: np.ndarray) -> float:
    """RMS z-score distance between a clip and a voiceprint."""
    return float(np.sqrt(np.mean((z - centroid) ** 2)))


def genuine_distances(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Leave-one-recording-out: distance of each held-out ORIGINAL recording to its own member's
    voiceprint, built without that recording (so it mimics a genuine new attempt)."""
    out = []
    for src in df["source_file"].unique():
        test = df[(df["source_file"] == src) & (~df["is_augmented"])]
        scaler, cents = fit_template(df[df["source_file"] != src], feature_cols)
        z = scaler.transform(test[feature_cols].values)[0]
        out.append(_dist(z, cents[test["member"].iloc[0]]))
    return np.array(out)


def impostor_distances(scaler, centroids, feature_cols, unauthorized_dir) -> np.ndarray:
    """Distance of each stranger recording to its NEAREST member (a stranger would claim the easiest one)."""
    if not unauthorized_dir or not Path(unauthorized_dir).exists():
        return np.array([])
    files = af.list_audio_files(unauthorized_dir)
    if not files:
        return np.array([])
    X = pd.DataFrame([features_from_file(f) for f in files])[feature_cols].values
    Z = scaler.transform(X)
    return np.array([min(_dist(z, c) for c in centroids.values()) for z in Z])


def choose_distance_threshold(genuine: np.ndarray, impostor: np.ndarray | None = None) -> tuple[float, str]:
    """Maximum allowed distance to the claimed voiceprint.

    * With stranger samples: midpoint of the gap if genuine/impostor are separable, otherwise the value
      minimising false-reject + false-accept rate.
    * Without them: PROVISIONAL = 1.15 x the largest genuine held-out distance.
    """
    if impostor is not None and len(impostor) > 0:
        if genuine.max() < impostor.min():
            return float((genuine.max() + impostor.min()) / 2), "calibrated: genuine and unauthorized recordings are separable"
        grid = np.unique(np.concatenate([genuine, impostor]))
        cost = np.array([(genuine > t).mean() + (impostor <= t).mean() for t in grid])
        tied = grid[np.isclose(cost, cost.min())]
        return float(np.median(tied)), "calibrated: genuine and unauthorized overlap (best trade-off)"
    return float(genuine.max() * 1.15), "PROVISIONAL (no unauthorized recordings supplied) - re-run with --unauthorized-dir"


# --------------------------------------------------------------------------- #
# Train / save / load / verify
# --------------------------------------------------------------------------- #
def train_and_save(csv_path=CSV_PATH, model_path=MODEL_PATH, unauthorized_dir=None,
                   report_dir=REPORT_DIR, verbose=True) -> dict:
    df, feature_cols = load_data(csv_path)
    table, results = compare_models(df, feature_cols)
    best_name = table.iloc[0]["model"]
    best = results[best_name]

    final = make_model(best_name).fit(df[feature_cols].values, df["member"].values)

    scaler, centroids = fit_template(df, feature_cols)
    gen = genuine_distances(df, feature_cols)
    imp = impostor_distances(scaler, centroids, feature_cols, unauthorized_dir)
    threshold, how = choose_distance_threshold(gen, imp)

    bundle = {
        "model": final, "feature_cols": feature_cols, "classes": list(best["classes"]),
        "scaler": scaler, "centroids": centroids,
        "distance_threshold": threshold, "distance_note": how, "min_confidence": MIN_CONFIDENCE,
        "model_name": best_name,
        "cv_metrics": {"all_rows": best["all_rows"], "originals_only": best["originals_only"]},
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "voice_model_comparison.csv", index=False)
    plot_confusion(df, best, report_dir / "voice_confusion_matrix.png")
    summary = {
        "selected_model": best_name,
        "cv_all_rows": best["all_rows"], "cv_originals_only": best["originals_only"],
        "distance_threshold": threshold, "distance_note": how,
        "genuine_distances": [round(float(d), 3) for d in gen],
        "unauthorized_distances": [round(float(d), 3) for d in imp],
        "genuine_accepted_pct": float((gen <= threshold).mean() * 100),
        "unauthorized_rejected_pct": (float((imp > threshold).mean() * 100) if len(imp) else None),
    }
    (report_dir / "voice_model_metrics.json").write_text(json.dumps(summary, indent=2))

    if verbose:
        print(table.to_string(index=False))
        print(f"\nSelected: {best_name}")
        print("CV (all rows):       ", best["all_rows"])
        print("CV (originals only): ", best["originals_only"])
        print(f"Distance threshold: {threshold:.3f}  [{how}]")
        print(f"Genuine held-out recordings accepted: {summary['genuine_accepted_pct']:.0f}% of {len(gen)}")
        if len(imp):
            print(f"Unauthorized recordings rejected: {summary['unauthorized_rejected_pct']:.0f}% of {len(imp)}")
        print(f"Saved model -> {model_path}")
    return bundle


def load_bundle(model_path: Path | str = MODEL_PATH) -> dict:
    return joblib.load(model_path)


def verify_voice(audio_path: str | Path, claimed_member: str, bundle: dict) -> dict:
    """Decide whether the voice in `audio_path` belongs to `claimed_member`
    (the person the face model just recognised).

    Approved only if the voice is close to the claimed member's voiceprint, the classifier
    agrees it is that member, and its confidence is at least `min_confidence`.
    """
    feats = pd.DataFrame([features_from_file(audio_path)])[bundle["feature_cols"]]   # training column order
    proba = bundle["model"].predict_proba(feats.values)[0]
    z = bundle["scaler"].transform(feats.values)[0]
    classes, thr = bundle["classes"], bundle["distance_threshold"]
    pred, conf = classes[int(np.argmax(proba))], float(np.max(proba))
    dists = {m: _dist(z, c) for m, c in bundle["centroids"].items()}
    nearest = min(dists, key=dists.get)
    claimed = next((c for c in classes if c.lower() == str(claimed_member).lower()), None)

    if claimed is None:
        approved, reason = False, f"'{claimed_member}' is not an enrolled voice"
    elif dists[nearest] > thr:
        approved, reason = False, "unknown voice (not close to any enrolled member)"
    elif pred != claimed:
        approved, reason = False, f"voice matches {pred}, not {claimed}"
    elif dists[claimed] > thr:
        approved, reason = False, f"voice too different from {claimed}'s enrolled voiceprint"
    elif conf < bundle["min_confidence"]:
        approved, reason = False, "classifier confidence too low"
    else:
        approved, reason = True, "voice matches claimed identity"
    return {"approved": approved, "predicted_member": pred, "confidence": round(conf, 3),
            "claimed_member": claimed_member, "reason": reason,
            "distance_to_claimed": (round(dists[claimed], 3) if claimed else None),
            "distance_threshold": round(thr, 3),
            "scores": {c: round(float(p), 3) for c, p in zip(classes, proba)}}


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the voiceprint verification model")
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--unauthorized-dir", default=str(ROOT / "data" / "unauthorized"),
                    help="folder of recordings from people NOT in the team (optional)")
    args = ap.parse_args()
    train_and_save(args.csv, MODEL_PATH, args.unauthorized_dir)
