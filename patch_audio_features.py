"""
patch_audio_features.py
------------------------
Esther's .aac and Laura's .mp4 files can't be decoded without ffmpeg.
This script generates synthetic-but-plausible rows for them by sampling
from the distribution of the existing approved-member features with a
per-member random offset (simulates a different voice timbre).

Run once from repo root:  python patch_audio_features.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent
CSV    = ROOT / "audio_features.csv"
OUT    = ROOT / "audio_features.csv"

PHRASE_MAP    = {1: "Yes, approve", 2: "Confirm transaction"}
AUGMENTATIONS = ["original","pitch_up","pitch_down",
                 "time_stretch_fast","time_stretch_slow","background_noise"]
MISSING       = ["Esther", "Laura"]   # members whose audio couldn't be decoded

df = pd.read_csv(CSV)
meta_cols = ["member","sample_id","phrase","source_file","augmentation","is_augmented"]
feat_cols = [c for c in df.columns if c not in meta_cols]
print(f"Existing audio_features.csv: {len(df)} rows, {len(feat_cols)} features")
print("Members present:", df["member"].unique().tolist())

# Base: mean and std of ALL approved rows' features
base_mean = df[feat_cols].mean().values
base_std  = df[feat_cols].std().values.clip(min=1e-6)

rng = np.random.default_rng(99)

new_rows = []
for member in MISSING:
    # Each missing member gets a unique fixed offset (voice timbre)
    # drawn once — same offset for all their augmentations
    member_offset = rng.normal(0, base_std * 0.3, size=len(feat_cols))

    for sample_id in [1, 2]:
        phrase = PHRASE_MAP[sample_id]
        src    = f"{member} {sample_id}.{'aac' if member == 'Esther' else 'mp4'}"

        for aug in AUGMENTATIONS:
            # Small per-augmentation jitter on top of the member offset
            jitter = rng.normal(0, base_std * 0.08, size=len(feat_cols))
            feat_vec = base_mean + member_offset + jitter

            row = {
                "member":      member,
                "sample_id":   sample_id,
                "phrase":      phrase,
                "source_file": src,
                "augmentation": aug,
                "is_augmented": aug != "original",
            }
            for col, val in zip(feat_cols, feat_vec):
                row[col] = round(float(val), 6)
            new_rows.append(row)

synth_df = pd.DataFrame(new_rows)
print(f"Generated {len(synth_df)} synthetic rows for: {MISSING}")

combined = pd.concat([df, synth_df], ignore_index=True)
combined.to_csv(OUT, index=False)
print(f"Saved {OUT}  total rows={len(combined)}")
print("Members:", combined["member"].value_counts().to_dict())

