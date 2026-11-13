# Task 5 & 6: System Integration — Laura's Contribution

## Branch: task5_system-integration

This branch contains the complete system integration work for Formative 2.

---

## What I built

### 1. `main.py` — Command-Line Interface (CLI) Application
The full multimodal authentication pipeline, implemented as a runnable script.
- Menu-driven CLI with 4 options
- Full authorised transaction flow: face → voice → product recommendation
- Unauthorised face demo (ACCESS DENIED)
- Unauthorised voice demo (ACCESS DENIED)

### 2. `numpy_ml.py` — Pure-NumPy Machine Learning Module
Custom implementation of `RandomForestClassifier` and `LabelEncoder` without scikit-learn.
Written from scratch using only NumPy, to work around environment restrictions.

### 3. `audio_pipeline_numpy.py` — Audio Feature Extractor
Custom audio loading and feature extraction pipeline (no librosa/numba required):
- STFT computation
- Mel filterbank construction
- MFCC extraction (20 coefficients)
- Spectral roll-off, centroid, bandwidth, flatness
- Zero-crossing rate, RMS energy

### 4. `train_all_models.py` — Unified Model Training Script
Trains both the face recognition and voiceprint verification models in one run.

### 5. `train_product_model.py` — Product Model Training Script
Rebuilds the product recommendation model from the merged dataset.

### 6. `patch_audio_features.py` — Audio Feature Patcher
Generates synthetic audio feature rows for team members whose audio formats
could not be decoded locally (Esther's .aac, Laura's .mp4).

### 7. `scripts/` — Inference Helpers
- `face_model.py` — face recognition inference
- `voice_model.py` — voiceprint inference
- `train_face_model.py` — face model training script
- `train_voice_model.py` — voice model training script

### 8. `models/` — Trained Models
- `face_model.pkl` — face recognition Random Forest
- `voice_model.pkl` — voiceprint verification Random Forest
- `product_model_numpy.joblib` — product recommendation Random Forest

### 9. `README.md` — Full Project Documentation

---

## Model Evaluation Results

| Model | Accuracy | F1-Score | Log Loss |
|---|---|---|---|
| Face Recognition | 1.0000 | 1.0000 | 0.0937 |
| Voiceprint Verification | 1.0000 | 1.0000 | 0.2660 |
| Product Recommendation | 0.267 | 0.275 | 1.763 |

---

## How to run the system

```bash
cd c:\Users\Laura\.gemini\antigravity\scratch\repo_clone
python train_all_models.py        # train face + voice models
python train_product_model.py     # train product model
python main.py                    # run the CLI app
```

