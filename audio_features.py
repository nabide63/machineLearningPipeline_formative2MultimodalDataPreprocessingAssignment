"""
audio_features.py
-----------------
Audio loading, augmentation and feature extraction for Formative 2
(Multimodal Data Preprocessing).

Used by:
  * task3_audio_processing.ipynb  -> builds audio_features.csv
  * the command-line app (Task 6) -> extract_features() on a live / test voice sample
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SR = 16_000          # common sample rate for every clip
N_MFCC = 13          # number of MFCC coefficients
N_FFT = 1024         # short window: clips are only a second or two long
HOP = 256
AUDIO_EXTS = {".ogg", ".opus", ".aac", ".mp4", ".m4a", ".mp3", ".wav", ".flac", ".webm"}

# Which phrase each recording number corresponds to ("<Name> 1", "<Name> 2").
# Edit this if the team recorded them in a different order.
PHRASE_MAP = {1: "Yes, approve", 2: "Confirm transaction"}


# --------------------------------------------------------------------------- #
# File handling
# --------------------------------------------------------------------------- #
def parse_filename(path: str | Path) -> tuple[str, int]:
    """'Rwema 2.ogg' -> ('Rwema', 2).  Falls back to (stem, 1) if no number."""
    stem = Path(path).stem
    m = re.match(r"^\s*(?P<member>.+?)[\s_\-]*(?P<n>\d+)\s*$", stem)
    if not m:
        return stem.strip().title(), 1
    return m["member"].strip().title(), int(m["n"])


def list_audio_files(audio_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(audio_dir).iterdir() if p.suffix.lower() in AUDIO_EXTS)


def convert_to_wav(src: str | Path, dst_dir: str | Path, sr: int = SR) -> Path:
    """Convert any supported container (ogg/aac/mp4/...) to mono 16 kHz WAV via ffmpeg.

    Mixed formats (.ogg, .aac, .mp4) are the main reason loading fails on some
    machines, so we normalise everything to WAV once, up front.
    """
    src, dst_dir = Path(src), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (src.stem.replace(" ", "_") + ".wav")
    if shutil.which("ffmpeg") is None:
        # Fallback: let librosa/soundfile try to decode it directly.
        y, _ = librosa.load(src, sr=sr, mono=True)
        sf.write(dst, y, sr)
        return dst
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(sr), str(dst)],
        check=True,
    )
    return dst


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(y)) if len(y) else 0.0
    return y if m == 0 else (y / m * peak).astype(np.float32)


def load_clean(path: str | Path, sr: int = SR, top_db: float = 30.0) -> np.ndarray:
    """Load -> mono -> trim leading/trailing silence -> peak-normalise."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trim) >= N_FFT:          # only keep the trim if something sensible is left
        y = y_trim
    if len(y) < 2 * N_FFT:            # pad extremely short clips so STFT works
        y = np.pad(y, (0, 2 * N_FFT - len(y)))
    return peak_normalize(y)


# --------------------------------------------------------------------------- #
# Augmentations
# --------------------------------------------------------------------------- #
def pitch_shift(y: np.ndarray, n_steps: float, sr: int = SR) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)


def add_background_noise(y: np.ndarray, snr_db: float = 20.0, seed: int = 42) -> np.ndarray:
    """Add white Gaussian noise at a given signal-to-noise ratio (dB)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(y ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    return y + rng.normal(0.0, np.sqrt(noise_power), size=y.shape)


AUGMENTATIONS = {
    "original":        lambda y: y,
    "pitch_up":        lambda y: pitch_shift(y, +2),
    "pitch_down":      lambda y: pitch_shift(y, -2),
    "time_stretch_fast": lambda y: time_stretch(y, 1.15),
    "time_stretch_slow": lambda y: time_stretch(y, 0.85),
    "background_noise": lambda y: add_background_noise(y, snr_db=20),
}


def augment(y: np.ndarray, name: str) -> np.ndarray:
    out = AUGMENTATIONS[name](y)
    return np.clip(peak_normalize(out), -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(y: np.ndarray, sr: int = SR) -> dict:
    """Return a flat dict of numeric features for one clip.

    MFCC mean/std (13 each), spectral roll-off, centroid, bandwidth, flatness,
    zero-crossing rate, RMS energy, total energy and duration.
    """
    f: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    for i in range(N_MFCC):
        f[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        f[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    spec_kw = dict(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    rolloff = librosa.feature.spectral_rolloff(**spec_kw, roll_percent=0.85)[0]
    centroid = librosa.feature.spectral_centroid(**spec_kw)[0]
    bandwidth = librosa.feature.spectral_bandwidth(**spec_kw)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]

    for name, arr in [("rolloff", rolloff), ("centroid", centroid), ("bandwidth", bandwidth),
                      ("flatness", flatness), ("zcr", zcr), ("rms", rms)]:
        f[f"{name}_mean"] = float(np.mean(arr))
        f[f"{name}_std"] = float(np.std(arr))

    f["energy"] = float(np.sum(y ** 2))
    f["duration_s"] = float(len(y) / sr)
    return f


def extract_features_from_file(path: str | Path, sr: int = SR) -> dict:
    """Convenience wrapper for the CLI app: file path -> feature dict (no augmentation)."""
    return extract_features(load_clean(path, sr=sr), sr=sr)


# --------------------------------------------------------------------------- #
# Whole pipeline
# --------------------------------------------------------------------------- #
def build_audio_dataset(
    audio_dir: str | Path,
    wav_dir: str | Path,
    aug_dir: str | Path,
    phrase_map: dict[int, str] = PHRASE_MAP,
    augmentations: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Convert -> clean -> augment -> extract features for every recording.

    Returns (features_df, clips) where `clips` maps '<source>|<aug>' to the waveform.
    """
    augmentations = augmentations or AUGMENTATIONS
    aug_dir = Path(aug_dir)
    aug_dir.mkdir(parents=True, exist_ok=True)

    rows, clips = [], {}
    for src in list_audio_files(audio_dir):
        member, n = parse_filename(src)
        wav = convert_to_wav(src, wav_dir)
        y = load_clean(wav)
        for aug_name in augmentations:
            y_aug = augment(y, aug_name)
            out = aug_dir / f"{wav.stem}__{aug_name}.wav"
            sf.write(out, y_aug, SR)
            clips[f"{src.name}|{aug_name}"] = y_aug
            rows.append({
                "member": member,
                "sample_id": n,
                "phrase": phrase_map.get(n, f"phrase_{n}"),
                "source_file": src.name,       # keep augmented copies grouped with their source
                "augmentation": aug_name,
                "is_augmented": aug_name != "original",
                **extract_features(y_aug),
            })
    return pd.DataFrame(rows), clips
