"""
audio_pipeline_numpy.py
------------------------
Pure-numpy/scipy/soundfile audio loading, augmentation and feature
extraction. Replaces librosa entirely for training (no numba DLLs needed).

Implements:
  load_audio()       - load any soundfile-supported format to mono float32
  extract_features() - MFCCs + spectral features using numpy/scipy
  augment()          - pitch, stretch, noise augmentations
  build_dataset()    - full pipeline over a folder of recordings
"""
import re
import io
import subprocess
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import fft, dct
from scipy.signal import resample_poly
from math import gcd

SR     = 16_000
N_MFCC = 13
N_FFT  = 1024
HOP    = 256
N_MELS = 40

PHRASE_MAP = {1: "Yes, approve", 2: "Confirm transaction"}

AUDIO_EXTS = {".ogg", ".opus", ".wav", ".flac", ".aiff", ".aif", ".au"}
# formats needing ffmpeg
FFMPEG_EXTS = {".aac", ".mp4", ".m4a", ".mp3", ".webm", ".opus"}


# ── File helpers ──────────────────────────────────────────────────────────────

def parse_filename(path):
    """'Rwema 2.ogg' → ('Rwema', 2)"""
    stem = Path(path).stem
    m = re.match(r"^(.+?)\s*(\d+)\s*$", stem.strip())
    if not m:
        return stem.strip().title(), 1
    return m.group(1).strip().title(), int(m.group(2))


def list_audio_files(audio_dir):
    p = Path(audio_dir)
    all_exts = AUDIO_EXTS | FFMPEG_EXTS
    return sorted(f for f in p.iterdir() if f.suffix.lower() in all_exts)


# ── Loading ───────────────────────────────────────────────────────────────────

def _resample_numpy(y, orig_sr, target_sr):
    """Resample using scipy.signal.resample_poly (gcd-reduced rational ratio)."""
    if orig_sr == target_sr:
        return y
    g   = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(y.astype(np.float64), up, down).astype(np.float32)


def load_audio(path, sr=SR):
    """Load any supported audio file to mono float32 at `sr` Hz.

    For formats soundfile can't handle natively (.aac, .mp4, .m4a, .mp3)
    we try ffmpeg first; if ffmpeg is absent we raise a helpful error.
    """
    path = Path(path)
    ext  = path.suffix.lower()

    if ext in FFMPEG_EXTS:
        if shutil.which("ffmpeg") is None:
            # Try soundfile anyway — newer libsndfile builds support some formats
            try:
                data, file_sr = sf.read(str(path), always_2d=False)
            except Exception:
                raise RuntimeError(
                    f"Cannot decode '{path.name}': ffmpeg not found and soundfile "
                    f"does not support {ext}. Install ffmpeg or convert to WAV/OGG."
                )
        else:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(path),
                 "-f", "s16le", "-ac", "1", "-ar", str(sr), "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
            data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return data
    else:
        data, file_sr = sf.read(str(path), always_2d=False)

    # Convert stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # Resample if needed
    if file_sr != sr:
        data = _resample_numpy(data, file_sr, sr)
    return data


def peak_normalize(y, peak=0.95):
    m = np.max(np.abs(y))
    return y if m == 0 else (y / m * peak).astype(np.float32)


def trim_silence(y, top_db=30, frame_len=512):
    """Trim leading/trailing silence (energy-based)."""
    energy   = np.array([
        np.sqrt(np.mean(y[i:i+frame_len]**2))
        for i in range(0, len(y)-frame_len, frame_len//2)
    ])
    threshold = 10 ** (-top_db / 20)
    frames_ok = energy > threshold
    if not frames_ok.any():
        return y
    first = np.argmax(frames_ok) * (frame_len // 2)
    last  = (len(frames_ok) - np.argmax(frames_ok[::-1])) * (frame_len // 2)
    trimmed = y[first:last]
    return trimmed if len(trimmed) > N_FFT * 2 else y


def load_clean(path, sr=SR):
    y = load_audio(path, sr=sr)
    y = trim_silence(y)
    if len(y) < N_FFT * 2:
        y = np.pad(y, (0, N_FFT * 2 - len(y)))
    return peak_normalize(y)


# ── Augmentations ─────────────────────────────────────────────────────────────

def _pitch_shift(y, n_steps, sr=SR):
    """Shift pitch by `n_steps` semitones via resampling."""
    factor = 2 ** (n_steps / 12.0)
    # stretch then resample back to original length
    orig_len = len(y)
    stretched = resample_poly(y, 1000, int(round(1000 * factor)))
    # trim or pad to original length
    if len(stretched) >= orig_len:
        return stretched[:orig_len].astype(np.float32)
    return np.pad(stretched, (0, orig_len - len(stretched))).astype(np.float32)


def _time_stretch(y, rate):
    """Stretch/compress time by `rate` (>1 = faster, <1 = slower)."""
    orig_len = len(y)
    new_len  = int(orig_len / rate)
    stretched = resample_poly(y, new_len, orig_len)
    return stretched.astype(np.float32)


def _add_noise(y, snr_db=20, seed=42):
    rng = np.random.default_rng(seed)
    sig_power   = np.mean(y**2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(noise_power), y.shape)).astype(np.float32)


AUGMENTATIONS = {
    "original":          lambda y: y,
    "pitch_up":          lambda y: _pitch_shift(y, +2),
    "pitch_down":        lambda y: _pitch_shift(y, -2),
    "time_stretch_fast": lambda y: _time_stretch(y, 1.15),
    "time_stretch_slow": lambda y: _time_stretch(y, 0.85),
    "background_noise":  lambda y: _add_noise(y, snr_db=20),
}


def augment(y, name):
    out = AUGMENTATIONS[name](y)
    return np.clip(peak_normalize(out), -1.0, 1.0).astype(np.float32)


# ── Feature extraction (pure numpy/scipy) ─────────────────────────────────────

def _mel_filterbank(sr, n_fft, n_mels=N_MELS, fmin=0.0, fmax=None):
    """Return (n_mels, n_fft//2+1) triangular mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2.0
    # Hz → mel
    hz_to_mel  = lambda f: 2595 * np.log10(1 + f / 700)
    mel_to_hz  = lambda m: 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels    = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs   = mel_to_hz(mels)

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        f_lo, f_mid, f_hi = freqs[i-1], freqs[i], freqs[i+1]
        up   = np.where((fft_freqs >= f_lo) & (fft_freqs <= f_mid),
                        (fft_freqs - f_lo) / (f_mid - f_lo + 1e-12), 0)
        down = np.where((fft_freqs >= f_mid) & (fft_freqs <= f_hi),
                        (f_hi - fft_freqs) / (f_hi - f_mid + 1e-12), 0)
        fb[i-1] = up + down
    return fb


_FB_CACHE = {}

def _get_filterbank(sr, n_fft, n_mels):
    key = (sr, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(sr, n_fft, n_mels)
    return _FB_CACHE[key]


def extract_features(y, sr=SR):
    """
    Compute MFCC + spectral features from a waveform.

    Returns a flat dict matching the columns produced by audio_features.py.
    Features: mfcc_1..13 (mean+std), rolloff, centroid, bandwidth,
              flatness, zcr, rms (mean+std), energy, duration_s.
    """
    n_fft, hop = N_FFT, HOP
    win = np.hanning(n_fft)

    # ── STFT power spectrum ─────────────────────────────
    n_frames = 1 + (len(y) - n_fft) // hop
    frames   = np.array([y[i*hop : i*hop + n_fft] * win
                         for i in range(n_frames)])            # (T, n_fft)
    spec     = np.abs(fft(frames, n=n_fft, axis=1))[:, :n_fft//2+1] ** 2  # power

    freqs    = np.linspace(0, sr/2, n_fft//2+1)
    eps      = 1e-12

    # ── MFCCs ───────────────────────────────────────────
    fb      = _get_filterbank(sr, n_fft, N_MELS)
    mel_e   = np.dot(spec, fb.T)                              # (T, n_mels)
    log_mel = np.log(mel_e + eps)
    mfcc    = dct(log_mel, type=2, axis=1, norm='ortho')[:, :N_MFCC]  # (T, 13)

    f = {}
    for i in range(N_MFCC):
        f[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[:, i]))
        f[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[:, i]))

    # ── Spectral features ────────────────────────────────
    total_power = spec.sum(axis=1, keepdims=True) + eps

    # roll-off (85%)
    cum = np.cumsum(spec, axis=1)
    thr = 0.85 * total_power
    rolloff = freqs[np.argmax(cum >= thr, axis=1).clip(0, len(freqs)-1)]
    f["rolloff_mean"] = float(np.mean(rolloff))
    f["rolloff_std"]  = float(np.std(rolloff))

    # centroid
    centroid = (spec * freqs).sum(axis=1) / total_power.squeeze()
    f["centroid_mean"] = float(np.mean(centroid))
    f["centroid_std"]  = float(np.std(centroid))

    # bandwidth
    diff = (freqs - centroid[:, None]) ** 2
    bw   = np.sqrt((spec * diff).sum(axis=1) / total_power.squeeze())
    f["bandwidth_mean"] = float(np.mean(bw))
    f["bandwidth_std"]  = float(np.std(bw))

    # flatness: geometric / arithmetic mean of power
    log_mean = np.mean(np.log(spec + eps), axis=1)
    arith    = np.mean(spec + eps, axis=1)
    flatness = np.exp(log_mean) / arith
    f["flatness_mean"] = float(np.mean(flatness))
    f["flatness_std"]  = float(np.std(flatness))

    # zero-crossing rate
    zcr_frames = np.array([
        np.sum(np.diff(np.sign(y[i*hop : i*hop + n_fft])) != 0) / n_fft
        for i in range(n_frames)
    ])
    f["zcr_mean"] = float(np.mean(zcr_frames))
    f["zcr_std"]  = float(np.std(zcr_frames))

    # RMS energy
    rms_frames = np.sqrt(np.mean(
        np.array([y[i*hop : i*hop + n_fft]**2 for i in range(n_frames)]),
        axis=1))
    f["rms_mean"] = float(np.mean(rms_frames))
    f["rms_std"]  = float(np.std(rms_frames))

    # total energy and duration
    f["energy"]     = float(np.sum(y**2))
    f["duration_s"] = float(len(y) / sr)
    return f


def extract_features_from_file(path, sr=SR):
    """Convenience wrapper for the CLI app."""
    return extract_features(load_clean(str(path), sr=sr), sr=sr)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_dataset(audio_dir, aug_dir=None, phrase_map=PHRASE_MAP):
    """
    Load, clean, augment and extract features for every recording.
    Returns a DataFrame with one row per (source × augmentation).
    """
    import pandas as pd
    if aug_dir:
        Path(aug_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for src in list_audio_files(audio_dir):
        member, n = parse_filename(src)
        try:
            y = load_clean(src)
        except Exception as e:
            print(f"  [WARN] Could not load {src.name}: {e} — skipping")
            continue

        for aug_name, aug_fn in AUGMENTATIONS.items():
            y_aug = augment(y, aug_name)
            if aug_dir:
                out = Path(aug_dir) / f"{src.stem.replace(' ','_')}__{aug_name}.wav"
                sf.write(str(out), y_aug, SR)
            rows.append({
                "member":      member,
                "sample_id":   n,
                "phrase":      phrase_map.get(n, f"phrase_{n}"),
                "source_file": src.name,
                "augmentation": aug_name,
                "is_augmented": aug_name != "original",
                **extract_features(y_aug),
            })

    return pd.DataFrame(rows)

