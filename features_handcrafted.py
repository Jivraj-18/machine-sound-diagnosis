"""Handcrafted features: Welch band energies + MFCC + spectral/time-domain statistics.

Usage: python features_handcrafted.py [--sr 50000]
Writes cache/emb_handcrafted_sr{sr//1000}k.npy (n_clips, n_features).
"""
import argparse

import librosa
import numpy as np
import scipy.signal as ss
import scipy.stats as st
from joblib import Parallel, delayed

from common import CACHE_DIR, ResourceMonitor, load_dataset

N_BANDS = 64
F_MAX_NORM = 0.32  # signal content ends ~0.31 cycles/sample (see REPORT.md, sample-rate section)


def clip_features(x, sr):
    x = x.astype(np.float32)
    # Welch PSD -> log energy in N_BANDS linear bands up to F_MAX_NORM * sr
    f, P = ss.welch(x, fs=sr, nperseg=2048)
    edges = np.linspace(0, F_MAX_NORM * sr, N_BANDS + 1)
    bands = [np.log(P[(f >= lo) & (f < hi)].sum() + 1e-12) for lo, hi in zip(edges[:-1], edges[1:])]
    # MFCC + deltas, mean/std over frames
    kw = dict(n_fft=2048, hop_length=512)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, n_mels=64, fmax=F_MAX_NORM * sr, **kw)
    dm = librosa.feature.delta(mfcc)
    S = np.abs(librosa.stft(x, **kw))
    spec = np.vstack([
        librosa.feature.spectral_centroid(S=S, sr=sr),
        librosa.feature.spectral_bandwidth(S=S, sr=sr),
        librosa.feature.spectral_rolloff(S=S, sr=sr),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.spectral_contrast(S=S, sr=sr, fmin=200.0 * sr / 50000),
        librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512),
        librosa.feature.rms(S=S, frame_length=2048),
    ])
    rms = np.sqrt(np.mean(x**2))
    time_stats = [rms, x.std(), st.skew(x), st.kurtosis(x), np.abs(x).max() / rms]
    return np.concatenate([bands, mfcc.mean(1), mfcc.std(1), dm.mean(1), dm.std(1),
                           spec.mean(1), spec.std(1), time_stats]).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=50000)
    args = ap.parse_args()
    X, y, _, _ = load_dataset()
    with ResourceMonitor(f"handcrafted_sr{args.sr // 1000}k", len(X), n_jobs=8):
        F = np.stack(Parallel(n_jobs=8)(delayed(clip_features)(x, args.sr) for x in X))
    out = CACHE_DIR / f"emb_handcrafted_sr{args.sr // 1000}k.npy"
    np.save(out, F)
    print(out, F.shape, "non-finite:", (~np.isfinite(F)).sum())
