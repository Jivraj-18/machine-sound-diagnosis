"""Look at the misclassified few-shot clips: are they atypical for their own class or borderline?
Also detects the mid-clip splice click and computes class-mean spectral distances.

Usage: python analyze_misses.py
Writes to results/investigation/:
  misclassified_spectra.png          spectrum vs class means + spectrogram, one row per miss
  misclassified_diagnostics.csv      per-miss distances, loudness, crank rate, click flag
  class_spectral_distance.csv        RMS dB distance between class-mean spectra (0-16 kHz)
  splice_click_clips.csv             every clip with the click at sample 25,475
"""
import re

import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss

from common import CLASSES, RESULTS_DIR, load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = RESULTS_DIR / "investigation"


def crank_cycles_per_clip(x):
    """Dominant envelope frequency in cycles per clip (mechanical cycle; ~7.3 for most clips)."""
    sos = ss.butter(4, [0.02, 0.12], "bandpass", output="sos", fs=1.0)
    env = np.abs(ss.hilbert(ss.sosfiltfilt(sos, x)))
    env = ss.decimate(env, 50)
    env -= env.mean()
    n = len(env)
    spec = np.abs(np.fft.rfft(env * np.hanning(n), n=16 * n)) ** 2
    f = np.fft.rfftfreq(16 * n) * n
    m = (f > 4) & (f < 12)
    return f[m][spec[m].argmax()]


def splice_click_flags(X, ratio=20):
    """True where 8-13 kHz (at 50 kHz) energy in the 10 ms windows around the clip midpoint is
    >ratio x the clip's median window energy: the step discontinuity at sample 25,475."""
    sos = ss.butter(6, [0.16, 0.26], "bandpass", output="sos", fs=1.0)
    H = ss.sosfiltfilt(sos, X, axis=1) ** 2
    win = 500
    E = H[:, : H.shape[1] // win * win].reshape(len(X), -1, win).mean(2)
    return E[:, 49:51].max(1) / np.median(E, 1) > ratio


def main():
    X, y, reading, _ = load_dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    f, P = ss.welch(X, fs=50000, nperseg=2048, axis=1)
    logP = 10 * np.log10(P + 1e-20)
    keep = f < 16000
    class_mean = np.stack([logP[y == c].mean(0) for c in range(len(CLASSES))])
    class_std = np.stack([logP[y == c].std(0) for c in range(len(CLASSES))])
    rms = np.sqrt((X ** 2).mean(1))
    click = splice_click_flags(X)

    D = np.sqrt(((class_mean[:, None, keep] - class_mean[None, :, keep]) ** 2).mean(-1))
    pd.DataFrame(D.round(2), index=CLASSES, columns=CLASSES).to_csv(OUT / "class_spectral_distance.csv")
    pd.DataFrame({"class": [CLASSES[c] for c in y[click]], "reading": reading[click]}).to_csv(
        OUT / "splice_click_clips.csv", index=False)
    print("splice click:", int(click.sum()), "clips;",
          {CLASSES[c]: int(click[y == c].sum()) for c in range(len(CLASSES))})

    rows = []
    for p in sorted((RESULTS_DIR / "misclassified").glob("*.wav")):
        t, pr, r = re.match(r"true-(\w+)_pred-(\w+)_Reading(\d+)", p.stem).groups()
        ti, pi = CLASSES.index(t), CLASSES.index(pr)
        i = np.where((y == ti) & (reading == int(r)))[0][0]
        d_true = np.sqrt(np.mean((logP[i, keep] - class_mean[ti, keep]) ** 2))
        d_pred = np.sqrt(np.mean((logP[i, keep] - class_mean[pi, keep]) ** 2))
        d_typ = np.median(np.sqrt(np.mean((logP[y == ti][:, keep] - class_mean[ti, keep]) ** 2, axis=1)))
        rows.append(dict(file=p.name, true=t, pred=pr, reading=int(r), idx=int(i),
                         dist_to_true_mean_db=round(d_true, 1), dist_to_pred_mean_db=round(d_pred, 1),
                         true_class_typical_dist_db=round(d_typ, 1),
                         class_means_apart_db=round(D[ti, pi], 1),
                         rms_z=round((rms[i] - rms[y == ti].mean()) / rms[y == ti].std(), 1),
                         crank_cycles_per_clip=round(crank_cycles_per_clip(X[i]), 2),
                         splice_click=bool(click[i])))
    df = pd.DataFrame(rows)
    df.drop(columns="idx").to_csv(OUT / "misclassified_diagnostics.csv", index=False)
    print(df.drop(columns=["idx", "file"]).to_string(index=False))

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.1 * len(rows)),
                             gridspec_kw=dict(width_ratios=[1.2, 1]))
    for ax, r in zip(axes, rows):
        ti, pi, i = CLASSES.index(r["true"]), CLASSES.index(r["pred"]), r["idx"]
        a = ax[0]
        a.fill_between(f[keep] / 1000, (class_mean - class_std)[ti, keep], (class_mean + class_std)[ti, keep],
                       color="tab:blue", alpha=.2)
        a.plot(f[keep] / 1000, class_mean[ti, keep], color="tab:blue", lw=1, label=f"{r['true']} mean (true)")
        a.plot(f[keep] / 1000, class_mean[pi, keep], color="tab:orange", lw=1, label=f"{r['pred']} mean (pred)")
        a.plot(f[keep] / 1000, logP[i, keep], color="black", lw=.7, label=f"clip Reading{r['reading']}")
        a.set_ylabel("dB")
        a.legend(fontsize=6, loc="upper right")
        a.set_title(f"true {r['true']} -> predicted {r['pred']} (Reading{r['reading']})", fontsize=8)
        ff, tt, S = ss.spectrogram(X[i], fs=50000, nperseg=1024, noverlap=768)
        sel = ff < 16000
        ax[1].pcolormesh(tt, ff[sel] / 1000, 10 * np.log10(S[sel] + 1e-12), shading="auto", cmap="magma",
                         vmin=-90, vmax=-20)
        ax[1].set_ylabel("kHz")
    axes[-1][0].set_xlabel("frequency (kHz, 50 kHz assumption)")
    axes[-1][1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "misclassified_spectra.png", dpi=90)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
