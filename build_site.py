"""Build the static site assets in docs/ from results/ (+ the local data cache for audio).

Usage: python build_site.py
Needs: results/results.csv, results/cost_at_scale.csv, results/investigation/*.csv (analyze_misses.py),
       cache/raw.npz and cache/eval_preds.npy (to cut audio and read the same predictions the
       confusion PNGs were drawn from).
Writes: docs/assets/data.js, docs/assets/audio/*.wav (48 kHz), docs/assets/img/spec_*.png,
        results/investigation/listen_clips.csv.
docs/index.html is hand-written and reads window.DATA from data.js. No build step is needed to serve it.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import scipy.signal as ss
import soundfile as sf

from common import CACHE_DIR, CLASSES, RESULTS_DIR, ROOT, load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS = ROOT / "docs"
AUDIO, IMG = DOCS / "assets" / "audio", DOCS / "assets" / "img"
INV = RESULTS_DIR / "investigation"
WEB_SR = 48000  # files are 50 kHz; 48 kHz plays everywhere (content ends ~15 kHz, so nothing is lost)
AMP_SCALE = 1 / 4.3  # same global scale as the rest of the project (keeps relative loudness)


def pct(x):
    """Fraction -> percent with 1 decimal, rounded half-up on the stored value (0.1625 -> 16.3)."""
    from decimal import ROUND_HALF_UP, Decimal
    return float((Decimal(str(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def write_wav(path, x):
    sf.write(path, np.clip(ss.resample_poly(x, 24, 25) * AMP_SCALE, -1, 1), WEB_SR, subtype="PCM_16")


def write_spectrogram(path, x):
    f, t, S = ss.spectrogram(x, fs=50000, nperseg=1024, noverlap=896)
    sel = f <= 16000
    fig = plt.figure(figsize=(4.8, 1.6), dpi=100)
    ax = fig.add_axes([0.11, 0.2, 0.87, 0.75])
    ax.pcolormesh(t, f[sel] / 1000, 10 * np.log10(S[sel] + 1e-12), shading="auto", cmap="magma",
                  vmin=-90, vmax=-20, rasterized=True)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(colors="#898781", labelsize=7, length=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("kHz", color="#898781", fontsize=7, labelpad=2)
    ax.set_xlabel("seconds", color="#898781", fontsize=7, labelpad=1)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def confusion(y_true, y_pred):
    m = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y_true, y_pred):
        if p >= 0:
            m[t, p] += 1
    return m.tolist()


def main():
    X, y, reading, _ = load_dataset()
    AUDIO.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    for old in list(AUDIO.glob("*.wav")) + list(IMG.glob("*.png")):
        old.unlink()

    res = pd.read_csv(RESULTS_DIR / "results.csv")
    preds = np.load(CACHE_DIR / "eval_preds.npy", allow_pickle=True).item()
    click = set(map(tuple, pd.read_csv(INV / "splice_click_clips.csv")[["class", "reading"]].values))

    # --- Listen: per class, the click-free clip whose spectrum is closest to the class mean
    f, P = ss.welch(X, fs=50000, nperseg=2048, axis=1)
    logP = 10 * np.log10(P[:, f < 16000] + 1e-20)
    listen = []
    for c, name in enumerate(CLASSES):
        idx = [i for i in np.where(y == c)[0] if (name, reading[i]) not in click]
        d = np.sqrt(((logP[idx] - logP[y == c].mean(0)) ** 2).mean(1))
        i = idx[int(np.argmin(d))]
        write_wav(AUDIO / f"listen_{name}.wav", X[i])
        write_spectrogram(IMG / f"spec_{name}.png", X[i])
        listen.append(dict(cls=name, reading=int(reading[i]), audio=f"assets/audio/listen_{name}.wav",
                           img=f"assets/img/spec_{name}.png"))
    pd.DataFrame(listen).to_csv(INV / "listen_clips.csv", index=False)

    # --- Misclassified clips (best k=5 few-shot model, protocol B repeat 0)
    diag = pd.read_csv(INV / "misclassified_diagnostics.csv")
    misses = []
    for _, r in diag.iterrows():
        i = np.where((y == CLASSES.index(r["true"])) & (reading == r["reading"]))[0][0]
        fn = f"miss_{r['true']}_as_{r['pred']}_R{r['reading']}.wav"
        write_wav(AUDIO / fn, X[i])
        misses.append({**{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()},
                       "audio": f"assets/audio/{fn}"})
    source = (RESULTS_DIR / "misclassified" / "SOURCE.txt").read_text().strip()

    # --- Numbers straight from results.csv
    def rows(**kw):
        m = np.ones(len(res), bool)
        for k, v in kw.items():
            m &= res[k].astype(str) == str(v)
        return res[m]

    models = ["fisher", "beats", "handcrafted", "dasheng", "echo", "clap"]
    fewshot = {m: {rate: {str(k): [pct(r.macro_acc_mean), pct(r.macro_acc_std)]
                          for _, r in rows(model=m, protocol="B_fewshot", classifier="logreg",
                                           sample_rate=rate).iterrows() for k in [r.k]}
                   for rate in ["50k", "16k"]} for m in models}
    protocols = []
    for m in models:
        for rate in ["50k", "16k"]:
            a = rows(model=m, protocol="A_5foldCV", classifier="svm", sample_rate=rate).iloc[0]
            b = rows(model=m, protocol="B_fewshot", classifier="logreg", sample_rate=rate, k=5).iloc[0]
            c = rows(model=m, protocol="C_blocksplit", classifier="svm", sample_rate=rate).iloc[0]
            protocols.append(dict(model=m, rate=rate, A=pct(a.macro_acc_mean),
                                  A_std=pct(a.macro_acc_std), B5=pct(b.macro_acc_mean),
                                  B5_std=pct(b.macro_acc_std), C=pct(c.macro_acc_mean)))
    zs = rows(protocol="D_zeroshot")
    zeroshot = [dict(model=r.model, rate=r.sample_rate, acc=pct(r.macro_acc_mean))
                for _, r in zs.iterrows()]

    best = rows(protocol="B_fewshot", k=5).sort_values("macro_acc_mean").iloc[-1]
    bi, bp = preds[(best.model, best.sample_rate, "B5", best.classifier)]
    gi, gp = preds[("gemini-3.1-pro-preview", "50k", "D", "zeroshot")]
    ci, cp = preds[("clap", "50k", "D", "zeroshot")]
    cms = {
        "best": dict(model=best.model, rate=best.sample_rate, clf=best.classifier,
                     mean10=pct(best.macro_acc_mean), m=confusion(y[bi], bp),
                     draw_acc=round(100 * np.mean([np.mean(bp[y[bi] == c] == c) for c in range(8)]), 1)),
        "gemini": dict(m=confusion(y[gi], gp)), "clap": dict(m=confusion(y[ci], cp)),
    }
    dist = pd.read_csv(INV / "class_spectral_distance.csv", index_col=0)
    cost = pd.read_csv(RESULTS_DIR / "cost_at_scale.csv")
    cost_rows = [dict(pipeline=r.pipeline, hardware=r.hardware, per_machine=r.usd_per_machine_month,
                      machines_per_box=None if pd.isna(r.machines_per_box_at_1clip_per_min)
                      else int(r.machines_per_box_at_1clip_per_min)) for _, r in cost.iterrows()]
    data = dict(classes=CLASSES, fewshot=fewshot, protocols=protocols, zeroshot=zeroshot, confusion=cms,
                listen=listen, misses=misses, miss_source=source, dist=dist.round(1).values.tolist(),
                click_counts={c: int(sum(1 for k in click if k[0] == c)) for c in CLASSES},
                n_click=len(click), cost=cost_rows)
    (DOCS / "assets" / "data.js").write_text(
        "// Generated by build_site.py from results/ - do not edit by hand.\nwindow.DATA = "
        + json.dumps(data, indent=1) + ";\n")
    n_audio = len(list(AUDIO.glob("*.wav")))
    print(f"wrote docs/assets: {n_audio} wav, {len(list(IMG.glob('*.png')))} png, data.js; "
          f"best k=5 model {best.model}@{best.sample_rate} {best.classifier} {best.macro_acc_mean:.4f}")


if __name__ == "__main__":
    main()
