"""Evaluate every cached feature set under protocols A (5-fold CV), B (few-shot), C (block split),
merge zero-shot predictions (D), write results/results.csv, confusion matrices and misclassified clips.

Usage: python evaluate.py [--only NAME ...]
Feature sets are discovered as cache/emb_<model>_sr<rate>.npy.
Zero-shot predictions are discovered as cache/zs_<method>_sr<rate>.npz (keys: idx, pred).
"""
import argparse
import re

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CACHE_DIR, CLASSES, RESULTS_DIR, block_split, load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEW_SHOT_K = [1, 3, 5, 10]
N_REPEATS = 10


def make_clf(name, k_neighbors=5):
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_neighbors))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    raise ValueError(name)


def protocol_a(F, y):
    """5-fold stratified CV. Returns {clf: (mean, std, pooled predictions)}."""
    out = {}
    for clf in ["svm", "knn", "logreg"]:
        scores, pred = [], np.empty_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F, y):
            pred[te] = make_clf(clf).fit(F[tr], y[tr]).predict(F[te])
            scores.append(balanced_accuracy_score(y[te], pred[te]))
        out[clf] = (np.mean(scores), np.std(scores), pred)
    return out


def protocol_b(F, y):
    """Few-shot: k clips per class train, rest test, N_REPEATS random draws.

    Returns {(clf, k): (mean, std)} and the k=5 predictions of repeat 0 for the misclassified dump.
    """
    out, k5_preds = {}, {}
    for k in FEW_SHOT_K:
        for clf in ["knn", "logreg"]:
            scores = []
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(1000 * k + rep)
                tr = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False)
                                     for c in np.unique(y)])
                te = np.setdiff1d(np.arange(len(y)), tr)
                pred = make_clf(clf, k_neighbors=min(5, k)).fit(F[tr], y[tr]).predict(F[te])
                scores.append(balanced_accuracy_score(y[te], pred))
                if k == 5 and rep == 0:
                    k5_preds[clf] = (te, pred)
            out[(clf, k)] = (np.mean(scores), np.std(scores))
    return out, k5_preds


def protocol_c(F, y, reading):
    tr, te = block_split(y, reading)
    out = {}
    for clf in ["svm", "knn", "logreg"]:
        pred = make_clf(clf).fit(F[tr], y[tr]).predict(F[te])
        out[clf] = (balanced_accuracy_score(y[te], pred), 0.0, (np.where(te)[0], pred))
    return out


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nmacro acc {balanced_accuracy_score(y_true, y_pred):.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def discover(only):
    sets = []
    for p in sorted(CACHE_DIR.glob("emb_*.npy")):
        m = re.fullmatch(r"emb_(.+)_sr(\w+)", p.stem)
        if m and (not only or m.group(1) in only):
            sets.append((m.group(1), m.group(2), p))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="model names to evaluate (default: all cached)")
    args = ap.parse_args()
    _, y, reading, _ = load_dataset()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, preds = [], {}
    for model, sr, path in discover(args.only):
        F = np.load(path).astype(np.float64)
        assert len(F) == len(y), (path, F.shape)
        print(f"== {model} @ {sr} {F.shape}")
        a = protocol_a(F, y)
        for clf, (m, s, pred) in a.items():
            rows.append(dict(model=model, protocol="A_5foldCV", classifier=clf, k="all",
                             sample_rate=sr, macro_acc_mean=m, macro_acc_std=s))
            preds[(model, sr, "A", clf)] = (np.arange(len(y)), pred)
            print(f"  A {clf:6s} {m:.4f} ± {s:.4f}")
        b, k5 = protocol_b(F, y)
        for (clf, k), (m, s) in b.items():
            rows.append(dict(model=model, protocol="B_fewshot", classifier=clf, k=k,
                             sample_rate=sr, macro_acc_mean=m, macro_acc_std=s))
        for clf, v in k5.items():
            preds[(model, sr, "B5", clf)] = v
        print("  B " + "  ".join(f"{clf}@{k}={m:.3f}" for (clf, k), (m, s) in b.items()))
        for clf, (m, s, v) in protocol_c(F, y, reading).items():
            rows.append(dict(model=model, protocol="C_blocksplit", classifier=clf, k="70%",
                             sample_rate=sr, macro_acc_mean=m, macro_acc_std=s))
            preds[(model, sr, "C", clf)] = v
            print(f"  C {clf:6s} {m:.4f}")
    # zero-shot predictions (protocol D)
    for p in sorted(CACHE_DIR.glob("zs_*.npz")):
        m = re.fullmatch(r"zs_(.+)_sr(\w+)", p.stem)
        if not m or (args.only and m.group(1) not in args.only):
            continue
        d = np.load(p)
        idx, pred = d["idx"], d["pred"]
        ok = pred >= 0  # -1 = no valid answer; counted as wrong below, reported separately
        acc = balanced_accuracy_score(y[idx], np.where(ok, pred, (y[idx] + 1) % len(CLASSES)))
        rows.append(dict(model=m.group(1), protocol="D_zeroshot", classifier="zeroshot", k=0,
                         sample_rate=m.group(2), macro_acc_mean=acc, macro_acc_std=0.0,
                         note=f"{(~ok).sum()} of {len(idx)} unparseable/failed"))
        preds[(m.group(1), m.group(2), "D", "zeroshot")] = (idx, pred)
        print(f"== zero-shot {m.group(1)} @ {m.group(2)}: {acc:.4f} ({(~ok).sum()} failed)")
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / ("results.csv" if not args.only else "results_partial.csv")
    df.to_csv(out, index=False, float_format="%.4f")
    print("wrote", out)
    np.save(CACHE_DIR / ("eval_preds.npy" if not args.only else "eval_preds_partial.npy"),
            preds, allow_pickle=True)
    if not args.only:
        write_artifacts(df, preds, y, reading)


def best_row(df, mask):
    sub = df[mask]
    return sub.loc[sub["macro_acc_mean"].idxmax()] if len(sub) else None


def write_artifacts(df, preds, y, reading):
    """Confusion matrices (best handcrafted, best pretrained, best few-shot k=5, each zero-shot)
    and misclassified clips of the best k=5 few-shot model as .wav."""
    import soundfile as sf

    cm_dir = RESULTS_DIR / "confusion"
    cm_dir.mkdir(exist_ok=True)
    is_hc = df["model"] == "handcrafted"
    picks = {
        "best_handcrafted_A": best_row(df, is_hc & (df["protocol"] == "A_5foldCV")),
        "best_pretrained_A": best_row(df, ~is_hc & (df["protocol"] == "A_5foldCV")),
        "best_pretrained_C": best_row(df, ~is_hc & (df["protocol"] == "C_blocksplit")),
        "best_fewshot_k5": best_row(df, (df["protocol"] == "B_fewshot") & (df["k"] == 5)),
        "best_pretrained_fewshot_k5": best_row(df, ~is_hc & (df["protocol"] == "B_fewshot") & (df["k"] == 5)),
    }
    for _, r in df[df["protocol"] == "D_zeroshot"].iterrows():
        picks[f"zeroshot_{r['model']}_sr{r['sample_rate']}"] = r
    proto_key = {"A_5foldCV": "A", "C_blocksplit": "C", "B_fewshot": "B5", "D_zeroshot": "D"}
    for tag, r in picks.items():
        if r is None:
            continue
        idx, pred = preds[(r["model"], r["sample_rate"], proto_key[r["protocol"]], r["classifier"])]
        ok = pred >= 0
        plot_confusion(y[idx][ok], pred[ok],
                       f"{tag}\n{r['model']} @ {r['sample_rate']}, {r['protocol']}, {r['classifier']}",
                       cm_dir / f"{tag}.png")
        print("confusion:", tag, r["model"], r["sample_rate"], r["classifier"], round(r["macro_acc_mean"], 4))
    # misclassified clips from the best k=5 few-shot model (repeat 0 of protocol B)
    r = picks["best_fewshot_k5"]
    idx, pred = preds[(r["model"], r["sample_rate"], "B5", r["classifier"])]
    X, _, _, _ = load_dataset()
    mis_dir = RESULTS_DIR / "misclassified"
    mis_dir.mkdir(exist_ok=True)
    for old in mis_dir.glob("*.wav"):
        old.unlink()
    for c in range(len(CLASSES)):
        wrong = [(i, p) for i, p in zip(idx, pred) if y[i] == c and p != c][:3]
        for i, p in wrong:
            # written at the 50 kHz assumption (1 s per clip), global amplitude scale
            sf.write(mis_dir / f"true-{CLASSES[c]}_pred-{CLASSES[p]}_Reading{reading[i]}.wav",
                     np.clip(X[i] / 4.3, -1, 1), 50000, subtype="PCM_16")
    (mis_dir / "SOURCE.txt").write_text(
        f"Misclassified clips from the best k=5 few-shot model: {r['model']} @ {r['sample_rate']} + "
        f"{r['classifier']} (protocol B, repeat 0; macro acc over 10 repeats {r['macro_acc_mean']:.3f}).\n"
        "WAVs are written at 50 kHz (1 s each), amplitude scaled by 1/4.3 (global, not per clip).\n")
    print("misclassified clips:", len(list(mis_dir.glob("*.wav"))), "from", r["model"], r["sample_rate"],
          r["classifier"])


if __name__ == "__main__":
    main()
