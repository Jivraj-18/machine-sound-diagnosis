"""Real-time check: once a 1-second clip has been recorded, how long until the answer comes back?

Usage: python latency_bench.py [--raw cache/raw.npz] [--n 100] [--threads 4] [--host "laptop CPU"] [--out CSV]
Runs the live pipeline one clip at a time (no batching, as a live system would):
  clip -> features/embedding -> classifier -> predicted state.
The classifier is logistic regression on 5 clips per state (the k=5, repeat-0 draw of evaluate.py), embedded by
the same live pipeline; the timed clips are other clips. FISHER runs if transformers 4.x is importable
(set FISHER_PYTHONPATH to a side folder, as in embed_pretrained.py); it is timed on CPU and, if present, on GPU.
Writes one row per pipeline and device: median / p95 / max ms per clip, and accuracy on the timed clips.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from evaluate import make_clf
from features_handcrafted import clip_features

AMP_SCALE = 1 / 4.3  # same scaling as embed_pretrained.py


def train_idx(y, k=5, rep=0):
    rng = np.random.default_rng(1000 * k + rep)  # the same draw evaluate.py uses
    return np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in range(8)])


def time_pipeline(name, device, embed_one, X, y, tr, test):
    clf = make_clf("logreg").fit(np.stack([embed_one(X[i]) for i in tr]), y[tr])
    for i in test[:5]:  # warm-up
        clf.predict(embed_one(X[i])[None])
    ms, correct = [], 0
    for i in test:
        t0 = time.perf_counter()
        pred = clf.predict(embed_one(X[i])[None])[0]
        ms.append(1000 * (time.perf_counter() - t0))
        correct += pred == y[i]
    ms = np.array(ms)
    row = dict(pipeline=name, device=device, n_clips=len(test), median_ms=round(np.median(ms), 1),
               p95_ms=round(np.percentile(ms, 95), 1), max_ms=round(ms.max(), 1),
               accuracy_on_timed_clips=round(correct / len(test), 3))
    print(row, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--host", default=f"laptop CPU ({os.cpu_count()} logical cores)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.raw:
        d = np.load(args.raw, allow_pickle=True)
        X, y = d["X"], d["y"]
    else:
        from common import load_dataset
        X, y, _, _ = load_dataset()
    tr = train_idx(y)
    test = np.random.default_rng(7).choice(np.setdiff1d(np.arange(len(y)), tr), args.n, replace=False)
    rows = [time_pipeline("handcrafted features + logreg", "cpu", lambda x: clip_features(x, 50000), X, y, tr, test)]

    if os.environ.get("FISHER_PYTHONPATH"):
        sys.path.insert(0, os.environ["FISHER_PYTHONPATH"])
    try:
        import torch
        import transformers
        assert transformers.__version__.startswith("4.")
    except (ImportError, AssertionError) as e:
        print("skipping FISHER:", repr(e))
    else:
        from embed_pretrained import load_fisher
        torch.set_num_threads(args.threads)
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            fn = load_fisher(device, None)[1](50000)

            def fisher_one(x, fn=fn, device=device):
                with torch.inference_mode():
                    e = fn((x * AMP_SCALE).astype(np.float32)[None])
                    return e.cpu().numpy()[0]  # .cpu() waits for the GPU, so the time is complete
            rows.append(time_pipeline("FISHER @50k + logreg", device, fisher_one, X, y, tr, test))
    df = pd.DataFrame(rows).assign(host=args.host, threads=args.threads)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "latency.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
