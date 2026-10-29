"""Zero-shot CLAP: cosine similarity between cached audio embeddings and class-prompt text embeddings.

Usage: python zeroshot_clap.py   (needs cache/emb_clap_sr*.npy and cache/clap_text_*.npz from embed_pretrained.py)
Writes cache/zs_clap_sr<rate>.npz on the protocol-D subset; also prints accuracy on all clips.
"""
import numpy as np

from common import CACHE_DIR, CLASSES, load_dataset, zeroshot_subset


def l2n(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


if __name__ == "__main__":
    _, y, _, _ = load_dataset()
    idx = zeroshot_subset(y)
    for rate in ["50k", "16k"]:
        a_path, t_path = CACHE_DIR / f"emb_clap_sr{rate}.npy", CACHE_DIR / f"clap_text_{rate}.npz"
        if not (a_path.exists() and t_path.exists()):
            print("missing", a_path.name, "or", t_path.name)
            continue
        A = l2n(np.load(a_path))
        T = l2n(l2n(np.load(t_path)["emb"]).mean(1))  # average the prompt templates per class
        pred_all = (A @ T.T).argmax(1)
        np.savez(CACHE_DIR / f"zs_clap_sr{rate}.npz", idx=idx, pred=pred_all[idx])
        print(f"CLAP zero-shot @ {rate}: subset acc {np.mean(pred_all[idx] == y[idx]):.3f}, "
              f"all-clips acc {np.mean(pred_all == y):.3f}")
        print("  predicted-label distribution (all clips):",
              {CLASSES[c]: int((pred_all == c).sum()) for c in range(len(CLASSES))})
