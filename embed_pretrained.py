"""Frozen embeddings from pretrained audio / industrial-signal models (run on a GPU box, e.g. Kaggle).

Usage:
  python embed_pretrained.py --raw path/to/raw.npz --out cache/            # all models, both rates
  python embed_pretrained.py --raw ... --out ... --model beats --sr 50k   # one model / rate

Sample-rate assumptions (see REPORT.md):
  50k: the 50,000-sample clip is 1 s at 50 kHz (most likely; resampled to each model's native rate)
  16k: the clip is 3.125 s at 16 kHz (MathWorks' interpretation; raw samples used as-is)
Outputs: <out>/emb_<model>_sr<rate>.npy, <out>/clap_text_<rate>.npz, <out>/resources.csv.
Each model runs in its own subprocess (BEATs/ECHO/FISHER all ship a top-level `modules.py`).
"""
import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from common import CLASSES, ResourceMonitor

MODELS = ["beats", "dasheng", "clap", "echo", "fisher"]
RATES = {"50k": 50000, "16k": 16000}
AMP_SCALE = 1 / 4.3  # global constant: max |x| over the dataset is 4.27; keeps relative loudness

CLAP_PROMPTS = {
    "Healthy": "a healthy air compressor running normally",
    "Bearing": "an air compressor with a faulty bearing",
    "Flywheel": "an air compressor with a damaged flywheel",
    "LIV": "an air compressor with a leaking inlet valve",
    "LOV": "an air compressor with a leaking outlet valve",
    "NRV": "an air compressor with a faulty non-return valve",
    "Piston": "an air compressor with a damaged piston",
    "Riderbelt": "an air compressor with a worn piston rider belt",
}
CLAP_TEMPLATES = ["{}", "the sound of {}", "a recording of {}"]


def to_rate(X, assumed_sr, target_sr):
    """Resample clips from the assumed source rate to the model's rate."""
    if assumed_sr == target_sr:
        return X.astype(np.float32)
    g = np.gcd(assumed_sr, target_sr)
    return resample_poly(X, target_sr // g, assumed_sr // g, axis=1).astype(np.float32)


def batched(fn, X, bs):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            out.append(fn(X[i:i + bs]).float().cpu().numpy())
    return np.concatenate(out)


# --- model loaders: each returns (native_sr or None for "any rate", embed_fn(np batch) -> torch (B, D)) ---

def load_beats(device, work):
    import torch
    src = work / "beats_src"
    src.mkdir(exist_ok=True)
    base = "https://raw.githubusercontent.com/microsoft/unilm/master/beats/"
    for f in ["BEATs.py", "backbone.py", "modules.py", "quantizer.py"]:
        if not (src / f).exists():
            subprocess.run(["wget", "-q", "-O", str(src / f), base + f], check=True)
    sys.path.insert(0, str(src))
    from BEATs import BEATs, BEATsConfig
    from huggingface_hub import hf_hub_download
    ckpt = torch.load(hf_hub_download("lpepino/beats_ckpts", "BEATs_iter3_plus_AS2M.pt"),
                      map_location="cpu", weights_only=False)
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    def embed(x):
        x = torch.from_numpy(x).to(device)
        rep = model.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0]
        return rep.mean(1)
    return 16000, embed


def load_dasheng(device, work):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained("mispeech/dasheng-base", trust_remote_code=True)
    model = AutoModel.from_pretrained("mispeech/dasheng-base", outputdim=None, trust_remote_code=True)
    model.eval().to(device)

    def embed(x):
        inp = fe(torch.from_numpy(x), sampling_rate=16000, return_tensors="pt")
        return model(**{k: v.to(device) for k, v in inp.items()}).logits
    return 16000, embed


def _as_tensor(o):
    import torch
    if isinstance(o, torch.Tensor):
        return o
    for k in ("audio_embeds", "text_embeds", "pooler_output"):
        if getattr(o, k, None) is not None:
            return getattr(o, k)
    return o[0]


def load_clap(device, work, rate_tag=None, out=None):
    import torch
    from transformers import ClapModel, ClapProcessor
    name = "laion/larger_clap_general"
    proc = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).eval().to(device)
    if out is not None:  # text side for zero-shot (independent of audio rate, saved once per rate tag)
        prompts = [t.format(CLAP_PROMPTS[c]) for c in CLASSES for t in CLAP_TEMPLATES]
        with torch.inference_mode():
            tok = proc.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            T = _as_tensor(model.get_text_features(**tok)).float().cpu().numpy()
        np.savez(out / f"clap_text_{rate_tag}.npz", emb=T.reshape(len(CLASSES), len(CLAP_TEMPLATES), -1),
                 prompts=np.array(prompts))

    def embed(x):
        inp = proc.feature_extractor(list(x), sampling_rate=48000, return_tensors="pt")
        return _as_tensor(model.get_audio_features(**{k: v.to(device) for k, v in inp.items()}))
    return 48000, embed


def load_echo(device, work):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    path = snapshot_download("yucongzh/echo-small-0824")
    sys.path.insert(0, path)
    from audioMAE_band_upgrade import AudioMAEWithBand
    model = AudioMAEWithBand(spec_len=2000, band_width=32, shift_size=16, in_chans=1, embed_dim=384,
                             encoder_depth=12, num_heads=6, mlp_ratio=4.0, freq_pos_emb_dim=384)
    res = model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=False)
    print("ECHO load_state_dict: missing", len(res.missing_keys), res.missing_keys[:5],
          "unexpected", len(res.unexpected_keys), res.unexpected_keys[:5])
    assert len(res.missing_keys) == 0, "ECHO weights did not load"
    model.eval().to(device)

    def embed_at(sr):
        def embed(x):  # ECHO's API is per clip
            return torch.stack([model.extract_features_from_audio(
                torch.from_numpy(c[None]).to(device), sample_rate=sr)[0] for c in x])
        return embed
    return None, embed_at


def load_fisher(device, work):
    import torch
    import torch.nn.functional as F
    import torchaudio
    from transformers import AutoModel
    model = AutoModel.from_pretrained("jiangab/FISHER-small-0723", trust_remote_code=True)
    model.eval().to(device)

    def embed_at(sr):
        stft = torchaudio.transforms.Spectrogram(n_fft=25 * sr // 1000, win_length=None,
                                                 hop_length=10 * sr // 1000, power=1,
                                                 center=False).to(device)

        def embed(x):
            w = torch.from_numpy(x).to(device)
            w = w - w.mean(1, keepdim=True)
            spec = torch.log(torch.abs(stft(w)) + 1e-10).transpose(-2, -1)  # [B, time, freq]
            spec = (spec + 3.017344307886898) / (2.1531635155379805 * 2)
            spec = spec[:, :1024]
            if spec.shape[-1] < model.cfg.band_width:
                spec = F.pad(spec, (0, model.cfg.band_width - spec.shape[-1]))
            with torch.autocast(device.split(":")[0], enabled=device.startswith("cuda")):
                return model.extract_features(spec.unsqueeze(1)).float()
        return embed
    return None, embed_at


LOADERS = dict(beats=load_beats, dasheng=load_dasheng, clap=load_clap, echo=load_echo, fisher=load_fisher)
BATCH = dict(beats=32, dasheng=32, clap=16, echo=1, fisher=32)


def run_one(model_name, rate_tag, raw, out, device, cpu_bench):
    import torch
    X = np.load(raw)["X"] * AMP_SCALE
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    assumed_sr = RATES[rate_tag]
    kw = dict(rate_tag=rate_tag, out=out) if model_name == "clap" else {}
    with ResourceMonitor(f"{model_name}_sr{rate_tag}_load", 0, out_csv=out / "resources.csv", device=device):
        native_sr, fn = LOADERS[model_name](device, work, **kw)
    if native_sr is None:  # arbitrary-rate model: feed the assumed rate directly
        Xm, fn = X.astype(np.float32), fn(assumed_sr)
    else:
        Xm = to_rate(X, assumed_sr, native_sr)
    print(f"{model_name} @ {rate_tag}: input {Xm.shape} at {native_sr or assumed_sr} Hz", flush=True)
    with ResourceMonitor(f"{model_name}_sr{rate_tag}", len(Xm), out_csv=out / "resources.csv",
                         device=device, input_seconds=round(Xm.shape[1] / (native_sr or assumed_sr), 3),
                         batch=BATCH[model_name]):
        E = batched(fn, Xm, BATCH[model_name])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    np.save(out / f"emb_{model_name}_sr{rate_tag}.npy", E)
    print(f"saved emb_{model_name}_sr{rate_tag}.npy {E.shape} finite={np.isfinite(E).all()}", flush=True)
    if cpu_bench and device != "cpu":  # CPU throughput for the cost-at-scale estimate
        torch.set_num_threads(os.cpu_count())
        native_sr, fn = LOADERS[model_name]("cpu", work)
        if native_sr is None:
            fn = fn(assumed_sr)
        with ResourceMonitor(f"{model_name}_sr{rate_tag}_cpu", cpu_bench, out_csv=out / "resources.csv",
                             device="cpu", threads=os.cpu_count(), batch=BATCH[model_name]):
            batched(fn, Xm[:cpu_bench], BATCH[model_name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="cache")
    ap.add_argument("--model", choices=MODELS, help="worker mode: run a single model")
    ap.add_argument("--models", nargs="*", choices=MODELS, default=MODELS, help="models to run (orchestrator)")
    ap.add_argument("--sr", choices=list(RATES))
    ap.add_argument("--device", default=None)
    ap.add_argument("--cpu-bench", type=int, default=64, help="clips to time on CPU (0 = skip)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.model:  # worker mode
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        run_one(args.model, args.sr, args.raw, out, device, args.cpu_bench)
        return
    status = {}
    for m in args.models:
        for r in RATES:
            if (out / f"emb_{m}_sr{r}.npy").exists():
                status[(m, r)] = "cached"
                continue
            cmd = [sys.executable, __file__, "--raw", args.raw, "--out", str(out), "--model", m,
                   "--sr", r, "--cpu-bench", str(args.cpu_bench)] + (["--device", args.device] if args.device else [])
            print(">>", " ".join(cmd), flush=True)
            env = dict(os.environ)
            # FISHER's remote code predates transformers 5 (never calls post_init()); if a
            # transformers-4.x install dir is provided, put it first on the path for FISHER only.
            if m == "fisher" and os.environ.get("FISHER_PYTHONPATH"):
                env["PYTHONPATH"] = os.environ["FISHER_PYTHONPATH"] + os.pathsep + env.get("PYTHONPATH", "")
            try:
                rc = subprocess.run(cmd, env=env).returncode
            except Exception:
                traceback.print_exc()
                rc = -1
            status[(m, r)] = "ok" if rc == 0 else f"FAILED rc={rc}"
    for k, v in status.items():
        print("STATUS", *k, v)


if __name__ == "__main__":
    main()
