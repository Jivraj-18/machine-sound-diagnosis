"""Zero-shot fault diagnosis with an audio-capable LLM (Gemini generateContent API).

Usage: python zeroshot_gemini.py [--model gemini-3.1-pro-preview] [--sr 50k] [--limit N]
Credentials (environment or .env):
  GEMINI_API_KEY                         call Google's API directly, or
  GEMINI_BASE_URL + GEMINI_GATEWAY_TOKEN call a Gemini-compatible gateway (Bearer token auth).
Each clip is sent as a 16-bit WAV with an identical prompt; the answer is constrained to the 8 labels.
Raw responses are cached in cache/gemini/ ; predictions go to cache/zs_<model>_sr<rate>.npz.
"""
import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf

from common import CACHE_DIR, CLASSES, ROOT, load_dataset, zeroshot_subset

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RATES = {"50k": 50000, "16k": 16000}
AMP_SCALE = 1 / 4.3

PROMPT = """You are an expert in machine condition monitoring.
The audio clip is a microphone recording of a single-stage reciprocating air compressor.
The compressor is in exactly one of these 8 states:
- Healthy: no fault
- Bearing: faulty bearing
- Flywheel: damaged flywheel
- LIV: leaking inlet valve
- LOV: leaking outlet valve
- NRV: faulty non-return (check) valve
- Piston: damaged piston (e.g. worn piston ring)
- Riderbelt: worn rider belt (piston rider band)
Listen to the clip and decide which state it is in. Answer with the single most likely label."""

SCHEMA = {"type": "OBJECT",
          "properties": {"label": {"type": "STRING", "enum": CLASSES},
                         "reason": {"type": "STRING"}},
          "required": ["label"]}


def setting(name):
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    return None


def endpoint():
    """Return (url template, headers) for the configured credentials."""
    if setting("GEMINI_API_KEY"):
        return GOOGLE_BASE_URL + "/models/{model}:generateContent", {"x-goog-api-key": setting("GEMINI_API_KEY")}
    if setting("GEMINI_BASE_URL") and setting("GEMINI_GATEWAY_TOKEN"):
        return (setting("GEMINI_BASE_URL").rstrip("/") + "/models/{model}:generateContent",
                {"Authorization": "Bearer " + setting("GEMINI_GATEWAY_TOKEN")})
    raise SystemExit("Set GEMINI_API_KEY, or GEMINI_BASE_URL and GEMINI_GATEWAY_TOKEN")


def wav_b64(x, sr):
    buf = io.BytesIO()
    sf.write(buf, np.clip(x * AMP_SCALE, -1, 1), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def classify(x, sr, model, api, cache_file):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    body = {"contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "audio/wav", "data": wav_b64(x, sr)}},
                {"text": PROMPT}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA}}
    for attempt in range(5):
        t0 = time.time()
        try:
            url, headers = api
            r = requests.post(url.format(model=model), json=body, timeout=180, headers=headers)
            if r.status_code == 200:
                resp = r.json()
                resp["_latency_s"] = round(time.time() - t0, 2)
                cache_file.write_text(json.dumps(resp))
                return resp
            print("HTTP", r.status_code, r.text[:300])
        except requests.RequestException as e:
            print("request error", e)
        time.sleep(5 * 2**attempt)
    return {"error": "failed after retries"}


def parse(resp):
    try:
        text = resp["candidates"][0]["content"]["parts"][-1]["text"]
        return CLASSES.index(json.loads(text)["label"])
    except (KeyError, IndexError, ValueError, TypeError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--sr", choices=list(RATES), default="50k")
    ap.add_argument("--limit", type=int, default=None, help="only first N subset clips (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    X, y, _, _ = load_dataset()
    idx = zeroshot_subset(y)[: args.limit]
    cdir = CACHE_DIR / "gemini"
    cdir.mkdir(parents=True, exist_ok=True)
    api = endpoint()
    jobs = [(X[i], RATES[args.sr], args.model, api, cdir / f"{args.model}_sr{args.sr}_{i}.json") for i in idx]
    with ThreadPoolExecutor(args.workers) as ex:
        resps = list(ex.map(lambda j: classify(*j), jobs))
    pred = np.array([parse(r) for r in resps])
    usage = [r.get("usageMetadata", {}) for r in resps]
    tot = {k: sum(u.get(k, 0) for u in usage) for k in
           ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"]}
    lat = [r.get("_latency_s") for r in resps if r.get("_latency_s")]
    print(f"{args.model} @ {args.sr}: n={len(idx)} acc={np.mean(pred == y[idx]):.3f} failed={(pred < 0).sum()}")
    print("pred distribution:", {CLASSES[c]: int((pred == c).sum()) for c in range(len(CLASSES))})
    print("tokens:", tot, "per clip:", {k: round(v / len(idx)) for k, v in tot.items()},
          "median latency s:", np.median(lat) if lat else None)
    if args.limit is None:
        np.savez(CACHE_DIR / f"zs_{args.model}_sr{args.sr}.npz", idx=idx, pred=pred)
        with open(CACHE_DIR / f"zs_{args.model}_sr{args.sr}_usage.json", "w") as fh:
            json.dump(dict(n=len(idx), tokens=tot, median_latency_s=float(np.median(lat)) if lat else None), fh)


if __name__ == "__main__":
    main()
