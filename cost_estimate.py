"""Turn measured resource logs into a clean resources table and a cost-at-scale estimate.

Usage: python cost_estimate.py
Reads results/resources_local.csv (this laptop) and results/resources_kaggle.csv (Kaggle T4 box),
cache/zs_*_usage.json (Gemini token counts). Writes results/resources.csv and results/cost_at_scale.csv.

Scenario: N machines, each sends one 1-second clip every CLIP_INTERVAL_S seconds.
Prices are list prices checked on 2026-10-29 (see REPORT.md for sources); edit PRICES to re-run.
"""
import csv
import json

import pandas as pd

from common import CACHE_DIR, RESULTS_DIR

PRICES = {  # USD per hour, Linux on-demand (us-east-1), from instances.vantage.sh / AWS price list
    "c7i.xlarge (4 vCPU, 8 GiB)": 0.1785,
    "g4dn.xlarge (4 vCPU, 1x T4)": 0.526,
}
GEMINI = {  # USD per 1M tokens, Google list price, standard tier, prompts <= 200k
    "gemini-3.1-pro-preview": dict(inp=2.00, out=12.00),
    "gemini-3-flash-preview": dict(inp=1.00, out=3.00),  # audio input price
}
CLIP_INTERVAL_S = 60          # one 1-s clip per machine per minute
UTILISATION = 0.5             # don't plan to run boxes at more than 50% of measured throughput
HOURS_PER_MONTH = 730
FIXED = ["name", "host", "n_clips", "wall_s", "cpu_s", "ms_per_clip", "avg_cpu_cores", "peak_rss_mb",
         "peak_gpu_mb"]


def read_ragged(path):
    """Older logs wrote extra fields without matching headers; keep the 9 fixed columns + raw extras."""
    rows = []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r)
        for line in r:
            if not line:
                continue
            row = dict(zip(FIXED, line[:9]))
            row["extra"] = json.dumps(line[9:]) if not line[9:10] or not line[9].startswith("{") else line[9]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    res = pd.concat([read_ragged(RESULTS_DIR / "resources_local.csv"),
                     read_ragged(RESULTS_DIR / "resources_kaggle.csv")], ignore_index=True)
    for c in FIXED[2:]:
        res[c] = pd.to_numeric(res[c])
    res.to_csv(RESULTS_DIR / "resources.csv", index=False)

    clips_per_hour_per_machine = 3600 / CLIP_INTERVAL_S
    rows = []

    def add(pipeline, box, ms_per_clip_on_box, note):
        clips_per_hour = UTILISATION * 3600 * 1000 / ms_per_clip_on_box
        machines_per_box = clips_per_hour / clips_per_hour_per_machine
        usd_month = PRICES[box] * HOURS_PER_MONTH
        rows.append(dict(pipeline=pipeline, hardware=box, measured_ms_per_clip=round(ms_per_clip_on_box, 2),
                         machines_per_box_at_1clip_per_min=int(machines_per_box),
                         usd_per_box_month=round(usd_month, 0),
                         usd_per_machine_month=round(usd_month / machines_per_box, 4),
                         usd_per_1000_machines_month=round(1000 * usd_month / machines_per_box, 1), note=note))

    get = lambda n: res.loc[res["name"] == n, "ms_per_clip"].iloc[0]  # noqa: E731
    # handcrafted: measured single-core; assume 4 vCPU box runs 4 workers in parallel (conservative: 2 phys cores)
    add("handcrafted + SVM", "c7i.xlarge (4 vCPU, 8 GiB)", get("handcrafted_sr50k_1core") / 2,
        "37 ms/clip on 1 laptop core; scaled to 2 physical cores")
    for model, rate in [("fisher", "50k"), ("beats", "16k"), ("dasheng", "50k"), ("echo", "50k"),
                        ("clap", "50k")]:
        add(f"{model} @{rate} + logreg", "c7i.xlarge (4 vCPU, 8 GiB)", get(f"{model}_sr{rate}_cpu"),
            "measured on Kaggle 4-vCPU CPU, batch inference")
        add(f"{model} @{rate} + logreg", "g4dn.xlarge (4 vCPU, 1x T4)", get(f"{model}_sr{rate}"),
            "measured on Kaggle T4, batch inference (GPU mostly idle at this scale)")
    # Gemini zero-shot: per-call token cost, no hardware
    for p in sorted(CACHE_DIR.glob("zs_gemini-*_usage.json")):
        u = json.loads(p.read_text())
        model = p.stem.replace("zs_", "").rsplit("_sr", 1)[0]
        if model not in GEMINI:
            continue
        t = u["tokens"]
        per_clip = (t["promptTokenCount"] * GEMINI[model]["inp"] +
                    (t["candidatesTokenCount"] + t["thoughtsTokenCount"]) * GEMINI[model]["out"]) / 1e6 / u["n"]
        per_machine_month = per_clip * clips_per_hour_per_machine * HOURS_PER_MONTH
        rows.append(dict(pipeline=f"{model} zero-shot ({p.stem.rsplit('_sr', 1)[1].split('_')[0]})",
                         hardware="API", measured_ms_per_clip=round(1000 * u["median_latency_s"]),
                         machines_per_box_at_1clip_per_min=None, usd_per_box_month=None,
                         usd_per_machine_month=round(per_machine_month, 2),
                         usd_per_1000_machines_month=round(1000 * per_machine_month, 0),
                         note=f"${per_clip:.5f}/clip from measured tokens; accuracy at chance"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "cost_at_scale.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["note"]).to_string(index=False))


if __name__ == "__main__":
    main()
