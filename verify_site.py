"""Check the published page in a real browser (Playwright driving the installed Chrome).

Usage: python verify_site.py URL [--shots DIR] [--validator path/to/validate_palette.js]
Checks: no console errors / failed requests; every <audio> decodes to ~1 s; every image loads;
no horizontal scroll at phone width (390 px); table numbers match results/results.csv;
chart palette passes the dataviz validator in light and dark mode. Exits non-zero on any failure.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

from build_site import pct
from common import RESULTS_DIR

LIGHT = "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300"
DARK = "#3987e5,#d95926,#199e70,#c98500,#d55181,#008300"
NAMES = {"fisher": "FISHER", "beats": "BEATs", "handcrafted": "Handcrafted", "dasheng": "Dasheng",
         "echo": "ECHO", "clap": "CLAP"}


def usd(v):
    """Same money format as the page's usd()."""
    if v >= 100:
        return f"${round(v):,}"
    if v >= 0.01:
        return f"${v:.2f}"
    return f"${v:#.2g}"  # keeps trailing zeros, like JS toPrecision(2)


def expected_protocol_rows():
    r = pd.read_csv(RESULTS_DIR / "results.csv")
    pick = lambda **kw: r[(r[list(kw)] == pd.Series(kw)).all(axis=1)].iloc[0]  # noqa: E731
    out = {}
    for m, name in NAMES.items():
        for rate in ["50k", "16k"]:
            a = pick(model=m, protocol="A_5foldCV", classifier="svm", sample_rate=rate).macro_acc_mean
            b = pick(model=m, protocol="B_fewshot", classifier="logreg", sample_rate=rate, k="5")
            c = pick(model=m, protocol="C_blocksplit", classifier="svm", sample_rate=rate).macro_acc_mean
            out[(name, rate)] = [f"{pct(a):.1f}%", f"{pct(b.macro_acc_mean):.1f}% ± {pct(b.macro_acc_std):.1f}",
                                 f"{pct(c):.1f}%"]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--shots", default=None)
    ap.add_argument("--validator", default=None)
    args = ap.parse_args()
    fails, notes = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--autoplay-policy=no-user-gesture-required"])
        for label, vp in [("desktop", dict(width=1280, height=900)), ("phone", dict(width=390, height=844))]:
            ctx = browser.new_context(viewport=vp, device_scale_factor=2 if label == "phone" else 1,
                                      is_mobile=label == "phone", has_touch=label == "phone")
            page = ctx.new_page()
            errors, bad = [], []
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("response", lambda r: r.status >= 400 and bad.append(f"{r.status} {r.url}"))
            page.goto(args.url, wait_until="networkidle")
            page.evaluate("document.querySelectorAll('img').forEach(i => i.loading = 'eager')")
            page.wait_for_function("[...document.images].every(i => i.complete)", timeout=30000)
            imgs = page.evaluate("[...document.images].map(i => [i.getAttribute('src'), i.naturalWidth])")
            for src, w in imgs:
                if not w:
                    fails.append(f"[{label}] image failed: {src}")
            sw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            if sw[0] > sw[1]:
                fails.append(f"[{label}] horizontal scroll: page {sw[0]}px > viewport {sw[1]}px")
            if label == "desktop":
                audio = page.evaluate("""async () => {
                  const ctx = new AudioContext(); const out = [];
                  for (const a of document.querySelectorAll('audio')) {
                    const src = a.getAttribute('src');
                    try {
                      const buf = await (await fetch(src)).arrayBuffer();
                      const d = await ctx.decodeAudioData(buf);
                      const ok = await new Promise(res => { a.preload = 'auto';
                        a.addEventListener('canplaythrough', () => res(true), {once: true});
                        a.addEventListener('error', () => res(false), {once: true}); a.load(); });
                      let played = false;
                      try { a.muted = true; await a.play(); played = !a.paused; a.pause(); } catch (e) {}
                      out.push([src, d.duration, d.sampleRate, ok, played]);
                    } catch (e) { out.push([src, 0, 0, false, false, String(e)]); }
                  }
                  return out; }""")
                for row in audio:
                    if not (0.95 < row[1] < 1.05 and row[3] and row[4]):
                        fails.append(f"audio problem: {row}")
                notes.append(f"audio: {len(audio)} clips decoded, loaded and played "
                             f"(durations {min(r[1] for r in audio):.3f}-{max(r[1] for r in audio):.3f} s, "
                             f"{audio[0][2]} Hz)")
                # numbers in the protocol table vs results.csv
                rows = page.evaluate("[...document.querySelectorAll('#protoTable tbody tr')]"
                                     ".map(tr => [...tr.cells].map(td => td.textContent.trim()))")
                exp = expected_protocol_rows()
                n_ok = 0
                for r in rows:
                    key = (r[0], r[1])
                    if key in exp:
                        if r[2:5] != exp[key]:
                            fails.append(f"protocol table mismatch {key}: page {r[2:5]} vs results.csv {exp[key]}")
                        else:
                            n_ok += 1
                notes.append(f"protocol table: {n_ok}/{len(exp)} model rows match results.csv")
                # few-shot table, in each chart mode
                res = pd.read_csv(RESULTS_DIR / "results.csv")
                fs = res[(res.protocol == "B_fewshot") & (res.classifier == "logreg")]
                checked = 0
                for mode in ["best", "50k", "16k"]:
                    page.click(f".seg button[data-mode='{mode}']")
                    trs = page.evaluate("[...document.querySelectorAll('#fewTable tbody tr')]"
                                        ".map(tr => [...tr.cells].map(td => td.textContent.trim()))")
                    for tr in trs:
                        mid = [k for k, v in NAMES.items() if v == tr[0]][0]
                        for k, cell in zip([1, 3, 5, 10], tr[2:]):
                            row = fs[(fs.model == mid) & (fs.sample_rate == tr[1]) & (fs.k.astype(str) == str(k))].iloc[0]
                            want = f"{pct(row.macro_acc_mean):.1f}% ± {pct(row.macro_acc_std):.1f}"
                            checked += 1
                            if cell != want:
                                fails.append(f"few-shot table mismatch {tr[0]}@{tr[1]} k={k}: {cell} vs {want}")
                page.click(".seg button[data-mode='best']")
                notes.append(f"few-shot table: {checked} cells checked across 3 modes")
                # zero-shot table
                zs = page.evaluate("[...document.querySelectorAll('#zsTable tbody tr')]"
                                   ".map(tr => [...tr.cells].map(td => td.textContent.trim()))")
                for tr, mid in zip(zs, ["clap", "gemini-3.1-pro-preview"]):
                    for rate, cell in zip(["50k", "16k"], tr[1:3]):
                        v = res[(res.protocol == "D_zeroshot") & (res.model == mid) & (res.sample_rate == rate)]
                        if cell != f"{pct(v.macro_acc_mean.iloc[0]):.1f}%":
                            fails.append(f"zero-shot mismatch {mid}@{rate}: {cell}")
                notes.append("zero-shot table checked")
                # accuracy-vs-cost table, every sample-rate mode x hardware
                cost = pd.read_csv(RESULTS_DIR / "cost_at_scale.csv")
                n_cost = 0
                for mode in ["best", "50k", "16k"]:
                    for hw in ["cpu", "gpu"]:
                        page.click(f".seg button[data-mode='{mode}']")
                        page.click(f".seg button[data-hw='{hw}']")
                        trs = page.evaluate("[...document.querySelectorAll('#costCmpTable tbody tr')].map(tr => "
                                            "[tr.dataset.model, tr.dataset.method, tr.dataset.rate, tr.dataset.hw, "
                                            "...[...tr.cells].map(td => td.textContent.trim())])")
                        for model, method, rate, rhw, _name, acc, month, per1000, _on, _per_server in trs:
                            c = cost[(cost.model == model) & (cost.method == method) & (cost.sample_rate == rate)]
                            c = c[c.hardware.str.startswith({"cpu": "c7i", "gpu": "g4dn", "api": "API"}[rhw])].iloc[0]
                            want = [usd(c.usd_per_machine_month), usd(c.usd_per_1000_clips)]
                            src = res[(res.model == model) & (res.sample_rate == rate) & (
                                (res.protocol == "D_zeroshot") if method == "zero-shot" else
                                ((res.protocol == "B_fewshot") & (res.classifier == "logreg") & (res.k.astype(str) == "5")))]
                            want_acc = f"{pct(src.macro_acc_mean.iloc[0]):.1f}%"
                            n_cost += 1
                            if [month, per1000] != want or acc != want_acc:
                                fails.append(f"cost table mismatch {model}/{method}@{rate}/{rhw}: page {[acc, month, per1000]}"
                                             f" vs csv {[want_acc] + want}")
                page.click(".seg button[data-mode='best']")
                page.click(".seg button[data-hw='cpu']")
                notes.append(f"accuracy-vs-cost table: {n_cost} rows checked across 3 rates x 2 hardware")
                # real-time table vs latency csvs
                lat = pd.concat([pd.read_csv(RESULTS_DIR / f) for f in ["latency_laptop.csv", "latency_kaggle.csv"]
                                 if (RESULTS_DIR / f).exists()])
                rts = page.evaluate("[...document.querySelectorAll('#rtTable tbody tr')]"
                                    ".map(tr => [...tr.cells].map(td => td.textContent.trim()))")
                meds = {f"{v:.1f}" if v < 10 else str(round(v)) for v in lat.median_ms}
                gem_ms = cost[cost.model.str.startswith("gemini") & (cost.sample_rate == "50k")].measured_ms_per_clip.iloc[0]
                for tr in rts:
                    cell = tr[2]
                    ok = (cell.endswith(" ms") and cell[:-3] in meds) or cell == f"{gem_ms / 1000:.1f} s"
                    if not ok:
                        fails.append(f"real-time table cell not in latency csv: {tr}")
                notes.append(f"real-time table: {len(rts)} rows checked (latency csv: {len(lat)} rows)")
                page.hover("#scatter circle[tabindex]", force=True)
                if page.is_hidden("#tip2"):
                    fails.append("scatter tooltip did not appear on hover")
                page.mouse.move(0, 0)
                # bound headline numbers
                notes.append("verdict: " + page.inner_text("#verdict").replace("\n", " ")[:400])
                # tooltip interaction
                page.hover("#chart rect")
                if page.is_hidden("#tip"):
                    fails.append("chart tooltip did not appear on hover")
                page.mouse.move(0, 0)
                if args.validator:
                    src = Path(args.validator).read_text()
                    res_v = page.evaluate("""async ([src, light, dark]) => {
                        const url = URL.createObjectURL(new Blob([src], {type: 'text/javascript'}));
                        const m = await import(url);
                        const brief = r => JSON.stringify(r).slice(0, 4000);
                        return [brief(m.validate(light.split(','), {mode: 'light', surface: '#fcfcfb'})),
                                brief(m.validate(dark.split(','), {mode: 'dark', surface: '#1a1a19'}))]; }""",
                                          [src, LIGHT, DARK])
                    for mode, r in zip(["light", "dark"], res_v):
                        notes.append(f"palette validator ({mode}): {r[:900]}")
                        if '"FAIL"' in r or '"fail"' in r:
                            fails.append(f"palette validator reported FAIL in {mode} mode")
            if args.shots:
                Path(args.shots).mkdir(parents=True, exist_ok=True)
                page.screenshot(path=f"{args.shots}/{label}_light.png", full_page=True)
                page.click("#themeBtn")
                page.wait_for_timeout(200)
                page.screenshot(path=f"{args.shots}/{label}_dark.png", full_page=True)
                page.click("#themeBtn")
            if errors:
                fails.append(f"[{label}] console errors: {errors}")
            if bad:
                fails.append(f"[{label}] failed requests: {bad}")
            ctx.close()
        browser.close()
    for n in notes:
        print("OK  ", n)
    for f in fails:
        print("FAIL", f)
    print("RESULT:", "PASS" if not fails else f"{len(fails)} problem(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
