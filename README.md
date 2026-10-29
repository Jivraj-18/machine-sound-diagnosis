# Machine sound diagnosis: which fault does the compressor have?

Exploratory baseline: can existing models tell **which** fault an air compressor has from a 1-second sound clip
(diagnosis, not just normal/abnormal)?

**Live write-up:** https://jivraj-18.github.io/machine-sound-diagnosis/ · Technical report: [`results/REPORT.md`](results/REPORT.md)

## Results in one table

Macro accuracy on the IIT Kanpur air-compressor dataset (8 states, 225 one-second clips each; chance = 12.5%).
Source: [`results/results.csv`](results/results.csv).

| | 5-fold CV | 5 examples per fault | 1 example per fault | Zero-shot |
|---|---|---|---|---|
| FISHER (industrial model, 50 kHz) + logistic regression | 100.0% | **97.4%** | 77.8% | – |
| Handcrafted FFT/MFCC features | 100.0% | 96.0% | 74.2% | – |
| BEATs (16 kHz) | 100.0% | 96.4% | 80.1% | – |
| CLAP | 99.2% | 80.1% | 51.5% | 16.3% |
| Gemini 3.1 Pro (audio LLM) | – | – | – | 16.3% |

- **Supervised diagnosis works on this data; zero-shot does not.**
- **Healthy is never confused with a fault.** The errors are valve faults with each other and mechanical faults (Bearing/Flywheel/Piston) with each other.
- **All recordings come from one compressor in one lab**, so transfer to other machines is untested.

## Repository layout

| File | What it does |
|---|---|
| `common.py` | data loading, block split, zero-shot subset, resource monitor |
| `features_handcrafted.py` | Welch band energies + MFCC + spectral/time statistics |
| `embed_pretrained.py` | frozen embeddings: BEATs, Dasheng, CLAP, ECHO, FISHER (GPU; ran on Kaggle via `kaggle/build_kernel.py`) |
| `zeroshot_clap.py`, `zeroshot_gemini.py` | zero-shot classification |
| `evaluate.py` | protocols A–D → `results/results.csv`, confusion matrices, misclassified clips |
| `analyze_misses.py` | spectral analysis of the errors, splice-click detection |
| `cost_estimate.py` | measured compute → cost-at-scale table |
| `build_site.py`, `docs/` | the static page (GitHub Pages serves `docs/`) |
| `verify_site.py` | browser check of the page: audio, images, phone width, numbers vs `results.csv` |

## Reproduce

1. Download `AirCompressor_Data.zip` from the [IIT Kanpur datasets page](https://iitk.ac.in/iil/datasets/) and unzip it to `data/` (giving `data/AirCompressor_Data/<Class>/*.dat`). The dataset is not included here.
2. Install: `pip install -r requirements.txt` (analysis, Python 3.11). Install `requirements-gpu.txt` on the GPU machine.
3. Run:

```
python features_handcrafted.py --sr 50000 && python features_handcrafted.py --sr 16000
python embed_pretrained.py --raw cache/raw.npz --out cache     # GPU
python zeroshot_clap.py
python zeroshot_gemini.py --sr 50k && python zeroshot_gemini.py --sr 16k   # GEMINI_API_KEY in env or .env
python evaluate.py
python analyze_misses.py && python cost_estimate.py && python build_site.py
python verify_site.py http://127.0.0.1:8000/    # after: python -m http.server 8000 --directory docs
```

## Data credit

N. K. Verma, R. K. Sevakula, S. Dixit and A. Salour, "Intelligent Condition Based Monitoring Using Acoustic Signals for
Air Compressors", *IEEE Transactions on Reliability*, 65(1), 2016. Data: https://iitk.ac.in/iil/datasets/.
This repository contains only the 21 one-second clips that the page plays: 8 "listen" examples and 13 misclassified clips.
The misclassified clips appear twice, as 48 kHz copies in `docs/assets/audio/` and at the original 50 kHz in `results/misclassified/`.
