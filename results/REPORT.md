# Can models tell *which* fault an air compressor has from its sound?

Exploratory baseline on the IIT Kanpur air-compressor dataset (Verma et al., IEEE Trans. Reliability 2016).
1,800 one-second recordings, 8 classes (Healthy + 7 seeded faults), 225 per class. Run 2026-10-28/29.

## TL;DR

| Question | Answer |
|---|---|
| Can it be diagnosed at all? | **Yes, on this dataset, trivially.** Simple FFT/MFCC features + SVM: **100%** (5-fold CV; published ≈99.8%). Every frozen pretrained model: 98.5–100%. |
| Is that because neighbouring recordings leak? | **No.** Training on the first 70% of recordings and testing on the last 30% still gives 100% (handcrafted, BEATs, Dasheng, FISHER). |
| How many labelled examples per fault? | **~5 per fault gives ~96–97%; 10 gives ~99%.** With 1 example: ~75–80%. Best: FISHER (industrial-signal model) at 97.4% with 5. |
| Which faults get confused? | Valve faults with each other (LIV ↔ NRV ↔ LOV) and mechanical faults with each other (Bearing ↔ Flywheel ↔ Piston); for the best model also Riderbelt → LOV. Healthy is never confused. |
| Does zero-shot work? | **No.** CLAP text prompts: 14–16%. Gemini 3.1 Pro listening to the clip: 16%. Chance is 12.5%. |
| Demo-worthy? | **Yes, as a "teach it 5 examples per fault" demo. No, as a "the AI recognises faults on its own" demo.** The main open risk is that everything comes from one compressor in one lab (see caveats). |

## 1. What I did

- **Models.** Every pretrained model was used frozen (no fine-tuning): the model turns each clip into a numeric "fingerprint" vector (an *embedding*), and a small classifier is trained on those vectors.
  - Handcrafted features: 64 Welch band energies, MFCC mean/std + deltas, spectral and time-domain statistics.
  - General audio models: BEATs (iter3+ AS2M), Dasheng-base and CLAP (LAION `larger_clap_general`).
  - Industrial-signal models: ECHO-small and FISHER-small.
  - Classifiers on top: SVM, kNN, logistic regression.
- **Protocols.** A = 5-fold stratified CV. B = few-shot, k = 1/3/5/10 clips per class, 10 random draws, the same draws for every model. C = block split by Reading number. D = zero-shot on a fixed 20-per-class subset (160 clips).
- **Metric.** Macro (class-balanced) accuracy everywhere.
- **Where it ran.**
  - Handcrafted features, evaluation and the Gemini calls ran on the laptop.
  - Pretrained embeddings were extracted on a Kaggle T4 GPU, in about 7 minutes of GPU work in total. The Kaggle dataset and notebook are private.

### Sample rate: resolved to 50 kHz, i.e. each file is 1 second

The files have 50,000 samples. MathWorks treats them as 16 kHz, but the evidence points to the native 50 kHz:

- The original recordings are documented as 5 s at 50 kHz (250,000 samples), from the arXiv 2402.09957 description of this dataset. So 50,000 samples = 1 s at the original rate.
- **Physics check.** The compressor's piston cycle shows up in every class as a strong envelope rhythm of 7.3 cycles per file.
  - At 50 kHz that is 7.3 Hz ≈ **440 rpm**, a normal crank speed for a belt-driven single-stage compressor.
  - At 16 kHz it would be ~140 rpm; at 10 kHz (a decimated 5 s file) ~90 rpm. Both are implausibly slow.
- The spectrum shows the fingerprints of undocumented preprocessing: a high-pass filter at ~375 Hz, an exact notch at 10 kHz, and a roll-off to a flat floor above ~15 kHz (plot: `investigation/psd_fine.png`).
- These don't settle the question alone, but they are consistent with 50 kHz and leave no reason to assume 16 kHz.

As requested, I ran every pretrained model under both assumptions. **The conclusions don't change**: supervised works, zero-shot doesn't, and Valve↔Valve / Mechanical↔Mechanical are the confusions. What does change:

- **The general audio models (BEATs, Dasheng, ECHO) do 2–8 points better in few-shot under "16 kHz".** Reading the file as 16 kHz plays the recording ~3× slower, which shifts all the content into the frequency range these models were trained on. It also means the general models see all 50,000 samples instead of a 16 kHz resample that drops two thirds of them. This is a useful engineering trick, not evidence about the true rate.
- **FISHER, the model built for arbitrary sample rates, is best at the true 50 kHz.**

## 2. Results

**Supervised, all data (A) and block split (C)**, SVM, macro accuracy %:

| Features | A @50k | C @50k | A @16k | C @16k |
|---|---|---|---|---|
| Handcrafted | 100.0 | 100.0 | 100.0 | 100.0 |
| BEATs | 99.9 | 100.0 | 100.0 | 100.0 |
| Dasheng | 99.9 | 100.0 | 100.0 | 100.0 |
| FISHER | 100.0 | 100.0 | 100.0 | 98.9 |
| ECHO | 98.5 | 99.1 | 99.6 | 99.8 |
| CLAP | 99.2 | 98.1 | 98.9 | 97.8 |

**Few-shot (B)**, logistic regression on frozen features, macro accuracy % (mean ± std over 10 draws):

| Features (best rate) | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| **FISHER @50k** | 77.8 ± 5.2 | **94.1 ± 1.8** | **97.4 ± 1.6** | **99.1 ± 0.8** |
| BEATs @16k | **80.1 ± 4.9** | 93.6 ± 2.2 | 96.4 ± 1.6 | 98.3 ± 1.1 |
| Handcrafted @50k | 74.2 ± 5.0 | 90.0 ± 3.0 | 96.0 ± 2.0 | 98.7 ± 0.9 |
| Dasheng @16k | 73.1 ± 6.1 | 90.8 ± 3.1 | 94.8 ± 2.2 | 98.3 ± 1.6 |
| ECHO @16k | 56.0 ± 4.2 | 81.6 ± 4.2 | 90.9 ± 2.9 | 97.3 ± 1.2 |
| CLAP @16k | 56.1 ± 4.3 | 74.6 ± 3.3 | 83.3 ± 3.0 | 89.9 ± 1.6 |

kNN is consistently 5–15 points worse than logistic regression at small k. Every row and both rates are in `results.csv`.

- **Comparing the top models.** FISHER beats handcrafted features in 8–9 of the 10 identical draws, but only by +1 to +4 points.
- **Practical reading.** The best pretrained model buys a few points when labels are very scarce. It is not the difference between working and not working on this data.

**Zero-shot (D)**, 160 clips:

| Method | @50k | @16k | Behaviour |
|---|---|---|---|
| CLAP, 3 prompt templates per class | 16.3% | 14.4% | Assigns nearly every clip to 2 classes (Bearing/Riderbelt or Healthy/Riderbelt) |
| Gemini 3.1 Pro (via an API gateway), identical prompt for every clip, answer constrained to the 8 labels | 16.3% | 15.6% | Says "Flywheel" for 43% of clips; confident, plausible-sounding explanations that are wrong |

Chance is 12.5%. Confusion matrices for both are in `confusion/`.

## 3. What gets diagnosed reliably and what gets confused, and why

With 5 examples per fault, the best model (FISHER @50k, `confusion/best_fewshot_k5.png`) gets:

- **Healthy: 100%**, and Healthy is never missed by *any* model, so fault vs no-fault is solved. Healthy's spectrum is 4.7–8 dB away from every fault class, roughly double the gap between any two faults.
- **NRV, Flywheel, Piston, LIV:** 99–100%.
- **Bearing (95%, mostly called Flywheel), LOV (95%, called LIV/NRV) and Riderbelt (95.5%, called LOV):** the weak spots.
- **The other models confuse things in two families** (aggregated over 10 draws):
  - **Valve faults with each other** (LIV ↔ NRV for the pretrained models, LIV ↔ LOV for handcrafted). Spectrally, LIV and NRV differ by 2.4 dB on average, and that difference sits almost entirely above 10 kHz. The general audio models resample to 16 kHz and partly or wholly discard that band, which plausibly explains why they confuse this pair most. A leaking valve of any kind mostly adds flow noise, so this confusion is physically expected.
  - **Mechanical/rotating faults with each other** (Bearing ↔ Flywheel ↔ Piston). These are the closest classes in the dataset: Bearing vs Flywheel differ by 1.7 dB, mostly below 500 Hz. That is less than the spread of recordings *within* a single class (1.5–2.5 dB).
- **The misses themselves** (`misclassified/` WAVs, plus `investigation/misclassified_spectra.png` with each miss's spectrum against both class means and its spectrogram):
  - Almost every misclassified clip is still *closer to its own class average* than to the predicted class. These are borderline recordings, not mislabelled files.
  - A few are louder or quieter than their class (|z| > 1.5) or run at a slightly different crank speed.
  - Please listen to them yourself. I compared them numerically and visually; I cannot listen to audio.

**Data-quality finding: a splice click in 13% of files.** 227 of the 1,800 files contain a step discontinuity at exactly sample 25,475 (the middle of the clip), heard as a click. This looks like a preprocessing artefact where two segments were joined.

- It is unevenly distributed: 0 Healthy files, 61 NRV, 56 LIV, 6 LOV.
- It makes clips *harder*, not easier: the error rate is 1.4–1.8× higher on clicked clips for every model checked. So it is not a shortcut the models exploit.
- It is still a sign the "preprocessed" files were edited in undocumented ways.

## 4. Why this result is weaker than "100%" suggests

1. **One compressor, one lab, one microphone position.** Every result here is "telling apart recordings from the same machine". Nothing says a model trained here would work on another compressor, another room or another microphone. That is the claim a customer cares about, and it is untested.
2. **Each fault was most likely recorded as its own session.** A fault had to be physically introduced, then 225 recordings taken. The class label is therefore mixed up with everything else about that session: background noise, mic placement, temperature, tank pressure.
   - The block split rules out drift *within* a session, but nothing in this dataset can separate "the fault" from "the day it was recorded".
   - The fact that the 64-band average spectrum alone gives 100% is consistent with both explanations.
3. **The task is easy.** Handcrafted features already score 100%, so this dataset cannot show what pretrained models add, except under label scarcity (k ≤ 3).
4. **The ~440 rpm figure is an inference.** The compressor's rated speed isn't documented in anything I could access.

## 5. Cost at scale (measured)

Measured on Kaggle (4 vCPU + T4) and the laptop (`resources.csv`). The scenario is each monitored machine sending one 1-s clip per minute, with boxes planned at 50% load. Prices are list prices checked 2026-10-29 (`cost_at_scale.csv`; edit `cost_estimate.py` to change assumptions).

| Pipeline | Compute per 1-s clip | One box handles | Cost per machine / month |
|---|---|---|---|
| Handcrafted + SVM, CPU | 37 ms on 1 core | ~1,600 machines per c7i.xlarge ($130/mo) | **≈ $0.08** |
| Dasheng + logreg, CPU | 20 ms on 4 vCPU | ~1,500 machines | ≈ $0.09 |
| FISHER + logreg, CPU | 63 ms on 4 vCPU | ~480 machines | ≈ $0.27 |
| FISHER + logreg, T4 GPU | 1.9 ms | ~16,000 machines per g4dn.xlarge ($384/mo) | ≈ $0.02 (only at large scale) |
| BEATs + logreg, CPU | 153 ms on 4 vCPU | ~200 machines | ≈ $0.66 |
| Gemini 3.1 Pro zero-shot | ~6 s latency; ~25 audio + 140 text in, ~450 out/thinking tokens | API | **≈ $254** (at chance accuracy) |

- **Model sizes are small.** Peak GPU memory was 0.1–1.3 GB and peak RAM 2–3.7 GB per model. Model load takes 1–35 s, once.
- **Training cost is negligible.** Fitting the classifier on 40 labelled clips takes milliseconds.
- **This experiment's paid cost** was ≈ $1.94 of Gemini tokens at Google list price (320 calls). The gateway's internal charge may differ. Kaggle GPU time was free.
- **Bottom line:** a demo runs on a laptop with no cloud at all. At fleet scale the ML compute is cents per machine per month on CPU. Audio capture, transport and storage, not ML, will dominate the cost.

## 6. Verdict

- **Worth a demo, if it is framed as "few-shot enrolment".**
  - The pitch: record a handful of examples of each known fault, and the system recognises them from then on (~97% here with 5 examples per fault).
  - FISHER, or just handcrafted features, on a laptop, in real time.
  - Healthy vs faulty is rock-solid; "which valve" is the weakest part.
- **Not worth a demo if the pitch is "an AI that recognises faults it hasn't been shown".** Both zero-shot routes are at chance. Gemini in particular produces confident, plausible-sounding explanations for wrong answers, which is the worst possible demo failure.
- **Before anyone makes claims to a customer, run one cheap test:** record a second machine (or the same machine on a different day or with a different mic position), train on this data plus k examples from the new setting, and test on the rest. If accuracy holds, there is a product story. If it collapses, the 100% here was "this machine, this session".

## Reproduce

```
.venv\Scripts\python features_handcrafted.py --sr 50000   # and --sr 16000
python kaggle/build_kernel.py && kaggle kernels push -p kaggle/kernel   # GPU embeddings -> copy emb_*.npy into cache/
.venv\Scripts\python zeroshot_clap.py
.venv\Scripts\python zeroshot_gemini.py --sr 50k ; --sr 16k   # needs GEMINI_API_KEY (or a gateway URL + token) in .env
.venv\Scripts\python evaluate.py          # results.csv, confusion/, misclassified/
.venv\Scripts\python analyze_misses.py ; .venv\Scripts\python cost_estimate.py
```

Implementation notes:
- **Dasheng was chosen over EAT** (task said "one of"), because it loads cleanly from Hugging Face.
- **BEATs weights** come from the `lpepino/beats_ckpts` Hugging Face mirror, because Microsoft's original links are OneDrive.
- **FISHER fails under `transformers` 5.x.** Its remote code never calls `post_init()`, giving `'FISHERModel' object has no attribute 'all_tied_weights_keys'`. It is run in its own process with `transformers==4.57.1` (see `requirements-gpu.txt`).
- **Amplitude** is scaled by one global constant (1/4.3), not per clip, so loudness differences are preserved.

Sources for prices and data facts:
- [Google Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [g4dn.xlarge pricing](https://instances.vantage.sh/aws/ec2/g4dn.xlarge)
- [c7i.xlarge pricing](https://instances.vantage.sh/aws/ec2/c7i.xlarge)
- [arXiv 2402.09957: dataset recording parameters](https://arxiv.org/pdf/2402.09957)
- [MathWorks example (16 kHz assumption)](https://www.mathworks.com/help/audio/ug/acoustics-based-machine-fault-recognition.html)
- [IIT Kanpur dataset page](https://iitk.ac.in/iil/datasets/)
