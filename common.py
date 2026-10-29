"""Shared data loading for the IIT Kanpur air compressor dataset."""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "AirCompressor_Data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
CLASSES = ["Healthy", "Bearing", "Flywheel", "LIV", "LOV", "NRV", "Piston", "Riderbelt"]
N_SAMPLES = 50_000


def reading_number(path):
    return int(re.search(r"Reading(\d+)", Path(path).name).group(1))


def load_dataset():
    """Return X (n_clips, 50000) float32, y (int class index), reading numbers, file paths.

    Parsed once from the .dat text files, then cached as .npz.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "raw.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["reading"], list(d["paths"])
    X, y, reading, paths = [], [], [], []
    for ci, cls in enumerate(CLASSES):
        files = sorted((DATA_DIR / cls).glob("*.dat"), key=reading_number)
        for f in files:
            sig = np.array(f.read_text().strip().split(","), dtype=np.float32)
            assert sig.size == N_SAMPLES, (f, sig.size)
            X.append(sig)
            y.append(ci)
            reading.append(reading_number(f))
            paths.append(str(f.relative_to(ROOT)))
    X, y, reading = np.stack(X), np.array(y), np.array(reading)
    np.savez(cache, X=X, y=y, reading=reading, paths=np.array(paths))
    return X, y, reading, paths


class ResourceMonitor:
    """Context manager logging wall time, process CPU time, peak RSS and peak GPU memory.

    Appends one row to <out_csv> so cost-at-scale estimates use measured numbers.
    """

    def __init__(self, name, n_clips, out_csv=None, **extra):
        self.name, self.n_clips, self.extra = name, n_clips, extra
        self.out_csv = Path(out_csv) if out_csv else RESULTS_DIR / "resources.csv"

    def __enter__(self):
        import platform
        import threading
        import time

        import psutil
        self._proc = psutil.Process()
        self._peak_rss = self._proc.memory_info().rss
        self._stop = threading.Event()

        def poll():
            while not self._stop.wait(0.2):
                self._peak_rss = max(self._peak_rss, self._proc.memory_info().rss)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        self._cpu0 = sum(self._proc.cpu_times()[:2])
        self._t0 = time.perf_counter()
        self.host = f"{platform.node()} {psutil.cpu_count()}cpu"
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.host += " " + torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return self

    def __exit__(self, *exc):
        import csv
        import json
        import time
        wall = time.perf_counter() - self._t0
        cpu = sum(self._proc.cpu_times()[:2]) - self._cpu0
        self._stop.set()
        self._thread.join()
        gpu_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mb = torch.cuda.max_memory_allocated() / 2**20
        except ImportError:
            pass
        row = dict(name=self.name, host=self.host, n_clips=self.n_clips, wall_s=round(wall, 2),
                   cpu_s=round(cpu, 2), ms_per_clip=round(1000 * wall / max(self.n_clips, 1), 2),
                   avg_cpu_cores=round(cpu / max(wall, 1e-9), 2),
                   peak_rss_mb=round(self._peak_rss / 2**20), peak_gpu_mb=round(gpu_mb),
                   extra=json.dumps(self.extra))  # one column, so every row has the same fields
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        new = not self.out_csv.exists()
        with open(self.out_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print("[resources]", row)


def block_split(y, reading, train_frac=0.7):
    """Protocol C: lowest-numbered 70% of readings per class train, highest 30% test."""
    train = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        order = idx[np.argsort(reading[idx])]
        train[order[: int(round(train_frac * len(order)))]] = True
    return train, ~train


def zeroshot_subset(y, n_per_class=20, seed=0):
    """Protocol D: fixed stratified subset of clip indices."""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate(
        [rng.choice(np.where(y == c)[0], n_per_class, replace=False) for c in np.unique(y)]))
