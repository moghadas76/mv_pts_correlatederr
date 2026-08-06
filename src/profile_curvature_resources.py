"""
Profile runtime and memory for the Brussels/xLSTM curvature rewiring method.

The measured "curvature core" is the graph-theoretic method used by the
curvature covariance head:

    W -> Balanced Forman curvature kappa -> bottleneck score b -> W'

The script also reports the downstream graph diagnostics used in the reviewer
table, but keeps them separate from the core method cost.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from curvature_graph_diagnostics import (
    DEFAULT_CHECKPOINT,
    balanced_forman_curvature,
    bottleneck_indicator,
    connected_components,
    laplacian,
    load_curvature_state,
    scaled_kirchhoff,
    sweep_cut_rows,
)


DEFAULT_OUT = "metrics_stat/xLSTM/brussels_xlstm_curvature_resources.json"


def current_rss_mb() -> float:
    """Return current resident set size in MiB on Linux."""
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


class MemorySampler:
    """Poll process RSS while a short benchmark function runs."""

    def __init__(self, interval_s: float = 0.001):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples: List[float] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(current_rss_mb())
            time.sleep(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        self.samples.append(current_rss_mb())
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(current_rss_mb())

    @property
    def peak_mb(self) -> float:
        return max(self.samples) if self.samples else float("nan")


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def benchmark(
    name: str,
    fn: Callable[[], object],
    repeats: int,
    warmup: int,
) -> Tuple[object, Dict[str, float]]:
    for _ in range(warmup):
        fn()

    outputs = None
    times_ms: List[float] = []
    peak_rss_mb: List[float] = []
    rss_delta_mb: List[float] = []

    for _ in range(repeats):
        before = current_rss_mb()
        start = time.perf_counter()
        with MemorySampler() as sampler:
            outputs = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        after = current_rss_mb()

        times_ms.append(elapsed_ms)
        peak_rss_mb.append(sampler.peak_mb)
        rss_delta_mb.append(max(0.0, sampler.peak_mb - before, after - before))

    stats = {
        "name": name,
        "repeats": repeats,
        "warmup": warmup,
        "time_mean_ms": statistics.mean(times_ms),
        "time_std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "time_median_ms": statistics.median(times_ms),
        "time_p95_ms": percentile(times_ms, 95),
        "rss_peak_mean_mb": statistics.mean(peak_rss_mb),
        "rss_peak_max_mb": max(peak_rss_mb),
        "rss_delta_mean_mb": statistics.mean(rss_delta_mb),
        "rss_delta_max_mb": max(rss_delta_mb),
    }
    return outputs, stats


def curvature_core(W: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (W > 0).astype(float)
    kappa = balanced_forman_curvature(W)
    b = bottleneck_indicator(kappa, mask, params["kappa_0"], params["tau"])
    W_prime = W * (1.0 + params["lambda"] * b)
    W_prime = 0.5 * (W_prime + W_prime.T)
    return kappa, b, W_prime


def spectral_kirchhoff(W: np.ndarray, W_prime: np.ndarray) -> Dict[str, float]:
    giant = max(connected_components(W), key=len)
    W_giant = W[np.ix_(giant, giant)]
    Wp_giant = W_prime[np.ix_(giant, giant)]
    eig = np.linalg.eigvalsh(laplacian(W_giant))
    eig_prime = np.linalg.eigvalsh(laplacian(Wp_giant))
    return {
        "n_component": int(len(giant)),
        "lambda_2_original": float(eig[1]),
        "lambda_2_reweighted": float(eig_prime[1]),
        "scaled_kirchhoff_original": scaled_kirchhoff(eig, len(giant)),
        "scaled_kirchhoff_reweighted": scaled_kirchhoff(eig_prime, len(giant)),
    }


def conductance_diagnostics(W: np.ndarray, W_prime: np.ndarray, b: np.ndarray, top_k: int) -> Dict[str, float]:
    giant = max(connected_components(W), key=len)
    W_giant = W[np.ix_(giant, giant)]
    Wp_giant = W_prime[np.ix_(giant, giant)]
    b_giant = b[np.ix_(giant, giant)]
    rows = sweep_cut_rows(W_giant, Wp_giant, b_giant)
    top_rows = sorted(rows, key=lambda row: row["mean_boundary_b"], reverse=True)[:top_k]
    return {
        "top_k": top_k,
        "phi_original_mean": float(np.mean([row["phi_original"] for row in top_rows])),
        "phi_reweighted_mean": float(np.mean([row["phi_reweighted"] for row in top_rows])),
        "phi_boundary_only_mean": float(np.mean([row["phi_boundary_only"] for row in top_rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_start = time.perf_counter()
    load_rss_before = current_rss_mb()
    W, params = load_curvature_state(Path(args.checkpoint))
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    load_rss_after = current_rss_mb()

    (kappa, b, W_prime), core_stats = benchmark(
        "curvature_core_kappa_b_reweight",
        lambda: curvature_core(W, params),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    _, spectral_stats = benchmark(
        "spectral_scaled_kirchhoff",
        lambda: spectral_kirchhoff(W, W_prime),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    _, conductance_stats = benchmark(
        "local_conductance_fiedler_sweep",
        lambda: conductance_diagnostics(W, W_prime, b, args.top_k),
        repeats=args.repeats,
        warmup=args.warmup,
    )

    edge_mask = np.triu(W > 0, k=1)
    result = {
        "checkpoint": args.checkpoint,
        "graph": {
            "n_nodes": int(W.shape[0]),
            "n_edges": int(edge_mask.sum()),
            "dtype": str(W.dtype),
        },
        "load": {
            "time_ms": load_time_ms,
            "rss_before_mb": load_rss_before,
            "rss_after_mb": load_rss_after,
            "rss_delta_mb": max(0.0, load_rss_after - load_rss_before),
        },
        "learned_parameters": params,
        "benchmarks": {
            core_stats["name"]: core_stats,
            spectral_stats["name"]: spectral_stats,
            conductance_stats["name"]: conductance_stats,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Graph: {result['graph']['n_nodes']} nodes, {result['graph']['n_edges']} undirected edges")
    print(f"Checkpoint load: {load_time_ms:.2f} ms, RSS delta {result['load']['rss_delta_mb']:.2f} MiB")
    for stats in result["benchmarks"].values():
        print(
            f"{stats['name']}: "
            f"{stats['time_mean_ms']:.2f} +/- {stats['time_std_ms']:.2f} ms "
            f"(median {stats['time_median_ms']:.2f}, p95 {stats['time_p95_ms']:.2f}); "
            f"peak RSS delta {stats['rss_delta_max_mb']:.2f} MiB"
        )


if __name__ == "__main__":
    main()
