"""
Cross-lag residual correlation, lags 1..K (reviews.md R2 point 1).

R2's complaint: "The four images in Figure 5(b) are almost identical, and thus
fail to exhibit both the contemporaneous spatial structure and the cross-lag
temporal persistence in the traffic residuals."

The contemporaneous (lag-0) numbers already exist from the kgxJ W1 study
(curvature_bottleneck_residual_corr_timeresolved.py, whose JSON output this
script's checkpoint/span/seed match exactly, so lag-0 here should reproduce
that study's pooled contemporaneous correlation as a consistency check). This
script reuses the SAME residual array (via the same collect_residuals() call
-- it is not cached to disk anywhere, so it is recomputed, not re-derived) and
adds the lagged version: for lag ell = 1..K native 5-minute steps,

    corr_lag[ell]_ij = corrcoef( residual_i(t), residual_j(t+ell) )

over the (strictly increasing, contiguous) rolling-origin time axis. Reported
overall (mean |corr| vs lag, a decay curve directly answering R2's complaint
that persistence isn't shown) and stratified by the curvature bottleneck
quartile (reusing the W1 study's edge/bottleneck machinery), to see whether
bottleneck edges carry more or less lagged cross-correlation than the rest of
the network -- extending, not just repeating, the contemporaneous W1 finding.

CAVEAT (inherited from curvature_bottleneck_residual_corr_timeresolved.py):
the evaluation span is IN-SAMPLE (the held-out test tail is only 2.9 hours;
see that module's docstring for why). This script reports the same caveat
because it reuses the same residuals.

Usage:
    python src/curvature_cross_lag_residual_corr.py \\
        --checkpoint logs/xLSTM/brussels_batch_curvature_.../checkpoints/epoch=17-val_loss=64.74.ckpt \\
        --dataset brussels --model xLSTM --span_start 2023-11-20 --span_end 2023-12-10 --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

if not hasattr(np, "float"):
    np.float = float

import pandas as pd
import pickle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_graph_diagnostics import load_curvature_state
from curvature_bottleneck_residual_corr import edge_arrays, independence_noise_floor, quantile_bin_summary
from curvature_bottleneck_residual_corr_timeresolved import (
    build_edge_covariates,
    build_time_index,
    collect_residuals,
)

DEFAULT_CHECKPOINT = (
    "logs/xLSTM/brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_"
    "LossLRw1.0_RegW2.5_TrainL_False/checkpoints/epoch=17-val_loss=64.74.ckpt"
)
JSON_OUT = "metrics_stat/xLSTM/brussels_xlstm_cross_lag_residual_corr.json"
FIG_OUT = "visualizations/brussels_xlstm_cross_lag_residual_corr.pdf"
K_MAX_LAG = 12  # matches Q=12, i.e. up to the full forecast horizon (1 hour @ 5min)


def contiguous_runs(origin_time_idx: np.ndarray) -> List[np.ndarray]:
    """Indices (into the origin axis) of maximal runs with unit spacing."""
    diffs = np.diff(origin_time_idx)
    breaks = np.flatnonzero(diffs != 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [len(origin_time_idx)]])
    return [np.arange(s, e) for s, e in zip(starts, stops)]


def lagged_cross_corr(residuals_h1: np.ndarray, runs: List[np.ndarray], lag: int) -> np.ndarray:
    """corr_ij = corrcoef(residual_i(t), residual_j(t+lag)), pooling pairs across
    all contiguous runs long enough to supply at least one (t, t+lag) pair."""
    N = residuals_h1.shape[1]
    X_list, Y_list = [], []
    for run in runs:
        if len(run) <= lag:
            continue
        X_list.append(residuals_h1[run[:-lag]])
        Y_list.append(residuals_h1[run[lag:]])
    if not X_list:
        raise ValueError(f"no contiguous run long enough for lag={lag}")
    X = np.concatenate(X_list, axis=0)  # (T', N) at time t
    Y = np.concatenate(Y_list, axis=0)  # (T', N) at time t+lag

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cov = Xc.T @ Yc / (X.shape[0] - 1)                      # (N, N), cov[i,j] = cov(X_i, Y_j)
    std_x = Xc.std(axis=0, ddof=1)
    std_y = Yc.std(axis=0, ddof=1)
    corr = cov / np.outer(std_x, std_y)
    return corr, X.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", default="brussels")
    parser.add_argument("--model", default="xLSTM")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--span_start", default="2023-11-20")
    parser.add_argument("--span_end", default="2023-12-10")
    parser.add_argument("--chunk_origins", type=int, default=288)
    parser.add_argument("--k_max_lag", type=int, default=K_MAX_LAG)
    parser.add_argument("--n_qbins", type=int, default=4)
    parser.add_argument("--locations-path", default="./datasets/locations.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=JSON_OUT)
    parser.add_argument("--fig-out", default=FIG_OUT)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    with open("./datasets/pred_horizon_dict_v1.pkl", "rb") as f:
        prediction_horizon = pickle.load(f)[args.dataset]
    with open("./datasets/pred_rolling_dict_v1.pkl", "rb") as f:
        num_pred_rolling = pickle.load(f)[args.dataset]

    data = pd.read_csv(f"./datasets/{args.dataset}.csv")
    time_index = build_time_index(args.dataset, data)

    max_idx = int(data["time_idx"].max())
    validation_cutoff = max_idx - prediction_horizon - num_pred_rolling + 1
    training_cutoff = validation_cutoff - (max_idx - validation_cutoff)

    span_start = pd.Timestamp(args.span_start)
    span_end = pd.Timestamp(args.span_end) + pd.Timedelta(days=1)
    in_span = (time_index >= span_start) & (time_index < span_end)
    span_idx = time_index.index[in_span].to_numpy()
    first_origin = int(span_idx.min())
    last_origin = min(int(span_idx.max()) - prediction_horizon + 1, training_cutoff - prediction_horizon + 1)
    print(f"Evaluation span: origins {first_origin}-{last_origin} "
          f"({time_index.loc[first_origin]} -> {time_index.loc[last_origin + prediction_horizon - 1]})")

    print("Running inference to collect residuals (same call as the W1 timeresolved study) ...")
    residuals, origin_time_idx, node_order = collect_residuals(
        args.checkpoint, args.model, args.dataset, device,
        args.batch_size or int(data["sensor"].nunique()),
        args.n_samples, first_origin, last_origin, args.chunk_origins, args.seed,
    )
    R, N, Q = residuals.shape
    print(f"residuals: R={R} origins, N={N} nodes, Q={Q} horizons")
    residuals_h1 = residuals[:, :, 0]

    runs = contiguous_runs(origin_time_idx)
    run_lengths = [len(r) for r in runs]
    print(f"{len(runs)} contiguous run(s) of origins, lengths={run_lengths[:5]}"
          f"{'...' if len(runs) > 5 else ''} (total covered={sum(run_lengths)}/{R})")

    print("Loading curvature module state + edges ...")
    W, params = load_curvature_state(Path(args.checkpoint))
    assert W.shape == (N, N)
    edge_i, edge_j, b_edge, _ = edge_arrays(W, params["kappa_0"], params["tau"])
    print(f"{len(edge_i)} edges; bottleneck score b in [{b_edge.min():.3f}, {b_edge.max():.3f}]")

    lag_results = []
    for lag in range(0, args.k_max_lag + 1):
        corr, n_pairs_time = lagged_cross_corr(residuals_h1, runs, lag) if lag > 0 else (
            np.corrcoef(residuals_h1, rowvar=False), R,
        )
        offdiag_mask = ~np.eye(N, dtype=bool)
        mean_abs_offdiag = float(np.abs(corr[offdiag_mask]).mean())
        mean_abs_diag = float(np.abs(np.diag(corr)).mean())  # node self-lag autocorrelation, reference only

        corr_abs_edge = np.abs(corr[edge_i, edge_j])
        floor = independence_noise_floor(n_pairs_time)
        qbins = quantile_bin_summary(b_edge, corr_abs_edge, args.n_qbins, floor=floor)

        lag_results.append({
            "lag_native_steps": lag,
            "lag_minutes": lag * 5,
            "n_time_pairs": int(n_pairs_time),
            "independence_noise_floor": floor,
            "mean_abs_corr_offdiag_all_pairs": mean_abs_offdiag,
            "mean_abs_corr_self_lag_reference": mean_abs_diag,
            "mean_abs_corr_edges_only": float(corr_abs_edge.mean()),
            "quantile_bins_by_bottleneck_score": qbins,
        })
        print(
            f"lag={lag:2d} ({lag*5:3d} min, T'={n_pairs_time:5d}): "
            f"mean|corr| offdiag(all pairs)={mean_abs_offdiag:.4f}  "
            f"edges-only={lag_results[-1]['mean_abs_corr_edges_only']:.4f}  "
            f"self-lag(ref)={mean_abs_diag:.4f}  floor={floor:.4f}  "
            f"Q0..Q{args.n_qbins-1}=" + ",".join(f"{r['mean_abs_corr']:.3f}" for r in qbins)
        )

    results = {
        "script": "curvature_cross_lag_residual_corr.py",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "model": args.model,
        "n_nodes": N,
        "n_edges": int(len(edge_i)),
        "n_samples_mc": args.n_samples,
        "seed": args.seed,
        "sample_status": "IN-SAMPLE (same span/checkpoint as curvature_bottleneck_residual_corr_timeresolved.py; see that module's docstring for why the held-out tail is too short to use here)",
        "span": {
            "span_start": args.span_start, "span_end": args.span_end,
            "n_origins_used": int(R), "n_contiguous_runs": len(runs), "run_lengths": run_lengths,
        },
        "k_max_lag_native_steps": args.k_max_lag,
        "definition": (
            "corr_lag[ell]_ij = corrcoef(residual_i(t), residual_j(t+ell)) over the pooled "
            "(t, t+ell) pairs from all contiguous rolling-origin runs; lag=0 is the plain "
            "contemporaneous correlation (consistency check against the W1 study). "
            "'edges-only' restricts to graph edges (i,j); 'offdiag(all pairs)' is over every "
            "node pair, not just edges. 'self-lag' is corr(residual_i(t), residual_i(t+ell)), "
            "reported as a reference, not a cross-sensor quantity."
        ),
        "lag_results": lag_results,
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {json_out}")

    make_figure(results, Path(args.fig_out))
    print(f"Wrote {args.fig_out}")


def make_figure(results: Dict, fig_out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["lag_results"]
    lags = [r["lag_minutes"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot(lags, [r["mean_abs_corr_offdiag_all_pairs"] for r in rows], marker="o", color="crimson", label="all pairs (off-diag)")
    ax.plot(lags, [r["mean_abs_corr_edges_only"] for r in rows], marker="s", color="darkorange", label="graph edges only")
    ax.plot(lags, [r["mean_abs_corr_self_lag_reference"] for r in rows], marker="^", color="steelblue", linestyle="--", label="self-lag (reference)")
    ax.axhline(rows[0]["independence_noise_floor"], color="black", linestyle=":", linewidth=1, label="independence floor")
    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel(r"mean $|\rho|$ (residuals)")
    ax.set_title("Cross-lag residual correlation decay")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    n_bins = len(rows[0]["quantile_bins_by_bottleneck_score"])
    cmap = plt.get_cmap("viridis")
    for b in range(n_bins):
        y = []
        for r in rows:
            match = [q for q in r["quantile_bins_by_bottleneck_score"] if q["bin"] == b]
            y.append(match[0]["mean_abs_corr"] if match else np.nan)
        ax.plot(lags, y, marker="o", color=cmap(b / max(n_bins - 1, 1)), label=f"bottleneck Q{b}")
    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel(r"mean $|\rho_{ij}|$ (edges in bin)")
    ax.set_title("By bottleneck-score quartile")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Cross-lag residual correlation — {results['dataset']}/{results['model']}")
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
