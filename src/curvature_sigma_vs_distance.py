"""
Implied node-level spatial covariance vs. graph distance / effective resistance.

Rebuttal Priority 3(a) (kgxJ W1, AC condition 3): the resolvent-decay argument
for Sigma is proven in node space (Q = beta*L + c*I), but the deployed model
builds its precision in a projected rank-R latent space:

    L'_R = P^T L' P                      (R, R)
    Q_R  = alpha*I_R + beta*L'_R
    G    = Q_R^{-1}                      (R, R)   <- what the model actually uses

so the node-space identity does not transfer exactly. This script constructs
the natural pullback of G to node space,

    Sigma_node = P G P^T                 (N, N)

(PSD because Sigma_node = (P G^{1/2})(P G^{1/2})^T) and checks empirically
whether |Sigma_node_ij| decays more slowly with graph distance / effective
resistance once the curvature-based edge reweighting (lambda) is switched on,
i.e. whether the tail thickens as the motivation section claims.

"With rewiring" uses the checkpoint's learned lambda, kappa_0, tau (the
deployed L'). "Without rewiring" recomputes the identical forward pass with
lambda pinned to 0 (so L' collapses to the original graph Laplacian L),
holding P, alpha, beta fixed at their learned values -- isolating the causal
effect of curvature-guided reweighting from everything else the model learned.

Both conditions call the model's own `_reweight_laplacian` (dynamic_graph.py)
so this script can never silently diverge from what the checkpoint actually
computes at inference time.

Usage:
    python src/curvature_sigma_vs_distance.py \\
        --checkpoint logs/xLSTM/brussels_batch_curvature_.../checkpoints/epoch=17-val_loss=64.74.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_graph_diagnostics import find_curvature_prefix, torch_load_checkpoint
from dynamic_graph import _reweight_laplacian

DEFAULT_CHECKPOINT = (
    "logs/xLSTM/brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_"
    "LossLRw1.0_RegW2.5_TrainL_False/checkpoints/epoch=17-val_loss=64.74.ckpt"
)
DEFAULT_JSON_OUT = "metrics_stat/xLSTM/brussels_xlstm_sigma_vs_distance.json"
DEFAULT_FIG_OUT = "visualizations/brussels_xlstm_sigma_vs_distance.pdf"
# Matches the class default in dynamic_graph.CurvatureAwareGraphPrecision_LearnP
# and the (unoverridden) curvature_sigma_min=1e-4 default threaded through
# BatchMGDCurvature_Kernel in loss.py -- not present in the checkpoint's
# state_dict because it is a Python float, not a Parameter/buffer.
SIGMA_MIN = 1e-4


def load_curvature_module_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = find_curvature_prefix(state_dict)

    static_adj = state_dict[prefix + "static_adj"].detach().double()
    P = state_dict[prefix + "P"].detach().double()
    log_alpha = state_dict[prefix + "log_alpha"].detach().double()
    log_beta = state_dict[prefix + "log_beta"].detach().double()
    kappa_0 = state_dict[prefix + "kappa_0"].detach().double()
    log_tau = state_dict[prefix + "log_tau"].detach().double()

    if prefix + "lam" in state_dict:
        lam = state_dict[prefix + "lam"].detach().double()
    elif prefix + "log_lam" in state_dict:
        lam = torch.exp(state_dict[prefix + "log_lam"].detach().double())
    else:
        raise KeyError("Could not find either learned `lam` or `log_lam`")

    return {
        "static_adj": static_adj,
        "P": P,
        "alpha": torch.exp(log_alpha),
        "beta": torch.exp(log_beta),
        "lam": lam,
        "kappa_0": kappa_0,
        "tau": torch.exp(log_tau),
    }


def implied_node_sigma(
    static_adj: torch.Tensor,
    P: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lam: torch.Tensor,
    kappa_0: torch.Tensor,
    tau: torch.Tensor,
) -> np.ndarray:
    """Exactly reproduces CurvatureAwareGraphPrecision_LearnP.forward(), then
    pulls G (R, R) back to node space via Sigma_node = P_norm G P_norm^T."""
    rank = P.shape[1]
    L_prime = _reweight_laplacian(static_adj, lam, kappa_0, tau)  # (N, N), same fn the model calls

    P_norm = F.normalize(P, dim=0)  # (N, R), matches forward() exactly
    L_R = P_norm.t() @ L_prime @ P_norm
    Q = alpha * torch.eye(rank, dtype=L_R.dtype) + beta * L_R
    Q = 0.5 * (Q + Q.t()) + SIGMA_MIN * torch.eye(rank, dtype=Q.dtype)

    G = torch.linalg.inv(Q)
    G = 0.5 * (G + G.t())

    Sigma_node = P_norm @ G @ P_norm.t()  # (N, N)
    return Sigma_node.numpy()


def hop_distance_matrix(W: np.ndarray) -> np.ndarray:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    adjacency = csr_matrix((W > 0).astype(np.float64))
    dist = shortest_path(adjacency, directed=False, unweighted=True)
    return dist


def effective_resistance_matrix(W: np.ndarray) -> np.ndarray:
    """R_eff(i,j) = L+_ii + L+_jj - 2 L+_ij, via the Moore-Penrose pseudo-inverse
    of the combinatorial Laplacian of the ORIGINAL (un-reweighted) graph -- the
    fixed physical reference frame both conditions are compared against."""
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    L_pinv = np.linalg.pinv(L, hermitian=True)
    diag = np.diag(L_pinv)
    R_eff = diag[:, None] + diag[None, :] - 2.0 * L_pinv
    return np.clip(R_eff, 0.0, None)  # numerical noise can give tiny negatives


def bin_by_x(x: np.ndarray, y_dict: Dict[str, np.ndarray], edges: np.ndarray) -> Dict:
    bin_idx = np.digitize(x, edges[1:-1])
    rows = []
    for b in range(len(edges) - 1):
        sel = bin_idx == b
        if sel.sum() == 0:
            continue
        row = {
            "bin": b,
            "x_lo": float(edges[b]),
            "x_hi": float(edges[b + 1]),
            "x_mean": float(x[sel].mean()),
            "n_pairs": int(sel.sum()),
        }
        for name, y in y_dict.items():
            row[f"{name}_mean"] = float(y[sel].mean())
            row[f"{name}_std"] = float(y[sel].std())
        rows.append(row)
    return rows


def decay_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of log(y) on x for y > 0; steeper (more negative) = faster decay."""
    valid = y > 0
    if valid.sum() < 3:
        raise ValueError("Not enough positive points to fit a decay slope")
    slope, _ = np.polyfit(x[valid], np.log(y[valid]), 1)
    return float(slope)


def tail_mass_fraction(x: np.ndarray, y_abs: np.ndarray, quantile: float) -> float:
    threshold = np.quantile(x, quantile)
    long_range = x >= threshold
    return float(y_abs[long_range].sum() / y_abs.sum())


def compute(checkpoint_path: Path, n_dist_bins: int, n_reff_bins: int, tail_quantile: float) -> Dict:
    state = load_curvature_module_state(checkpoint_path)
    N = state["static_adj"].shape[0]
    assert state["P"].shape[0] == N

    Sigma_rewired = implied_node_sigma(**state)
    state_no_rewire = dict(state)
    state_no_rewire["lam"] = torch.zeros_like(state["lam"])
    Sigma_original = implied_node_sigma(**state_no_rewire)

    W = 0.5 * (state["static_adj"] + state["static_adj"].t())
    W = (W - torch.diag(torch.diag(W))).numpy()
    assert (W >= 0).all(), "expected a non-negative adjacency"

    hop_dist = hop_distance_matrix(W)
    r_eff = effective_resistance_matrix(W)

    triu = np.triu_indices(N, k=1)
    hop_pairs = hop_dist[triu]
    reachable = np.isfinite(hop_pairs)
    n_unreachable = int((~reachable).sum())

    reff_pairs = r_eff[triu][reachable]
    hop_pairs = hop_pairs[reachable]
    sigma_rw_pairs = np.abs(Sigma_rewired[triu])[reachable]
    sigma_orig_pairs = np.abs(Sigma_original[triu])[reachable]

    hop_edges = np.arange(hop_pairs.min(), hop_pairs.max() + 2) - 0.5
    hop_bins_rw = bin_by_x(hop_pairs, {"sigma": sigma_rw_pairs}, hop_edges)
    hop_bins_orig = bin_by_x(hop_pairs, {"sigma": sigma_orig_pairs}, hop_edges)

    reff_edges = np.quantile(reff_pairs, np.linspace(0, 1, n_reff_bins + 1))
    reff_edges[0] -= 1e-9
    reff_edges[-1] += 1e-9
    reff_bins_rw = bin_by_x(reff_pairs, {"sigma": sigma_rw_pairs}, reff_edges)
    reff_bins_orig = bin_by_x(reff_pairs, {"sigma": sigma_orig_pairs}, reff_edges)

    hop_means_rw = np.array([r["sigma_mean"] for r in hop_bins_rw])
    hop_x_rw = np.array([r["x_mean"] for r in hop_bins_rw])
    hop_means_orig = np.array([r["sigma_mean"] for r in hop_bins_orig])
    hop_x_orig = np.array([r["x_mean"] for r in hop_bins_orig])

    slope_rw = decay_slope(hop_x_rw, hop_means_rw)
    slope_orig = decay_slope(hop_x_orig, hop_means_orig)

    tail_frac_rw = tail_mass_fraction(hop_pairs, sigma_rw_pairs, tail_quantile)
    tail_frac_orig = tail_mass_fraction(hop_pairs, sigma_orig_pairs, tail_quantile)

    return {
        "checkpoint": str(checkpoint_path),
        "n_nodes": int(N),
        "n_pairs_total": int(len(triu[0])),
        "n_pairs_unreachable_excluded": n_unreachable,
        "lam_learned": float(state["lam"]),
        "tail_quantile": tail_quantile,
        "hop_distance": {
            "bins_rewired": hop_bins_rw,
            "bins_original": hop_bins_orig,
            "decay_slope_rewired": slope_rw,
            "decay_slope_original": slope_orig,
            "decay_slope_ratio_rewired_over_original": slope_rw / slope_orig,
            "tail_mass_fraction_rewired": tail_frac_rw,
            "tail_mass_fraction_original": tail_frac_orig,
            "tail_mass_fraction_ratio": tail_frac_rw / tail_frac_orig,
        },
        "effective_resistance": {
            "bins_rewired": reff_bins_rw,
            "bins_original": reff_bins_orig,
        },
        "raw_pairs": {
            "hop_dist": hop_pairs.tolist(),
            "r_eff": reff_pairs.tolist(),
            "sigma_rewired_abs": sigma_rw_pairs.tolist(),
            "sigma_original_abs": sigma_orig_pairs.tolist(),
        },
    }


def make_figure(results: Dict, fig_out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, key, xlabel in [
        (axes[0], "hop_distance", "Graph distance (hops)"),
        (axes[1], "effective_resistance", "Effective resistance $R_{\\mathrm{eff}}$"),
    ]:
        block = results[key]
        for label, bins_key, style in [
            ("With curvature reweighting", "bins_rewired", dict(marker="o", color="crimson")),
            ("Without reweighting ($\\lambda=0$)", "bins_original", dict(marker="s", color="steelblue")),
        ]:
            rows = block[bins_key]
            x = [r["x_mean"] for r in rows]
            y = [r["sigma_mean"] for r in rows]
            yerr = [r["sigma_std"] / max(r["n_pairs"], 1) ** 0.5 for r in rows]
            ax.errorbar(x, y, yerr=yerr, label=label, **style)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"mean $|\Sigma^{\mathrm{node}}_{ij}|$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("Implied node-level spatial covariance vs. graph distance — Brussels/xLSTM")
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--fig-out", default=DEFAULT_FIG_OUT)
    parser.add_argument("--n-dist-bins", type=int, default=0, help="unused for hops (one bin per integer hop); kept for CLI symmetry")
    parser.add_argument("--n-reff-bins", type=int, default=10)
    parser.add_argument("--tail-quantile", type=float, default=0.75)
    args = parser.parse_args()

    results = compute(Path(args.checkpoint), args.n_dist_bins, args.n_reff_bins, args.tail_quantile)

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    raw_pairs = results.pop("raw_pairs")
    json_out.write_text(json.dumps(results, indent=2) + "\n")
    npz_out = json_out.with_suffix(".npz")
    np.savez(
        npz_out,
        hop_dist=np.array(raw_pairs["hop_dist"]),
        r_eff=np.array(raw_pairs["r_eff"]),
        sigma_rewired_abs=np.array(raw_pairs["sigma_rewired_abs"]),
        sigma_original_abs=np.array(raw_pairs["sigma_original_abs"]),
    )
    results["raw_pairs"] = raw_pairs  # restore for figure/printing

    make_figure(results, Path(args.fig_out))

    hd = results["hop_distance"]
    print(f"Wrote {json_out}")
    print(f"Wrote {npz_out}")
    print(f"Wrote {args.fig_out}")
    print(f"lambda (learned) = {results['lam_learned']:.4f}")
    print(
        f"Decay slope (log|Sigma| vs hop): rewired={hd['decay_slope_rewired']:.4f}, "
        f"original={hd['decay_slope_original']:.4f} "
        f"(closer to 0 = thicker tail; ratio={hd['decay_slope_ratio_rewired_over_original']:.3f})"
    )
    print(
        f"Tail mass fraction (top {int((1-results['tail_quantile'])*100)}% farthest pairs): "
        f"rewired={hd['tail_mass_fraction_rewired']:.4f}, original={hd['tail_mass_fraction_original']:.4f} "
        f"(ratio={hd['tail_mass_fraction_ratio']:.3f})"
    )


if __name__ == "__main__":
    main()
