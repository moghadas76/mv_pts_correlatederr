"""
k-hop propagation "Jacobian" proxy vs. graph distance, before/after Teger's
actual reweighting (reviews.md R2 point 2, rebuttal.md Priority 3).

R2's complaint: "the method never performs the multi-layer message passing in
which over-squashing is actually defined ... the theoretical results merely
show that edge reweighting improves standard graph proxies ... without ever
connecting these quantities to the Jacobian decay that constitutes
over-squashing."

Teger genuinely does not run k rounds of message passing -- it builds a single
resolvent (alpha*I + beta*L'_R)^{-1} in a projected R-dim space. There is no
real ∂h_v^(k)/∂h_u^(0) tensor anywhere in this codebase to "read off" (checked:
no Jacobian/propagation matrix of that kind exists in dynamic_graph.py or
loss.py). So this script does not pretend to; it computes the standard
analytical SURROGATE for what a k-layer GCN's Jacobian sensitivity would be if
one were built on this (reweighted) graph:

    M^(k)_ij = (A_hat)^k_ij,   A_hat = D^{-1/2} W D^{-1/2}

which is the propagation operator underlying essentially every GCN variant. It
is a genuine analytical bound, not just a metaphor: for a GCN with symmetric-
normalised propagation, k layers, 1-Lipschitz elementwise nonlinearity, and
weight matrices with spectral norm <= 1 at every layer, the Jacobian obeys
||∂h_v^(k)/∂h_u^(0)|| <= C * |(A_hat^k)_{vu}| (Topping et al. 2022; Chamberlain
et al. 2021 make the same substitution). Comparing (A_hat_before)^k against
(A_hat_after)^k -- both built from matrices Algorithm 1 already computes
(the static and reweighted adjacency) -- is therefore a legitimate, if
partial, empirical connection between the paper's graph-theoretic results and
Jacobian-type over-squashing decay: it shows what WOULD happen to message-
passing sensitivity if the reweighted graph were used for propagation, without
claiming Teger itself performs that propagation.

Usage:
    python src/curvature_jacobian_distance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_graph_diagnostics import (
    DEFAULT_CHECKPOINT,
    connected_components,
    find_curvature_prefix,
    torch_load_checkpoint,
)
from curvature_sigma_vs_distance import decay_slope, hop_distance_matrix
from dynamic_graph import _reweight_laplacian

JSON_OUT = "metrics_stat/xLSTM/brussels_xlstm_jacobian_vs_distance.json"
FIG_OUT = "visualizations/brussels_xlstm_jacobian_vs_distance.pdf"
K_MAX = 6


def normalized_adjacency(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    return (deg_inv_sqrt[:, None] * W) * deg_inv_sqrt[None, :]


def bin_by_hop(hop: np.ndarray, values: np.ndarray) -> List[Dict]:
    rows = []
    for h in range(int(hop.min()), int(hop.max()) + 1):
        sel = hop == h
        if sel.sum() == 0:
            continue
        rows.append({
            "hop": h,
            "n_pairs": int(sel.sum()),
            "mean_abs_J": float(values[sel].mean()),
            "std_abs_J": float(values[sel].std()),
        })
    return rows


def main() -> None:
    sd = torch_load_checkpoint(Path(DEFAULT_CHECKPOINT))
    sd = sd.get("state_dict", sd)
    prefix = find_curvature_prefix(sd)

    static_adj = sd[prefix + "static_adj"].detach().double()
    W_full = (0.5 * (static_adj + static_adj.t())).numpy()
    np.fill_diagonal(W_full, 0.0)

    lam = sd[prefix + "lam"].detach().double()
    kappa_0 = sd[prefix + "kappa_0"].detach().double()
    tau = torch.exp(sd[prefix + "log_tau"].detach().double())
    L_prime_full = _reweight_laplacian(torch.tensor(W_full), lam, kappa_0, tau).numpy()
    W_prime_full = np.diag(np.diag(L_prime_full)) - L_prime_full
    np.fill_diagonal(W_prime_full, 0.0)
    W_prime_full = 0.5 * (W_prime_full + W_prime_full.T)

    giant = max(connected_components(W_full), key=len)
    idx = np.ix_(giant, giant)
    W = W_full[idx]
    W_prime = W_prime_full[idx]
    N = len(giant)
    print(f"N_total={W_full.shape[0]}, giant component={N} nodes, "
          f"learned lambda={float(lam):.4f} kappa_0={float(kappa_0):.4f} tau={float(tau):.4f}")

    A_before = normalized_adjacency(W)
    A_after = normalized_adjacency(W_prime)

    hop = hop_distance_matrix(W)  # topology unchanged by reweighting -> shared
    triu = np.triu_indices(N, k=1)
    hop_pairs = hop[triu]
    reachable = np.isfinite(hop_pairs)
    n_unreachable = int((~reachable).sum())
    hop_pairs = hop_pairs[reachable]

    results_by_k = []
    Ak_before = np.eye(N)
    Ak_after = np.eye(N)
    for k in range(1, K_MAX + 1):
        Ak_before = Ak_before @ A_before
        Ak_after = Ak_after @ A_after

        j_before = np.abs(Ak_before[triu])[reachable]
        j_after = np.abs(Ak_after[triu])[reachable]

        bins_before = bin_by_hop(hop_pairs, j_before)
        bins_after = bin_by_hop(hop_pairs, j_after)
        x_before = np.array([r["hop"] for r in bins_before])
        y_before = np.array([r["mean_abs_J"] for r in bins_before])
        x_after = np.array([r["hop"] for r in bins_after])
        y_after = np.array([r["mean_abs_J"] for r in bins_after])

        try:
            slope_before = decay_slope(x_before, y_before)
        except ValueError:
            slope_before = float("nan")
        try:
            slope_after = decay_slope(x_after, y_after)
        except ValueError:
            slope_after = float("nan")

        far_quantile = 0.75
        threshold = np.quantile(hop_pairs, far_quantile)
        far = hop_pairs >= threshold
        far_mass_before = float(j_before[far].sum() / j_before.sum())
        far_mass_after = float(j_after[far].sum() / j_after.sum())

        results_by_k.append({
            "k": k,
            "bins_before": bins_before,
            "bins_after": bins_after,
            "decay_slope_before": slope_before,
            "decay_slope_after": slope_after,
            "decay_slope_closer_to_zero_after": bool(abs(slope_after) < abs(slope_before)) if np.isfinite(slope_before) and np.isfinite(slope_after) else None,
            "far_hop_mass_fraction_before": far_mass_before,
            "far_hop_mass_fraction_after": far_mass_after,
            "far_hop_mass_ratio_after_over_before": far_mass_after / far_mass_before if far_mass_before > 0 else float("nan"),
        })
        print(
            f"k={k}: decay slope before={slope_before:+.4f} after={slope_after:+.4f} "
            f"(closer to 0 = thicker tail); far-hop(>{far_quantile*100:.0f}%ile) mass "
            f"before={far_mass_before:.4f} after={far_mass_after:.4f} "
            f"(ratio={results_by_k[-1]['far_hop_mass_ratio_after_over_before']:.3f})"
        )

    results = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "n_total": int(W_full.shape[0]),
        "n_giant_component": N,
        "n_pairs_unreachable_excluded": n_unreachable,
        "k_max": K_MAX,
        "learned_params": {"lambda": float(lam), "kappa_0": float(kappa_0), "tau": float(tau)},
        "definition": (
            "M^(k)_ij = (A_hat)^k_ij with A_hat = D^{-1/2} W D^{-1/2}; A_hat_before from the "
            "static (unreweighted) graph, A_hat_after from Teger's actual deployed reweighted "
            "graph (dynamic_graph._reweight_laplacian, matching CurvatureAwareGraphPrecision_"
            "LearnP.forward() exactly). This is the standard GCN-propagation surrogate for "
            "message-passing Jacobian sensitivity used in the over-squashing literature; Teger "
            "itself does not run k-layer propagation, so this is a partial answer to R2 point 2, "
            "not a claim that Teger performs multi-layer message passing."
        ),
        "results_by_k": results_by_k,
    }

    json_out = Path(JSON_OUT)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {json_out}")

    make_figure(results, Path(FIG_OUT))
    print(f"Wrote {FIG_OUT}")


def make_figure(results: Dict, fig_out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks_to_plot = [k for k in (1, 2, 3, K_MAX) if k <= results["k_max"]]
    fig, axes = plt.subplots(1, len(ks_to_plot), figsize=(4.2 * len(ks_to_plot), 3.8), sharey=False)
    if len(ks_to_plot) == 1:
        axes = [axes]

    by_k = {r["k"]: r for r in results["results_by_k"]}
    for ax, k in zip(axes, ks_to_plot):
        r = by_k[k]
        for label, key, style in [
            ("before reweighting", "bins_before", dict(marker="s", color="steelblue")),
            ("after reweighting (Teger)", "bins_after", dict(marker="o", color="crimson")),
        ]:
            rows = r[key]
            x = [row["hop"] for row in rows]
            y = [row["mean_abs_J"] for row in rows]
            ax.plot(x, y, label=label, **style)
        ax.set_yscale("log")
        ax.set_xlabel("Graph distance (hops)")
        ax.set_title(f"k = {k}", fontsize=10)
        if k == ks_to_plot[0]:
            ax.set_ylabel(r"mean $|(\hat A^k)_{ij}|$")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

    fig.suptitle("k-hop propagation sensitivity vs. graph distance — Brussels/xLSTM")
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
