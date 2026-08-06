"""
Graph-theoretic diagnostics for curvature-aware rewiring.

This script is intentionally checkpoint-driven: it reads the learned
curvature module parameters, reconstructs W and W', and writes reviewer-facing
numbers for local bottleneck conductance and scaled Kirchhoff index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


DEFAULT_CHECKPOINT = (
    "logs/xLSTM/brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_"
    "LossLRw1.0_RegW2.5_TrainL_False/checkpoints/epoch=17-val_loss=64.74.ckpt"
)
DEFAULT_JSON_OUT = "metrics_stat/xLSTM/brussels_xlstm_graph_diagnostics.json"
DEFAULT_TEX_OUT = "NeurIPS/brussels_xlstm_graph_diagnostics.tex"


def torch_load_checkpoint(path: Path) -> Dict:
    """Load across torch versions where `weights_only` may or may not exist."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_curvature_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    suffix = "curvature_precision.static_adj"
    for key in state_dict:
        if key.endswith(suffix):
            return key[: -len("static_adj")]
    raise KeyError("Could not find a curvature_precision.static_adj tensor in the checkpoint")


def load_curvature_state(checkpoint_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = find_curvature_prefix(state_dict)

    adj_key = prefix + "static_adj"
    adj = state_dict[adj_key].detach().cpu().double().numpy()
    adj = 0.5 * (adj + adj.T)
    np.fill_diagonal(adj, 0.0)

    if prefix + "lam" in state_dict:
        lam = abs(float(state_dict[prefix + "lam"].detach().cpu()))
    elif prefix + "log_lam" in state_dict:
        lam = float(torch.exp(state_dict[prefix + "log_lam"].detach().cpu()))
    else:
        raise KeyError("Could not find either learned `lam` or `log_lam`")

    params = {
        "lambda": lam,
        "kappa_0": float(state_dict[prefix + "kappa_0"].detach().cpu()),
        "tau": float(torch.exp(state_dict[prefix + "log_tau"].detach().cpu())),
        "alpha": float(torch.exp(state_dict[prefix + "log_alpha"].detach().cpu())),
        "beta": float(torch.exp(state_dict[prefix + "log_beta"].detach().cpu())),
    }
    return adj, params


def balanced_forman_curvature(W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    W_clean = W.copy()
    np.fill_diagonal(W_clean, 0.0)
    mask = (W_clean > 0).astype(float)

    deg = W_clean.sum(axis=-1).clip(min=eps)
    inv_sqrt_deg = 1.0 / np.sqrt(deg)

    term1 = W_clean * (inv_sqrt_deg[:, None] + inv_sqrt_deg[None, :])
    W_normed = W_clean / np.sqrt(deg)[:, None].clip(min=eps)
    triangle_weight = W_normed @ W_normed.T
    triangles = mask @ mask
    term2 = W_clean * triangle_weight * (triangles > 0).astype(float)

    return (term1 + term2 - W_clean) * mask


def bottleneck_indicator(
    kappa: np.ndarray,
    mask: np.ndarray,
    kappa_0: float,
    tau: float,
) -> np.ndarray:
    edge_kappas = kappa[mask > 0]
    if edge_kappas.size == 0:
        return np.zeros_like(kappa)

    z = (edge_kappas.mean() - kappa) / max(float(edge_kappas.std()), 1e-6)
    x = tau * (z + kappa_0)
    softplus = np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -60.0, 30.0))))
    return softplus * mask


def connected_components(W: np.ndarray) -> List[np.ndarray]:
    adjacency = W > 0
    seen = np.zeros(W.shape[0], dtype=bool)
    components: List[np.ndarray] = []

    for start in range(W.shape[0]):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in np.flatnonzero(adjacency[node]):
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(int(nbr))
        components.append(np.array(component, dtype=int))
    return components


def laplacian(W: np.ndarray) -> np.ndarray:
    return np.diag(W.sum(axis=1)) - W


def conductance(W: np.ndarray, S: np.ndarray) -> float:
    S = np.asarray(S, dtype=bool)
    cut = W[np.ix_(S, ~S)].sum()
    degree = W.sum(axis=1)
    denom = min(degree[S].sum(), degree[~S].sum())
    return float(cut / denom) if denom > 0 else float("nan")


def scaled_kirchhoff(eigenvalues: np.ndarray, n_nodes: int, tol: float = 1e-8) -> float:
    positive = eigenvalues[eigenvalues > tol]
    return float(np.sum(1.0 / positive) / n_nodes)


def pct_change(before: float, after: float) -> float:
    return 100.0 * (after - before) / before


def sweep_cut_rows(W: np.ndarray, W_prime: np.ndarray, b: np.ndarray) -> List[Dict[str, float]]:
    eigvals, eigvecs = np.linalg.eigh(laplacian(W))
    if len(eigvals) < 2 or eigvals[1] <= 1e-8:
        raise ValueError("Fiedler sweep requires a connected component with positive lambda_2")

    order = np.argsort(eigvecs[:, 1])
    rows = []
    for size in range(1, W.shape[0]):
        S = np.zeros(W.shape[0], dtype=bool)
        S[order[:size]] = True

        phi = conductance(W, S)
        phi_full = conductance(W_prime, S)

        boundary = np.outer(S, ~S) | np.outer(~S, S)
        W_boundary = W.copy()
        W_boundary[boundary] = W_prime[boundary]
        phi_boundary = conductance(W_boundary, S)

        boundary_weights = W[np.ix_(S, ~S)]
        boundary_scores = b[np.ix_(S, ~S)]
        mean_boundary_b = (
            float(boundary_scores[boundary_weights > 0].mean())
            if np.any(boundary_weights > 0)
            else 0.0
        )

        rows.append(
            {
                "size": int(size),
                "phi_original": float(phi),
                "phi_reweighted": float(phi_full),
                "phi_boundary_only": float(phi_boundary),
                "pct_reweighted": pct_change(phi, phi_full),
                "pct_boundary_only": pct_change(phi, phi_boundary),
                "mean_boundary_b": mean_boundary_b,
            }
        )
    return rows


def summarize_rows(rows: Iterable[Dict[str, float]], top_k: int) -> Dict[str, float]:
    rows = list(rows)
    return {
        "n_cuts": len(rows),
        "cut_sizes": [row["size"] for row in rows],
        "phi_original": float(np.mean([row["phi_original"] for row in rows])),
        "phi_reweighted": float(np.mean([row["phi_reweighted"] for row in rows])),
        "phi_boundary_only": float(np.mean([row["phi_boundary_only"] for row in rows])),
        "pct_reweighted": float(np.mean([row["pct_reweighted"] for row in rows])),
        "pct_boundary_only": float(np.mean([row["pct_boundary_only"] for row in rows])),
        "mean_boundary_b": float(np.mean([row["mean_boundary_b"] for row in rows])),
        "top_k": top_k,
    }


def compute_diagnostics(checkpoint_path: Path, top_k: int) -> Dict:
    W, params = load_curvature_state(checkpoint_path)
    mask = (W > 0).astype(float)
    kappa = balanced_forman_curvature(W)
    b = bottleneck_indicator(kappa, mask, params["kappa_0"], params["tau"])
    W_prime = W * (1.0 + params["lambda"] * b)
    W_prime = 0.5 * (W_prime + W_prime.T)

    components = connected_components(W)
    giant = max(components, key=len)
    W_giant = W[np.ix_(giant, giant)]
    Wp_giant = W_prime[np.ix_(giant, giant)]
    b_giant = b[np.ix_(giant, giant)]

    eig = np.linalg.eigvalsh(laplacian(W_giant))
    eig_prime = np.linalg.eigvalsh(laplacian(Wp_giant))
    sweep_rows = sweep_cut_rows(W_giant, Wp_giant, b_giant)

    curvature_loaded = sorted(
        sweep_rows,
        key=lambda row: row["mean_boundary_b"],
        reverse=True,
    )[:top_k]
    min_conductance = sorted(sweep_rows, key=lambda row: row["phi_original"])[:top_k]

    edge_mask = np.triu(W > 0, k=1)
    edge_kappas = kappa[edge_mask]
    edge_b = b[edge_mask]
    edge_ratio = W_prime[edge_mask] / np.clip(W[edge_mask], 1e-12, None)

    n_component = int(len(giant))
    diagnostics = {
        "checkpoint": str(checkpoint_path),
        "n_total": int(W.shape[0]),
        "n_component": n_component,
        "n_components": int(len(components)),
        "n_edges": int(edge_mask.sum()),
        "parameters": params,
        "curvature": {
            "min": float(edge_kappas.min()),
            "max": float(edge_kappas.max()),
            "mean": float(edge_kappas.mean()),
            "std": float(edge_kappas.std()),
            "frac_negative": float((edge_kappas < 0).mean()),
        },
        "bottleneck": {
            "mean": float(edge_b.mean()),
            "max": float(edge_b.max()),
            "p90": float(np.percentile(edge_b, 90)),
            "mean_reweight_ratio": float(edge_ratio.mean()),
            "max_reweight_ratio": float(edge_ratio.max()),
        },
        "spectral": {
            "lambda_2_original": float(eig[1]),
            "lambda_2_reweighted": float(eig_prime[1]),
            "lambda_2_pct_change": pct_change(float(eig[1]), float(eig_prime[1])),
            "scaled_kirchhoff_original": scaled_kirchhoff(eig, n_component),
            "scaled_kirchhoff_reweighted": scaled_kirchhoff(eig_prime, n_component),
        },
        "conductance": {
            "curvature_loaded_fiedler_top_k": summarize_rows(curvature_loaded, top_k),
            "min_conductance_fiedler_top_k": summarize_rows(min_conductance, top_k),
        },
    }
    spectral = diagnostics["spectral"]
    spectral["scaled_kirchhoff_pct_change"] = pct_change(
        spectral["scaled_kirchhoff_original"],
        spectral["scaled_kirchhoff_reweighted"],
    )
    return diagnostics


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}\\%"


def build_latex_table(diagnostics: Dict) -> str:
    spectral = diagnostics["spectral"]
    conductance = diagnostics["conductance"]["curvature_loaded_fiedler_top_k"]
    n_comp = diagnostics["n_component"]
    n_total = diagnostics["n_total"]
    top_k = conductance["top_k"]

    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Brussels/xLSTM graph diagnostics for the learned curvature reweighting. "
            f"Conductance values average over the top {top_k} Fiedler sweep cuts with the largest mean boundary bottleneck score on the largest connected component "
            f"({n_comp}/{n_total} sensors). Lower is better for the scaled Kirchhoff index; higher is better for conductance and $\\lambda_2$.}}",
            "\\label{tab:brussels-xlstm-graph-diagnostics}",
            "\\small",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Quantity & Original $\\mathbf W$ & Reweighted $\\mathbf W'$ & Change \\\\",
            "\\midrule",
            "Scaled Kirchhoff $\\tr(\\mathbf L^+)/N_c$ & "
            f"${fmt(spectral['scaled_kirchhoff_original'])}$ & "
            f"$\\mathbf{{{fmt(spectral['scaled_kirchhoff_reweighted'])}}}$ & "
            f"${fmt_pct(spectral['scaled_kirchhoff_pct_change'])}$ \\\\",
            "$\\lambda_2(\\mathbf L)$ & "
            f"${fmt(spectral['lambda_2_original'])}$ & "
            f"$\\mathbf{{{fmt(spectral['lambda_2_reweighted'])}}}$ & "
            f"${fmt_pct(spectral['lambda_2_pct_change'])}$ \\\\",
            "Local bottleneck conductance $\\phi(S)$ & "
            f"${fmt(conductance['phi_original'])}$ & "
            f"$\\mathbf{{{fmt(conductance['phi_reweighted'])}}}$ & "
            f"${fmt_pct(conductance['pct_reweighted'])}$ \\\\",
            "Boundary-only local $\\phi_{\\partial S}(S)$ & "
            f"${fmt(conductance['phi_original'])}$ & "
            f"$\\mathbf{{{fmt(conductance['phi_boundary_only'])}}}$ & "
            f"${fmt_pct(conductance['pct_boundary_only'])}$ \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--tex-out", default=DEFAULT_TEX_OUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    diagnostics = compute_diagnostics(Path(args.checkpoint), args.top_k)

    json_out = Path(args.json_out)
    tex_out = Path(args.tex_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    tex_out.write_text(build_latex_table(diagnostics))

    spectral = diagnostics["spectral"]
    conductance = diagnostics["conductance"]["curvature_loaded_fiedler_top_k"]
    print(f"Wrote {json_out}")
    print(f"Wrote {tex_out}")
    print(
        "Scaled Kirchhoff: "
        f"{spectral['scaled_kirchhoff_original']:.4f} -> "
        f"{spectral['scaled_kirchhoff_reweighted']:.4f} "
        f"({spectral['scaled_kirchhoff_pct_change']:+.1f}%)"
    )
    print(
        "Local conductance: "
        f"{conductance['phi_original']:.4f} -> "
        f"{conductance['phi_reweighted']:.4f} "
        f"({conductance['pct_reweighted']:+.1f}%)"
    )


if __name__ == "__main__":
    main()
