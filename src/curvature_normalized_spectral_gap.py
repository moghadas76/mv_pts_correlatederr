"""
Normalised spectral gap before/after reweighting, and curvature-vs-Fiedler-
sensitivity Spearman correlation, on the Brussels/xLSTM ablation ladder.

Two rebuttal asks answered together because they share one eigendecomposition
(rebuttal.md Priority 3 / Priority 1; reviews.md R2 point 5 and kgxJ W2):

  (1) Normalised spectral gap lambda_2(D^{-1/2} L D^{-1/2}) before/after
      Algorithm 1, on the four already-built ablation-ladder graphs
      (ablation_graphs/brussels_xlstm_row{3,4,5,6}_*.pt via
      build_ablation_multipliers.py). This directly answers the "you just
      multiplied every weight by ~1.4" reading of Table 2 columns 1-2: for a
      UNIFORM rescaling W -> cW, the normalised Laplacian is invariant
      (D -> cD, so D^{-1/2}WD^{-1/2} is unchanged), so row4_uniform is
      mathematically guaranteed to show ~0% change here even though its
      unnormalised lambda_2 and Kirchhoff index move (Table 2 is unnormalised).
      A nontrivial normalised-lambda_2 shift therefore cannot come from "just
      adding mass everywhere" -- it requires non-uniform, targeted placement.

  (2) Spearman rho between -kappa_ij (Balanced Forman curvature, as specified
      in Algorithm 1 / Section 3.3.1) and the Fiedler sensitivity
      (v_i - v_j)^2, where v is the Fiedler vector of the ORIGINAL
      (pre-rewiring) unnormalised Laplacian -- reusing the eigenvectors from
      (1)'s "before" eigendecomposition ("same cost"). Since d(lambda_2)/d(w_ij)
      = (v_i-v_j)^2 (first-order eigenvalue perturbation for a simple lambda_2),
      this checks whether curvature is a good *local* proxy for the *global*
      optimal-reweighting direction -- the "curvature as a proxy, not a
      mystery" argument for kgxJ W2.

SCOPE NOTE: static_adj is frozen (TrainL_False) and verified bit-identical
across all four ablation-ladder checkpoints, so kappa_ij and the "before"
Fiedler vector are the same physical quantity regardless of which of the four
checkpoints they are read from. The four numbers reported for (2) are
therefore a robustness/reproducibility check across four independently-saved
checkpoints, not four different underlying answers -- reported as such, not
disguised as four independent trials.

FORMULA CAVEAT (found while building this): the *deployed* model
(CurvatureAwareGraphPrecision_LearnP.forward(), mode="curvature", i.e. row 7 /
Teger) reweights using dynamic_graph._reweight_laplacian, which calls the
*lightweight* curvature proxy dynamic_graph._balanced_forman
(kappa = 2/d_max * triangles/(d_i+d_j-2)) -- NOT the Algorithm-1 Balanced
Forman formula (term1 + term2 - w_ij) implemented in
curvature_graph_diagnostics.balanced_forman_curvature, which is what produced
the existing brussels_xlstm_graph_diagnostics.json (Table 2) and what
build_ablation_multipliers.py used to mass-match rows 3-6. This script:
  - uses the checkpoint's own frozen `fixed_multiplier` for rows 3-6 (exact,
    formula-agnostic: W'_row = W*(1+fixed_multiplier_row), matching
    CurvatureAwareGraphPrecision_LearnP.forward()'s "fixed" branch verbatim);
  - uses the Algorithm-1 (paper-specified) Balanced Forman curvature for
    kappa_ij in (2), since that is the quantity the reviewer is asking whether
    the paper's formula is a sensible signal (not the internal shortcut);
  - additionally reconstructs the row-7/Teger reweighted graph via the
    *actual* deployed dynamic_graph._reweight_laplacian for a reference row,
    explicitly labelled so it is not confused with rows 3-6.
This mismatch between the paper's stated Algorithm 1 and the deployed
forward() is real and worth fixing before camera-ready; it is flagged again
in the printed output rather than silently resolved.

Usage:
    python src/curvature_normalized_spectral_gap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_graph_diagnostics import (
    balanced_forman_curvature,
    connected_components,
    find_curvature_prefix,
    laplacian,
    pct_change,
    torch_load_checkpoint,
)
from dynamic_graph import _reweight_laplacian

BASE_CHECKPOINT = (
    "logs/xLSTM/student_brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_"
    "LossLRw1.0_RegW2.5_TrainL_False/checkpoints/epoch=17-val_loss=64.74.ckpt"
)
ROW_DIRS = {
    "row3_none": "logs/xLSTM/student_brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_fixed_brussels_xlstm_row3_none",
    "row4_uniform": "logs/xLSTM/student_brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_fixed_brussels_xlstm_row4_uniform",
    "row5_permuted": "logs/xLSTM/student_brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_fixed_brussels_xlstm_row5_permuted",
    "row6_inverse": "logs/xLSTM/student_brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_fixed_brussels_xlstm_row6_inverse",
}
JSON_OUT = "metrics_stat/xLSTM/brussels_xlstm_normalized_spectral_gap.json"
TEX_OUT = "NeurIPS/brussels_xlstm_normalized_spectral_gap.tex"


def latest_checkpoint(row_dir: str) -> Path:
    ckpts = sorted(Path(row_dir).glob("checkpoints/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {row_dir}")
    # Pick the lowest val_loss (filenames end in val_loss=X.XX.ckpt).
    def val_loss(p: Path) -> float:
        return float(p.stem.split("val_loss=")[-1].split("-v")[0])
    return min(ckpts, key=val_loss)


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch_load_checkpoint(path)
    return ckpt.get("state_dict", ckpt)


def giant_component_index(W: np.ndarray) -> np.ndarray:
    components = connected_components(W)
    return max(components, key=len)


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    N = W.shape[0]
    return np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    base_sd = load_state(Path(BASE_CHECKPOINT))
    prefix = find_curvature_prefix(base_sd)
    static_adj = base_sd[prefix + "static_adj"].detach().double().numpy()
    W_full = 0.5 * (static_adj + static_adj.T)
    np.fill_diagonal(W_full, 0.0)

    giant = giant_component_index(W_full)
    idx = np.ix_(giant, giant)
    W = W_full[idx]
    n_total, n_giant = W_full.shape[0], len(giant)
    print(f"N_total={n_total}, giant component={n_giant} nodes")

    # ---- "before" (shared across all rows): one normalised + one
    # unnormalised eigendecomposition of the original graph. ----
    L_norm_before = normalized_laplacian(W)
    eig_norm_before, _ = np.linalg.eigh(L_norm_before)
    lambda2_norm_before = float(eig_norm_before[1])

    L_unnorm_before = laplacian(W)
    eig_unnorm_before, vec_unnorm_before = np.linalg.eigh(L_unnorm_before)
    lambda2_unnorm_before = float(eig_unnorm_before[1])
    fiedler = vec_unnorm_before[:, 1]

    print(f"before: lambda2_normalised={lambda2_norm_before:.6f}, "
          f"lambda2_unnormalised={lambda2_unnorm_before:.6f}")

    # ---- (2) curvature vs Fiedler sensitivity, computed once, reported once
    # per checkpoint for the robustness check. ----
    kappa = balanced_forman_curvature(W)
    edge_mask = np.triu(W > 0, k=1)
    ei, ej = np.where(edge_mask)
    kappa_edge = kappa[ei, ej]
    fiedler_sens_edge = (fiedler[ei] - fiedler[ej]) ** 2
    rho_shared = spearman(-kappa_edge, fiedler_sens_edge)
    print(f"Spearman(-kappa_ij, Fiedler sensitivity) [shared original graph] = {rho_shared:+.4f}  "
          f"(n_edges={len(ei)})")

    # ---- (1) four ablation-ladder graphs: one normalised eigendecomposition
    # each, "after" reweighting. ----
    rows_out: List[Dict] = []
    for name, row_dir in ROW_DIRS.items():
        ckpt_path = latest_checkpoint(row_dir)
        sd = load_state(ckpt_path)
        row_prefix = find_curvature_prefix(sd)
        row_static_adj = sd[row_prefix + "static_adj"].detach().double().numpy()
        row_static_adj = 0.5 * (row_static_adj + row_static_adj.T)
        np.fill_diagonal(row_static_adj, 0.0)
        identical_topology = np.allclose(row_static_adj, W_full)

        mult_full = sd[row_prefix + "fixed_multiplier"].detach().double().numpy()
        mult = mult_full[idx]
        W_prime = W * (1.0 + mult)
        W_prime = 0.5 * (W_prime + W_prime.T)

        L_norm_after = normalized_laplacian(W_prime)
        eig_norm_after, _ = np.linalg.eigh(L_norm_after)
        lambda2_norm_after = float(eig_norm_after[1])

        # Robustness check for (2): same graph -> same kappa/Fiedler -> same rho.
        # Recomputed independently per checkpoint rather than assumed.
        row_giant = giant_component_index(row_static_adj)
        same_giant = np.array_equal(row_giant, giant)
        L_unnorm_row_before = laplacian(row_static_adj[np.ix_(row_giant, row_giant)])
        _, vec_row = np.linalg.eigh(L_unnorm_row_before)
        fiedler_row = vec_row[:, 1]
        kappa_row = balanced_forman_curvature(row_static_adj[np.ix_(row_giant, row_giant)])
        ei_r, ej_r = np.where(np.triu(row_static_adj[np.ix_(row_giant, row_giant)] > 0, k=1))
        rho_row = spearman(-kappa_row[ei_r, ej_r], (fiedler_row[ei_r] - fiedler_row[ej_r]) ** 2)

        mult_edge = mult[ei, ej]
        row_result = {
            "row": name,
            "checkpoint": str(ckpt_path),
            "static_adj_identical_to_base": bool(identical_topology),
            "same_giant_component": bool(same_giant),
            "multiplier_stats": {
                "min": float(mult_edge.min()), "max": float(mult_edge.max()),
                "mean_nonzero": float(mult_edge[mult_edge > 0].mean()) if (mult_edge > 0).any() else 0.0,
                "total_added_mass": float((W * mult).sum()),
            },
            "spectral": {
                "lambda2_normalised_before": lambda2_norm_before,
                "lambda2_normalised_after": lambda2_norm_after,
                "lambda2_normalised_pct_change": pct_change(lambda2_norm_before, lambda2_norm_after),
            },
            "curvature_vs_fiedler_sensitivity": {
                "spearman_rho": rho_row,
                "n_edges": int(len(ei_r)),
            },
        }
        rows_out.append(row_result)
        print(
            f"{name:14s} mass={row_result['multiplier_stats']['total_added_mass']:9.3f}  "
            f"lambda2_norm: {lambda2_norm_before:.6f} -> {lambda2_norm_after:.6f} "
            f"({row_result['spectral']['lambda2_normalised_pct_change']:+.2f}%)  "
            f"rho(-kappa,Fiedler_sens)={rho_row:+.4f}  "
            f"[static_adj identical to base: {identical_topology}]"
        )

    # ---- Extra context (not one of the "four"): the ACTUAL deployed Teger
    # (row 7) reweighting, via the real forward-pass formula. ----
    lam = base_sd[prefix + "lam"].detach().double()
    kappa_0 = base_sd[prefix + "kappa_0"].detach().double()
    tau = torch.exp(base_sd[prefix + "log_tau"].detach().double())
    static_adj_full_t = torch.tensor(W_full, dtype=torch.float64)
    L_prime_teger_full = _reweight_laplacian(static_adj_full_t, lam, kappa_0, tau).numpy()
    W_prime_teger_full = np.diag(np.diag(L_prime_teger_full)) - L_prime_teger_full
    np.fill_diagonal(W_prime_teger_full, 0.0)
    W_prime_teger = W_prime_teger_full[idx]
    W_prime_teger = 0.5 * (W_prime_teger + W_prime_teger.T)

    L_norm_teger_after = normalized_laplacian(W_prime_teger)
    eig_norm_teger_after, _ = np.linalg.eigh(L_norm_teger_after)
    lambda2_norm_teger_after = float(eig_norm_teger_after[1])
    teger_reference = {
        "row": "row7_teger_reference_actual_deployed_formula",
        "checkpoint": BASE_CHECKPOINT,
        "note": "NOT one of the four requested ablation graphs; reference context using "
                "dynamic_graph._reweight_laplacian (the true forward() formula) so this "
                "number matches what the checkpoint actually computes at inference.",
        "learned_params": {"lambda": float(lam), "kappa_0": float(kappa_0), "tau": float(tau)},
        "spectral": {
            "lambda2_normalised_before": lambda2_norm_before,
            "lambda2_normalised_after": lambda2_norm_teger_after,
            "lambda2_normalised_pct_change": pct_change(lambda2_norm_before, lambda2_norm_teger_after),
        },
    }
    print(
        f"{'row7_teger':14s} (reference, actual deployed formula)  lambda2_norm: "
        f"{lambda2_norm_before:.6f} -> {lambda2_norm_teger_after:.6f} "
        f"({teger_reference['spectral']['lambda2_normalised_pct_change']:+.2f}%)"
    )

    results = {
        "checkpoint_base": BASE_CHECKPOINT,
        "n_total": n_total,
        "n_giant_component": n_giant,
        "formula_caveat": (
            "The deployed CurvatureAwareGraphPrecision_LearnP.forward() (mode='curvature', "
            "row 7) reweights via dynamic_graph._reweight_laplacian, which internally calls "
            "the lightweight dynamic_graph._balanced_forman curvature proxy, NOT the "
            "Algorithm-1 Balanced Forman formula implemented in "
            "curvature_graph_diagnostics.balanced_forman_curvature (used here for kappa_ij "
            "in the Fiedler-sensitivity correlation, and used to build the ablation-ladder "
            "mass-matching in build_ablation_multipliers.py). Rows 3-6 sidestep this because "
            "their fixed_multiplier is read directly from the checkpoint (formula-agnostic); "
            "the row7 reference number uses the true deployed formula and is reported "
            "separately for that reason."
        ),
        "before": {
            "lambda2_normalised": lambda2_norm_before,
            "lambda2_unnormalised": lambda2_unnorm_before,
        },
        "curvature_vs_fiedler_sensitivity_shared_graph": {
            "spearman_rho": rho_shared,
            "n_edges": int(len(ei)),
            "note": "static_adj is bit-identical across all four ablation checkpoints "
                    "(verified per-row below), so kappa_ij and the Fiedler vector of the "
                    "pre-rewiring graph are the same physical quantity in every row; the "
                    "per-row numbers are a reproducibility check, not four different answers.",
        },
        "ablation_ladder_rows": rows_out,
        "teger_row7_reference": teger_reference,
    }

    json_out = Path(JSON_OUT)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {json_out}")

    tex_lines = [
        "% Auto-generated by src/curvature_normalized_spectral_gap.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Normalised spectral gap $\\lambda_2(D^{-1/2}LD^{-1/2})$ before/after reweighting, "
        "Brussels/xLSTM ablation ladder. Uniform mass addition (row 4) leaves the normalised gap "
        "unchanged by construction (a global rescaling $W\\to cW$ is a similarity that cancels "
        "in $D^{-1/2}WD^{-1/2}$); only non-uniform, targeted placement moves it.}",
        "\\label{tab:normalized-spectral-gap}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Row & Added mass & $\\lambda_2$ before & $\\lambda_2$ after & Change \\\\",
        "\\midrule",
    ]
    for row in rows_out:
        s = row["spectral"]
        tex_lines.append(
            f"{row['row'].replace('_', ' ')} & {row['multiplier_stats']['total_added_mass']:.2f} & "
            f"${s['lambda2_normalised_before']:.4f}$ & ${s['lambda2_normalised_after']:.4f}$ & "
            f"${s['lambda2_normalised_pct_change']:+.2f}\\%$ \\\\"
        )
    s = teger_reference["spectral"]
    tex_lines.append(
        f"row 7 (Teger, actual) & --- & ${s['lambda2_normalised_before']:.4f}$ & "
        f"${s['lambda2_normalised_after']:.4f}$ & ${s['lambda2_normalised_pct_change']:+.2f}\\%$ \\\\"
    )
    tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    tex_out = Path(TEX_OUT)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text("\n".join(tex_lines))
    print(f"Wrote {tex_out}")


if __name__ == "__main__":
    main()
