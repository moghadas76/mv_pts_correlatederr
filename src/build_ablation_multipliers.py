"""
Rebuttal Priority 1 (ablation ladder), rows 3-6, Brussels / xLSTM.

Derives mass-matched, FROZEN edge-multiplier matrices b_ij from the latest
trained curvature (Teger, row 7) checkpoint, so that rows 4-6 add exactly the
same total edge mass as Teger and only differ in *where* that mass is placed:

    row 3  none      : b_ij = 0                                  (capacity control)
    row 4  uniform   : b_ij = c  for every edge, mass-matched to Teger
    row 5  permuted  : Teger's own b_ij multiset, reassigned to a random
                        permutation of the edges (mass + distribution exact)
    row 6  inverse   : sign of the curvature term flipped (amplifies
                        *positively* curved edges instead), mass-matched

Each matrix is saved as a plain (N, N) torch.FloatTensor under
ablation_graphs/, to be passed to train_batch.py via
--reweight_mode fixed --fixed_multiplier_path <file>.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from curvature_graph_diagnostics import (
    balanced_forman_curvature,
    bottleneck_indicator,
    load_curvature_state,
)

CHECKPOINT = (
    "logs/xLSTM/student_brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_"
    "LossLRw1.0_RegW2.5_TrainL_False/checkpoints/epoch=17-val_loss=64.74.ckpt"
)
OUT_DIR = "ablation_graphs"
SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    W, params = load_curvature_state(CHECKPOINT)
    mask = (W > 0).astype(float)
    n_edges_directed = int(mask.sum())
    print(f"Loaded {CHECKPOINT}")
    print(f"N={W.shape[0]}, directed edge entries={n_edges_directed}, params={params}")

    kappa = balanced_forman_curvature(W)

    # Teger's actual multiplier (row 7), reproducing training-time formula exactly.
    b_teger = bottleneck_indicator(kappa, mask, params["kappa_0"], params["tau"])
    mult_teger = params["lambda"] * b_teger
    total_mass = float((W * mult_teger).sum())
    print(f"Teger total added edge mass (sum W*b*lambda) = {total_mass:.6f}")

    # --- row 3: no reweighting ---
    mult_none = np.zeros_like(W)

    # --- row 4: uniform, mass-matched ---
    edge_mass_base = float(W[mask > 0].sum())
    c = total_mass / edge_mass_base
    mult_uniform = c * mask
    print(f"row4 uniform constant c={c:.6f}, mass={float((W*mult_uniform).sum()):.6f}")

    # --- row 5: permuted (mass-matched null) ---
    # Permute the *added mass* m_ij = W_ij * b_ij across edge positions (upper
    # triangle, symmetrised), then re-derive b'_ij = m'_ij / W_ij at the new
    # position. This matches total mass EXACTLY regardless of how
    # heterogeneous the base edge weights W_ij are (a plain b_ij shuffle would
    # not, since mass = sum(W_ij * b_ij) mixes weights and values).
    iu = np.triu_indices_from(W, k=1)
    edge_present = mask[iu] > 0
    idx_i = iu[0][edge_present]
    idx_j = iu[1][edge_present]
    added_mass = (W * mult_teger)[iu][edge_present].copy()
    perm = rng.permutation(len(added_mass))
    mass_shuffled = added_mass[perm]
    mult_permuted = np.zeros_like(W)
    b_shuffled = mass_shuffled / W[idx_i, idx_j]
    mult_permuted[idx_i, idx_j] = b_shuffled
    mult_permuted[idx_j, idx_i] = b_shuffled
    print(f"row5 permuted mass={float((W*mult_permuted).sum()):.6f} (should match Teger)")

    # --- row 6: inverse curvature, mass-matched ---
    # Flip which edges are treated as "bottlenecks": amplify above-average
    # (positive) curvature instead of below-average (negative) curvature.
    edge_kappas = kappa[mask > 0]
    mu, sigma = edge_kappas.mean(), max(float(edge_kappas.std()), 1e-6)
    z_inv = (kappa - mu) / sigma          # sign flipped vs bottleneck_indicator
    x = params["tau"] * (z_inv + params["kappa_0"])
    b_inverse_raw = np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -60.0, 30.0)))) * mask
    raw_mass = float((W * b_inverse_raw).sum())
    s = total_mass / raw_mass if raw_mass > 0 else 0.0
    mult_inverse = s * b_inverse_raw
    print(f"row6 inverse scale s={s:.6f}, mass={float((W*mult_inverse).sum()):.6f}")

    for name, mult in [
        ("row3_none", mult_none),
        ("row4_uniform", mult_uniform),
        ("row5_permuted", mult_permuted),
        ("row6_inverse", mult_inverse),
    ]:
        path = os.path.join(OUT_DIR, f"brussels_xlstm_{name}.pt")
        torch.save(torch.tensor(mult, dtype=torch.float32), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
