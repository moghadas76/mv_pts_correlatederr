import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
import numpy as np

from dynamic_graph import _balanced_forman as _balanced_forman_curvature, _bottleneck_indicator, laplacian_from_adj


# =====================================================================
# Fig 1 (Revised): κ → b → ΔW → spectral → G_t pipeline  (6-panel)
# Addresses reviewer §1.i–iv and §4
# =====================================================================

def visualize_curvature_precision_effectiveness(loss_module, node_subset=None, save_path=None):
    """
    Revised Fig 1: 6-panel figure addressing all reviewer concerns.

    (a) Edge curvature with DIVERGING colormap (shows negative curvature)
    (b) Bottleneck score b_ij (should now be O(1), not O(1e-5))
    (c) Relative change (W' - W) / W  (reviewer §1.iii demanded this)
    (d) Spectral comparison: eigenvalues of L vs L' (validates λ₂ increase)
    (e) G_t with appropriate colorbar (reveals off-diagonals)
    (f) G_t off-diagonal magnitude histogram
    """
    cp = loss_module.curvature_precision
    device = next(cp.parameters()).device

    with torch.no_grad():
        W = cp.static_adj_sym.clone()

        # Compute curvature (uses revised self-loop-free version)
        kappa = _balanced_forman_curvature(W)                  # (N, N)

        # Edge mask (no self-loops)
        W_clean = W.clone()
        W_clean = W_clean - torch.diag(torch.diag(W_clean))
        mask = (W_clean > 0).float()

        # Bottleneck score (revised: distribution-aware)
        tau = cp.log_tau.exp()
        b_ij = _bottleneck_indicator(kappa, cp.kappa_0, tau, mask)

        # Reweighted adjacency
        lam = cp.log_lam.exp()
        W_prime = W_clean * (1.0 + lam * b_ij)
        W_prime = 0.5 * (W_prime + W_prime.T)

        # Relative change  (only where W_clean > 0)
        delta_W = torch.where(
            W_clean > 1e-8,
            (W_prime - W_clean) / W_clean,
            torch.zeros_like(W_clean),
        )

        # Spectral comparison
        L_orig = laplacian_from_adj(W_clean)
        L_prime = laplacian_from_adj(W_prime)
        eig_orig = torch.linalg.eigvalsh(L_orig.float()).cpu().numpy()
        eig_prime = torch.linalg.eigvalsh(L_prime.float()).cpu().numpy()

        # G_t via full forward pass
        G_t = cp()

    N = W.shape[0]
    n_vis = min(N, 50)
    idx = slice(0, n_vis)

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ------ (a) Edge curvature graph with diverging colormap ------
    ax_a = fig.add_subplot(gs[0, 0])
    G_nx = nx.from_numpy_array(W_clean[idx, idx].cpu().numpy())
    pos = nx.spring_layout(G_nx, seed=42)
    edge_curv = [kappa[u, v].item() for u, v in G_nx.edges()]
    if len(edge_curv) == 0:
        ax_a.text(0.5, 0.5, "No edges in subgraph", ha='center', transform=ax_a.transAxes)
    else:
        e_min, e_max = min(edge_curv), max(edge_curv)
        if e_min < 0:
            norm = TwoSlopeNorm(vmin=e_min, vcenter=0.0, vmax=max(e_max, 0.01))
        else:
            norm = plt.Normalize(vmin=e_min, vmax=e_max)
        nx.draw_networkx_nodes(G_nx, pos, node_size=30, ax=ax_a, node_color='steelblue')
        nx.draw_networkx_edges(
            G_nx, pos, ax=ax_a,
            edge_color=edge_curv, edge_cmap=plt.cm.RdYlGn,
            width=1.5, alpha=0.8,
        )
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
        plt.colorbar(sm, ax=ax_a, fraction=0.046, label=r'$\kappa$ (Balanced Forman)')
        frac_neg = sum(1 for c in edge_curv if c < 0) / max(len(edge_curv), 1)
        ax_a.set_title(
            f'(a) Edge Curvature\n'
            f'range: [{e_min:.3f}, {e_max:.3f}], {100*frac_neg:.0f}% negative',
            fontsize=11,
        )
    ax_a.axis('off')

    # ------ (b) Bottleneck score — should now be O(1) ------
    ax_b = fig.add_subplot(gs[0, 1])
    b_vis = b_ij[idx, idx].cpu().numpy()
    im_b = ax_b.imshow(b_vis, cmap='hot', aspect='auto')
    plt.colorbar(im_b, ax=ax_b, fraction=0.046, label=r'$b_{ij}$')
    nonzero_b = b_vis[b_vis > 1e-8]
    stats = (f'mean={nonzero_b.mean():.3f}, max={nonzero_b.max():.3f}'
             if len(nonzero_b) > 0 else 'all zero')
    ax_b.set_title(f'(b) Bottleneck Score $b_{{ij}}$\n{stats}', fontsize=11)
    ax_b.set_xlabel('Node j'); ax_b.set_ylabel('Node i')

    # ------ (c) Relative weight change ΔW/W ------
    ax_c = fig.add_subplot(gs[0, 2])
    dW_vis = delta_W[idx, idx].cpu().numpy()
    nz_dw = dW_vis[dW_vis != 0]
    if len(nz_dw) > 0:
        vabs = np.percentile(np.abs(nz_dw), 99)
    else:
        vabs = 1.0
    im_c = ax_c.imshow(dW_vis, cmap='PuOr', aspect='auto', vmin=-vabs, vmax=vabs)
    plt.colorbar(im_c, ax=ax_c, fraction=0.046, label=r"$(W'-W)/W$")
    mean_pct = np.abs(nz_dw).mean() * 100 if len(nz_dw) > 0 else 0
    ax_c.set_title(
        rf"(c) Relative Weight Change $(W'-W)/W$" + f'\nmean |change|={mean_pct:.1f}%',
        fontsize=11,
    )
    ax_c.set_xlabel('Node j'); ax_c.set_ylabel('Node i')

    # ------ (d) Spectral comparison ------
    ax_d = fig.add_subplot(gs[1, 0])
    k_show = min(20, len(eig_orig) - 1)
    ax_d.plot(range(1, k_show + 1), eig_orig[1:k_show + 1], 'o-',
              label=r'$L$ (original)', alpha=0.8, markersize=4)
    ax_d.plot(range(1, k_show + 1), eig_prime[1:k_show + 1], 's-',
              label=r"$L'$ (reweighted)", alpha=0.8, markersize=4)
    lam2_orig = eig_orig[1] if len(eig_orig) > 1 else 0
    lam2_prime = eig_prime[1] if len(eig_prime) > 1 else 0
    pct = 100 * (lam2_prime - lam2_orig) / max(abs(lam2_orig), 1e-10)
    ax_d.set_xlabel('Eigenvalue index $k$')
    ax_d.set_ylabel(r'$\lambda_k$')
    ax_d.set_title(
        f'(d) Laplacian Spectrum\n'
        rf'$\lambda_2$: {lam2_orig:.4f} → {lam2_prime:.4f} ({pct:+.1f}%)',
        fontsize=11,
    )
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # ------ (e) G_t with enhanced colormap ------
    ax_e = fig.add_subplot(gs[1, 1])
    G_np = G_t.cpu().float().numpy()
    diag_vals = np.diag(G_np)
    offdiag_abs = np.abs(G_np - np.diag(diag_vals))
    offdiag_max = offdiag_abs.max() if offdiag_abs.max() > 0 else 1e-6
    abs_max = np.max(np.abs(G_np))
    im_e = ax_e.imshow(
        G_np, cmap='RdBu_r', aspect='auto',
        norm=TwoSlopeNorm(vmin=-offdiag_max * 5, vcenter=0, vmax=abs_max),
    )
    plt.colorbar(im_e, ax=ax_e, fraction=0.046, label='Covariance')
    ax_e.set_title(
        r'(e) $G_t = Q_t^{-1}$' + f' (R={G_t.shape[0]})\n'
        f'diag: [{diag_vals.min():.4f}, {diag_vals.max():.4f}]\n'
        f'off-diag max: {offdiag_max:.4f}',
        fontsize=10,
    )
    ax_e.set_xlabel('Latent dim j'); ax_e.set_ylabel('Latent dim i')

    # ------ (f) Off-diagonal distribution ------
    ax_f = fig.add_subplot(gs[1, 2])
    offdiag_vals = G_np[~np.eye(G_np.shape[0], dtype=bool)]
    ax_f.hist(offdiag_vals, bins=40, alpha=0.7, label='Off-diagonal',
              color='steelblue', density=True)
    ax_f.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_f.axvline(diag_vals.mean(), color='red', linestyle='-', alpha=0.8,
                 label=f'Diag mean={diag_vals.mean():.4f}')
    od_ratio = np.abs(offdiag_vals).mean() / max(np.abs(diag_vals).mean(), 1e-10)
    ax_f.set_xlabel('Covariance value')
    ax_f.set_ylabel('Density')
    ax_f.set_title(
        f'(f) $G_t$ Entry Distribution\n|off-diag|/|diag| ratio: {od_ratio:.4f}',
        fontsize=10,
    )
    ax_f.legend(fontsize=8)
    ax_f.grid(True, alpha=0.3)

    plt.suptitle(
        r'Curvature $\to$ Bottleneck $\to$ Precision Pipeline (Revised)',
        fontsize=14, y=1.01,
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# =====================================================================
# Fig 2 (Revised): G_t ablation with off-diagonal visibility
# Addresses reviewer §2
# =====================================================================

def visualize_ablation_Gt_vs_IR(loss_curvature, loss_kernel, loss_graph, save_path=None):
    """
    Revised ablation: 2-row figure.
    Top: G_t heatmap with diverging colormap (reveals off-diagonals).
    Bottom: Row-0 bar profile showing coupling decay.
    Also reports condition number and off-diagonal/diagonal ratio.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                              gridspec_kw={'height_ratios': [3, 1]})

    configs = [
        (loss_kernel,    r'$G_t = I_R$ (Baseline)'),
        (loss_graph,     r'$G_t$ (GraphFactorKernel)'),
        (loss_curvature, r'$G_t = Q_t^{-1}$ (Ours)'),
    ]

    for col, (loss_mod, title) in enumerate(configs):
        cov_mats = loss_mod.get_covariance_matrices()
        G = cov_mats['G_t'].cpu().float()
        G_np = G.numpy()

        diag = np.diag(G_np)
        offdiag = G_np.copy()
        np.fill_diagonal(offdiag, 0)
        abs_offdiag_max = np.abs(offdiag).max()

        # ---- Top row: heatmap ----
        ax_top = axes[0, col]
        if abs_offdiag_max < 1e-10:
            im = ax_top.imshow(G_np, cmap='gray_r', aspect='auto',
                                vmin=0, vmax=max(diag.max(), 1e-6))
        else:
            im = ax_top.imshow(
                G_np, cmap='RdBu_r', aspect='auto',
                norm=TwoSlopeNorm(
                    vmin=min(G_np.min(), -abs_offdiag_max),
                    vcenter=0,
                    vmax=G_np.max(),
                ),
            )
        plt.colorbar(im, ax=ax_top, fraction=0.046)

        eigvals = torch.linalg.eigvalsh(G.float())
        eff_rank = (eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()
        cond = (eigvals.max() / eigvals.min().clamp(min=1e-10)).item()
        od_ratio = (np.abs(offdiag).mean() / max(np.abs(diag).mean(), 1e-10))

        ax_top.set_title(
            f'{title}\nEff.rank={eff_rank:.1f},  κ(G)={cond:.1f}\n'
            f'|off-diag|/|diag|={od_ratio:.4f}',
            fontsize=10,
        )
        ax_top.set_xlabel('Latent dim j')
        ax_top.set_ylabel('Latent dim i')

        # ---- Bottom row: row-0 bar profile ----
        ax_bot = axes[1, col]
        row0 = G_np[0, :]
        colors = ['red' if i == 0 else 'steelblue' for i in range(len(row0))]
        ax_bot.bar(range(len(row0)), row0, color=colors)
        ax_bot.set_xlabel('Latent dim j')
        ax_bot.set_ylabel(r'$G_t[0, j]$')
        ax_bot.set_title('Row 0 profile', fontsize=9)
        ax_bot.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_bot.grid(True, alpha=0.3)

    plt.suptitle(r'Factor Covariance $G_t$ Ablation (Revised)', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# =====================================================================
# Fig 3 (Revised): Converged parameter snapshot
# Addresses reviewer §4 (now includes γ, offdiag ratio, grad norm)
# =====================================================================

def track_curvature_params_during_training(loss_module, optimizer, dataloader,
                                            n_steps=200, save_path=None):
    """
    Revised parameter tracking: records curvature params over dataloader batches.
    Now includes γ (diffusion coupling), off-diag/diag ratio, and per-param values.
    At inference this shows the *converged* values are not at init.
    """
    history = {k: [] for k in [
        'alpha', 'beta', 'gamma', 'lam', 'kappa_0', 'tau',
        'eff_rank_Gt', 'offdiag_ratio',
    ]}

    cp = loss_module.curvature_precision
    for step, (x, y) in enumerate(dataloader):
        if step >= n_steps:
            break
        with torch.no_grad():
            history['alpha'].append(cp.log_alpha.exp().item())
            history['beta'].append(cp.log_beta.exp().item())
            history['gamma'].append(cp.log_gamma.exp().item())
            history['lam'].append(cp.log_lam.exp().item())
            history['kappa_0'].append(cp.kappa_0.item())
            history['tau'].append(cp.log_tau.exp().item())

            G_t = cp()
            G_np = G_t.cpu().float().numpy()
            eigvals = torch.linalg.eigvalsh(G_t.float())
            eff_rank = (eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()
            history['eff_rank_Gt'].append(eff_rank)

            diag = np.diag(G_np)
            offdiag = G_np.copy()
            np.fill_diagonal(offdiag, 0)
            ratio = np.abs(offdiag).mean() / max(np.abs(diag).mean(), 1e-10)
            history['offdiag_ratio'].append(ratio)

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    param_configs = [
        ('alpha',        r'$\alpha$ (precision base)',       '#e41a1c'),
        ('beta',         r'$\beta$ (Laplacian weight)',      '#377eb8'),
        ('gamma',        r'$\gamma$ (diffusion coupling)',   '#4daf4a'),
        ('lam',          r'$\lambda$ (bottleneck amplifier)', '#984ea3'),
        ('kappa_0',      r'$\kappa_0$ (curvature bias)',     '#ff7f00'),
        ('tau',          r'$\tau$ (softplus sharpness)',      '#a65628'),
        ('eff_rank_Gt',  r'Eff. Rank of $G_t$',             '#f781bf'),
        ('offdiag_ratio', r'$|$off-diag$|/|$diag$|$',       '#999999'),
    ]

    for ax, (key, label, color) in zip(axes.flat, param_configs):
        data = history[key]
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(label, fontsize=10)
            continue
        ax.plot(data, color=color, linewidth=1.5)
        ax.axhline(data[0], linestyle='--', color='gray', alpha=0.5,
                    label=f'val={data[0]:.4g}')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Batch')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('CurvatureAwareGraphPrecision: Converged Parameters', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return history


# =====================================================================
# Fig 4: Kronecker decomposition C_t ⊗ G_t  (addresses reviewer §3)
# =====================================================================

def visualize_batch_covariance_Sigma(loss_module, sample_pred=None, save_path=None):
    """
    Full Σ^{bat} = L(C_t ⊗ G_t)L^T + diag(d) structure.
    Decomposes into C_t, G_t, and the full C_t ⊗ G_t.
    """
    cov_mats = loss_module.get_covariance_matrices()
    C_t = cov_mats['C_t'].cpu().float()
    G_t = cov_mats['G_t'].cpu().float()

    D, R = C_t.shape[0], G_t.shape[0]

    # Compute C_t ⊗ G_t
    CkronG = torch.kron(C_t.contiguous(), G_t.contiguous())  # (D*R, D*R)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              gridspec_kw={'width_ratios': [D, R, D * R]})

    im0 = axes[0].imshow(C_t.numpy(), cmap='coolwarm', aspect='auto',
                          vmin=-1, vmax=1)
    axes[0].set_title(rf'$C_t$ (Temporal, {D}×{D})', fontsize=12)
    axes[0].set_xlabel('Time step j'); axes[0].set_ylabel('Time step i')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Use diverging colormap for G_t (shows off-diagonals better)
    G_np = G_t.numpy()
    abs_max_G = max(abs(G_np.max()), abs(G_np.min()), 1e-6)
    im1 = axes[1].imshow(G_np, cmap='RdBu_r', aspect='auto',
                          vmin=-abs_max_G, vmax=abs_max_G)
    axes[1].set_title(rf'$G_t = Q_t^{{-1}}$ (Factor cov, {R}×{R})', fontsize=12)
    axes[1].set_xlabel('Latent dim j'); axes[1].set_ylabel('Latent dim i')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(CkronG.numpy(), cmap='coolwarm', aspect='auto')
    axes[2].set_title(rf'$C_t \otimes G_t$ ({D * R}×{D * R})', fontsize=12)
    axes[2].set_xlabel('Time×Latent j'); axes[2].set_ylabel('Time×Latent i')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Add block separators on Kronecker panel
    for k in range(1, D):
        axes[2].axhline(k * R - 0.5, color='white', linewidth=0.5, alpha=0.6)
        axes[2].axvline(k * R - 0.5, color='white', linewidth=0.5, alpha=0.6)

    plt.suptitle(r'Decomposition of Batch Covariance $\Sigma^{bat}_t$',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# =====================================================================
# Rev 6: Quantitative effectiveness metrics  (addresses reviewer §general)
# =====================================================================

def compute_curvature_effectiveness_metrics(loss_module) -> dict:
    """
    Table 1 (supplementary): Quantitative metrics proving curvature module
    is active and produces meaningful changes.

    Returns a dict suitable for printing or inclusion in a LaTeX table.
    """
    cp = loss_module.curvature_precision

    with torch.no_grad():
        W = cp.static_adj_sym.clone()
        W_clean = W - torch.diag(torch.diag(W))
        mask = (W_clean > 0).float()

        kappa = _balanced_forman_curvature(W)
        tau = cp.log_tau.exp()
        b_ij = _bottleneck_indicator(kappa, cp.kappa_0, tau, mask)

        lam = cp.log_lam.exp()
        W_prime = W_clean * (1.0 + lam * b_ij)
        W_prime = 0.5 * (W_prime + W_prime.T)

        # Original Laplacian
        L = laplacian_from_adj(W_clean)
        eig_L = torch.linalg.eigvalsh(L.float())

        # Reweighted Laplacian
        L_p = laplacian_from_adj(W_prime)
        eig_Lp = torch.linalg.eigvalsh(L_p.float())

        # G_t properties
        G_t = cp()
        G_np = G_t.cpu().float().numpy()
        diag = np.diag(G_np)
        offdiag = G_np.copy()
        np.fill_diagonal(offdiag, 0)
        eigvals_G = torch.linalg.eigvalsh(G_t.float())

        edge_kappas = kappa[mask > 0]
        edge_b = b_ij[mask > 0]

    metrics = {
        # Curvature statistics
        'kappa_min':               edge_kappas.min().item(),
        'kappa_max':               edge_kappas.max().item(),
        'kappa_mean':              edge_kappas.mean().item(),
        'kappa_std':               edge_kappas.std().item(),
        'frac_negative_kappa':     (edge_kappas < 0).float().mean().item(),
        # Bottleneck scores
        'b_ij_mean':               edge_b.mean().item(),
        'b_ij_max':                edge_b.max().item(),
        'b_ij_nonzero_frac':       (edge_b > 0.01).float().mean().item(),
        # Spectral change
        'lambda_2_original':       eig_L[1].item(),
        'lambda_2_reweighted':     eig_Lp[1].item(),
        'lambda_2_pct_change':     100 * (eig_Lp[1] - eig_L[1]).item() / max(eig_L[1].item(), 1e-10),
        # G_t structure
        'G_t_offdiag_diag_ratio':  np.abs(offdiag).mean() / max(np.abs(diag).mean(), 1e-10),
        'G_t_eff_rank':            (eigvals_G.sum() ** 2 / (eigvals_G ** 2).sum()).item(),
        'G_t_condition_number':    (eigvals_G.max() / eigvals_G.min().clamp(min=1e-10)).item(),
        # Learned parameters
        'alpha':                   cp.log_alpha.exp().item(),
        'beta':                    cp.log_beta.exp().item(),
        'gamma':                   cp.log_gamma.exp().item(),
        'lambda':                  cp.log_lam.exp().item(),
        'kappa_0':                 cp.kappa_0.item(),
        'tau':                     cp.log_tau.exp().item(),
    }

    # Pretty print
    print("\n" + "=" * 60)
    print("Curvature Module Effectiveness Metrics")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:35s}: {v:>12.6f}")
    print("=" * 60 + "\n")

    return metrics