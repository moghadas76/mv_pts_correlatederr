"""
Dynamic Graph Module with SAGSAM (Semi-autonomous Generation Spatial Adjacency Matrix)

Implements the DG-LoGraP methodology for dynamic-graph-aware spatio-temporal covariance modeling.

Reference:
- SAGSAM from https://arxiv.org/pdf/2205.01480
- DG-LoGraP methodology for grouped latent spatio-temporal residual models
"""

import pickle
import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_static_graph(graph_path: str, num_nodes: int) -> torch.Tensor:
    """
    Load static graph from CSV file and convert to adjacency matrix.
    
    Args:
        graph_path: Path to the graph CSV file (from, to, distance format)
        num_nodes: Number of nodes in the graph
        
    Returns:
        adj_matrix: Static adjacency matrix (N, N)
    """
    df = pd.read_csv(graph_path)
    
    # Get unique node IDs and create mapping
    all_nodes = pd.concat([df['from'], df['to']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    
    # Initialize adjacency matrix
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    
    # Fill adjacency matrix with distance-based weights
    for _, row in df.iterrows():
        from_idx = node_to_idx.get(row['from'], None)
        to_idx = node_to_idx.get(row['to'], None)
        
        if from_idx is not None and to_idx is not None and from_idx < num_nodes and to_idx < num_nodes:
            # Use Gaussian kernel for distance-based weight
            weight = np.exp(-row['distance'] ** 2 / 2)
            adj_matrix[from_idx, to_idx] = weight
            adj_matrix[to_idx, from_idx] = weight  # Make symmetric
    
    # Add self-loops
    adj_matrix = adj_matrix + torch.eye(num_nodes)
    
    # Normalize (symmetric normalization)
    degree = adj_matrix.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    adj_matrix = degree_inv_sqrt.unsqueeze(1) * adj_matrix * degree_inv_sqrt.unsqueeze(0)
    
    return adj_matrix

def load_static_graph_from_pickle(pickle_path: str) -> torch.Tensor:
    """
    Load static graph adjacency matrix from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file containing the adjacency matrix
    Returns:
        adj_matrix: Static adjacency matrix (N, N)
    """
    _, _, adj_matrix = pickle.load(open(pickle_path, "rb"))
    adj_matrix = torch.from_numpy(adj_matrix).float()
    return adj_matrix


import torch
import torch.nn.functional as F

def symmetrize_adj(A: torch.Tensor) -> torch.Tensor:
    # A: (N,N)
    return 0.5 * (A + A.transpose(-1, -2))

def laplacian_from_adj(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # W: (N,N) symmetric nonnegative
    deg = W.sum(dim=-1)  # (N,)
    D = torch.diag(deg)
    L = D - W
    return L

def precision_from_adj(
    A: torch.Tensor,
    alpha: float = 1e-2,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    A: (N,N) adjacency (not necessarily symmetric)
    returns Q = alpha I + beta L, PD if alpha>0
    """
    W = symmetrize_adj(A)
    L = laplacian_from_adj(W, eps=eps)
    N = A.shape[-1]
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    Q = alpha * I + beta * L
    return Q


# =====================================================================
# Curvature-Aware Graph Precision  (Steps A → B → C of the proposal)
# =====================================================================

def _balanced_forman_curvature(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the Balanced Forman curvature proxy for every edge of an
    undirected weighted graph.

    W : (N, N) symmetric non-negative weight matrix (may contain self-loops).
    Returns κ : (N, N) curvature matrix (only meaningful where W > 0).

    **Revision (addressing reviewer §1.i):**
    Self-loops are removed before computing degrees and curvature, because
    self-loop weights inflate d_i and shift all κ positive, hiding the
    bridge / bottleneck edges that should have *negative* curvature.

    The Balanced Forman curvature κ_{ij} for edge (i,j) with weight w_{ij}:

        κ_{ij} = w_{ij} · ( 1/√d_i + 1/√d_j )
               + w_{ij} · Σ_{k ∈ Δ(i,j)} ( √(w_ik/d_i) + √(w_jk/d_j) )
               − w_{ij}

    where d_i = Σ_{k≠i} w_{ik}  (no self-loop) and Δ(i,j) is the set of
    common neighbours.
    """
    # Remove self-loops so degrees reflect only inter-node connectivity.
    # .clone() keeps the autograd graph alive for differentiable W.
    W_clean = W.clone()
    W_clean = W_clean - torch.diag(torch.diag(W_clean))        # zero diagonal

    N = W_clean.shape[0]
    mask = (W_clean > 0).float().detach()                      # topology fixed

    deg = W_clean.sum(dim=-1).clamp(min=eps)                   # (N,)
    inv_sqrt_deg = torch.rsqrt(deg)                            # (N,)

    # Term 1:  w_{ij} * (1/√d_i + 1/√d_j)
    term1 = W_clean * (inv_sqrt_deg.unsqueeze(1) + inv_sqrt_deg.unsqueeze(0))

    # Triangle term (common-neighbour contribution)
    A_bin = mask                                               # binary topology
    triangles = A_bin @ A_bin                                  # #common neighbours
    W_normed = W_clean / deg.unsqueeze(1).clamp(min=eps).sqrt()
    triangle_weight = W_normed @ W_normed.T
    term2 = W_clean * triangle_weight * (triangles > 0).float().detach()

    kappa = (term1 + term2 - W_clean) * mask                   # (N, N)
    return kappa

def _balanced_forman(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Lightweight Balanced-Forman curvature proxy.
        κ_{ij} = 2/d_max * (#triangles through (i,j)) / (d_i + d_j - 2)
    Returns edge-curvature matrix  (N, N), same sparsity pattern as W.
    """
    # degree vector
    deg = W.sum(dim=-1)                          # (N,)
    # triangle count proxy: (W^2)_{ij}
    tri  = torch.mm(W, W)                        # (N, N)
    d_i  = deg.unsqueeze(1).expand_as(W)
    d_j  = deg.unsqueeze(0).expand_as(W)
    denom = (d_i + d_j - 2).clamp(min=eps)
    d_max = deg.max().clamp(min=eps)
    kappa = (2.0 / d_max) * tri / denom
    return kappa * (W > 0).float()               # zero out non-edges


def _reweight_laplacian(
    static_adj: torch.Tensor,     # (N, N)  fixed binary/weighted adjacency
    lam:        torch.Tensor,     # scalar learnable
    kappa_0:    torch.Tensor,     # scalar learnable
    tau:        torch.Tensor,     # scalar learnable
) -> torch.Tensor:
    """
    Steps A-C from the paper:
        W'_{ij} = W_{ij} (1 + λ · softplus(τ(κ₀ − κ_{ij})))
    Returns the reweighted normalised Laplacian  L'_t  (N, N).
    """
    N = static_adj.shape[0]

    # Step A: symmetrise
    W = 0.5 * (static_adj + static_adj.t())

    # Step B: curvature + bottleneck score
    kappa = _balanced_forman(W)
    b     = F.softplus(tau * (kappa_0 - kappa))  # (N, N)

    # Step C: reweight edges → Laplacian
    W_prime = W * (1.0 + lam.abs() * b)          # abs() keeps lam sign-free
    deg     = W_prime.sum(dim=-1)
    L_prime = torch.diag(deg) - W_prime           # combinatorial Laplacian
    return L_prime    


def _bottleneck_indicator(
    kappa: torch.Tensor,
    kappa_0: torch.Tensor,
    tau: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Distribution-aware soft indicator of bottleneck edges  (Step B).

    **Revision (addressing reviewer §1.ii + §4):**
    Instead of  b_ij = softplus(τ(κ₀ − κ_ij))  which saturates to ~0 when
    all κ > κ₀ (reviewer: b_ij ≈ 1e-5, effectively inactive), we use a
    *relative* formulation based on the edge-curvature distribution:

        z_ij = (μ_κ − κ_ij) / (σ_κ + ε)        # standardised below-mean score
        b_ij = softplus(τ · (z_ij + κ₀)) · mask  # κ₀ is now a learnable bias

    Edges with *below-average* curvature (relative bottlenecks) get b_ij > 0.
    The z-score ensures the softplus input is centered near 0 regardless of
    the absolute curvature magnitude, keeping gradients alive for τ and κ₀.
    """
    edge_kappas = kappa[mask > 0]
    if edge_kappas.numel() == 0:
        return torch.zeros_like(kappa)

    mu_kappa = edge_kappas.mean()
    sigma_kappa = edge_kappas.std().clamp(min=1e-6)

    # Standardised below-mean score: positive for below-average curvature
    z_ij = (mu_kappa - kappa) / sigma_kappa                    # (N, N)

    # κ₀ acts as a learnable bias (shifts the bottleneck threshold)
    b_ij = F.softplus(tau * (z_ij + kappa_0)) * mask
    return b_ij


class CurvatureAwareGraphPrecision_LearnP(nn.Module):
    """
    G_t = (α I_R + β P^T L'_t P)^{-1}

    KEY CHANGE vs original:
        P ∈ R^{N×R}  is a *learnable* nn.Parameter initialised with the
        spectral eigenvectors but allowed to move freely during training.

    Because P is no longer constrained to eigenvectors of L, after a few
    gradient steps P^T L'_t P develops genuine off-diagonal entries
    → G_t gets non-trivial cross-latent coupling.

    Drop-in usage
    -------------
        self.curvature_precision = CurvatureAwareGraphPrecision_LearnP(
            rank=rank, num_nodes=num_nodes, static_adj=static_graph, ...
        )
        G_t = self.curvature_precision()   # (R, R)
    """

    def __init__(
        self,
        rank:       int,
        num_nodes:  int,
        static_adj: torch.Tensor,
        alpha:      float = 0.01,
        beta:       float = 1.0,
        lam:        float = 1.0,
        kappa_0:    float = 0.0,
        tau:        float = 5.0,
        sigma_min:  float = 1e-4,
        init_with_eigenvectors: bool = True,   # warm-start from spectral P
        fixed_multiplier: torch.Tensor = None,
        # --- ablation-ladder controls (rebuttal Priority 1) ---
        # 'curvature' (default, row 7 / Teger): b_ij learned end-to-end.
        # 'fixed'    : b_ij supplied externally via `fixed_multiplier` and
        #              frozen for the whole run (rows 3/4/5/6 — no-reweight,
        #              uniform, permuted-null, inverse-curvature; all
        #              mass-matched to a reference Teger run offline).
        reweight_mode: str = "curvature",
    ):
        super().__init__()
        self.rank      = rank
        self.num_nodes = num_nodes
        self.sigma_min = sigma_min
        self.reweight_mode = reweight_mode

        # Register static adjacency as a buffer (not trained, moves with device)
        self.register_buffer("static_adj", static_adj.float())
        self.register_buffer("static_adj_sym", 0.5 * (static_adj + static_adj.T))
        self.register_buffer("log_gamma", torch.tensor(math.log(max(1e-8, 1.0))))  # Default log_gamma = 0
        self.register_buffer("log_lam", torch.tensor(math.log(max(1e-8, 1.0))))    # Default log_lam = 0

        if reweight_mode == "fixed":
            # Frozen, externally-supplied edge multiplier b_ij (rows 3-6 of the
            # ablation ladder). W'_ij = W_ij * (1 + fixed_multiplier_ij).
            if fixed_multiplier is None:
                raise ValueError("reweight_mode='fixed' requires `fixed_multiplier`.")
            self.register_buffer("fixed_multiplier", fixed_multiplier.float())
            # lam/kappa_0/tau are inert placeholders in this mode (kept so
            # checkpoints/state_dicts stay shape-compatible with row 7).
            self.register_buffer("lam", torch.tensor(0.0))
            self.register_buffer("kappa_0", torch.tensor(0.0))
            self.register_buffer("log_tau", torch.tensor(0.0))
        else:
            # ── Learnable curvature parameters (row 7 / Teger, default) ──────
            self.lam       = nn.Parameter(torch.tensor(lam))
            self.kappa_0   = nn.Parameter(torch.tensor(kappa_0))
            self.log_tau   = nn.Parameter(torch.tensor(tau).log())

        self.log_alpha = nn.Parameter(torch.tensor(alpha).log())
        self.log_beta  = nn.Parameter(torch.tensor(beta).log())

        # ── THE FIX: learnable P ─────────────────────────────────────────────
        if init_with_eigenvectors:
            P_init = self._spectral_init()      # (N, R)  — warm start
        else:
            P_init = torch.randn(num_nodes, rank) / (num_nodes ** 0.5)

        # Free parameter — no orthogonality constraint intentionally
        # (gradient will shape it toward useful directions)
        self.register_buffer("P_init_cache", P_init.clone())
        self.P = nn.Parameter(P_init)           # (N, R)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _spectral_init(self) -> torch.Tensor:
        """Compute bottom-R eigenvectors of the graph Laplacian for warm-start."""
        W   = 0.5 * (self.static_adj + self.static_adj.t())
        deg = W.sum(dim=-1)
        L   = torch.diag(deg) - W
        try:
            # eigh returns eigenvalues in ascending order → take first R
            _, vecs = torch.linalg.eigh(L)
            return vecs[:, :self.rank].clone()
        except Exception:
            return torch.randn(self.num_nodes, self.rank) / (self.num_nodes ** 0.5)

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """Returns G_t = Q_t^{-1}  of shape  (R, R)."""
        alpha = self.log_alpha.exp()
        beta  = self.log_beta.exp()

        if self.reweight_mode == "fixed":
            # Rows 3-6: W' is frozen and supplied externally; only alpha,
            # beta and P (below) remain learnable, isolating the effect of
            # *where* the edge mass sits from everything else in row 7.
            W = 0.5 * (self.static_adj + self.static_adj.t())
            W_prime = W * (1.0 + self.fixed_multiplier)
            deg = W_prime.sum(dim=-1)
            L_prime = torch.diag(deg) - W_prime
        else:
            tau = self.log_tau.exp()
            # Reweighted Laplacian  L'_t  (N, N)
            L_prime = _reweight_laplacian(
                self.static_adj, self.lam, self.kappa_0, tau
            )

        # ── Project with LEARNED P  →  genuinely non-diagonal ───────────────
        # Normalise P columns so scale doesn't explode  (optional but stable)
        P_norm = F.normalize(self.P, dim=0)          # (N, R)
        L_R    = P_norm.t() @ L_prime @ P_norm        # (R, R)  — NOT diagonal

        # Precision matrix  Q_t = α I_R + β L'_R
        Q_t = alpha * torch.eye(self.rank, device=L_R.device, dtype=L_R.dtype) \
            + beta  * L_R

        # Symmetrise for numerical safety before inversion
        Q_t = 0.5 * (Q_t + Q_t.t())

        # Add floor to diagonal so Q_t is strictly PD
        Q_t = Q_t + self.sigma_min * torch.eye(
            self.rank, device=Q_t.device, dtype=Q_t.dtype
        )

        G_t = torch.linalg.inv(Q_t)                  # (R, R)
        return G_t
    
    def get_P_regularization_loss(self) -> torch.Tensor:
        """
        Optional regularization to encourage P to stay close to the initial
        spectral eigenvectors (prevents drifting too far from a good starting
        point early in training).

        This can be weighted by a hyperparameter λ_p when added to the main loss.
        """
        with torch.no_grad():
            P_init = self._spectral_init()  # (N, R)
        return F.mse_loss(self.P, P_init)
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Encourages P columns to stay orthonormal: ||P^T P - I||^2_F
        
        This allows P to rotate freely away from the spectral basis
        (which is what we want — to break the diagonal structure)
        while preventing columns from collapsing or becoming redundant.
        """
        R = self.P.shape[1]
        I = torch.eye(R, device=self.P.device, dtype=self.P.dtype)
        PtP = self.P.t() @ self.P          # (R, R)
        return torch.norm(PtP - I, p='fro') ** 2


# ─────────────────────────────────────────────────────────────────────────────
# FIX B  ── Spectral P  +  learned dense correction  V Φ V^T
# ─────────────────────────────────────────────────────────────────────────────

class CurvatureAwareGraphPrecision_LearnedRotation(nn.Module):
    """
    G_t = (α I_R + β P^T L'_t P)^{-1}  +  V Φ V^T

    KEY CHANGE vs original:
        After computing the (diagonal) spectral precision, we ADD a low-rank
        dense correction  V Φ V^T  where:
            V ∈ R^{R×R}  is a learnable orthonormal basis (parameterised via
                          Cayley map from a skew-symmetric matrix)
            Φ = diag(softplus(φ))  are learnable positive eigenvalues

        This correction is initialised near zero so training starts from the
        original spectral G_t and gradually learns off-diagonal structure.

    Drop-in usage
    -------------
        self.curvature_precision = CurvatureAwareGraphPrecision_LearnedRotation(
            rank=rank, num_nodes=num_nodes, static_adj=static_graph, ...
        )
        G_t = self.curvature_precision()   # (R, R)
    """

    def __init__(
        self,
        rank:       int,
        num_nodes:  int,
        static_adj: torch.Tensor,
        alpha:      float = 0.01,
        beta:       float = 1.0,
        lam:        float = 1.0,
        kappa_0:    float = 0.0,
        tau:        float = 5.0,
        sigma_min:  float = 1e-4,
        correction_rank: int = None,    # rank of VΦV^T, defaults to R
    ):
        super().__init__()
        self.rank      = rank
        self.num_nodes = num_nodes
        self.sigma_min = sigma_min
        self.cr        = correction_rank if correction_rank is not None else rank

        self.register_buffer("static_adj", static_adj.float())

        # ── Learnable curvature parameters ──────────────────────────────────
        self.log_alpha = nn.Parameter(torch.tensor(alpha).log())
        self.log_beta  = nn.Parameter(torch.tensor(beta).log())
        self.lam       = nn.Parameter(torch.tensor(lam))
        self.kappa_0   = nn.Parameter(torch.tensor(kappa_0))
        self.log_tau   = nn.Parameter(torch.tensor(tau).log())

        # ── Fixed spectral projection (same as original, NOT trained) ────────
        self.register_buffer("P", self._spectral_init())  # (N, R)

        # ── THE FIX: learned rotation + eigenvalues ──────────────────────────
        # Skew-symmetric matrix → Cayley map → orthonormal V
        # Initialise near zero so correction starts near 0
        self.skew_raw = nn.Parameter(
            torch.zeros(rank, rank)            # anti-symmetric entries
        )
        # Correction eigenvalues — initialised small so G_t starts spectral
        self.log_phi  = nn.Parameter(
            torch.full((self.cr,), fill_value=-4.0)   # softplus(-4) ≈ 0.018
        )
        self.register_buffer("static_adj_sym", 0.5 * (static_adj + static_adj.T))
        self.register_buffer("log_gamma", torch.tensor(math.log(max(1e-8, 1.0))))  # Default log_gamma = 0
        self.register_buffer("log_lam", torch.tensor(math.log(max(1e-8, 1.0))))    # Default log_lam = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _spectral_init(self) -> torch.Tensor:
        W   = 0.5 * (self.static_adj + self.static_adj.t())
        deg = W.sum(dim=-1)
        L   = torch.diag(deg) - W
        try:
            _, vecs = torch.linalg.eigh(L)
            return vecs[:, :self.rank].clone()
        except Exception:
            return torch.randn(self.num_nodes, self.rank) / (self.num_nodes ** 0.5)

    def _cayley_orthogonal(self) -> torch.Tensor:
        """
        Cayley map: skew-symmetric A  →  orthonormal V = (I-A)(I+A)^{-1}
        Guarantees V^T V = I so VΦV^T is a valid symmetric PSD matrix.
        """
        A   = self.skew_raw - self.skew_raw.t()         # enforce skew-symmetry
        I   = torch.eye(self.rank, device=A.device, dtype=A.dtype)
        V   = torch.linalg.solve(I + A, I - A)          # (I+A)^{-1}(I-A)
        return V                                         # (R, R) orthonormal

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """Returns G_t = spectral_G_t + VΦV^T  of shape  (R, R)."""
        alpha = self.log_alpha.exp()
        beta  = self.log_beta.exp()
        tau   = self.log_tau.exp()

        # ── Spectral part (same as original, diagonal) ───────────────────────
        L_prime = _reweight_laplacian(
            self.static_adj, self.lam, self.kappa_0, tau
        )
        L_R = self.P.t() @ L_prime @ self.P
        Q_t = alpha * torch.eye(self.rank, device=L_R.device, dtype=L_R.dtype) \
            + beta  * L_R
        Q_t = 0.5 * (Q_t + Q_t.t()) \
            + self.sigma_min * torch.eye(self.rank, device=Q_t.device, dtype=Q_t.dtype)
        G_spectral = torch.linalg.inv(Q_t)

        # ── Dense correction  V Φ V^T ────────────────────────────────────────
        V   = self._cayley_orthogonal()
        phi = F.softplus(self.log_phi)

        V_cr       = V[:, :self.cr]
        correction = V_cr @ torch.diag(phi) @ V_cr.t()

        G_t = G_spectral + correction

        # ── Normalize to unit trace so scale matches I_R ─────────────────────
        G_t = G_t * (self.rank / G_t.trace().clamp(min=1e-6))
        
        return G_t
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Encourages P columns to stay orthonormal: ||P^T P - I||^2_F
        
        This allows P to rotate freely away from the spectral basis
        (which is what we want — to break the diagonal structure)
        while preventing columns from collapsing or becoming redundant.
        """
        R = self.P.shape[1]
        I = torch.eye(R, device=self.P.device, dtype=self.P.dtype)
        PtP = self.P.t() @ self.P          # (R, R)
        return torch.norm(PtP - I, p='fro') ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Learnable Factor Covariance  (bypass Laplacian entirely)
# ─────────────────────────────────────────────────────────────────────────────

class LearnableFactorCovariance(nn.Module):
    """
    Directly parameterize G_t as a learnable PSD matrix:

        G_t = I_R  +  U diag(softplus(φ)) U^T

    where U ∈ R^{R×cr} is a learnable low-rank basis and
    φ ∈ R^{cr} are eigenvalues.

    **Time-varying mode** (hidden_dim > 0):
        φ(h) = softplus( log_phi_base  +  phi_net(h) )

    The network produces a *residual correction* to the base log-eigenvalues,
    so at initialisation (phi_net ≈ 0) the module reduces to the static
    version.  The basis U is shared across all time steps; only the
    eigenvalue magnitudes vary with hidden state → the *pattern* of
    cross-factor correlation is learned once, its *strength* adapts to
    each sequence context.

    Key design choices:
        · Initialized at I_R so training starts from the kernel baseline.
        · No dependency on graph Laplacian (which may be degenerate).
        · Backward-compatible: hidden_dim=0 recovers the original static G_t.

    Drop-in usage
    -------------
        # static
        self.learnable_factor_cov = LearnableFactorCovariance(rank=10, correction_rank=4)
        G_t = self.learnable_factor_cov()           # (R, R) PD

        # time-varying (hidden_dim > 0)
        self.learnable_factor_cov = LearnableFactorCovariance(rank=10, correction_rank=4, hidden_dim=16)
        G_t = self.learnable_factor_cov(h)          # h: (hidden_dim,)  →  (R, R) PD
    """

    def __init__(
        self,
        rank: int,
        correction_rank: int = None,
        sigma_min: float = 1e-4,
        hidden_dim: int = 0,       # 0 = static (backward compat), >0 = time-varying
    ):
        super().__init__()
        self.rank = rank
        self.sigma_min = sigma_min
        self.cr = correction_rank if correction_rank is not None else max(1, rank // 2)
        self.hidden_dim = hidden_dim

        self.U = nn.Parameter(torch.randn(rank, self.cr) * 0.01)   # small random init
        self.log_phi = nn.Parameter(torch.zeros(self.cr))           # softplus(0)≈0.69, meaningful signal

        # --- Time-varying: hidden → φ residual correction ---
        if hidden_dim > 0:
            self.phi_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ELU(),
                nn.Linear(hidden_dim * 2, self.cr),
            )
            # Init output layer near zero so G_t ≈ static G at start
            nn.init.zeros_(self.phi_net[-1].weight)
            nn.init.zeros_(self.phi_net[-1].bias)


    # ---- helpers --------------------------------------------------------
    def _cayley_orthogonal(self) -> torch.Tensor:
        """Cayley map: skew A → orthonormal V = (I−A)(I+A)^{-1}."""
        A = self.skew_raw - self.skew_raw.t()          # enforce skew-symmetry
        I = torch.eye(self.rank, device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I + A, I - A)        # (R, R) orthonormal

    # ---- forward --------------------------------------------------------
    def forward(self, h: torch.Tensor = None) -> torch.Tensor:
        """
        Returns G_t: (R, R) symmetric positive-definite.

        Parameters
        ----------
        h : (hidden_dim,) or None.
            Pooled hidden-state features.  When provided (and hidden_dim > 0),
            eigenvalues are modulated:  φ = softplus(log_phi + phi_net(h)).
            When None, falls back to the static base eigenvalues.
        """
        # Compute eigenvalues — static base + optional hidden correction
        if h is not None and self.hidden_dim > 0:
            phi = F.softplus(self.log_phi + self.phi_net(h))    # (cr,) time-varying
        else:
            phi = F.softplus(self.log_phi)                      # (cr,) static fallback

        I_R = torch.eye(self.rank, device=self.U.device, dtype=self.U.dtype)

        correction = self.U @ torch.diag(phi) @ self.U.t()     # (R, R)

        G_t = I_R + correction
        G_t = G_t + self.sigma_min * I_R
        return G_t

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Not needed (Cayley guarantees V^T V = I), included for API compat."""
        return torch.tensor(0.0, device=self.log_phi.device, dtype=self.log_phi.dtype)


class CurvatureAwareGraphPrecision(nn.Module):
    """
    Curvature-aware graph precision for factor covariance G_t.

    Implements Steps A → B → C of the proposal:

        Step A  :  W_t = ½(Ã_t + Ã_t^T)        (symmetrised batch-mean adj)
        Step B  :  κ_{ij,t} = BalancedForman(W_t)
                   b_{ij,t} = softplus(τ · (z_{ij} + κ_0))  (distribution-aware)
        Step C  :  W'_{ij,t} = W_{ij,t} · (1 + λ · b_{ij,t})
                   L'_t  from  W'_t  (graph Laplacian)
                   Q_t = α I + β L'_R + γ (L'_R)^k  (precision, PD)
                   G_t = Q_t^{-1}                    (factor covariance)

    The resulting G_t (R×R) replaces I_R in the Kronecker product
    C_t ⊗ G_t that appears in the batch covariance Σ^{bat}_t.

    **Revisions (addressing reviewer §1-4):**
    - Self-loop removal before curvature (Rev 3 → negative κ on bridges)
    - Distribution-aware bottleneck score (Rev 1 → b_ij ~ O(1))
    - Multi-hop diffusion term γ(L'_R)^k in precision (Rev 2 → off-diagonal G_t)
    - Fully differentiable forward (Rev 4 → no .item() detaching)

    Parameters
    ----------
    rank          : Latent rank R (dimension of G_t).
    num_nodes     : Number of spatial nodes N.
    static_adj    : (N, N) static adjacency (used when no dynamic adj is available).
    alpha         : Ridge regulariser in Q_t = αI + βL' (ensures PD).
    beta          : Weight of the Laplacian in Q_t.
    gamma         : Weight of multi-hop diffusion term (L'_R)^k.
    diffusion_hops: Number of hops k for multi-hop term.
    lam           : Strength of bottleneck reweighting  W'=W·(1+λ·b).
    kappa_0       : Learnable bias in bottleneck indicator (shifts threshold).
    tau           : Temperature of softplus bottleneck indicator.
    sigma_min     : Jitter added to G_t diagonal for numerical safety.
    """

    def __init__(
        self,
        rank: int,
        num_nodes: int,
        static_adj: Optional[torch.Tensor] = None,
        alpha: float = 0.01,
        beta: float = 1.0,
        gamma: float = 0.5,
        diffusion_hops: int = 2,
        lam: float = 1.0,
        kappa_0: float = 0.0,
        tau: float = 5.0,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.rank = rank
        self.num_nodes = num_nodes
        self.sigma_min = sigma_min
        self.diffusion_hops = diffusion_hops

        # Learnable precision hyper-parameters
        self.log_alpha = nn.Parameter(torch.tensor(math.log(max(alpha, 1e-8))))
        self.log_beta = nn.Parameter(torch.tensor(math.log(max(beta, 1e-8))))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(max(gamma, 1e-8))))
        self.log_lam = nn.Parameter(torch.tensor(math.log(max(lam, 1e-8))))
        self.kappa_0 = nn.Parameter(torch.tensor(float(kappa_0)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau, 1e-8))))

        # Learnable projection  P : N → R   (init from Fiedler eigvecs)
        if static_adj is not None:
            L_graph = self._normalised_laplacian(static_adj)
            _, eigvecs = torch.linalg.eigh(L_graph)
            P_init = eigvecs[:, :rank].clone()
        else:
            L_graph = torch.eye(num_nodes)
            P_init = torch.randn(num_nodes, rank) * (1.0 / math.sqrt(num_nodes))
        self.projector = nn.Parameter(P_init)                   # (N, R)
        self.register_buffer("L_graph_static", L_graph)

        # Raw static adjacency (for curvature computation — includes self-loops)
        if static_adj is not None:
            self.register_buffer("static_adj_raw", static_adj.clone())
        else:
            self.register_buffer("static_adj_raw", torch.eye(num_nodes))

        # Fallback static adj (symmetrised)
        if static_adj is not None:
            self.register_buffer("static_adj_sym", 0.5 * (static_adj + static_adj.T))
        else:
            self.register_buffer("static_adj_sym", torch.eye(num_nodes))

    
    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _normalised_laplacian(adj: torch.Tensor) -> torch.Tensor:
        W = 0.5 * (adj + adj.T)
        deg = W.sum(-1)
        d_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        N = W.shape[0]
        D_isqrt = torch.diag(d_inv_sqrt)
        return torch.eye(N, device=W.device, dtype=W.dtype) - D_isqrt @ W @ D_isqrt

    def _project_to_R(self, L_full: torch.Tensor) -> torch.Tensor:
        """L_R = P^T L P  (R×R), symmetrised."""
        P = self.projector                                      # (N, R)
        L_R = P.T @ L_full @ P                                 # (R, R)
        return 0.5 * (L_R + L_R.T)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(
        self,
        dynamic_adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute  G_t = Q_t^{-1}  with curvature-aware edge reweighting.

        **Revision (Rev 2+4):**
        - Precision now includes multi-hop diffusion:  Q_t = αI + βL'_R + γ(L'_R)^k
        - All operations are differentiable (no .item() calls that detach).
        - The mask is detached but curvature → b_ij → W' → L' → Q → G chain
          preserves gradient flow through τ, κ₀, λ, α, β, γ, P.

        Parameters
        ----------
        dynamic_adj : (B, N, N) or (N, N) or None.
                      If batched, we average over B first (Choice 2 / Step A).

        Returns
        -------
        G_t : (R, R)   positive-definite factor covariance.
        """
        # --- Step A: batch-mean symmetrised adjacency --------------------
        if dynamic_adj is not None:
            if dynamic_adj.dim() == 3:
                W_t = dynamic_adj.mean(dim=0)                   # (N, N)
            else:
                W_t = dynamic_adj                               # (N, N)
            W_t = 0.5 * (W_t + W_t.T)                          # symmetrise
        else:
            W_t = self.static_adj_sym                           # fallback

        # --- Step B: curvature + bottleneck indicator --------------------
        # Rev 4: keep tau, kappa_0 as tensors (no .item()) for gradient flow
        tau = self.log_tau.exp()
        kappa = _balanced_forman_curvature(W_t)                 # (N, N)
        mask = (W_t - torch.diag(torch.diag(W_t)) > 0).float().detach()
        b = _bottleneck_indicator(kappa, self.kappa_0, tau, mask)  # (N, N)

        # --- Step C: reweight, Laplacian, precision, invert --------------
        lam = self.log_lam.exp()
        W_prime = W_t * (1.0 + lam * b)                        # (N, N)
        W_prime = 0.5 * (W_prime + W_prime.T)                  # enforce symmetry
        # Zero out self-loops for Laplacian computation
        W_prime = W_prime - torch.diag(torch.diag(W_prime))

        L_prime = laplacian_from_adj(W_prime)                   # (N, N)

        # Project to latent space  L'_R = P^T L' P
        L_R = self._project_to_R(L_prime)                      # (R, R)

        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()
        gamma = self.log_gamma.exp()
        eye_R = torch.eye(self.rank, device=L_R.device, dtype=L_R.dtype)

        # Rev 2: Multi-hop diffusion term for off-diagonal coupling
        L_R_power = L_R.clone()
        for _ in range(self.diffusion_hops - 1):
            L_R_power = L_R_power @ L_R

        Q_t = alpha * eye_R + beta * L_R + gamma * L_R_power   # (R, R)  PD

        # Enforce symmetry and PD
        Q_t = 0.5 * (Q_t + Q_t.T)
        Q_t = Q_t + self.sigma_min * eye_R                     # jitter

        # G_t = Q_t^{-1}
        G_t = torch.linalg.inv(Q_t).contiguous()               # (R, R)
        G_t = 0.5 * (G_t + G_t.T)                              # enforce symmetry
        G_t = G_t + self.sigma_min * eye_R                     # jitter

        return G_t


class SAGSAM(nn.Module):
    """
    Semi-autonomous Generation Spatial Adjacency Matrix (SAGSAM)
    
    Generates time-varying weighted adjacency matrices by combining:
    - Learnable node embeddings 
    - Predefined static adjacency matrix
    
    Formula: Ã = softmax(ReLU(E_A · E_A^T)) · A
    
    where E_A ∈ R^{N×d} is the learnable node embedding dictionary.
    """
    
    def __init__(
        self,
        num_nodes: int,
        embed_dim: int,
        static_adj: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_nodes: Number of nodes N
            embed_dim: Node embedding dimension d
            static_adj: Predefined static adjacency matrix (N, N)
            dropout: Dropout rate for embeddings
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        
        # Learnable node embeddings E_A ∈ R^{N×d}
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        
        # Register static adjacency as buffer (non-trainable)
        if static_adj is not None:
            self.register_buffer('static_adj', static_adj)
        else:
            self.register_buffer('static_adj', torch.eye(num_nodes))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate dynamic adjacency matrix.
        
        Args:
            hidden_states: Optional hidden states from base model (B, N, H)
                          If provided, modulates node embeddings
                          
        Returns:
            dynamic_adj: Dynamic adjacency matrix (N, N) or (B, N, N)
        """
        # Get node embeddings, optionally modulated by hidden states
        if hidden_states is not None:
            # Project hidden states to embedding dimension
            # hidden_states: (B, N, H) -> (B, N, d)
            batch_size = hidden_states.shape[0]
            # Use base embeddings modulated by hidden states
            node_emb = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, d)
        else:
            node_emb = self.node_embedding  # (N, d)
        
        node_emb = self.dropout(node_emb)
        
        # Compute attention: softmax(ReLU(E_A · E_A^T))
        if hidden_states is not None:
            attention = torch.bmm(node_emb, node_emb.transpose(1, 2))  # (B, N, N)
        else:
            attention = torch.mm(node_emb, node_emb.T)  # (N, N)
        
        attention = F.relu(attention)
        attention = F.softmax(attention, dim=-1)
        
        # Combine with static adjacency: Ã = attention · A
        if hidden_states is not None:
            dynamic_adj = attention * self.static_adj.unsqueeze(0)  # (B, N, N)
        else:
            dynamic_adj = attention * self.static_adj  # (N, N)
        
        return dynamic_adj


class DynamicGraphLoader(nn.Module):
    """
    Wrapper module that provides dynamic-graph-aware loading matrix L^{(g)}_t
    for the DG-LoGraP methodology.
    
    Implements: L^{(g)}_t = U^{(g)}_t S^{(g)}_t
    
    where U^{(g)}_t comes from the leading eigenvectors of the dynamic covariance graph,
    and S^{(g)}_t is a learnable scaling matrix.
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_groups: int,
        ranks_per_group: List[int],
        embed_dim: int,
        static_adj: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            num_nodes: Number of nodes N
            num_groups: Number of groups G
            ranks_per_group: List of ranks [R_1, R_2, ..., R_G] for each group
            embed_dim: Embedding dimension for SAGSAM
            static_adj: Static adjacency matrix
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.ranks_per_group = ranks_per_group
        self.total_rank = sum(ranks_per_group)
        
        # SAGSAM for dynamic adjacency
        self.sagsam = SAGSAM(num_nodes, embed_dim, static_adj)
        
        # Learnable scaling matrices S^{(g)} for each group
        # Using Cholesky factor for positive definiteness
        self.scale_factors = nn.ParameterList([
            nn.Parameter(torch.eye(r) * 0.1) for r in ranks_per_group
        ])
        
    def compute_dynamic_eigenbasis(
        self,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Compute dynamic graph eigenbasis from SAGSAM-generated adjacency.
        
        Returns:
            List of U^{(g)}_t matrices for each group
        """
        # Get dynamic adjacency
        dynamic_adj = self.sagsam(hidden_states)  # (N, N) or (B, N, N)
        
        # Make symmetric covariance-like matrix
        if dynamic_adj.dim() == 2:
            cov_graph = (dynamic_adj + dynamic_adj.T) / 2
            # Eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_graph)
            # Sort in descending order
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, idx]
            
            # Extract leading eigenvectors for each group
            U_list = []
            start_idx = 0
            for r in self.ranks_per_group:
                U_g = eigenvectors[:, start_idx:start_idx + r]  # (N, R_g)
                U_list.append(U_g)
                start_idx += r
        else:
            batch_size = dynamic_adj.shape[0]
            cov_graph = (dynamic_adj + dynamic_adj.transpose(1, 2)) / 2  # (B, N, N)
            
            U_list = [[] for _ in range(self.num_groups)]
            for b in range(batch_size):
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_graph[b])
                idx = torch.argsort(eigenvalues, descending=True)
                eigenvectors = eigenvectors[:, idx]
                
                start_idx = 0
                for g, r in enumerate(self.ranks_per_group):
                    U_g = eigenvectors[:, start_idx:start_idx + r]
                    U_list[g].append(U_g)
                    start_idx += r
            
            U_list = [torch.stack(u, dim=0) for u in U_list]  # List of (B, N, R_g)
        
        return U_list
    
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute loading matrices L^{(g)}_t = U^{(g)}_t S^{(g)}_t for all groups.
        
        Args:
            hidden_states: Hidden states from base model (B, N, H)
            
        Returns:
            L_list: List of loading matrices L^{(g)}_t for each group
            dynamic_adj: The dynamic adjacency matrix
        """
        # Get eigenbasis
        U_list = self.compute_dynamic_eigenbasis(hidden_states)
        
        # Apply scaling: L^{(g)} = U^{(g)} @ S^{(g)}
        L_list = []
        for g, (U_g, S_g) in enumerate(zip(U_list, self.scale_factors)):
            # Ensure positive definiteness via softplus on diagonal
            S_g_scaled = S_g.tril(-1) + torch.diag(F.softplus(torch.diag(S_g)))
            
            if U_g.dim() == 2:
                L_g = torch.mm(U_g, S_g_scaled)  # (N, R_g)
            else:
                L_g = torch.bmm(U_g, S_g_scaled.unsqueeze(0).expand(U_g.shape[0], -1, -1))  # (B, N, R_g)
            
            L_list.append(L_g)
        
        dynamic_adj = self.sagsam(hidden_states)
        
        return L_list, dynamic_adj


class SAGS_GCN(nn.Module):
    """
    Semi-autonomous Generative Spatial Graph Convolution Network (SAGS-GCN)
    
    Combines SAGSAM with graph convolution for spatial feature extraction.
    
    Formula: Z_t = (I_N + softmax(ReLU(E_At · E_At^T)) · A) X_t E_At · W_At + E_At · b_At
    
    Where:
        - E_At: Time-varying node embedding matrix
        - W_At ∈ R^{d×C×F}: Shared weight pool
        - b_At ∈ R^{d×F}: Shared bias pool
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        out_features: int,
        embed_dim: int,
        static_adj: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_nodes: Number of nodes N
            in_features: Input feature dimension C
            out_features: Output feature dimension F
            embed_dim: Node embedding dimension d
            static_adj: Predefined static adjacency matrix
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.embed_dim = embed_dim
        
        # SAGSAM module
        self.sagsam = SAGSAM(num_nodes, embed_dim, static_adj, dropout)
        
        # Shared weight pool W_At ∈ R^{d×C×F}
        self.weight_pool = nn.Parameter(
            torch.randn(embed_dim, in_features, out_features) * (1.0 / math.sqrt(in_features))
        )
        
        # Shared bias pool b_At ∈ R^{d×F}
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, out_features))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of SAGS-GCN.
        
        Args:
            x: Input features X_t ∈ R^{B×N×C} or R^{N×C}
            
        Returns:
            z: Output features Z_t ∈ R^{B×N×F} or R^{N×F}
            dynamic_adj: Dynamic adjacency matrix
        """
        # Get node embeddings from SAGSAM
        node_emb = self.sagsam.node_embedding  # (N, d)
        node_emb = self.dropout(node_emb)
        
        # Compute attention weights
        attention = F.relu(torch.mm(node_emb, node_emb.T))  # (N, N)
        attention = F.softmax(attention, dim=-1)
        
        # Dynamic adjacency: Ã = attention · A
        dynamic_adj = attention * self.sagsam.static_adj  # (N, N)
        
        # Graph convolution with identity: (I_N + Ã)
        adj_with_identity = torch.eye(self.num_nodes, device=x.device) + dynamic_adj  # (N, N)
        
        # Compute node-specific weights: W_t = E_At · W_At  (N, C, F)
        # node_emb: (N, d), weight_pool: (d, C, F)
        node_weights = torch.einsum('nd,dcf->ncf', node_emb, self.weight_pool)  # (N, C, F)
        
        # Compute node-specific biases: b_t = E_At · b_At  (N, F)
        node_biases = torch.mm(node_emb, self.bias_pool)  # (N, F)
        
        # Apply graph convolution: (I_N + Ã) @ X_t
        if x.dim() == 2:
            # x: (N, C)
            x_conv = torch.mm(adj_with_identity, x)  # (N, C)
            # Apply node-specific transformation
            z = torch.einsum('nc,ncf->nf', x_conv, node_weights) + node_biases  # (N, F)
        else:
            # x: (B, N, C)
            batch_size = x.shape[0]
            x_conv = torch.bmm(
                adj_with_identity.unsqueeze(0).expand(batch_size, -1, -1),
                x
            )  # (B, N, C)
            # Apply node-specific transformation
            z = torch.einsum('bnc,ncf->bnf', x_conv, node_weights)  # (B, N, F)
            z = z + node_biases.unsqueeze(0)  # (B, N, F)
        
        return z, dynamic_adj


class TemporalCorrelationKernel(nn.Module):
    """
    Dynamic kernel-mixture parameterization for temporal correlation C^{(g)}_t.
    
    C^{(g)}_t = Σ_m w^{(g)}_{m,t} K_m
    
    where K_m are fixed kernel matrices and w^{(g)}_{m,t} are softmax weights
    produced from hidden states.
    """
    
    def __init__(
        self,
        horizon: int,
        num_kernels: int = 4,
        min_lengthscale: float = 0.5,
        max_lengthscale: float = 8.0,
        hidden_dim: int = 32,
        include_identity: bool = True,
    ):
        """
        Args:
            horizon: Temporal horizon D
            num_kernels: Number of base kernel matrices M
            min_lengthscale: Minimum lengthscale for SE kernels
            max_lengthscale: Maximum lengthscale for SE kernels
            hidden_dim: Hidden dimension for weight network
            include_identity: Whether to include identity kernel
        """
        super().__init__()
        
        self.horizon = horizon
        self.num_kernels = num_kernels
        self.include_identity = include_identity
        
        # Generate lengthscales
        if include_identity:
            lengthscales = torch.linspace(min_lengthscale, max_lengthscale, num_kernels - 1)
        else:
            lengthscales = torch.linspace(min_lengthscale, max_lengthscale, num_kernels)
        
        # Pre-compute base kernel matrices
        self.register_buffer('time_dist', torch.arange(horizon).float())
        
        kernel_list = []
        for l in lengthscales:
            # Squared exponential kernel
            dist_sq = (self.time_dist.unsqueeze(0) - self.time_dist.unsqueeze(1)) ** 2
            K = torch.exp(-dist_sq / (2 * l ** 2))
            kernel_list.append(K)
        
        if include_identity:
            kernel_list.append(torch.eye(horizon))
        
        # Stack kernels: (M, D, D)
        self.register_buffer('base_kernels', torch.stack(kernel_list, dim=0))
        
        # Weight network: produces mixing weights from hidden state
        total_kernels = num_kernels
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_kernels),
        )
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic temporal correlation matrix.
        
        Args:
            hidden_state: Hidden representation (B, H) or (H,)
            
        Returns:
            C: Temporal correlation matrix (B, D, D) or (D, D)
        """
        # Compute mixture weights
        logits = self.weight_net(hidden_state)  # (B, M) or (M,)
        weights = F.softmax(logits, dim=-1)  # (B, M) or (M,)
        
        # Mix kernels
        if weights.dim() == 1:
            C = torch.einsum('m,mij->ij', weights, self.base_kernels)  # (D, D)
        else:
            C = torch.einsum('bm,mij->bij', weights, self.base_kernels)  # (B, D, D)
        
        return C


class DiffusionWithSourceCovariance(nn.Module):
    """
    Graph Laplacian Diffusion with Source Term for spatial error covariance.
    
    Models the spatial error covariance as arising from a heat diffusion process
    on the graph with a source term:
    
        de/dτ = -α L e + s
    
    The stationary covariance (at equilibrium) is:
        Σ_spatial = (2α L)^{-1} diag(s²)  (continuous-time steady state)
    
    For finite diffusion time τ, the covariance kernel is:
        K(τ) = e^{-α τ L}  (diffusion component)
    
    And the full spatial covariance becomes:
        G = e^{-α τ L} G_0 e^{-α τ L}^T + S  (diffusion + source floor)
    
    where:
        - G_0 is the initial error covariance (learned or identity)
        - S = diag(s²) with learnable per-node source strengths
        - The source term S captures the irreducible error floor
    
    Projected to rank space: G_R = P^T G P  ∈ R^{R×R}
    """
    
    def __init__(
        self,
        rank: int,
        num_nodes: int,
        static_adj: torch.Tensor = None,
        alpha_init: float = 1.0,
        tau_init: float = 1.0,
        sigma_min: float = 1e-4,
        learnable_projection: bool = True,
        source_parameterization: str = "per_node",  # "per_node" | "scalar" | "per_rank"
        hidden_proj_dim: int = 0,
        ema_rate: float = 0.9,
    ):
        """
        Args:
            rank: R — dimension of factor space
            num_nodes: N — number of graph nodes
            static_adj: (N, N) static adjacency matrix
            alpha_init: Initial diffusivity parameter
            tau_init: Initial diffusion time
            sigma_min: Minimum diagonal jitter for PD
            learnable_projection: Whether P is learnable or fixed (top-R eigenvectors)
            source_parameterization: How to parameterize the source term
            hidden_proj_dim: Dimension of projected hidden features for state-aware
                diffusion (Revision 1). Set > 0 to enable. When enabled, the
                effective Laplacian blends the graph Laplacian with a hidden-state
                similarity Laplacian: L_eff = L_graph + gamma * L_hidden(h).
            ema_rate: EMA coefficient for updating node_repr_buffer (0.9 = slow decay).
        """
        super().__init__()
        self.rank = rank
        self.num_nodes = num_nodes
        self.sigma_min = sigma_min
        self.source_param = source_parameterization
        
        # --- Compute graph Laplacian ---
        if static_adj is None:
            static_adj = torch.eye(num_nodes)
        
        A_sym = 0.5 * (static_adj + static_adj.T)
        A_sym.fill_diagonal_(0.0)
        D_deg = A_sym.sum(dim=-1)
        L = torch.diag(D_deg) - A_sym  # Unnormalized Laplacian
        
        # Normalized Laplacian for numerical stability
        D_inv_sqrt = torch.diag(1.0 / (D_deg.sqrt() + 1e-8))
        L_norm = torch.eye(num_nodes) - D_inv_sqrt @ A_sym @ D_inv_sqrt
        
        self.register_buffer('laplacian', L_norm)
        
        # --- Eigendecomposition of L for efficient matrix exponential ---
        # L = U Λ U^T  =>  e^{-α τ L} = U e^{-α τ Λ} U^T
        eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
        eigenvalues = eigenvalues.clamp(min=0.0)  # Ensure non-negative
        
        self.register_buffer('L_eigenvalues', eigenvalues)   # (N,)
        self.register_buffer('L_eigenvectors', eigenvectors)  # (N, N)
        
        # --- Projection matrix P: N → R ---
        # Initialize with top-R eigenvectors (smallest non-zero eigenvalues)
        # These capture the smoothest graph modes
        # Skip the constant eigenvector (eigenvalue ≈ 0)
        idx = torch.argsort(eigenvalues)
        # Take eigenvectors corresponding to smallest non-trivial eigenvalues
        start_idx = 1 if eigenvalues[idx[0]] < 1e-6 else 0
        selected = idx[start_idx:start_idx + rank]
        P_init = eigenvectors[:, selected]  # (N, R)
        
        if learnable_projection:
            self.P = nn.Parameter(P_init.clone())
        else:
            self.register_buffer('P', P_init)
        
        # --- Learnable diffusion parameters ---
        # α: diffusivity (controls how fast correlation decays with graph distance)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
        
        # τ: diffusion time (controls the effective range of spatial correlation)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        
        # --- Source term s: irreducible error floor ---
        if source_parameterization == "per_node":
            # One source strength per node, projected to rank space
            self.log_source = nn.Parameter(torch.zeros(num_nodes))
        elif source_parameterization == "scalar":
            # Single scalar source strength
            self.log_source = nn.Parameter(torch.tensor(0.0))
        elif source_parameterization == "per_rank":
            # One source strength per rank dimension (after projection)
            self.log_source = nn.Parameter(torch.zeros(rank))
        else:
            raise ValueError(f"Unknown source_parameterization: {source_parameterization}")

        # --- Revision 1: State-Aware Diffusion ---
        # Maintains a per-node EMA representation buffer updated each training batch.
        # The effective Laplacian blends the graph Laplacian with a hidden-state
        # similarity Laplacian, correcting for xLSTM's content-based (not graph-based)
        # error correlation structure.
        self.hidden_proj_dim = hidden_proj_dim
        self.ema_rate = ema_rate
        if hidden_proj_dim > 0:
            self.register_buffer('node_repr_buffer', torch.zeros(num_nodes, hidden_proj_dim))
            # log_gamma: weight for the hidden-state Laplacian term. Init near -1
            # so gamma ≈ 0.37 — starts as a small correction, learns to grow.
            self.log_gamma = nn.Parameter(torch.tensor(-1.0))
            # log_sigma_hidden: RBF lengthscale for hidden-state similarity.
            # Init = 0 → sigma = 1.0.  Larger sigma = smoother similarity kernel.
            self.log_sigma_hidden = nn.Parameter(torch.tensor(0.0))

    # -----------------------------------------------------------------
    # Revision 1 helpers
    # -----------------------------------------------------------------
    @torch.no_grad()
    def update_node_repr(self, features: torch.Tensor) -> None:
        """
        Update EMA node representation buffer.

        Called once per training batch from BatchMGDDiffusion_Kernel.loss().
        The buffer is *detached* from the gradient graph — it carries information
        about xLSTM's hidden-state distribution without adding a gradient path
        from the buffer into the model weights.

        Args:
            features: (num_nodes, hidden_proj_dim) projected hidden features
                      pooled over the current batch.
        """
        self.node_repr_buffer.mul_(self.ema_rate).add_(
            (1.0 - self.ema_rate) * features.to(self.node_repr_buffer.device)
        )

    def _compute_hidden_laplacian(self) -> torch.Tensor:
        """
        Build a symmetric-normalized Laplacian from node hidden representations.

        Uses an RBF kernel on the EMA buffer to measure hidden-state similarity:
            W_h[i,j] = exp(-||h_i - h_j||^2 / (2 sigma^2))
            L_h = I - D^{-1/2} W_h D^{-1/2}   (eigenvalues in [0, 2])

        The lengthscale sigma = exp(log_sigma_hidden) is learnable, allowing the
        model to adapt the width of the similarity neighbourhood.

        Gradients flow through log_sigma_hidden (via sigma → dist_sq scaling)
        but NOT through node_repr_buffer (it is a detached buffer).
        """
        H = self.node_repr_buffer  # (N, d) — detached buffer
        sigma = self.log_sigma_hidden.exp()
        dist_sq = torch.cdist(H, H, p=2).pow(2)            # (N, N)
        W_h = torch.exp(-dist_sq / (2.0 * sigma ** 2))     # RBF weights
        # Remove self-loops so degree reflects only inter-node similarity
        eye = torch.eye(self.num_nodes, device=W_h.device, dtype=W_h.dtype)
        W_h = W_h * (1.0 - eye)
        deg = W_h.sum(dim=-1).clamp(min=1e-8)
        d_inv_sqrt = deg.pow(-0.5)
        # Symmetric normalized Laplacian: L_h = I - D^{-1/2} W_h D^{-1/2}
        L_h = eye - (d_inv_sqrt.unsqueeze(1) * W_h * d_inv_sqrt.unsqueeze(0))
        return L_h

    def _get_diffusion_kernel_full(self) -> torch.Tensor:
        """
        Compute the full (N×N) diffusion kernel e^{-α τ L_eff}.

        **Revision 1 (State-Aware Diffusion):**
        When hidden_proj_dim > 0 and the node_repr_buffer has been populated,
        the effective Laplacian is:

            L_eff = L_graph + gamma * L_hidden(h)

        where L_hidden is the symmetric-normalized Laplacian derived from
        pairwise hidden-state similarities (RBF kernel).  This corrects for
        xLSTM's content-based error correlation structure: nodes with similar
        hidden states get stronger diffusive coupling regardless of graph distance.

        Gradient flow: alpha, tau, gamma, log_sigma_hidden all receive gradients.
        node_repr_buffer is detached (EMA update, no backprop through it).

        Returns:
            (N, N) diffusion kernel matrix
        """
        alpha = self.log_alpha.exp()
        tau = self.log_tau.exp()

        # --- Revision 1: blend graph + hidden-state Laplacian ---
        buffer_active = (
            self.hidden_proj_dim > 0
            and self.node_repr_buffer.abs().max().item() > 1e-6
        )
        if buffer_active:
            gamma = self.log_gamma.exp()
            L_h = self._compute_hidden_laplacian()          # (N, N), grad through sigma
            L_eff = self.laplacian + gamma * L_h            # grad through gamma
            L_eff = 0.5 * (L_eff + L_eff.T)               # enforce symmetry
            # Use matrix_exp instead of eigh + manual exponentiation.
            # eigh backward involves 1/(λ_i − λ_j) terms that blow up for the
            # near-degenerate eigenvalues typical of road-network Laplacians,
            # corrupting gradients after a few steps.  torch.linalg.matrix_exp
            # uses Padé approximants whose backward is numerically stable.
            K = torch.linalg.matrix_exp(-alpha * tau * L_eff)
        else:
            # Fall back to pre-stored graph Laplacian eigenbasis (no hidden info yet)
            exp_eigenvalues = torch.exp(-alpha * tau * self.L_eigenvalues)
            K = self.L_eigenvectors @ torch.diag(exp_eigenvalues) @ self.L_eigenvectors.T
        return K
    
    def _get_source_covariance_full(self) -> torch.Tensor:
        """
        Compute the full (N×N) source covariance S = diag(s²).
        
        This represents the irreducible error floor — the spatial correlation
        that persists even at large graph distances.
        
        Returns:
            (N, N) source covariance (diagonal)
        """
        if self.source_param == "per_node":
            s_squared = F.softplus(self.log_source)  # (N,)
            return torch.diag(s_squared)
        elif self.source_param == "scalar":
            s_squared = F.softplus(self.log_source)
            return s_squared * torch.eye(self.num_nodes, device=self.log_source.device)
        else:
            return None  # Handled in rank space directly
    
    def _project_to_rank(self, M_full: torch.Tensor) -> torch.Tensor:
        """
        Project an (N, N) matrix to (R, R) via P^T M P.
        
        If P is learnable, we orthonormalize via QR to maintain well-conditioned
        projection.
        """
        if isinstance(self.P, nn.Parameter):
            # Orthonormalize P via QR decomposition
            P_orth, _ = torch.linalg.qr(self.P)
        else:
            P_orth = self.P
        
        return P_orth.T @ M_full @ P_orth  # (R, R)
    
    def forward(self, hidden: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the (R, R) factor covariance G_t.
        
        G_full = K @ G_0 @ K^T + S
        
        where K = e^{-α τ L} is the diffusion kernel and S is the source floor.
        Then G_t = P^T G_full P  (projected to rank space).
        
        For the initial covariance G_0, we use the identity (error starts
        independent, then diffusion induces correlation).
        
        Args:
            hidden: Optional hidden state for time-varying α, τ (not used in 
                    static version)
        
        Returns:
            G_t: (R, R) positive-definite factor covariance matrix
        """
        # Diffusion kernel in full node space
        K = self._get_diffusion_kernel_full()  # (N, N)
        
        # G_full = K @ I @ K^T + S = K @ K^T + S
        # This is the covariance after diffusing independent errors + source
        G_full = K @ K.T  # (N, N) — diffused component
        
        # Add source term (the error floor)
        S_full = self._get_source_covariance_full()
        if S_full is not None:
            G_full = G_full + S_full
        
        # Project to rank space
        G_t = self._project_to_rank(G_full)  # (R, R)
        
        # Handle per_rank source (added directly in rank space)
        if self.source_param == "per_rank":
            s_squared = F.softplus(self.log_source)  # (R,)
            G_t = G_t + torch.diag(s_squared)
        
        # Ensure PD with minimum jitter
        G_t = G_t + self.sigma_min * torch.eye(self.rank, device=G_t.device)
        
        # Symmetrize for numerical safety
        G_t = 0.5 * (G_t + G_t.T)
        
        return G_t
    
    def get_diffusion_parameters(self) -> Dict[str, float]:
        """Return interpretable diffusion parameters for logging."""
        params = {
            'alpha': self.log_alpha.exp().item(),
            'tau': self.log_tau.exp().item(),
            'alpha_tau': (self.log_alpha.exp() * self.log_tau.exp()).item(),
            'source_mean': F.softplus(self.log_source).mean().item(),
        }
        if self.hidden_proj_dim > 0:
            params['gamma'] = self.log_gamma.exp().item()
            params['sigma_hidden'] = self.log_sigma_hidden.exp().item()
            params['buffer_norm'] = self.node_repr_buffer.norm().item()
        return params


class GARCHDiffusionCovariance(DiffusionWithSourceCovariance):
    """
    GARCH(1,1)-inspired dynamic spatial factor covariance with graph diffusion.

    The factor covariance G_t follows a GARCH(1,1) recursion on top of the
    graph Laplacian diffusion covariance:

        G_t = ω · G_∞ + α · Shock_t + β · G_{t-1}

    where [ω, α, β] = softmax([raw_ω, raw_α, raw_β])  (stationarity by construction)

    Components:
        G_∞    = P^T (K K^T + S) P   — long-run diffusion covariance (parent class)
        Shock_t = P^T diag(ε̄²_t) P  — ARCH term: projected EMA of squared residuals
        G_{t-1}                       — GARCH persistence: lagged factor covariance

    The stationarity condition α + β < 1 is automatically satisfied because
    [ω, α, β] sum to 1 (softmax) and ω > 0.  The unconditional (long-run) mean is:

        E[G_t] = G_∞   (since E[Shock_t] ≈ G_∞ at stationarity)

    Physical interpretation:
        - α controls how fast the model reacts to new spatial error patterns
          (high α → quick adaptation to volatility spikes).
        - β controls persistence: how long a volatility burst lingers
          (high β → slow decay, long memory).
        - ω controls mean-reversion strength back to the graph diffusion prior.

    Buffers (no-grad, updated once per batch after forward()):
        resid_sq_buffer : (N,)   EMA of per-node squared residuals
        G_prev_buffer   : (R, R) last observed factor covariance matrix
    """

    def __init__(
        self,
        rank: int,
        num_nodes: int,
        static_adj: torch.Tensor = None,
        alpha_init: float = 1.0,
        tau_init: float = 1.0,
        sigma_min: float = 1e-4,
        learnable_projection: bool = True,
        source_parameterization: str = "per_node",
        hidden_proj_dim: int = 0,
        ema_rate: float = 0.9,
        # --- GARCH-specific ---
        garch_alpha_init: float = 0.1,
        garch_beta_init: float = 0.8,
        ema_resid_rate: float = 0.9,
    ):
        """
        Args:
            garch_alpha_init: Initial ARCH coefficient (reaction to past shocks).
            garch_beta_init:  Initial GARCH coefficient (volatility persistence).
            ema_resid_rate:   EMA decay rate for the squared-residual buffer.
                              Close to 1 → slow-moving shock; close to 0 → reactive.

        All other arguments are forwarded to DiffusionWithSourceCovariance.
        """
        super().__init__(
            rank=rank,
            num_nodes=num_nodes,
            static_adj=static_adj,
            alpha_init=alpha_init,
            tau_init=tau_init,
            sigma_min=sigma_min,
            learnable_projection=learnable_projection,
            source_parameterization=source_parameterization,
            hidden_proj_dim=hidden_proj_dim,
            ema_rate=ema_rate,
        )
        self.ema_resid_rate = ema_resid_rate

        # GARCH weights as a 3-way softmax to guarantee ω + α + β = 1.
        # Initialize raw logits so that softmax ≈ [1-α-β, α, β].
        omega_init = max(1.0 - garch_alpha_init - garch_beta_init, 1e-4)
        raw_init = torch.tensor(
            [math.log(omega_init),
             math.log(garch_alpha_init + 1e-8),
             math.log(garch_beta_init + 1e-8)]
        )
        self.garch_raw = nn.Parameter(raw_init)  # (3,) learnable

        # Buffers: no gradient, updated via update_garch_state()
        # resid_sq_buffer: EMA of per-node squared residuals (initialised to 1)
        self.register_buffer('resid_sq_buffer', torch.ones(num_nodes))
        # G_prev_buffer: lagged factor covariance (initialised to identity)
        self.register_buffer('G_prev_buffer', torch.eye(rank))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_garch_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (ω, α, β) via 3-way softmax — always sum to 1, all > 0."""
        weights = F.softmax(self.garch_raw, dim=0)
        return weights[0], weights[1], weights[2]

    def _compute_shock_R(self) -> torch.Tensor:
        """
        Compute ARCH shock in rank space: Shock = P^T diag(ε̄²) P ∈ R^{R×R}.

        Uses the EMA squared-residual buffer (detached from grad graph).
        Equivalent to projecting the per-node sample variance into factor space.
        """
        sq_resid = self.resid_sq_buffer.clamp(min=0.0)  # (N,) non-negative
        if isinstance(self.P, nn.Parameter):
            P_orth, _ = torch.linalg.qr(self.P)
        else:
            P_orth = self.P
        # P_orth^T diag(sq_resid) P_orth = (P_orth * sq_resid[:, None]).T @ P_orth
        shock_R = (P_orth * sq_resid.unsqueeze(1)).T @ P_orth  # (R, R)
        return shock_R

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the (R, R) factor covariance G_t via GARCH(1,1) recursion.

        Step-by-step:
          1.  G_∞  ← parent diffusion covariance  (long-run spatial structure)
          2.  Shock ← P^T diag(resid_sq_buffer) P  (ARCH shock from EMA residuals)
          3.  G_t  = ω·G_∞  +  α·Shock  +  β·G_{t-1}
          4.  Clamp to PD and symmetrize.

        The GARCH buffers are *not* updated here; call update_garch_state()
        after the loss backward to advance the state for the next batch.

        Returns:
            G_t: (R, R) positive-definite factor covariance.
        """
        # --- Long-run diffusion covariance G_∞ ---
        K = self._get_diffusion_kernel_full()       # (N, N)
        G_full_inf = K @ K.T                        # diffused component
        S_full = self._get_source_covariance_full()
        if S_full is not None:
            G_full_inf = G_full_inf + S_full
        G_inf = self._project_to_rank(G_full_inf)   # (R, R)
        if self.source_param == "per_rank":
            G_inf = G_inf + torch.diag(F.softplus(self.log_source))

        # --- GARCH coefficients ---
        omega_w, alpha_w, beta_w = self._get_garch_weights()

        # --- ARCH shock: projected squared residuals ---
        shock_R = self._compute_shock_R()           # (R, R), from detached buffer

        # --- GARCH persistence: lagged covariance ---
        # .clone() gives G_prev its own storage so the subsequent in-place
        # copy_() inside update_garch_state() doesn't bump the version counter
        # of a tensor that is already in the backward graph.
        G_prev = self.G_prev_buffer.clone()         # (R, R)

        # --- GARCH(1,1) recursion ---
        G_t = omega_w * G_inf + alpha_w * shock_R + beta_w * G_prev

        # Ensure PD and symmetry
        G_t = G_t + self.sigma_min * torch.eye(self.rank, device=G_t.device, dtype=G_t.dtype)
        G_t = 0.5 * (G_t + G_t.T)

        # Cache for state update (detached — buffer update must not affect gradients)
        self._last_G_t = G_t.detach()

        return G_t

    # ------------------------------------------------------------------
    # State update (called once per batch, outside autograd)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_garch_state(self, sq_resid_per_node: torch.Tensor) -> None:
        """
        Advance the GARCH state after each training batch.

        1. EMA-update resid_sq_buffer with the batch's per-node squared residuals.
        2. Copy _last_G_t → G_prev_buffer for use in the next forward().

        Args:
            sq_resid_per_node: (N,) or (batch_size,) squared residuals averaged
                               over the decoder horizon.  Should be detached.

        Note on mini-batch mode: when the dataloader batch size is smaller than
        num_nodes (e.g. batch_size=20, num_nodes=195), the incoming tensor has
        fewer elements than resid_sq_buffer.  In that case the batch-mean is
        broadcast to all nodes — spatial resolution is lost but temporal GARCH
        dynamics (volatility clustering and persistence) are preserved intact.
        Full spatial resolution is recovered when batch_size == num_nodes.
        """
        sq = sq_resid_per_node.to(self.resid_sq_buffer.device)
        if sq.shape[0] != self.num_nodes:
            # Mini-batch: broadcast mean residual energy to all nodes
            sq = sq.mean().expand(self.num_nodes)
        # Update EMA of squared residuals
        self.resid_sq_buffer.mul_(self.ema_resid_rate).add_(
            (1.0 - self.ema_resid_rate) * sq
        )
        # Advance lagged covariance
        if hasattr(self, '_last_G_t'):
            self.G_prev_buffer.copy_(self._last_G_t.to(self.G_prev_buffer.device))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diffusion_parameters(self) -> Dict[str, float]:
        params = super().get_diffusion_parameters()
        omega_w, alpha_w, beta_w = self._get_garch_weights()
        params.update({
            'garch_omega':           omega_w.item(),
            'garch_alpha':           alpha_w.item(),
            'garch_beta':            beta_w.item(),
            'garch_persistence':     (alpha_w + beta_w).item(),
            'garch_resid_sq_mean':   self.resid_sq_buffer.mean().item(),
            'garch_G_prev_trace':    self.G_prev_buffer.trace().item(),
        })
        return params


class GraphFactorKernelMixture(nn.Module):
    """
    Graph-driven kernel mixture for factor covariance  G_t ∈ R^{R×R}.

    Mirrors the temporal TemporalCorrelationKernel but in the latent-factor (R)
    space, using **graph diffusion kernels** instead of temporal SE kernels:

        G_t = Σ_{m=1}^M  v_{m,t} H_m,    v_{m,t} ≥ 0,  Σ_m v_{m,t} = 1,

    where {H_m} are positive-definite base kernels built from the projected
    graph Laplacian  L_R = P^T L P  ∈ R^{R×R}:

        H_m = exp(−α_m  L_R)          (diffusion kernel at scale α_m)

    P ∈ R^{N×R} is a learnable projection initialised from the leading
    eigenvectors of L (Fiedler basis).  Because R is small the matrix
    exponential is negligible.

    The resulting G_t plugs into Eq. (9'):
        Σ^{bat}_t = L^{bat}_t (C_t ⊗ G_t) (L^{bat}_t)^T + diag(d^{bat}_t)
    replacing the identity I_R that appeared in the original formulation.
    """

    def __init__(
        self,
        rank: int,
        num_nodes: int,
        num_kernels: int = 4,
        static_adj: Optional[torch.Tensor] = None,
        diffusion_scales: Optional[List[float]] = None,
        hidden_dim: int = 32,
        include_identity: bool = True,
        sigma_min: float = 1e-4,
        dynamic_weights: bool = False,
    ):
        """
        Args:
            rank:             Latent rank R  (size of G_t).
            num_nodes:        Number of spatial nodes N.
            num_kernels:      Total number of base kernels M (incl. identity if used).
            static_adj:       (N, N) static adjacency (symmetrised inside).
            diffusion_scales: Explicit list of diffusion scales [α_1,…].
                              If *None*, linearly spaced in [0.5, 4.0].
            hidden_dim:       Hidden dimension for the weight MLP (dynamic mode).
            include_identity: Whether to include I_R as the last base kernel.
            sigma_min:        Jitter added to the diagonal for PD guarantee.
            dynamic_weights:  If True, mixing weights come from a hidden-state
                              MLP; otherwise learnable static logits are used.
        """
        super().__init__()
        self.rank = rank
        self.num_nodes = num_nodes
        self.num_kernels = num_kernels
        self.include_identity = include_identity
        self.sigma_min = sigma_min
        self.dynamic_weights = dynamic_weights

        # -- learnable projection  P : N → R  --
        if static_adj is not None:
            L_graph = self._compute_normalized_laplacian(static_adj)
            # initialise P with the smallest-eigenvalue eigenvectors (Fiedler)
            _, eigvecs = torch.linalg.eigh(L_graph)
            P_init = eigvecs[:, :rank].clone()
        else:
            L_graph = torch.eye(num_nodes)
            P_init = torch.randn(num_nodes, rank) * (1.0 / math.sqrt(num_nodes))
        self.projector = nn.Parameter(P_init)
        self.register_buffer("L_graph", L_graph)

        # -- diffusion scales --
        num_se = (num_kernels - 1) if include_identity else num_kernels
        if diffusion_scales is None:
            diffusion_scales = torch.linspace(0.5, 4.0, max(num_se, 1)).tolist()
        else:
            diffusion_scales = list(diffusion_scales)[:num_se]
        self.register_buffer("_diffusion_scales", torch.tensor(diffusion_scales))

        # -- mixing-weight head --
        if dynamic_weights:
            self.weight_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_kernels),
            )
        else:
            self.mixture_logits = nn.Parameter(torch.zeros(num_kernels))

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric-normalised Laplacian from a (possibly asymmetric) adjacency."""
        W = 0.5 * (adj + adj.T)
        deg = W.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        # L_sym = I - D^{-1/2} W D^{-1/2}
        N = W.shape[0]
        D_isqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(N, device=W.device, dtype=W.dtype) - D_isqrt @ W @ D_isqrt
        return L

    def _projected_laplacian(self, L: Optional[torch.Tensor] = None) -> torch.Tensor:
        """L_R = P^T L P  ∈ R^{R×R}."""
        if L is None:
            L = self.L_graph
        P = self.projector                                     # (N, R)
        L_R = P.T @ L @ P                                     # (R, R)
        return 0.5 * (L_R + L_R.T)                            # symmetrise

    def _build_base_kernels(self, L_R: torch.Tensor) -> torch.Tensor:
        """
        Returns (M, R, R) stack of PD base kernels.
        H_m = exp(−α_m L_R) for each diffusion scale, plus optional I_R.
        """
        kernels = []
        for alpha in self._diffusion_scales:
            H_m = torch.matrix_exp(-alpha * L_R)               # (R, R)
            H_m = 0.5 * (H_m + H_m.T)                         # ensure exact symmetry
            kernels.append(H_m)
        if self.include_identity:
            kernels.append(
                torch.eye(self.rank, device=L_R.device, dtype=L_R.dtype)
            )
        return torch.stack(kernels, dim=0)                     # (M, R, R)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(
        self,
        mixture_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute  G_t = Σ_m v_{m,t} H_m  (R, R) or (B, R, R).

        Args:
            hidden_state: (B, H) — only used when *dynamic_weights=True*.
            dynamic_adj:  (N, N) or (B, N, N) — optional time-varying adjacency.
                          If *None*, the static graph is used.
        Returns:
            G_t: positive-definite factor covariance, (R, R) or (B, R, R).
        """
        
        # # ---- projected Laplacian -----------------------------------------
        # if dynamic_adj is not None and dynamic_adj.dim() == 3:
        #     # batched dynamic adjacency  →  batched base kernels
        #     B = dynamic_adj.shape[0]
        #     bk_list = []
        #     for b in range(B):
        #         L_dyn = self._compute_normalized_laplacian(dynamic_adj[b])
        #         L_R = self._projected_laplacian(L_dyn)
        #         bk_list.append(self._build_base_kernels(L_R))  # (M, R, R)
        #     base_kernels = torch.stack(bk_list, dim=0)         # (B, M, R, R)
        # else:
        #     if dynamic_adj is not None:
                # L_dyn = self._compute_normalized_laplacian(dynamic_adj)
                # L_R = self._projected_laplacian(L_dyn)
            # else:
        L_R = self._projected_laplacian()
        base_kernels = self._build_base_kernels(L_R)       # (M, R, R)

        # # ---- mixing weights ----------------------------------------------
        # if self.dynamic_weights and hidden_state is not None:
        #     weights = F.softmax(self.weight_net(hidden_state), dim=-1)  # (B, M)
        # else:
        #     weights = F.softmax(self.mixture_logits, dim=-1)            # (M,)
        weights = mixture_weights
        # ---- mix kernels -------------------------------------------------
        # weights may be (M,), (B, M), or (B, D, M); collapse to (M,)
        # because G_t must be a single (R, R) matrix shared across the batch
        # (the distribution is constructed over all batch elements jointly).
        if weights.dim() == 3:
            weights = weights.mean(dim=(0, 1))            # (B, D, M) -> (M,)
        elif weights.dim() == 2:
            weights = weights.mean(dim=0)                 # (B, M) -> (M,)
        G_t = torch.einsum("m,mij->ij", weights, base_kernels)  # (R, R)

        # ---- ensure PD via symmetry + jitter -----------------------------
        G_t = 0.5 * (G_t + G_t.transpose(-1, -2))
        eye = torch.eye(self.rank, device=G_t.device, dtype=G_t.dtype)
        if G_t.dim() == 2:
            G_t = G_t + self.sigma_min * eye
        else:
            G_t = G_t + self.sigma_min * eye.unsqueeze(0)
        return G_t


class DGLoGraPDistribution(nn.Module):
    """
    DG-LoGraP (Dynamic-Graph Low-Rank-plus-Diagonal with Grouped Latent Spatio-Temporal Processes)
    
    Implements the batch covariance:
    
    Σ^{bat}_t = Σ_g L^{(g),bat}_t (C^{(g)}_t ⊗ I_{R_g}) L^{(g),bat⊤}_t + diag(d^{bat}_t)
    
    with efficient computation via Woodbury and determinant lemma.
    """
    
    def __init__(
        self,
        num_nodes: int,
        horizon: int,
        num_groups: int,
        ranks_per_group: List[int],
        embed_dim: int,
        num_kernels: int = 4,
        static_adj: Optional[torch.Tensor] = None,
        sigma_minimum: float = 1e-3,
    ):
        """
        Args:
            num_nodes: Number of nodes N
            horizon: Temporal horizon D
            num_groups: Number of groups G
            ranks_per_group: Ranks per group [R_1, ..., R_G]
            embed_dim: Embedding dimension for SAGSAM
            num_kernels: Number of temporal kernel mixtures
            static_adj: Static adjacency matrix
            sigma_minimum: Minimum diagonal variance
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.num_groups = num_groups
        self.ranks_per_group = ranks_per_group
        self.total_rank = sum(ranks_per_group)
        self.sigma_minimum = sigma_minimum
        
        # Dynamic graph loader
        self.graph_loader = DynamicGraphLoader(
            num_nodes, num_groups, ranks_per_group, embed_dim, static_adj
        )
        
        # Temporal correlation kernels for each group
        self.temporal_kernels = nn.ModuleList([
            TemporalCorrelationKernel(
                horizon, num_kernels, hidden_dim=embed_dim
            )
            for _ in range(num_groups)
        ])
        
    def compute_batch_covariance(
        self,
        hidden_states: torch.Tensor,
        diag_variance: torch.Tensor,
        temporal_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute batch covariance matrix using Woodbury identity.
        
        Args:
            hidden_states: Hidden states (B, N, H)
            diag_variance: Diagonal variance d^{bat} (B, DN)
            temporal_hidden: Hidden state for temporal kernel (B, H)
            
        Returns:
            cov_inv: Precision matrix
            log_det: Log determinant
        """
        batch_size = hidden_states.shape[0]
        DN = self.horizon * self.num_nodes
        
        # Get loading matrices
        L_list, _ = self.graph_loader(hidden_states)
        
        # Compute temporal correlations
        C_list = [
            kernel(temporal_hidden) for kernel in self.temporal_kernels
        ]
        
        # Build block-diagonal loading matrix A = [L^{(1),bat}, ..., L^{(G),bat}]
        # L^{(g),bat} = blkdiag(L^{(g)}_{t-D+1}, ..., L^{(g)}_t)
        
        # For simplicity, assume L is constant over the horizon (can be extended)
        A_list = []
        for g, (L_g, C_g) in enumerate(zip(L_list, C_list)):
            R_g = self.ranks_per_group[g]
            
            # Build block diagonal: (DN, DR_g)
            if L_g.dim() == 2:
                L_bat = torch.block_diag(*[L_g for _ in range(self.horizon)])  # (DN, DR_g)
            else:
                # Batched version
                L_bat_list = []
                for b in range(batch_size):
                    L_bat_b = torch.block_diag(*[L_g[b] for _ in range(self.horizon)])
                    L_bat_list.append(L_bat_b)
                L_bat = torch.stack(L_bat_list, dim=0)  # (B, DN, DR_g)
            
            A_list.append(L_bat)
        
        # Concatenate all groups: A = [A_1, ..., A_G]
        if A_list[0].dim() == 2:
            A = torch.cat(A_list, dim=-1)  # (DN, Σ DR_g)
        else:
            A = torch.cat(A_list, dim=-1)  # (B, DN, Σ DR_g)
        
        # Build block-diagonal correlation C = blkdiag(C^{(1)} ⊗ I_{R_1}, ..., C^{(G)} ⊗ I_{R_G})
        C_blocks = []
        for g, C_g in enumerate(C_list):
            R_g = self.ranks_per_group[g]
            eye_R = torch.eye(R_g, device=C_g.device)
            
            if C_g.dim() == 2:
                C_kron = torch.kron(C_g, eye_R)  # (DR_g, DR_g)
            else:
                C_kron_list = []
                for b in range(batch_size):
                    C_kron_b = torch.kron(C_g[b], eye_R)
                    C_kron_list.append(C_kron_b)
                C_kron = torch.stack(C_kron_list, dim=0)  # (B, DR_g, DR_g)
            
            C_blocks.append(C_kron)
        
        if C_blocks[0].dim() == 2:
            C = torch.block_diag(*C_blocks)  # (Σ DR_g, Σ DR_g)
        else:
            C_list_batched = []
            for b in range(batch_size):
                C_b = torch.block_diag(*[c[b] for c in C_blocks])
                C_list_batched.append(C_b)
            C = torch.stack(C_list_batched, dim=0)  # (B, Σ DR_g, Σ DR_g)
        
        # Apply Woodbury identity for efficient computation
        # Σ = E + A C A^T, where E = diag(d^{bat})
        # (E + A C A^T)^{-1} = E^{-1} - E^{-1} A (C^{-1} + A^T E^{-1} A)^{-1} A^T E^{-1}
        
        E = diag_variance + self.sigma_minimum ** 2  # (B, DN)
        E_inv = 1.0 / E  # (B, DN)
        
        # Compute capacitance matrix: C^{-1} + A^T E^{-1} A
        C_inv = torch.linalg.inv(C)
        
        if A.dim() == 2:
            At_Einv = A.T * E_inv.unsqueeze(-2)  # (Σ DR_g, DN)
            capacitance = C_inv + At_Einv @ A  # (Σ DR_g, Σ DR_g)
            capacitance_chol = torch.linalg.cholesky(capacitance)
            
            # Log determinant via determinant lemma
            # log|Σ| = log|C| + log|E| + log|capacitance|
            log_det_C = torch.linalg.slogdet(C)[1]
            log_det_E = torch.log(E).sum()
            log_det_cap = 2 * torch.log(torch.diag(capacitance_chol)).sum()
            log_det = log_det_cap + log_det_E - log_det_C
        else:
            At_Einv = A.transpose(1, 2) * E_inv.unsqueeze(-2)  # (B, Σ DR_g, DN)
            capacitance = C_inv + torch.bmm(At_Einv, A)  # (B, Σ DR_g, Σ DR_g)
            capacitance_chol = torch.linalg.cholesky(capacitance)
            
            # Log determinant
            log_det_C = torch.linalg.slogdet(C)[1]  # (B,)
            log_det_E = torch.log(E).sum(dim=-1)  # (B,)
            log_det_cap = 2 * torch.log(
                capacitance_chol.diagonal(dim1=-2, dim2=-1)
            ).sum(dim=-1)  # (B,)
            log_det = log_det_cap + log_det_E - log_det_C  # (B,)
        
        return {
            'A': A,
            'C': C,
            'C_inv': C_inv,
            'E': E,
            'E_inv': E_inv,
            'capacitance_chol': capacitance_chol,
            'log_det': log_det,
        }
    
    def log_prob(
        self,
        residuals: torch.Tensor,
        hidden_states: torch.Tensor,
        diag_variance: torch.Tensor,
        temporal_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of residuals under the DG-LoGraP model.
        
        Args:
            residuals: Batch residuals η^{bat} (B, DN) or (DN,)
            hidden_states: Hidden states (B, N, H)
            diag_variance: Diagonal variance (B, DN)
            temporal_hidden: Temporal hidden state (B, H)
            
        Returns:
            log_prob: Log probability (B,) or scalar
        """
        cov_info = self.compute_batch_covariance(
            hidden_states, diag_variance, temporal_hidden
        )
        
        A = cov_info['A']
        E_inv = cov_info['E_inv']
        capacitance_chol = cov_info['capacitance_chol']
        log_det = cov_info['log_det']
        
        # Compute Mahalanobis term using Woodbury identity
        # x^T Σ^{-1} x = x^T E^{-1} x - x^T E^{-1} A cap^{-1} A^T E^{-1} x
        x = residuals
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if A.dim() == 2:
            x_Einv = x * E_inv  # (B, DN)
            mahal_term1 = (x * x_Einv).sum(dim=-1)  # (B,)
            
            At_Einv_x = torch.mv(A.T, x_Einv.squeeze(0))  # (Σ DR_g,)
            v = torch.linalg.solve_triangular(capacitance_chol, At_Einv_x, upper=False)
            mahal_term2 = (v ** 2).sum()
        else:
            x_Einv = x * E_inv  # (B, DN)
            mahal_term1 = (x * x_Einv).sum(dim=-1)  # (B,)
            
            At_Einv_x = torch.bmm(A.transpose(1, 2), x_Einv.unsqueeze(-1)).squeeze(-1)  # (B, Σ DR_g)
            v = torch.linalg.solve_triangular(capacitance_chol, At_Einv_x.unsqueeze(-1), upper=False)
            mahal_term2 = (v.squeeze(-1) ** 2).sum(dim=-1)  # (B,)
        
        mahal = mahal_term1 - mahal_term2
        
        DN = x.shape[-1]
        log_prob = -0.5 * (DN * math.log(2 * math.pi) + log_det + mahal)
        
        return log_prob
