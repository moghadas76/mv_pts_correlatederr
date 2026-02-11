"""
Dynamic Graph Module with SAGSAM (Semi-autonomous Generation Spatial Adjacency Matrix)

Implements the DG-LoGraP methodology for dynamic-graph-aware spatio-temporal covariance modeling.

Reference:
- SAGSAM from https://arxiv.org/pdf/2205.01480
- DG-LoGraP methodology for grouped latent spatio-temporal residual models
"""

import math
from typing import Optional, Tuple, List

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
