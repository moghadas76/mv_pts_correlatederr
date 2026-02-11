# DG-LoGraP: Dynamic-Graph Low-Rank-plus-Diagonal with Grouped Latent Spatio-Temporal Processes

Implementation of the DG-LoGraP methodology for spatio-temporal forecasting with dynamic graph-aware covariance modeling.

## Overview

This implementation combines:

1. **SAGSAM (Semi-autonomous Generation Spatial Adjacency Matrix)**: Dynamically generates weighted adjacency matrices based on learnable node embeddings and a predefined static adjacency matrix.

2. **SAGS-GCN (Semi-autonomous Generative Spatial Graph Convolution Network)**: Graph convolution layer that uses SAGSAM for spatial message passing with node-specific weights.

3. **DG-LoGraP**: Grouped latent spatio-temporal residual model that captures:
   - Contemporaneous covariance across nodes (low-rank-plus-diagonal)
   - Spatio-temporal cross-covariance via grouped latent processes
   - Dynamic-graph-aware parameterization of loading matrices

## Mathematical Formulation

### Residual Decomposition
$$\eta_t = \sum_{g=1}^{G} L^{(g)}_t r^{(g)}_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \text{diag}(d_t))$$

### SAGSAM Dynamic Adjacency
$$\tilde{A} = \text{softmax}(\text{ReLU}(E_A \cdot E_A^T)) \cdot A$$

where $E_A \in \mathbb{R}^{N \times d}$ is the learnable node embedding dictionary.

### Dynamic-Graph-Aware Loading Matrices
$$L^{(g)}_t = U^{(g)}_t S^{(g)}_t$$

where $U^{(g)}_t$ comes from the leading eigenvectors of the dynamic covariance graph, and $S^{(g)}_t$ is a learnable scaling matrix.

### Temporal Correlation (Kernel Mixture)
$$C^{(g)}_t = \sum_{m=1}^{M} w^{(g)}_{m,t} K_m$$

where $K_m$ are fixed kernel matrices and $w^{(g)}_{m,t}$ are softmax weights.

### Batch Covariance
$$\Sigma^{\text{bat}}_t = \sum_{g=1}^{G} L^{(g),\text{bat}}_t (C^{(g)}_t \otimes I_{R_g}) L^{(g),\text{bat}\top}_t + \text{diag}(d^{\text{bat}}_t)$$

Efficient computation via Woodbury identity and determinant lemma.

## File Structure

```
src/
├── dynamic_graph.py        # SAGSAM, SAGS-GCN, and core DG-LoGraP modules
├── dg_lograp_model.py      # DeepAR and Transformer models with DG-LoGraP
├── loss.py                 # DGLoGraP_Loss class (extended)
├── train_dg_lograp.py      # Training script for PEMS03 dataset
└── ...
```

## Components

### 1. `dynamic_graph.py`

- **`load_static_graph`**: Loads static graph from CSV and converts to normalized adjacency matrix
- **`SAGSAM`**: Semi-autonomous generation spatial adjacency matrix module
- **`SAGS_GCN`**: Graph convolution with SAGSAM and node-specific weights
- **`DynamicGraphLoader`**: Computes loading matrices from dynamic graph eigenbasis
- **`TemporalCorrelationKernel`**: Dynamic kernel-mixture for temporal correlation
- **`DGLoGraPDistribution`**: Full DG-LoGraP distribution with Woodbury computation

### 2. `dg_lograp_model.py`

- **`DGLoGraPDeepAR`**: DeepAR variant with SAGSAM, SAGS-GCN, and DG-LoGraP
- **`DGLoGraPTransformer`**: Transformer variant with DG-LoGraP components

### 3. `loss.py` (extended)

- **`DGLoGraP_Loss`**: Loss function implementing DG-LoGraP batch covariance

## Usage

### Training on PEMS03

```bash
cd src
python train_dg_lograp.py \
    --dataset pems03_flow \
    --model deepar \
    --num_nodes 358 \
    --num_groups 2 \
    --rank_per_group 5 \
    --num_kernels 4 \
    --graph_embed_dim 32 \
    --batch_cov_horizon 12 \
    --hidden_size 64 \
    --lr 1e-3 \
    --device 0
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_nodes` | Number of spatial nodes | 358 |
| `--num_groups` | Number of latent groups G | 2 |
| `--rank_per_group` | Rank per group R_g | 5 |
| `--num_kernels` | Number of temporal kernel mixtures | 4 |
| `--graph_embed_dim` | SAGSAM embedding dimension | 32 |
| `--batch_cov_horizon` | Temporal horizon D | 12 |
| `--use_graph_conv` | Use SAGS-GCN | True |

## PEMS03 Dataset

The PEMS03 dataset contains traffic flow data from 358 sensors in California.

- **Graph**: `datasets/PEMS03_graph.csv` (sensor connectivity with distances)
- **Data**: `datasets/pems03_flow.csv` (5-minute interval traffic flow)

## References

1. SAGSAM from: [Traffic Flow Forecasting with SAGSAM](https://arxiv.org/pdf/2205.01480)

2. DG-LoGraP methodology from the paper draft for grouped latent spatio-temporal residual models

## Example

```python
from dynamic_graph import SAGSAM, SAGS_GCN, load_static_graph
import torch

# Load static graph
static_adj = load_static_graph('datasets/PEMS03_graph.csv', num_nodes=358)

# Create SAGSAM
sagsam = SAGSAM(num_nodes=358, embed_dim=32, static_adj=static_adj)

# Get dynamic adjacency
dynamic_adj = sagsam()  # (358, 358)

# Create SAGS-GCN
sags_gcn = SAGS_GCN(
    num_nodes=358,
    in_features=64,
    out_features=64,
    embed_dim=32,
    static_adj=static_adj
)

# Apply graph convolution
x = torch.randn(8, 358, 64)  # (batch, nodes, features)
z, adj = sags_gcn(x)  # z: (8, 358, 64)
```
