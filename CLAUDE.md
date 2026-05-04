# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch/PyTorch Lightning implementation of **"Multivariate Probabilistic Time Series Forecasting with Correlated Errors"** (NeurIPS 2024). Extends DeepAR, Transformer, and xLSTM backbones with multivariate Gaussian distributions and learned covariance structures (including spatial/graph-aware variants).

## Setup

```bash
pip install -r requirements.txt  # Python 3.10+, PyTorch 1.13.1+cu117
```

Datasets are generated via the Jupyter notebook:
```bash
jupyter notebook exps/generate_datasets.ipynb
```

## Common Commands

### Training the proposed model (kernel-based correlated errors)
```bash
python src/train_batch.py --model deepar --dataset exchange_rate_nips --loss kernel --num_mixture_r 4 --device 0
```

### Training with different loss functions
```bash
# --loss options: kernel, ar, graph_kernel, curvature_kernel, learnable_kernel, diffusion_kernel
python src/train_batch.py --model deepar --dataset pems03_flow --loss graph_kernel --num_mixture_r 4 --batch_cov_horizon 12
```

### Training a baseline (standard multivariate normal)
```bash
python src/train_baseline.py --model deepar --dataset exchange_rate_nips --batch_size 20 --hidden_size 40 --lr 1e-3
```

### DeepVAR baseline (CRPS benchmarking)
```bash
python src/deep_var.py --dataset exchange_rate_nips --device 0 --num_repeat 5 --n_samples 100
```

### DG-LoGraP (spatial graph integration)
```bash
python src/train_dg_lograp.py --dataset pems03_flow --model deepar --num_nodes 358 --num_groups 2 --rank_per_group 5 --num_kernels 4 --graph_embed_dim 32 --batch_cov_horizon 12 --hidden_size 64 --lr 1e-3 --device 0
```

### Tests
```bash
python src/test_dg_lograp.py   # DG-LoGraP component tests
python test_xlstm_init.py      # xLSTM initialization test
```

## Architecture

### Data Flow
```
Multivariate Time Series
  → Encoder (DeepAR LSTM / ARTransformer / xLSTMBackbone)
  → Hidden state projected to: mean, scale, low-rank factors (mixture components)
  → Loss function constructs covariance matrix
  → Negative log-likelihood of MultivariateNormal
```

### Key Source Files

- **[src/train_batch.py](src/train_batch.py)** — Main entry point for the proposed model. Loads config, builds dataset, calls `BatchedEstimator.fit()`.
- **[src/train_baseline.py](src/train_baseline.py)** — Baseline training with standard `MultivariateNormalDistributionLoss`.
- **[src/batched_model.py](src/batched_model.py)** — `BatchedEstimator`: extends `AutoRegressiveBaseModelWithCovariates` from pytorch-forecasting. Adds mixture projector heads and plugs in custom loss functions.
- **[src/model.py](src/model.py)** — Backbone model classes: extended `DeepAR`, `ARTransformer`, `xLSTMBackbone`.
- **[src/loss.py](src/loss.py)** — All loss implementations: `BatchMGD_Kernel`, `BatchMGDGraph_Kernel`, `BatchMGDCurvature_Kernel`, `BatchMGDLearnable_Kernel`, `BatchMGDDiffusion_Kernel`, `BatchMGD_AR`, `BatchMGD_Toeplitz`.
- **[src/dynamic_graph.py](src/dynamic_graph.py)** — Spatial graph components: `SAGSAM` (dynamic adjacency), `SAGS_GCN`, `DGLoGraPDistribution`, `GraphFactorKernelMixture`, `TemporalCorrelationKernel`.
- **[src/general_lr_multivariate_normal.py](src/general_lr_multivariate_normal.py)** — Custom `GeneralLowRankMultivariateNormal` using Woodbury identity and matrix determinant lemma for efficient low-rank covariance.
- **[src/metrics.py](src/metrics.py)** — Evaluation: `eval_crps_mean()`, `eval_crps_sum()`, `eval_rrmse()`, `eval_energy_score()`.

### Covariance Parameterization

Each loss function constructs a covariance matrix differently:
- **kernel** — Kernel mixture: `Σ = Σ_i w_i * (C_i ⊗ I_R)` where `C_i` are temporal kernel matrices
- **ar** — AR(1) structured covariance via recurrence
- **graph_kernel** — Spatial-aware: `Σ = Σ_i w_i * (C_i ⊗ G_i)` where `G_i` uses graph Laplacian
- **curvature_kernel** — Adds curvature regularization to precision matrix
- **learnable_kernel** — End-to-end learned kernel parameters
- **diffusion_kernel** — Graph diffusion process for spatial covariance
- **DG-LoGraP** — Full dynamic graph model with `SAGSAM` → `SAGS-GCN` → `GraphFactorKernelMixture`

### Configuration

[config/config.yaml](config/config.yaml) stores per-dataset hyperparameters (prediction horizon `H`, learning rate `lr`, hidden size, etc.) organized by model type (`deepar`, `gpt`, `xLSTM`). `train_batch.py` reads these at startup and CLI args override them.

### Outputs

- **Checkpoints**: `logs/<model_name>/` (pytorch-lightning format)
- **Metrics**: `metrics/` directory (JSON/CSV per run)
- **TensorBoard**: `lightning_logs/`
