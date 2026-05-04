# TEGER: Temporal Graph-based Error Correction for Multivariate Time-Series Forecasting

![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 1.13](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic)

> Anonymous code release accompanying a NeurIPS submission. Do not redistribute.

PyTorch implementation of **TEGER**, a backbone-agnostic uncertainty module for probabilistic multivariate time-series forecasting that jointly models temporal *and* spatial residual correlations on dynamic graphs.

TEGER parameterizes the joint residual covariance over a horizon $D$ as

$$\Sigma_t^{\mathrm{bat}} = \mathbf{L}_t^{\mathrm{bat}}(\mathbf{C}_t \otimes \mathbf{G}_t)(\mathbf{L}_t^{\mathrm{bat}})^\top + \mathrm{diag}(\mathbf{d}_t),$$

where $\mathbf{C}_t$ is a dynamic temporal kernel mixture and $\mathbf{G}_t$ is a graph-structured spatial factor covariance. We provide two variants:

- **kTEGER** — graph-diffusion kernel for spatial error propagation.
- **cTEGER** — curvature-aware rewiring that strengthens edges with highly negative Balanced Forman curvature ($\kappa(i,j) \le -2 + \delta$) to mitigate oversquashing. We prove this rewiring monotonically increases the Fiedler value, reduces effective resistance, and tightens $\mathrm{CRPS}_{\mathrm{sum}}$ calibration bounds.

The low-rank-plus-diagonal structure preserves $\mathcal{O}(NR^2)$ inference via the Woodbury identity, and the module plugs into any autoregressive backbone (LSTM, Transformer, xLSTM).

## Requirements

Python 3.10+, PyTorch 1.13.1+cu117.

```bash
pip install -r requirements.txt
```

## Datasets

Generate the datasets (PeMS03, PeMS04, PeMS07, Brussels, and the GluonTS benchmarks) by running the notebook:

```bash
jupyter notebook exps/generate_datasets.ipynb
```

## Training

### kTEGER (graph-diffusion kernel)
```bash
python src/train_batch.py --model deepar --dataset pems03_flow \
    --loss kernel --num_mixture_r 4 --batch_cov_horizon 12 --device 0
```

### cTEGER (curvature-aware rewiring)
```bash
python src/train_batch.py --model deepar --dataset pems03_flow \
    --loss curvature --num_mixture_r 4 --batch_cov_horizon 12 --device 0
```

### Baselines
```bash
# Diagonal Gaussian backbone
python src/train_baseline.py --model deepar --dataset pems03_flow

# Temporal-only correlated-error baseline
python src/train_batch.py --model deepar --dataset pems03_flow --loss ar

# DeepVAR (CRPS benchmark)
python src/deep_var.py --dataset pems03_flow --num_repeat 5 --n_samples 100
```

Backbones are swapped via `--model {deepar, gpt, xLSTM}`. Per-dataset hyperparameters live in [config/config.yaml](config/config.yaml) and are overridden by CLI flags.

## Repository Layout

| File | Role |
|---|---|
| [src/train_batch.py](src/train_batch.py) | Main training entry point for TEGER variants |
| [src/batched_model.py](src/batched_model.py) | `BatchedEstimator` with mixture projector heads |
| [src/model.py](src/model.py) | DeepAR / Transformer / xLSTM backbones |
| [src/loss.py](src/loss.py) | Kernel, curvature, AR, learnable, diffusion losses |
| [src/dynamic_graph.py](src/dynamic_graph.py) | SAGSAM, GraphFactorKernelMixture, DG-LoGraP |
| [src/general_lr_multivariate_normal.py](src/general_lr_multivariate_normal.py) | Woodbury-based low-rank multivariate normal |
| [src/metrics.py](src/metrics.py) | CRPS, RRMSE, energy score evaluation |

Outputs: checkpoints in `logs/<model>/`, metrics in `metrics/`, TensorBoard logs in `lightning_logs/`.

## Reference

```bibtex
@inproceedings{teger_anonymous,
  title     = {TEGER: Temporal Graph-based Error Correction for Multivariate Time-Series Forecasting},
  author    = {Anonymous},
  booktitle = {Submitted to NeurIPS},
  year      = {2026},
  note      = {Under review. Author information withheld for double-blind review.}
}
```
