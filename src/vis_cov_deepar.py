"""
Visualize contemporaneous and cross-covariance of one-step-ahead prediction
residuals from the stock pytorch_forecasting DeepAR model.

Reproduces a figure analogous to Figure 1 of the paper:
  Cov(η_t, η_t)  and  Cov(η_{t-Δ}, η_t) for Δ = 1, 2, 3

Usage (from repo root):
    python ./src/vis_cov_deepar.py \
        --checkpoint logs/pf_deepar/pf_deepar_pems03_flow_B20_Q12_H40_lr0.01/checkpoints/epoch=46-val_loss=86.02.ckpt \
        --dataset pems03_flow \
        --n_samples 500 \
        --out visualizations/pf_deepar_cov.pdf
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.models.deepar import DeepAR
from batched_model import BatchxLSTMEstimator

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────── CLI ────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/pf_deepar/pf_deepar_pems03_flow_B20_Q12_H40_lr0.01/checkpoints/epoch=46-val_loss=86.02.ckpt",
)
parser.add_argument("--dataset",   type=str, default="pems03_flow")
parser.add_argument("--device",    type=int, default=0)
parser.add_argument("--batch_size",type=int, default=20)
parser.add_argument("--n_samples", type=int, default=500,
                    help="MC samples to draw for empirical covariance estimation")
parser.add_argument("--max_lags",  type=int, default=3,
                    help="Number of lagged cross-covariance panels (Δ=1..max_lags)")
parser.add_argument("--clim",      type=float, default=0.6,
                    help="Colour-axis limit (covariances clipped to [-clim, clim])")
parser.add_argument("--out", type=str,
                    default="visualizations/pf_deepar_cov.pdf")
args = parser.parse_args()

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

# ────────────────────────────── dataset metadata ────────────────────────────
with open("./datasets/pred_horizon_dict.pkl", "rb") as f:
    pred_horizon_dict = pickle.load(f)
with open("./datasets/pred_rolling_dict.pkl", "rb") as f:
    pred_rolling_dict = pickle.load(f)
with open("./datasets/dataset_freq.pkl", "rb") as f:
    dataset_freq_dict = pickle.load(f)

default_rolling = {"B": 5, "30min": 56, "M": 1, "W": 3,
                   "5min": 56, "D": 5, "Q": 1, "H": 7, "Y": 1}
freq = dataset_freq_dict[args.dataset]
pred_horizon  = pred_horizon_dict[args.dataset]
num_rolling   = (pred_rolling_dict[args.dataset]
                 if args.dataset in pred_rolling_dict
                 else default_rolling[freq])

with open("./config/config.yaml") as f:
    configs = yaml.load(f, Loader=yaml.Loader)

# ────────────────────────────── build dataset ───────────────────────────────
data = pd.read_csv(f"./datasets/{args.dataset}.csv")

if freq in ["30min", "5min", "H", "T"]:
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["tod"] = (
        (data["datetime"].values - data["datetime"].values.astype("datetime64[D]"))
        / np.timedelta64(1, "D")
    )
    data["dow"] = data["datetime"].dt.weekday
    tcat = ["tod", "dow"]
    data = data.astype(dict(sensor=str, tod=str, dow=str))
    lags = {"value": [24, 168]} if freq == "H" else {"value": [2, 4, 12, 24, 48]}
elif freq in ["B", "D"]:
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["dow"] = data["datetime"].dt.weekday
    tcat = ["dow"]
    data = data.astype(dict(sensor=str, dow=str))
    lags = {"value": [7, 14]}
else:
    tcat = []
    data = data.astype(dict(sensor=str))
    lags = {}

validation_cutoff = data["time_idx"].max() - pred_horizon - num_rolling + 1
training_cutoff   = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    target_normalizer=GroupNormalizer(groups=["sensor"], transformation=None),
    categorical_encoders={"sensor": NaNLabelEncoder().fit(data.sensor)},
    group_ids=["sensor"],
    static_categoricals=["sensor"],
    time_varying_known_categoricals=tcat,
    time_varying_unknown_reals=["value"],
    lags=lags,
    min_encoder_length=pred_horizon,
    max_encoder_length=pred_horizon,
    min_prediction_length=pred_horizon,
    max_prediction_length=pred_horizon,
    allow_missing_timesteps=False,
)
testing = TimeSeriesDataSet.from_dataset(
    training, data, min_prediction_idx=validation_cutoff + 1
)
test_loader = testing.to_dataloader(
    train=False, batch_size=args.batch_size, num_workers=0,
    batch_sampler="synchronized",
)

# ────────────────────────────── load model ──────────────────────────────────
print(f"Loading checkpoint: {args.checkpoint}")
model = BatchxLSTMEstimator.load_from_checkpoint(args.checkpoint, map_location=device).to(device)
model.eval()

# ────────────────────────────── predict ─────────────────────────────────────
print(f"Running prediction with {args.n_samples} MC samples …")
with torch.no_grad():
    raw_predictions, x = model.predict(
        test_loader,
        mode="raw",
        n_samples=args.n_samples,
        show_progress_bar=True,
        return_x=True,
    )

# preds:   (B_total, Q, n_samples)
# actuals: (B_total, Q)
preds   = raw_predictions["prediction"].cpu().float()   # (B_total, Q, S)
actuals = x["decoder_target"].cpu().float()             # (B_total, Q)


def temporal_rolling_info():
    # Reshape: (num_rolling, N_sensors, Q, S) and (num_rolling, N_sensors, Q)
    try:
        N = preds.shape[0] // num_rolling
        preds   = preds.reshape(num_rolling, N, preds.shape[1], preds.shape[2])
        actuals = actuals.reshape(num_rolling, N, actuals.shape[1])
    except Exception:
        N = preds.shape[0]
        preds   = preds.unsqueeze(0)
        actuals = actuals.unsqueeze(0)
        num_rolling = 1

    print(f"  num_rolling={num_rolling}, N_sensors={N}, Q={preds.shape[2]}, S={preds.shape[3]}")

    # ──────────────────── compute residuals η_{i,t,s} ───────────────────────────
    # η shape: (num_rolling, N, Q, S)
    # residual = sample - actual (deviation of each MC draw from the observation)
    eta = preds - actuals.unsqueeze(-1)          # (R, N, Q, S)

    # ─────────────────── empirical covariance helpers ────────────────────────────
    def emp_cov(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute N×N empirical cross-covariance averaged over rolling windows and
        timesteps.

        A, B : (num_rolling, N, T, S)   — residuals at two (possibly different) lags
        Returns: (N, N)
        """
        # Centre within each (rolling, timestep) context across samples
        A_c = A - A.mean(-1, keepdim=True)   # (R, N, T, S)
        B_c = B - B.mean(-1, keepdim=True)

        # (R, N, T, S) → (R, T, N, S)
        A_c = A_c.permute(0, 2, 1, 3)
        B_c = B_c.permute(0, 2, 1, 3)

        S = A_c.shape[-1]
        # cov per (rolling, timestep): (R, T, N, N)
        cov = torch.einsum("rtnS,rtmS->rtnm", A_c, B_c) / (S - 1)

        # Average over rolling windows and timesteps that exist for both lags
        return cov.mean(dim=(0, 1))   # (N, N)


    # ─────── build covariance matrices ──────────────────────────────────────────
    Q = preds.shape[2]

    # Contemporaneous Cov(η_t, η_t)  — use all Q timesteps
    cov_0 = emp_cov(eta, eta).numpy()

    # Lagged  Cov(η_{t-Δ}, η_t)  — shift along Q dimension
    lagged = []
    for lag in range(1, args.max_lags + 1):
        # η at t-Δ:  timesteps [0 .. Q-lag-1]
        # η at t  :  timesteps [lag .. Q-1]
        if lag >= Q:
            lagged.append(np.zeros((N, N)))
            continue
        A_lag = eta[:, :, :Q - lag, :]    # (R, N, Q-lag, S)
        B_cur = eta[:, :, lag:,     :]    # (R, N, Q-lag, S)
        lagged.append(emp_cov(A_lag, B_cur).numpy())

    # ─────────────────────────── plotting ────────────────────────────────────────
    n_panels = 1 + args.max_lags
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.2))

    cmap = "YlOrRd"   # warm palette matching the paper figure

    def _plot_panel(ax, mat, title):
        mat_clipped = np.clip(mat, 0, args.clim)          # paper clips to [0, clim]
        im = ax.imshow(
            mat_clipped,
            cmap=cmap,
            vmin=0,
            vmax=args.clim,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _plot_panel(axes[0], cov_0, r"$\mathrm{Cov}(\mathbf{\eta}_t,\, \mathbf{\eta}_t)$")
    for i, (lag, mat) in enumerate(zip(range(1, args.max_lags + 1), lagged), start=1):
        _plot_panel(
            axes[i],
            mat,
            r"$\mathrm{Cov}(\mathbf{\eta}_{t-%d},\, \mathbf{\eta}_t)$" % lag,
        )

    fig.suptitle(
        f"Prediction residual covariance — DeepAR (PF) on {args.dataset}\n"
        f"({args.n_samples} MC samples, covariances clipped to [0, {args.clim}])",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to: {args.out}")

    # Also save the raw matrices for later analysis
    out_stem = os.path.splitext(args.out)[0]
    np.save(f"{out_stem}_cov0.npy", cov_0)
    for lag, mat in zip(range(1, args.max_lags + 1), lagged):
        np.save(f"{out_stem}_cov_lag{lag}.npy", mat)
    print("Raw covariance .npy arrays saved alongside the figure.")
