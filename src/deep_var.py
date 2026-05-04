"""
Train and evaluate pytorch_forecasting's built-in DeepAR on CRPS_mean and CRPS_sum.

Usage (run from repo root):
    python ./src/deep_var.py --dataset pems03_flow
    python ./src/deep_var.py --dataset exchange_rate_nips --device 0
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.metrics.distributions import MultivariateNormalDistributionLoss

sys.path.insert(0, os.path.dirname(__file__))
from metrics import get_metrics

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Stock pytorch_forecasting DeepAR — CRPS_mean / CRPS_sum evaluation"
)
parser.add_argument("--device",      type=int,   default=0)
parser.add_argument("--dataset",     type=str,   default="exchange_rate_nips")
parser.add_argument("--batch_size",  type=int,   default=20)
parser.add_argument("--hidden_size", type=int,   default=None,
                    help="Overrides config.yaml when set.")
parser.add_argument("--lr",          type=float, default=None,
                    help="Overrides config.yaml when set.")
parser.add_argument("--num_repeat",  type=int,   default=5,
                    help="Independent prediction runs used for mean/std reporting.")
parser.add_argument("--n_samples",   type=int,   default=100,
                    help="MC samples drawn per prediction call.")
parser.add_argument("--seed",        type=int,   default=42)
args = parser.parse_args()

pl.seed_everything(args.seed)

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
with open("./datasets/pred_horizon_dict.pkl", "rb") as f:
    pred_horizon_dict = pickle.load(f)
with open("./datasets/pred_rolling_dict.pkl", "rb") as f:
    pred_rolling_dict = pickle.load(f)
with open("./datasets/dataset_freq.pkl", "rb") as f:
    dataset_freq_dict = pickle.load(f)

default_rolling = {"B": 5, "30min": 56, "M": 1, "W": 3,
                   "5min": 56, "D": 5, "Q": 1, "H": 7, "Y": 1}

args.prediction_horizon = pred_horizon_dict[args.dataset]
args.num_pred_rolling = (
    pred_rolling_dict[args.dataset]
    if args.dataset in pred_rolling_dict
    else default_rolling[dataset_freq_dict[args.dataset]]
)

# Load YAML config
with open("./config/config.yaml") as f:
    configs = yaml.load(f, Loader=yaml.Loader)

_deepar_cfg = configs.get("deepar", {}).get("dataset", {}).get(args.dataset, {})
hidden_size    = args.hidden_size if args.hidden_size is not None else int(_deepar_cfg.get("H", 40))
learning_rate  = args.lr         if args.lr         is not None else float(_deepar_cfg.get("lr", 1e-3))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_datasets(data: pd.DataFrame):
    """Return (training, validation, testing) TimeSeriesDataSet objects."""
    freq = dataset_freq_dict[args.dataset]

    if freq in ["30min", "5min", "H", "T"]:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["tod"] = (
            (data["datetime"].values - data["datetime"].values.astype("datetime64[D]"))
            / np.timedelta64(1, "D")
        )
        data["dow"] = data["datetime"].dt.weekday
        time_varying_known_cats = ["tod", "dow"]
        data = data.astype(dict(sensor=str, tod=str, dow=str))
        lags = {"value": [24, 168]} if freq == "H" else {"value": [2, 4, 12, 24, 48]}
    elif freq in ["B", "D"]:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["dow"] = data["datetime"].dt.weekday
        time_varying_known_cats = ["dow"]
        data = data.astype(dict(sensor=str, dow=str))
        lags = {"value": [7, 14]}
    else:
        time_varying_known_cats = []
        data = data.astype(dict(sensor=str))
        lags = {}

    validation_cutoff = (
        data["time_idx"].max() - args.prediction_horizon - args.num_pred_rolling + 1
    )
    training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        target_normalizer=GroupNormalizer(groups=["sensor"], transformation=None),
        categorical_encoders={"sensor": NaNLabelEncoder().fit(data.sensor)},
        group_ids=["sensor"],
        static_categoricals=["sensor"],
        time_varying_known_categoricals=time_varying_known_cats,
        time_varying_unknown_reals=["value"],
        lags=lags,
        min_encoder_length=args.prediction_horizon,
        max_encoder_length=args.prediction_horizon,
        min_prediction_length=args.prediction_horizon,
        max_prediction_length=args.prediction_horizon,
        allow_missing_timesteps=False,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data[lambda x: x.time_idx <= validation_cutoff],
        min_prediction_idx=training_cutoff + 1,
    )
    testing = TimeSeriesDataSet.from_dataset(
        training, data, min_prediction_idx=validation_cutoff + 1
    )
    return training, validation, testing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------ Data
    data = pd.read_csv("./datasets/%s.csv" % args.dataset)
    training, validation, testing = build_datasets(data)

    train_dataloader = training.to_dataloader(
        train=True, batch_size=args.batch_size, num_workers=0,
        batch_sampler="synchronized",
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0,
        batch_sampler="synchronized",
    )
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0,
        batch_sampler="synchronized",
    )

    # --------------------------------------------------------------- Logging
    log_file = "pf_deepar_%s_B%s_Q%s_H%s_lr%s" % (
        args.dataset, args.batch_size, args.prediction_horizon,
        hidden_size, learning_rate,
    )
    logger = TensorBoardLogger(save_dir="logs", name="pf_deepar", version=log_file)

    # ------------------------------------------------------ Model (stock DeepAR)
    net = DeepAR.from_dataset(
        training,
        cell_type=configs["deepar"].get("cell_type", "LSTM"),
        hidden_size=hidden_size,
        rnn_layers=configs["deepar"].get("rnn_layers", 2),
        dropout=configs["train"].get("dropout", 0.01),
        loss=MultivariateNormalDistributionLoss(),
        optimizer="adam",
        learning_rate=learning_rate,
        weight_decay=float(configs["train"].get("weight_decay", 1e-8)),
        reduce_on_plateau_patience=configs["train"].get("reduce_on_plateau_patience", 500),
    )
#     print(net.summarize())

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        logger=logger,
        max_steps=configs["train"]["max_steps"],
        accelerator="gpu",
        devices=[args.device],
        enable_model_summary=True,
        gradient_clip_val=configs["train"]["gradient_clip_val"],
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_train_batches=configs["train"]["limit_train_batches"],
        enable_checkpointing=True,
        accumulate_grad_batches=16,
    )

    trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # ----------------------------------------------------------------
    # Evaluation — load best checkpoint, repeat num_repeat times
    # ----------------------------------------------------------------
    device_str = "cuda:%d" % args.device
    best_model = DeepAR.load_from_checkpoint(
        checkpoint_callback.best_model_path, map_location=device_str
    ).to(device_str)
    best_model.eval()

    all_metrics = []
    crps_mean_noagg_all, crps_noagg_all, crps_sum_noagg_all = [], [], []

    for repeat in range(args.num_repeat):
        raw_predictions, x = best_model.predict(
            test_dataloader,
            mode="raw",
            n_samples=args.n_samples,
            show_progress_bar=True,
            return_x=True,
        )
        # raw_predictions["prediction"]: (batch, horizon, n_samples)
        # x["decoder_target"]:           (batch, horizon)
        preds   = raw_predictions["prediction"].cpu()   # (B, Q, S)
        actuals = x["decoder_target"].cpu()             # (B, Q)

        # Reshape to (num_pred_rolling, N_sensors, Q, n_samples) / (n_frcs, N, Q)
        try:
            preds = preds.reshape(
                args.num_pred_rolling,
                preds.shape[0] // args.num_pred_rolling,
                preds.shape[1],
                preds.shape[2],
            )
            actuals = actuals.reshape(
                args.num_pred_rolling,
                actuals.shape[0] // args.num_pred_rolling,
                actuals.shape[1],
            )
        except Exception:
            preds   = preds.unsqueeze(0)
            actuals = actuals.unsqueeze(0)

        agg_metric, crps_mean_noagg, crps_noagg, crps_sum_noagg = get_metrics(preds, actuals)
        crps_mean_noagg_all.append(crps_mean_noagg)
        crps_noagg_all.append(crps_noagg)
        crps_sum_noagg_all.append(crps_sum_noagg)
        all_metrics.append(agg_metric)

        print(
            f"[repeat {repeat+1}/{args.num_repeat}] "
            f"CRPS_mean={agg_metric[0]:.4f}  CRPS_sum={agg_metric[2]:.4f}"
        )

    # ----------------------------------------------------------------
    # Aggregate over repeats and save
    # ----------------------------------------------------------------
    # metric order from get_metrics: crps_mean, crps, crps_sum, p05_risk, p09_risk, energy_score, rrmse
    metric_names = ["crps_mean", "crps", "crps_sum", "p05_risk", "p09_risk", "energy_score", "rrmse"]

    metrics_arr     = np.array(all_metrics)                                        # (n_repeat, 7)
    metrics_summary = np.concatenate(
        [metrics_arr.mean(0).reshape(-1, 1), metrics_arr.std(0).reshape(-1, 1)],
        axis=1,
    )  # (7, 2)

    print("\n===== Final Evaluation (mean ± std over %d runs) =====" % args.num_repeat)
    for name, row in zip(metric_names, metrics_summary):
        print(f"  {name:<18s}: {row[0]:.4f} ± {row[1]:.4f}")

    # Save raw per-window tensors
    raw_dir = "./metrics_raw/pf_deepar"
    os.makedirs(raw_dir, exist_ok=True)
    torch.save(torch.stack(crps_mean_noagg_all), f"{raw_dir}/{log_file}_crps_mean.pt")
    torch.save(torch.stack(crps_noagg_all),      f"{raw_dir}/{log_file}_crps.pt")
    torch.save(torch.stack(crps_sum_noagg_all),  f"{raw_dir}/{log_file}_crps_sum.pt")

    # Save human-readable summary
    summary_dir = "./metrics/pf_deepar"
    os.makedirs(summary_dir, exist_ok=True)
    with open(f"{summary_dir}/{log_file}.txt", "w") as f:
        for name, row in zip(metric_names, metrics_summary):
            f.write(f"{name}: {row[0]:.4f} ± {row[1]:.4f}\n")
        # LaTeX-friendly line with CRPS_mean and CRPS_sum
        f.write(
            "\nLaTeX: & %.4f$\\pm$%.4f & %.4f$\\pm$%.4f \n" % (
                metrics_summary[0, 0], metrics_summary[0, 1],
                metrics_summary[2, 0], metrics_summary[2, 1],
            )
        )

    print(f"\nResults saved to {summary_dir}/{log_file}.txt")
    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    main()

