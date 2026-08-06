"""
Rebuttal Priority 1 (ablation ladder), row 1: "Backbone + diagonal Gaussian".

The true no-covariance baseline: the xLSTM backbone trained with
pytorch-forecasting's stock `NormalDistributionLoss` (independent Gaussian
per sensor per step), bypassing the BatchedEstimator / BatchCovLoss machinery
entirely. Data pipeline mirrors train_batch.py's Brussels branch exactly
(same tod/dow covariates, lags, rolling-window split) so CRPS is comparable
to rows 2-9.
"""
import argparse
import os
import pickle

import numpy as np
if not hasattr(np, "float"):
    # pytorch_forecasting's GroupNormalizer still references the deprecated
    # np.float alias (removed in numpy>=1.20); restore it in-process only.
    np.float = float
import pandas as pd
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import NormalDistributionLoss

from metrics import get_metrics
from model import xLSTMBackbone

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default="brussels")
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

pl.seed_everything(args.seed)

with open('./datasets/pred_horizon_dict_v1.pkl', 'rb') as f:
    pred_horizon_dict = pickle.load(f)
with open('./datasets/pred_rolling_dict_v1.pkl', 'rb') as f:
    pred_rolling_dict = pickle.load(f)
with open('./datasets/dataset_freq_v1.pkl', 'rb') as f:
    dataset_freq_dict = pickle.load(f)

args.prediction_horizon = pred_horizon_dict[args.dataset]
args.num_pred_rolling = pred_rolling_dict[args.dataset]

f = open("./config/config.yaml")
configs = yaml.load(f, Loader=yaml.Loader)
f.close()
args.hidden_size = configs['xLSTM']['dataset'][args.dataset]['H']
args.lr = float(configs['xLSTM']['dataset'][args.dataset]['lr'])

log_file = "%s_row1_diagonal_B%s_Q%s_H%s" % (
    args.dataset, args.batch_size, args.prediction_horizon, args.hidden_size
)


def main():
    data = pd.read_csv("./datasets/%s.csv" % (args.dataset))
    if dataset_freq_dict[args.dataset] in ['30min', '5min', 'H', 'T']:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["tod"] = (data["datetime"].values - data["datetime"].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        data['dow'] = data['datetime'].dt.weekday
        time_varying_known_cats = ['tod', 'dow']
        data = data.astype(dict(sensor=str, tod=str, dow=str))
        lags = {"value": [2, 4, 12, 24, 48]}
    else:
        time_varying_known_cats = []
        data = data.astype(dict(sensor=str))
        lags = {}

    validation_cutoff = data["time_idx"].max() - args.prediction_horizon - args.num_pred_rolling + 1
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
    validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.time_idx <= validation_cutoff], min_prediction_idx=training_cutoff + 1)
    testing = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=validation_cutoff + 1)

    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized")
    val_dataloader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized")
    test_dataloader = testing.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized")

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_top_k=1, monitor="val_loss", mode="min")
    logger = TensorBoardLogger(save_dir="logs", name="xLSTM", version=log_file)

    trainer = pl.Trainer(
        logger=logger,
        max_steps=configs['train']['max_steps'],
        accelerator='gpu',
        max_epochs=40,
        devices=[args.device],
        enable_model_summary=True,
        gradient_clip_val=configs['train']['gradient_clip_val'],
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_train_batches=configs['train']['limit_train_batches'],
        enable_checkpointing=True,
        accumulate_grad_batches=16,
    )

    net = xLSTMBackbone.from_dataset(
        training,
        hidden_size=args.hidden_size,
        rnn_layers=configs['xLSTM']['rnn_layers'],
        dropout=configs['train']['dropout'],
        loss=NormalDistributionLoss(),
        optimizer="adam",
        learning_rate=args.lr,
        weight_decay=float(configs['train']['weight_decay']),
        reduce_on_plateau_patience=configs['train']['reduce_on_plateau_patience'],
    )

    trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model = xLSTMBackbone.load_from_checkpoint(
        checkpoint_callback.best_model_path, map_location="cuda:%s" % (args.device)
    ).to("cuda:%s" % (args.device))

    metrics = []
    crps_mean_noagg_all, crps_noagg_all, crps_sum_noagg_all = [], [], []
    for i in range(args.num_repeat):
        raw_predictions, x = best_model.predict(test_dataloader, mode="raw", n_samples=100, show_progress_bar=True, return_x=True)
        preds = raw_predictions['prediction'].cpu()
        actuals = x['decoder_target']
        try:
            preds = preds.reshape(args.num_pred_rolling, preds.shape[0] // args.num_pred_rolling, preds.shape[1], preds.shape[2])
            actuals = actuals.reshape(args.num_pred_rolling, actuals.shape[0] // args.num_pred_rolling, actuals.shape[1])
        except Exception:
            preds = preds.unsqueeze(0)
            actuals = actuals.unsqueeze(0)

        agg_metric, crps_mean_noagg, crps_noagg, crps_sum_noagg = get_metrics(preds, actuals)
        crps_mean_noagg_all.append(crps_mean_noagg)
        crps_noagg_all.append(crps_noagg)
        crps_sum_noagg_all.append(crps_sum_noagg)
        metrics.append(agg_metric)

    metrics = np.array(metrics)
    metrics = np.concatenate([metrics.mean(0).reshape(-1, 1), metrics.std(0).reshape(-1, 1)], axis=1)

    crps_mean_noagg = torch.stack(crps_mean_noagg_all)
    crps_noagg = torch.stack(crps_noagg_all)
    crps_sum_noagg = torch.stack(crps_sum_noagg_all)

    os.makedirs("./metrics_raw/xLSTM", exist_ok=True)
    torch.save(crps_mean_noagg, './metrics_raw/xLSTM/%s_crps_mean.pt' % (log_file))
    torch.save(crps_noagg, './metrics_raw/xLSTM/%s_crps.pt' % (log_file))
    torch.save(crps_sum_noagg, './metrics_raw/xLSTM/%s_crps_sum.pt' % (log_file))

    os.makedirs("./metrics/xLSTM", exist_ok=True)
    with open('./metrics/xLSTM/%s.txt' % (log_file), 'w') as f:
        for i in range(metrics.shape[0]):
            sep = '\n' if i == metrics.shape[0] - 1 else ''
            f.write('& %.4f$\\pm$%.4f %s' % (metrics[i, 0], metrics[i, 1], sep))

    print("DONE", log_file, "crps_mean=%.4f crps_sum=%.4f" % (crps_mean_noagg.mean(), crps_sum_noagg.mean()))
    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    main()
