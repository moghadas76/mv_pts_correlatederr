"""Minimal repro for xLSTM segfault"""
import faulthandler
faulthandler.enable()
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch import nn
print("Step 1: basic imports done")

from _recurrent import SeriesDecomposition, mLSTMNetwork, sLSTMNetwork
print("Step 2: _recurrent imports done")

# Test SeriesDecomposition
print("Step 3: testing SeriesDecomposition...")
decomp = SeriesDecomposition(25)
x = torch.randn(4, 12, 3)
seasonal, trend = decomp(x)
print(f"  seasonal: {seasonal.shape}, trend: {trend.shape}")

# Test sLSTMNetwork standalone
print("Step 4: testing sLSTMNetwork standalone...")
net = sLSTMNetwork(input_size=10, hidden_size=10, num_layers=2, output_size=10, dropout=0.1)
h = net.init_hidden(4)
x_sf = torch.randn(12, 4, 10)
out, hid = net.slstm_layer(x_sf, *h)
print(f"  out: {out.shape}")

# Now test xLSTMBackbone directly (not via from_dataset)
print("Step 5: importing xLSTMBackbone...")
from model import xLSTMBackbone
print("  imported")

# Test from_dataset
print("Step 6: testing from_dataset...")
import pandas as pd
import numpy as np
import pickle, yaml
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from loss import BatchMGDGraph_Kernel

with open('./config/config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.Loader)
with open('./datasets/pred_horizon_dict.pkl', 'rb') as f:
    pred_horizon_dict = pickle.load(f)
with open('./datasets/dataset_freq.pkl', 'rb') as f:
    dataset_freq_dict = pickle.load(f)

data = pd.read_csv('./datasets/pems03_flow.csv')
prediction_horizon = pred_horizon_dict['pems03_flow']
data["datetime"] = pd.to_datetime(data["datetime"])
data["tod"] = (data["datetime"].values - data["datetime"].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
data['dow'] = data['datetime'].dt.weekday
data = data.astype(dict(sensor=str, tod=str, dow=str))
lags = {"value": [2, 4, 12, 24, 48]}

validation_cutoff = data["time_idx"].max() - prediction_horizon - 56 + 1
training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    target_normalizer=GroupNormalizer(groups=["sensor"], transformation=None),
    categorical_encoders={"sensor": NaNLabelEncoder().fit(data.sensor)},
    group_ids=["sensor"],
    static_categoricals=["sensor"],
    time_varying_known_categoricals=['tod', 'dow'],
    time_varying_unknown_reals=["value"],
    lags=lags,
    min_encoder_length=prediction_horizon,
    max_encoder_length=prediction_horizon,
    min_prediction_length=prediction_horizon,
    max_prediction_length=prediction_horizon,
    allow_missing_timesteps=False,
)

from batched_model import BatchxLSTMEstimator
print("Step 7: calling from_dataset...")

from dynamic_graph import SAGSAM, precision_from_adj, load_static_graph

loss = BatchMGDGraph_Kernel(
    D=prediction_horizon,
    K_r=4,
    delta_l=1.0,
    train_l=False,
    lr=0.01,
    wd=1e-08,
    reg_w=2.5,
    static=False,
    static_graph=load_static_graph("./datasets/PEMS03_graph.csv", 358)
)

net = BatchxLSTMEstimator.from_dataset(
    training,
    hidden_size=10,
    rnn_layers=2,
    dropout=0.01,
    loss=loss,
    optimizer="adam",
    learning_rate=0.01,
    weight_decay=1e-08,
    reduce_on_plateau_patience=500,
)
print("Step 8: model created!")
print(net)

# Quick forward test
print("Step 9: testing forward pass...")
train_dataloader = training.to_dataloader(train=True, batch_size=20, num_workers=0, batch_sampler="synchronized")
for x, y in train_dataloader:
    print("  x encoder_cont:", x["encoder_cont"].shape)
    net.train()
    out = net(x)
    print("  output:", out["prediction"].shape)
    break

# Test on GPU if available
if torch.cuda.is_available():
    print("Step 10: testing on GPU forward...")
    net = net.cuda()
    for x, y in train_dataloader:
        x = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        y = tuple(v.cuda() if isinstance(v, torch.Tensor) else v for v in y)
        out = net(x)
        print("  GPU output:", out["prediction"].shape)
        print("Step 11: testing backward on GPU...")
        pred = out["prediction"]
        fake_loss = pred.sum()
        fake_loss.backward()
        print("  backward OK")
        break

# Test with pl.Trainer for 1 step
print("Step 12: testing with PL Trainer...")
import pytorch_lightning as pl
net_cpu = BatchxLSTMEstimator.from_dataset(
    training,
    hidden_size=10,
    rnn_layers=2,
    dropout=0.01,
    loss=loss,
    optimizer="adam",
    learning_rate=0.01,
    weight_decay=1e-08,
    reduce_on_plateau_patience=500,
)
trainer = pl.Trainer(
    max_steps=2,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    enable_model_summary=False,
    gradient_clip_val=10.0,
    enable_checkpointing=False,
    limit_train_batches=2,
)
trainer.fit(net_cpu, train_dataloaders=train_dataloader)
print("ALL TESTS PASSED")
