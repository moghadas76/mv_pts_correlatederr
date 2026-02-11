"""
Training script for DG-LoGraP (Dynamic-Graph Low-Rank-plus-Diagonal with Grouped Latent Spatio-Temporal Processes)

This script trains the DG-LoGraP model on the PEMS03 traffic flow dataset with:
- SAGSAM for dynamic graph generation
- SAGS-GCN for spatial message passing
- Grouped latent spatio-temporal covariance structure
"""

import warnings
warnings.filterwarnings("ignore")
import pickle
import argparse
import os
import yaml

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
)

from metrics import get_metrics
from loss import DGLoGraP_Loss
from dg_lograp_model import DGLoGraPDeepAR, DGLoGraPTransformer


parser = argparse.ArgumentParser(description='Train DG-LoGraP model on PEMS03 dataset')
parser.add_argument('--device', type=int, default=0, help='GPU device index')
parser.add_argument('--model', type=str, default="deepar", choices=["deepar", "transformer"], help='Base model type')
parser.add_argument('--dataset', type=str, default="pems03_flow", help='Dataset name')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
parser.add_argument('--prediction_horizon', type=int, default=12, help='Prediction horizon')
parser.add_argument('--num_pred_rolling', type=int, default=7, help='Number of rolling predictions')
parser.add_argument('--batch_cov_horizon', type=int, default=12, help='Batch covariance horizon D')
parser.add_argument('--num_repeat', type=int, default=5, help='Number of evaluation repeats')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

# DG-LoGraP specific parameters
parser.add_argument('--num_nodes', type=int, default=358, help='Number of spatial nodes')
parser.add_argument('--graph_embed_dim', type=int, default=32, help='SAGSAM embedding dimension')
parser.add_argument('--num_groups', type=int, default=2, help='Number of latent groups G')
parser.add_argument('--rank_per_group', type=int, default=5, help='Rank per group R_g')
parser.add_argument('--num_kernels', type=int, default=4, help='Number of temporal kernel mixtures')
parser.add_argument('--graph_path', type=str, default='./datasets/PEMS03_graph.csv', help='Path to graph CSV')
parser.add_argument('--use_graph_conv', action='store_true', default=True, help='Use SAGS-GCN')

# Training parameters
parser.add_argument('--lr', type=float, default=1e-03, help='Learning rate')
parser.add_argument('--reg_w', type=float, default=0.1, help='Regularization weight')
parser.add_argument('--loss_lr_w', type=float, default=1.0, help='Loss learning rate weight')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

args = parser.parse_args()
pl.seed_everything(args.seed)


def load_pems03_data():
    """Load and preprocess PEMS03 traffic flow data."""
    data = pd.read_csv("./datasets/%s.csv" % args.dataset)
    
    # Process datetime
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["tod"] = (data["datetime"].values - data["datetime"].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    data['dow'] = data['datetime'].dt.weekday
    
    # Convert to categorical
    data = data.astype(dict(sensor=str, tod=str, dow=str))
    
    # Time-varying known categoricals
    time_varying_known_cats = ['tod', 'dow']
    
    # Lags for autoregressive features
    lags = {"value": [12, 288]}  # 1 hour and 1 day lags for 5-min data
    
    return data, time_varying_known_cats, lags


def create_dataloaders(data, time_varying_known_cats, lags):
    """Create train, validation, and test dataloaders."""
    # Split data
    validation_cutoff = data["time_idx"].max() - args.prediction_horizon - args.num_pred_rolling + 1
    training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)
    
    # Training dataset
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
    
    # Validation and test datasets
    validation = TimeSeriesDataSet.from_dataset(
        training, data[lambda x: x.time_idx <= validation_cutoff], 
        min_prediction_idx=training_cutoff + 1
    )
    testing = TimeSeriesDataSet.from_dataset(
        training, data, min_prediction_idx=validation_cutoff + 1
    )
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )
    
    return training, train_dataloader, val_dataloader, test_dataloader


def main():
    """Main training function."""
    print("=" * 60)
    print("DG-LoGraP Training on PEMS03 Dataset")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Number of nodes: {args.num_nodes}")
    print(f"Number of groups: {args.num_groups}")
    print(f"Rank per group: {args.rank_per_group}")
    print(f"Graph embedding dim: {args.graph_embed_dim}")
    print(f"Number of kernels: {args.num_kernels}")
    print(f"Use graph conv: {args.use_graph_conv}")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading PEMS03 data...")
    data, time_varying_known_cats, lags = load_pems03_data()
    print(f"Data shape: {data.shape}")
    print(f"Number of sensors: {data.sensor.nunique()}")
    print(f"Time range: {data.datetime.min()} to {data.datetime.max()}")
    
    # Create dataloaders
    print("\n[2] Creating dataloaders...")
    training, train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        data, time_varying_known_cats, lags
    )
    print(f"Training samples: {len(training)}")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    # Initialize loss function
    print("\n[3] Initializing DG-LoGraP loss...")
    ranks_per_group = [args.rank_per_group] * args.num_groups
    loss = DGLoGraP_Loss(
        num_nodes=args.num_nodes,
        D=args.batch_cov_horizon,
        num_groups=args.num_groups,
        ranks_per_group=ranks_per_group,
        embed_dim=args.graph_embed_dim,
        num_kernels=args.num_kernels,
        graph_path=args.graph_path,
        reg_w=args.reg_w,
    )
    print(f"Total rank: {sum(ranks_per_group)}")
    print(f"Distribution arguments: {len(loss.distribution_arguments)}")
    
    # Initialize model
    print("\n[4] Initializing model...")
    if args.model == "deepar":
        net = DGLoGraPDeepAR.from_dataset(
            training,
            cell_type="LSTM",
            hidden_size=args.hidden_size,
            rnn_layers=2,
            dropout=0.1,
            loss=loss,
            optimizer="adam",
            learning_rate=args.lr,
            weight_decay=1e-6,
            reduce_on_plateau_patience=5,
            num_nodes=args.num_nodes,
            graph_embed_dim=args.graph_embed_dim,
            num_groups=args.num_groups,
            ranks_per_group=ranks_per_group,
            num_kernels=args.num_kernels,
            graph_path=args.graph_path,
            use_graph_conv=args.use_graph_conv,
        )
    elif args.model == "transformer":
        net = DGLoGraPTransformer.from_dataset(
            training,
            hidden_size=args.hidden_size,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            loss=loss,
            optimizer="adam",
            learning_rate=args.lr,
            weight_decay=1e-6,
            reduce_on_plateau_patience=5,
            num_nodes=args.num_nodes,
            graph_embed_dim=args.graph_embed_dim,
            num_groups=args.num_groups,
            ranks_per_group=ranks_per_group,
            num_kernels=args.num_kernels,
            graph_path=args.graph_path,
            use_graph_conv=args.use_graph_conv,
        )
    
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=args.patience, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_loss:.4f}', 
        save_top_k=1, 
        monitor="val_loss", 
        mode="min"
    )
    
    # Logger
    log_file = f"{args.dataset}_dg_lograp_{args.model}_G{args.num_groups}_R{args.rank_per_group}_K{args.num_kernels}_H{args.hidden_size}_D{args.batch_cov_horizon}"
    logger = TensorBoardLogger(save_dir="logs", name="dg_lograp", version=log_file)
    
    # Trainer
    print("\n[5] Setting up trainer...")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        max_steps=10000,
        limit_train_batches=400,
        accelerator='gpu',
        devices=[args.device],
        enable_model_summary=True,
        gradient_clip_val=10.0,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
        accumulate_grad_batches=16,
    )
    
    # Train
    print("\n[6] Training...")
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Evaluate
    print("\n[7] Evaluating best model...")
    if args.model == "deepar":
        best_model = DGLoGraPDeepAR.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            map_location=f"cuda:{args.device}"
        ).to(f"cuda:{args.device}")
    else:
        best_model = DGLoGraPTransformer.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            map_location=f"cuda:{args.device}"
        ).to(f"cuda:{args.device}")
    
    best_model.eval()
    
    # Run evaluation
    metrics = []
    crps_mean_noagg_all, crps_noagg_all, crps_sum_noagg_all = [], [], []
    
    for i in range(args.num_repeat):
        print(f"  Evaluation run {i+1}/{args.num_repeat}")
        with torch.no_grad():
            raw_predictions, x = best_model.predict(
                test_dataloader, 
                mode="raw", 
                n_samples=100, 
                show_progress_bar=True, 
                return_x=True
            )
        
        preds = raw_predictions['prediction'].cpu()
        actuals = x['decoder_target']
        
        # Handle autoregressive output 
        # preds has shape (total_batches * n_samples, batch_size_per_batch, T, output_dim)
        if preds.dim() == 4:
            n_samples_pred = 100  # We used 100 samples
            
            # Take only the mean/loc (first parameter in output_dim)
            preds = preds[..., 0]  # (n_batches * n_samples, batch_size, T)
            
            # Reshape to separate samples and batches
            total_sample_batches = preds.shape[0]
            n_batches = total_sample_batches // n_samples_pred
            
            # Reshape: (n_batches * n_samples, batch_size, T) -> (n_batches, n_samples, batch_size, T)
            preds = preds.reshape(n_batches, n_samples_pred, preds.shape[1], preds.shape[2])
            
            # Flatten batches and batch_size together, keep samples last
            # (n_batches, n_samples, batch_size, T) -> (n_batches * batch_size, T, n_samples)
            preds = preds.permute(0, 2, 3, 1)  # (n_batches, batch_size, T, n_samples)
            preds = preds.reshape(-1, preds.shape[2], preds.shape[3])  # (N, T, n_samples)
        
        # Ensure preds and actuals have same number of samples
        min_n = min(preds.shape[0], actuals.shape[0])
        preds = preds[:min_n]
        actuals = actuals[:min_n]
        
        # Handle NaN values by replacing with median
        preds_np = preds.numpy()
        if np.any(np.isnan(preds_np)):
            median_val = np.nanmedian(preds_np)
            preds_np = np.nan_to_num(preds_np, nan=median_val)
            preds = torch.from_numpy(preds_np)
        
        try:
            # preds: (N, T, n_samples) -> (n_frcs, N/n_frcs, T, n_samples)
            preds = preds.reshape(
                args.num_pred_rolling, 
                preds.shape[0] // args.num_pred_rolling, 
                preds.shape[1], 
                preds.shape[2]
            )
            actuals = actuals.reshape(
                args.num_pred_rolling, 
                actuals.shape[0] // args.num_pred_rolling, 
                actuals.shape[1]
            )
        except Exception as e:
            preds = preds.unsqueeze(0)
            actuals = actuals.unsqueeze(0)
        
        agg_metric, crps_mean_noagg, crps_noagg, crps_sum_noagg = get_metrics(preds, actuals)
        crps_mean_noagg_all.append(crps_mean_noagg)
        crps_noagg_all.append(crps_noagg)
        crps_sum_noagg_all.append(crps_sum_noagg)
        metrics.append(agg_metric)
    
    # Aggregate metrics
    metrics = np.array(metrics)
    metrics = np.concatenate([
        metrics.mean(0).reshape(-1, 1), 
        metrics.std(0).reshape(-1, 1)
    ], axis=1)
    
    crps_mean_noagg = torch.stack(crps_mean_noagg_all)
    crps_noagg = torch.stack(crps_noagg_all)
    crps_sum_noagg = torch.stack(crps_sum_noagg_all)
    
    # Save metrics
    os.makedirs(f"./metrics_raw/dg_lograp", exist_ok=True)
    os.makedirs(f"./metrics/dg_lograp", exist_ok=True)
    
    torch.save(crps_mean_noagg, f'./metrics_raw/dg_lograp/{log_file}_crps_mean.pt')
    torch.save(crps_noagg, f'./metrics_raw/dg_lograp/{log_file}_crps.pt')
    torch.save(crps_sum_noagg, f'./metrics_raw/dg_lograp/{log_file}_crps_sum.pt')
    
    with open(f'./metrics/dg_lograp/{log_file}.txt', 'w') as f:
        for i in range(metrics.shape[0]):
            if i != metrics.shape[0] - 1:
                f.write(f'& {metrics[i, 0]:.4f}$\\pm${metrics[i, 1]:.4f}')
            else:
                f.write(f'& {metrics[i, 0]:.4f}$\\pm${metrics[i, 1]:.4f} \n')
    
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    metric_names = ["CRPS", "CRPS-Mean", "CRPS-Sum", "MAE", "RMSE"]
    for i, name in enumerate(metric_names[:min(len(metric_names), metrics.shape[0])]):
        print(f"{name}: {metrics[i, 0]:.4f} ± {metrics[i, 1]:.4f}")
    print("=" * 60)
    
    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    score = main()
    print(f"\nBest validation score: {score}")
