import warnings

import model
warnings.filterwarnings("ignore")
import pickle
import argparse
import os
import torch.nn.functional as F

import yaml

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
)

from metrics import get_metrics
from batched_model import BatchDeepAREstimator, BatchDeepARPredictor, BatchGPTEstimator, BatchGPTPredictor, BatchxLSTMPredictor, BatchxLSTMEstimator
from loss import BatchMGD_AR, BatchMGD_Kernel, BatchMGDGraph_Kernel,\
BatchMGDCurvature_Kernel, TStudent_BatchMGDCurvature_Kernel, BatchMGDLearnable_Kernel, BatchMGDDiffusion_Kernel
from dynamic_graph import load_static_graph, load_static_graph_from_pickle
from vis import (visualize_curvature_precision_effectiveness, visualize_ablation_Gt_vs_IR,
    visualize_batch_covariance_Sigma, track_curvature_params_during_training,
    compute_curvature_effectiveness_metrics)
def visualize_covariance_matrices(cov_matrices, save_path, title_prefix, clip_range=(0, 0.03)):
    """
    Visualize covariance matrices with clipping for interpretability.
    
    Args:
        cov_matrices: Tensor of covariance matrices - can be various shapes
        save_path: Base path to save visualizations
        title_prefix: Prefix for plot titles (e.g., 'sigma_D', 'C_t', 'G_t')
        clip_range: Tuple of (min, max) values to clip covariance values
    """
    if cov_matrices is None:
        return
    
    # Handle different input shapes
    if len(cov_matrices.shape) == 1:
        # 1D array - cannot visualize as matrix
        print(f"Warning: {title_prefix} is 1D with shape {cov_matrices.shape}, skipping visualization")
        return
    elif len(cov_matrices.shape) == 2:
        # 2D matrix [features, features] - single covariance matrix
        cov_matrix = cov_matrices.detach().cpu().numpy()
        matrices_to_plot = [cov_matrix]
        titles = [f'{title_prefix}']
    elif len(cov_matrices.shape) == 3:
        # 3D tensor [time, features, features] or [batch, features, features]
        cov_clipped = torch.clamp(cov_matrices, min=clip_range[0], max=clip_range[1])
        # Take first, middle, and last matrices
        indices = [0, cov_clipped.shape[0]//2, cov_clipped.shape[0]-1]
        matrices_to_plot = [cov_clipped[i].detach().cpu().numpy() for i in indices]
        titles = [f'{title_prefix} at index {i}' for i in indices]
    elif len(cov_matrices.shape) == 4:
        # 4D tensor [batch, time, features, features]
        cov_clipped = torch.clamp(cov_matrices, min=clip_range[0], max=clip_range[1])
        # Take median across batch dimension
        cov_median = torch.median(cov_clipped, dim=0)[0]  # [time, features, features]
        # Select first, middle, and last time steps
        indices = [0, cov_median.shape[0]//2, cov_median.shape[0]-1]
        matrices_to_plot = [cov_median[i].detach().cpu().numpy() for i in indices]
        titles = [f'{title_prefix} at t={i}' for i in indices]
    elif len(cov_matrices.shape) == 5:
        # 5D tensor [num_repeat, batch, time, features, features]
        cov_clipped = torch.clamp(cov_matrices, min=clip_range[0], max=clip_range[1])
        # Take median across both num_repeat and batch dimensions
        cov_median = torch.median(cov_clipped.flatten(0, 1), dim=0)[0]  # [time, features, features]
        # Select first, middle, and last time steps
        indices = [0, cov_median.shape[0]//2, cov_median.shape[0]-1]
        matrices_to_plot = [cov_median[i].detach().cpu().numpy() for i in indices]
        titles = [f'{title_prefix} at t={i}' for i in indices]
    else:
        print(f"Warning: Unsupported tensor shape {cov_matrices.shape} for {title_prefix}")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, len(matrices_to_plot), figsize=(5*len(matrices_to_plot), 5))
    if len(matrices_to_plot) == 1:
        axes = [axes]
    
    for i, (cov_matrix, title) in enumerate(zip(matrices_to_plot, titles)):
        # Clip values for visualization clarity
        cov_matrix_clipped = np.clip(cov_matrix, clip_range[0], clip_range[1])
        
        im = axes[i].imshow(cov_matrix_clipped, cmap='viridis', vmin=clip_range[0], vmax=clip_range[1])
        axes[i].set_title(title)
        axes[i].set_xlabel('Feature Index')
        axes[i].set_ylabel('Feature Index')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}_{title_prefix}_covariance_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_prediction_video(prediction_actual_pairs, save_path, fps=2):
    """
    Create a video visualization of predictions vs actuals over time.
    Each node gets its own subplot.
    
    Args:
        prediction_actual_pairs: List of tuples (preds, actuals) for each time step
        save_path: Path to save the video
        fps: Frames per second for the video
    """
    import matplotlib.animation as animation

    # prediction_actual_pairs[0] is a list of preds tensors (one per batch),
    # each shaped [num_rolling, nodes, time, samples]. Use the first batch.
    preds_all = prediction_actual_pairs[0][batch_idx]   # [24, nodes, 12, 100]
    actuals_all = prediction_actual_pairs[1][batch_idx]  # [24, nodes, 12]
    num_rolling = preds_all.shape[0]
    num_nodes = preds_all.shape[1]
    # take [113, 153, 47, 101, 118]
    # selected_nodes = np.random.choice(num_nodes, size=min(5, num_nodes), replace=False)
    selected_nodes = np.array([113, 153, 47, 101, 118], dtype=int)
    

    # Create subplots - one for each selected node
    num_subplots = len(selected_nodes)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3*num_subplots))
    if num_subplots == 1:
        axes = [axes]

    def update(frame):
        # frame indexes the rolling window dimension
        preds = preds_all[frame]    # [nodes, 12, 100]
        actuals = actuals_all[frame]  # [nodes, 12]
        time_steps = np.arange(preds.shape[1])

        for subplot_idx, node_id in enumerate(selected_nodes):
            ax = axes[subplot_idx]
            ax.clear()

            node_preds = preds[node_id, :, :].cpu().numpy()    # [12, 100]
            node_actuals = actuals[node_id, :].cpu().numpy()   # [12]

            # Compute percentiles across samples
            median_pred = np.median(node_preds, axis=1)  # [12]
            p10_pred = np.percentile(node_preds, 10, axis=1)  # [12]
            p90_pred = np.percentile(node_preds, 90, axis=1)  # [12]

            # Plot for this node
            ax.plot(time_steps, node_actuals, 'o-', color='red',
                   label='Actual', linewidth=2, markersize=4)
            ax.plot(time_steps, median_pred, '--', color='blue',
                   label='Median', linewidth=1.5, alpha=0.7)
            ax.fill_between(time_steps, p10_pred, p90_pred, color='blue',
                           alpha=0.15, label='10-90th percentile')

            ax.set_title(f'Node {node_id} - Rolling Window {frame+1}/{num_rolling}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(fig, update, frames=num_rolling, repeat=False)
    anim.save(save_path, writer='ffmpeg', fps=fps)
    plt.close()



def visualize_prediction_images(batch_idx, prediction_actual_pairs, save_dir, dpi=150):
    """
    Save per-node prediction interval images for all rolling windows.

    Uses the same input convention as ``visualize_prediction_video``:
    - prediction_actual_pairs[0][batch_idx]: [num_rolling, nodes, horizon, samples]
    - prediction_actual_pairs[1][batch_idx]: [num_rolling, nodes, horizon]

    For each node and rolling window, saves one PNG:
        {save_dir}/node_{node_id}/window_{frame}.png

    Args:
        batch_idx: Index of the batch to visualize.
        prediction_actual_pairs: Pair of prediction/actual tensors collected in evaluation loop.
        save_dir: Output directory for images.
        dpi: Saved image DPI.
    """
    if prediction_actual_pairs is None or len(prediction_actual_pairs) < 2:
        print("Warning: Invalid prediction_actual_pairs, skipping image visualization")
        return

    if not prediction_actual_pairs[0] or not prediction_actual_pairs[1]:
        print("Warning: Empty prediction/actual pairs, skipping image visualization")
        return

    preds_all = prediction_actual_pairs[0][batch_idx]    # [num_rolling, nodes, horizon, samples]
    actuals_all = prediction_actual_pairs[1][batch_idx]  # [num_rolling, nodes, horizon]

    if len(preds_all.shape) != 4 or len(actuals_all.shape) != 3:
        print(
            f"Warning: Unexpected shapes for image visualization: "
            f"preds={preds_all.shape}, actuals={actuals_all.shape}"
        )
        return

    num_rolling = preds_all.shape[0]
    num_nodes = preds_all.shape[1]
    save_dir = os.path.join(save_dir, f'batch_{batch_idx}')
    os.makedirs(save_dir, exist_ok=True)
    time_steps = np.arange(preds_all.shape[2])

    print(f"Saving prediction images for {num_nodes} nodes across {num_rolling} rolling windows...")

    for node_id in range(num_nodes):
        node_dir = os.path.join(save_dir, f"node_{node_id:04d}")
        os.makedirs(node_dir, exist_ok=True)

        for frame in range(num_rolling):
            node_preds = preds_all[frame, node_id, :, :].detach().cpu().numpy()   # [horizon, samples]
            node_actuals = actuals_all[frame, node_id, :].detach().cpu().numpy()  # [horizon]

            median_pred = np.median(node_preds, axis=1)
            p10_pred = np.percentile(node_preds, 10, axis=1)
            p90_pred = np.percentile(node_preds, 90, axis=1)

            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(time_steps, node_actuals, "o-", color="red", label="Actual", linewidth=2, markersize=4)
            ax.plot(time_steps, median_pred, "--", color="blue", label="Median", linewidth=1.5, alpha=0.8)
            ax.fill_between(
                time_steps,
                p10_pred,
                p90_pred,
                color="blue",
                alpha=0.15,
                label="10-90th percentile",
            )

            mae = np.mean(np.abs(node_actuals - median_pred))

            ax.set_title(f"Node {node_id} - Rolling Window {frame + 1}/{num_rolling}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
            ax.text(
                0.02,
                0.98,
                f"MAE: {mae:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()
            frame_path = os.path.join(node_dir, f"window_{frame + 1:03d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved all node prediction images to {save_dir}")


def visualize_prediction_intervals(predictions, actuals, save_path, num_series=5, x_data=None, save_all_nodes=True):
    """
    Visualize prediction intervals using median, 10th, and 90th percentiles.
    
    Args:
        predictions: Tensor [num_repeat, num_rolling, batch, time, features/samples] 
        actuals: Tensor [num_rolling, batch, time] 
        save_path: Path to save visualization
        num_series: Number of time series to visualize in combined plot
        x_data: Additional context data with encoder information
        save_all_nodes: If True, save individual forecasts for all nodes
    """
    print(f"Predictions shape: {predictions.shape}")
    print(f"Actuals shape: {actuals.shape}")
    
    # Handle different prediction tensor shapes
    # predictions: [num_repeat, num_rolling, batch, time, n_samples]
    # The last dimension (n_samples=100) is the samples from probabilistic model
    if len(predictions.shape) == 5:
        num_repeat, num_rolling, batch_size, pred_time, n_samples = predictions.shape
        
        # Select the first rolling window and first repeat
        # preds_selected: [batch, time, n_samples]
        preds_selected = predictions[0, 0, :, :, :]  # [batch, time, n_samples]
        actuals_selected = actuals[0, :, :]  # [batch, time]
        
    elif len(predictions.shape) == 4:
        # [batch, time, n_samples, ?] 
        preds_selected = predictions[0, :, :, :]  # [batch, time, n_samples]
        actuals_selected = actuals  # [batch, time]
    else:
        print(f"Warning: Unsupported prediction tensor shape {predictions.shape}")
        return
    
    # Extract context information if available
    context_length = 12  # Show last 12 time steps of context
    encoder_targets = None
    sensor_names = None
    
    if x_data is not None:
        print(f"x_data keys: {list(x_data.keys())}")
        
        # Get encoder targets for context
        if 'encoder_target' in x_data:
            encoder_raw = x_data['encoder_target']
            print(f"Encoder target shape: {encoder_raw.shape}")
            
            # Handle different possible shapes for encoder_target
            if len(encoder_raw.shape) == 2:
                encoder_targets = encoder_raw
            elif len(encoder_raw.shape) == 3:
                if encoder_raw.shape[0] == preds_selected.shape[0]:
                    encoder_targets = encoder_raw[:, :, 0] if encoder_raw.shape[2] > 1 else encoder_raw.squeeze(-1)
                else:
                    encoder_targets = encoder_raw[0]
            else:
                print(f"Unexpected encoder_target shape: {encoder_raw.shape}")
                encoder_targets = None
            
            if encoder_targets is not None and len(encoder_targets.shape) == 2:
                if encoder_targets.shape[1] >= context_length:
                    encoder_targets = encoder_targets[:, -context_length:]
            
        # Get sensor/node names if available
        if 'groups' in x_data:
            groups_raw = x_data['groups']
            print(f"Groups data shape: {groups_raw.shape}")
            if hasattr(groups_raw, 'cpu'):
                sensor_ids = groups_raw.cpu().numpy()[:, 0]
            else:
                sensor_ids = groups_raw[:, 0]
            sensor_names = [f"Node_{int(s)}" for s in sensor_ids[:preds_selected.shape[0]]]
        elif 'sensor' in x_data:
            sensor_raw = x_data['sensor']
            print(f"Sensor data shape: {sensor_raw.shape if hasattr(sensor_raw, 'shape') else type(sensor_raw)}")
            if hasattr(sensor_raw, 'cpu'):
                if len(sensor_raw.shape) >= 2:
                    sensor_ids = sensor_raw.cpu().numpy()[:, 0]
                else:
                    sensor_ids = sensor_raw.cpu().numpy()
            else:
                sensor_ids = np.arange(preds_selected.shape[0])
            sensor_names = [f"Node_{int(s)}" for s in sensor_ids[:preds_selected.shape[0]]]
        else:
            sensor_names = [f"Node_{i}" for i in range(preds_selected.shape[0])]
    else:
        sensor_names = [f"Node_{i}" for i in range(preds_selected.shape[0])]
    
    # ============== Save all nodes separately ==============
    if save_all_nodes:
        nodes_dir = os.path.dirname(save_path) + "/individual_nodes"
        if not os.path.isdir(nodes_dir):
            os.makedirs(nodes_dir)
        
        print(f"Saving individual forecasts for {preds_selected.shape[0]} nodes...")
        
        for node_idx in range(preds_selected.shape[0]):
            # Get predictions for this node: [time, n_samples]
            node_preds = preds_selected[node_idx, :, :].detach().cpu().numpy()  # [time, n_samples]
            node_actual = actuals_selected[node_idx, :].detach().cpu().numpy()  # [time]
            
            # Compute percentiles across samples for each time step
            median_pred = np.median(node_preds, axis=1)  # [time]
            p10_pred = np.percentile(node_preds, 10, axis=1)  # [time]
            p90_pred = np.percentile(node_preds, 90, axis=1)  # [time]
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            
            pred_time_steps = np.arange(len(median_pred))
            
            # Get context data if available
            if encoder_targets is not None and node_idx < encoder_targets.shape[0]:
                context_data = encoder_targets[node_idx, :].detach().cpu().numpy()
                context_time_steps = np.arange(-len(context_data), 0)
                ax.plot(context_time_steps, context_data, 'g-', label='Historical context', linewidth=2, alpha=0.7)
                ax.axvline(x=-0.5, color='black', linestyle='--', alpha=0.5, label='Prediction start')
            
            # Plot prediction intervals
            ax.fill_between(pred_time_steps, p10_pred, p90_pred, alpha=0.3, 
                           color='blue', label='10-90th percentile')
            ax.plot(pred_time_steps, median_pred, 'b-', label='Median prediction', linewidth=2)
            ax.plot(pred_time_steps, node_actual, 'r-', label='Actual', linewidth=2)
            
            node_name = sensor_names[node_idx] if node_idx < len(sensor_names) else f"Node_{node_idx}"
            ax.set_title(f'{node_name}: Prediction with 10-90th Percentile CI')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mape = np.mean(np.abs((node_actual - median_pred) / (node_actual + 1e-8))) * 100
            mae = np.mean(np.abs(node_actual - median_pred))
            ax.text(0.02, 0.98, f'MAPE: {mape:.2f}%, MAE: {mae:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{nodes_dir}/{node_name}_forecast.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {preds_selected.shape[0]} individual node forecasts to {nodes_dir}")
    
    # ============== Combined plot for selected nodes ==============
    num_series = min(num_series, preds_selected.shape[0])
    
    fig, axes = plt.subplots(num_series, 1, figsize=(14, 3*num_series))
    if num_series == 1:
        axes = [axes]
    
    for i in range(num_series):
        # Get predictions for this series: [time, n_samples]
        series_preds = preds_selected[i, :, :].detach().cpu().numpy()  # [time, n_samples]
        series_actual = actuals_selected[i, :].detach().cpu().numpy()  # [time]
        
        # Compute percentiles across samples for each time step
        median_pred = np.median(series_preds, axis=1)  # [time]
        p10_pred = np.percentile(series_preds, 10, axis=1)  # [time]
        p90_pred = np.percentile(series_preds, 90, axis=1)  # [time]
        
        pred_time_steps = np.arange(len(median_pred))
        
        # Plot context if available
        if encoder_targets is not None and i < encoder_targets.shape[0]:
            context_data = encoder_targets[i, :].detach().cpu().numpy()
            context_time_steps = np.arange(-len(context_data), 0)
            axes[i].plot(context_time_steps, context_data, 'g-', label='Historical context', 
                        linewidth=2, alpha=0.7)
            axes[i].axvline(x=-0.5, color='black', linestyle='--', alpha=0.5, 
                          label='Prediction start')
        
        # Plot prediction intervals
        axes[i].fill_between(pred_time_steps, p10_pred, p90_pred, alpha=0.3, 
                           color='blue', label='10-90th percentile')
        axes[i].plot(pred_time_steps, median_pred, 'b-', label='Median prediction', linewidth=2)
        axes[i].plot(pred_time_steps, series_actual, 'r-', label='Actual', linewidth=2)
        
        # Add title with node information
        if sensor_names is not None and i < len(sensor_names):
            title = f'{sensor_names[i]}: Prediction with 10-90th Percentile CI'
        else:
            title = f'Series {i+1}: Prediction with 10-90th Percentile CI'
        axes[i].set_title(title)
        
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mape = np.mean(np.abs((series_actual - median_pred) / (series_actual + 1e-8))) * 100
        mae = np.mean(np.abs(series_actual - median_pred))
        axes[i].text(0.02, 0.98, f'MAPE: {mape:.2f}%, MAE: {mae:.2f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}_prediction_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default="deepar")
parser.add_argument('--dataset', type=str, default="exchange_rate_nips")
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--hidden_size', type=int, default=40)
parser.add_argument('--prediction_horizon', type=int, default=24)
parser.add_argument('--num_pred_rolling', type=int, default=7)
parser.add_argument('--batch_cov_horizon', type=int, default=None)
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--loss', type=str, default="kernel")  # ar, kernel
parser.add_argument('--num_mixture_r', type=int, default=4)
parser.add_argument('--delta_l', type=float, default=1.0)
parser.add_argument('--train_l', action='store_true')
parser.add_argument('--use_garch', action='store_true')
parser.add_argument('--direct_inference', action='store_true')

parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--reg_w', type=float, default=2.5)
parser.add_argument('--loss_lr_w', type=float, default=1.0)
parser.add_argument('--loss_lr', type=float, default=1e-03)
parser.add_argument('--garch_alpha_init', type=float, default=1e-03)
parser.add_argument('--garch_beta_init', type=float, default=1e-03)
parser.add_argument('--loss_wd', type=float, default=1e-04)
parser.add_argument('--hidden_proj_dim', type=int, default=0,
                    help='Revision 1 state-aware diffusion: dim of projected hidden '
                         'features (0 = disabled). Only used with --loss kernel.')
args = parser.parse_args()
pl.seed_everything(args.seed)

with open('./datasets/pred_horizon_dict_v1.pkl', 'rb') as f:
    pred_horizon_dict = pickle.load(f)
with open('./datasets/pred_rolling_dict_v1.pkl', 'rb') as f:
    pred_rolling_dict = pickle.load(f)
with open('./datasets/dataset_freq_v1.pkl', 'rb') as f:
    dataset_freq_dict = pickle.load(f)

default_rolling = {'B':5, '30min':56, 'M':1, 'W':3, '5min':56, 'D':5, 'Q':1, 'H':7, 'Y':1}

args.prediction_horizon = pred_horizon_dict[args.dataset]
args.num_pred_rolling = pred_rolling_dict[args.dataset] if args.dataset in pred_rolling_dict.keys() else default_rolling[dataset_freq_dict[args.dataset]]
args.batch_cov_horizon = args.prediction_horizon if args.batch_cov_horizon is None else args.batch_cov_horizon
args.direct_inference = False
f = open("./config/config.yaml")
configs = yaml.load(f, Loader=yaml.Loader)
f.close()

args.hidden_size = configs[args.model]['dataset'][args.dataset]['H']
args.lr = float(configs[args.model]['dataset'][args.dataset]['lr'])
args.loss_lr = args.lr*args.loss_lr_w
args.loss_wd = float(configs['train']['weight_decay'])

video_predictions = True
# args.direct_inference = True
# args.checkpoint = "PATH/logs/gpt/pems03_flow_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False/checkpoints/epoch=41-val_loss=86.80.ckpt"
def main():
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Traffic data
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ##################################### Load Data ###################################
    data = pd.read_csv("./datasets/%s.csv"%(args.dataset))
    if dataset_freq_dict[args.dataset] in ['30min', '5min', 'H', 'T']:
        # data['datetime'] = pd.to_datetime(data['datetime'])
        # if args.dataset =='pems03_flow':
        #     data["datetime"] = pd.date_range(start='2012-01-01', periods=len(data['datetime']), freq='5min')
        # sanity check of dates
        # np.timedelta64(1, "D")
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["tod"] =(data["datetime"].values - data["datetime"].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        
        data['dow'] = data['datetime'].dt.weekday
        time_varying_known_cats = ['tod', 'dow']
        data = data.astype(dict(sensor=str, tod=str, dow=str))
        if dataset_freq_dict[args.dataset] == 'H':
            lags = {"value": [24, 168]}
        else:
            lags = {"value": [2, 4, 12, 24, 48]}
    elif dataset_freq_dict[args.dataset] in ['B', 'D']:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['dow'] = data['datetime'].dt.weekday
        time_varying_known_cats = ['dow']
        data = data.astype(dict(sensor=str, dow=str))
        lags = {"value": [7, 14]}
    else:
        time_varying_known_cats = []
        data = data.astype(dict(sensor=str))
        lags = {}

    ################################## Create Dataloaders ##################################
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

    validation = TimeSeriesDataSet.from_dataset(training, data[lambda x: x.time_idx <= validation_cutoff], min_prediction_idx=training_cutoff+1)
    testing = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=validation_cutoff + 1)

    train_dataloader = training.to_dataloader(
        train=True, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )
    test_dataloader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0, batch_sampler="synchronized"
    )

    ################################## Initialize Model ##################################
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}', save_top_k=1, monitor="val_loss", mode="min")
    if args.dataset == "pems03_flow":
        static_graph = load_static_graph("PATH/datasets/PEMS03_graph.csv", 358)
    elif args.dataset == "brussels":
        static_graph = load_static_graph_from_pickle("PATH/exps/adjacency_matrix.pkl")
    else:
        static_graph = None
    if args.loss == 'kernel':
        loss = BatchMGD_Kernel(
            D=args.batch_cov_horizon, 
            K_r=args.num_mixture_r, 
            delta_l=args.delta_l, 
            train_l=args.train_l, 
            lr=args.loss_lr, 
            wd=args.loss_wd, reg_w=args.reg_w)
        # loss = BatchMGDGraph_Kernel(
        #     D=args.batch_cov_horizon, 
        #     K_r=args.num_mixture_r, 
        #     delta_l=args.delta_l, 
        #     train_l=args.train_l, 
        #     lr=args.loss_lr, 
        #     wd=args.loss_wd, 
        #     reg_w=args.reg_w,
        #     static=False,
        #     static_graph=load_static_graph("PATH/datasets/PEMS03_graph.csv", 358)
        # )
        loss = BatchMGDDiffusion_Kernel(
            D=args.batch_cov_horizon,
            K_r=args.num_mixture_r,
            delta_l=args.delta_l,
            train_l=args.train_l,
            lr=args.loss_lr,
            wd=args.loss_wd,
            reg_w=args.reg_w,
            static_graph=static_graph,
            hidden_proj_dim=args.hidden_proj_dim,
            use_garch=args.use_garch,
            garch_alpha_init=args.garch_alpha_init,
            garch_beta_init=args.garch_beta_init,
        )
        # loss = GraphBatchMGD_Kernel(D=args.batch_cov_horizon, K_r=args.num_mixture_r, delta_l=args.delta_l, train_l=args.train_l, lr=args.loss_lr, wd=args.loss_wd, reg_w=args.reg_w)
        log_file = "%s_batch_%s_B%s_Q%s_H%s_D%s_Class_%s_Kr%s_DeltaL%s_LossLRw%s_RegW%s_TrainL_%s"%(args.dataset, args.loss, args.batch_size, args.prediction_horizon, args.hidden_size, args.batch_cov_horizon, loss.__class__.__name__+f"_use_garch_{args.use_garch}", args.num_mixture_r, args.delta_l, args.loss_lr_w, args.reg_w, args.train_l)
        logger = TensorBoardLogger(save_dir="logs", name=args.model, version=log_file)
    elif args.loss == 'curvature':
        log_file = "student_%s_batch_%s_B%s_Q%s_H%s_D%s_Kr%s_DeltaL%s_LossLRw%s_RegW%s_TrainL_%s"%(args.dataset, args.loss, args.batch_size, args.prediction_horizon, args.hidden_size, args.batch_cov_horizon, args.num_mixture_r, args.delta_l, args.loss_lr_w, args.reg_w, args.train_l)
        logger = TensorBoardLogger(save_dir="logs", name=args.model, version=log_file)
        loss = BatchMGDCurvature_Kernel(
            D=args.batch_cov_horizon,
            K_r=args.num_mixture_r,
            delta_l=args.delta_l,
            train_l=args.train_l,
            lr=args.loss_lr,
            wd=args.loss_wd,
            reg_w=args.reg_w,
            static=False,
            static_graph=static_graph,
        )
    elif args.loss == 'learnable':
        log_file = "%s_batch_%s_B%s_Q%s_H%s_D%s_Kr%s_DeltaL%s_LossLRw%s_RegW%s_TrainL_%s"%(args.dataset, args.loss, args.batch_size, args.prediction_horizon, args.hidden_size, args.batch_cov_horizon, args.num_mixture_r, args.delta_l, args.loss_lr_w, args.reg_w, args.train_l)
        logger = TensorBoardLogger(save_dir="logs", name=args.model, version=log_file)
        loss = BatchMGDLearnable_Kernel(
            D=args.batch_cov_horizon,
            K_r=args.num_mixture_r,
            delta_l=args.delta_l,
            train_l=args.train_l,
            lr=args.loss_lr,
            wd=args.loss_wd,
            reg_w=args.reg_w,
            static=False,
        )
    elif args.loss == 'ar':
        log_file = "%s_batch_%s_B%s_Q%s_H%s_D%s_AR%s_L2Reg%s"%(args.dataset, args.loss, args.batch_size, args.prediction_horizon, args.hidden_size, args.batch_cov_horizon, args.num_mixture_r-1, args.reg_w)
        logger = TensorBoardLogger(save_dir="logs", name=args.model, version=log_file)
        loss = BatchMGD_AR(D=args.batch_cov_horizon, K_r=args.num_mixture_r, reg_w=args.reg_w)
    else:
        raise ValueError("Not supported parameterization for Cov")
    
    trainer = pl.Trainer(
        logger=logger,
        max_steps=configs['train']['max_steps'],
        accelerator='gpu',
        max_epochs=40,
        devices=[0],
        enable_model_summary=True,
        gradient_clip_val=configs['train']['gradient_clip_val'],
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_train_batches=configs['train']['limit_train_batches'],
        enable_checkpointing=True,
        accumulate_grad_batches=16,
    )

    if args.model == "deepar":
        net = BatchDeepAREstimator.from_dataset(
            training,
            cell_type=configs[args.model]['cell_type'],
            hidden_size=args.hidden_size,
            rnn_layers=configs[args.model]['rnn_layers'],
            dropout=configs['train']['dropout'],
            loss=loss,
            optimizer="adam",
            learning_rate=args.lr,
            weight_decay=float(configs['train']['weight_decay']),
            reduce_on_plateau_patience=configs['train']['reduce_on_plateau_patience']
        )
    elif args.model == "gpt":
        net = BatchGPTEstimator.from_dataset(
            training,
            n_heads=configs[args.model]['n_heads'],
            hidden_size=args.hidden_size,
            rnn_layers=configs[args.model]['rnn_layers'],
            dropout=configs['train']['dropout'],
            loss=loss,
            optimizer="adam",
            learning_rate=args.lr,
            weight_decay=float(configs['train']['weight_decay']),
            reduce_on_plateau_patience=configs['train']['reduce_on_plateau_patience']
        )
    elif args.model == "xLSTM":
        net = BatchxLSTMEstimator.from_dataset(
            training,
            hidden_size=args.hidden_size,
            rnn_layers=configs[args.model]['rnn_layers'],
            dropout=configs['train']['dropout'],
            loss=loss,
            optimizer="adam",
            learning_rate=args.lr,
            weight_decay=float(configs['train']['weight_decay']),
            reduce_on_plateau_patience=configs['train']['reduce_on_plateau_patience']
        )

    if not args.direct_inference:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    ################################## Evaluate Model ##################################
    if args.model == "deepar":
        best_model = BatchDeepARPredictor.load_from_checkpoint(checkpoint_callback.best_model_path, map_location="cuda:%s"%(args.device)).to("cuda:%s"%(args.device))
    elif args.model == "gpt":
        if not args.direct_inference:
            best_model = BatchGPTPredictor.load_from_checkpoint(checkpoint_callback.best_model_path, map_location="cuda:%s"%(args.device)).to("cuda:%s"%(args.device))
        else:
            best_model = BatchGPTEstimator.load_from_checkpoint(args.checkpoint, map_location="cuda:%s"%(args.device)).to("cuda:%s"%(args.device))
    elif args.model == "xLSTM":
        if not args.direct_inference:
            best_model = BatchxLSTMPredictor.load_from_checkpoint(checkpoint_callback.best_model_path, map_location="cuda:%s"%(args.device)).to("cuda:%s"%(args.device))
        else:
            best_model = BatchxLSTMEstimator.load_from_checkpoint(args.checkpoint, map_location="cuda:%s"%(args.device)).to("cuda:%s"%(args.device))
    best_model.wReg = True

    metrics = []
    crps_mean_noagg_all, crps_noagg_all, crps_sum_noagg_all = [], [], []
    all_predictions = []
    all_pairs = [[], []]
    covariance_matrices = {'sigma_D': [], 'C_t': [], 'G_t': []}
    captured = {"L": [], "d": [], "w": [], "h": []}
    for i in range(args.num_repeat):
        (raw_predictions, x), param = best_model.predict(test_dataloader, mode="raw", n_samples=100, show_progress_bar=True, return_x=True)
        preds = raw_predictions['prediction'].cpu()
        actuals = x['decoder_target']
        # print("weights shape:", weights.shape)
        # Store x data for visualization (only need to store once)
        if i == 0:
            all_x_data = x
        
        # Extract covariance matrices if available
        if hasattr(best_model, 'loss') and hasattr(best_model.loss, 'get_covariance_matrices'):
            try:
                cov_mats = best_model.loss.get_covariance_matrices()
                if 'sigma_D' in cov_mats:
                    covariance_matrices['sigma_D'].append(cov_mats['sigma_D'].cpu())
                if 'C_t' in cov_mats:
                    covariance_matrices['C_t'].append(cov_mats['C_t'].cpu())
                if 'G_t' in cov_mats:
                    covariance_matrices['G_t'].append(cov_mats['G_t'].cpu())
            except Exception as e:
                print(f"Warning: Could not extract covariance matrices: {e}")

        try:
            preds = preds.reshape(args.num_pred_rolling, preds.shape[0]//args.num_pred_rolling, preds.shape[1], preds.shape[2])
            actuals = actuals.reshape(args.num_pred_rolling, actuals.shape[0]//args.num_pred_rolling, actuals.shape[1])
        except:
            preds = preds.unsqueeze(0)
            actuals = actuals.unsqueeze(0)

        all_predictions.append(preds)
        all_pairs[0].append(preds)
        all_pairs[1].append(actuals)

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

    if not os.path.isdir("./metrics_raw/%s"%(args.model)):
        os.makedirs("./metrics_raw/%s"%(args.model))

    torch.save(crps_mean_noagg, './metrics_raw/%s/%s_Reg_%s_crps_mean.pt'%(args.model, log_file, best_model.wReg))
    torch.save(crps_noagg, './metrics_raw/%s/%s_Reg_%s_crps.pt'%(args.model, log_file, best_model.wReg))
    torch.save(crps_sum_noagg, './metrics_raw/%s/%s_Reg_%s_crps_sum.pt'%(args.model, log_file, best_model.wReg))

    if not os.path.isdir("./metrics/%s"%(args.model)):
        os.makedirs("./metrics/%s"%(args.model))

    with open('./metrics/%s/%s_Reg_%s.txt'%(args.model, log_file, best_model.wReg), 'w') as f:
        for i in range(metrics.shape[0]):
            if i != metrics.shape[0]-1:
                f.write('& %.4f$\pm$%.4f'%(metrics[i, 0], metrics[i, 1]))
            else:
                f.write('& %.4f$\pm$%.4f \n'%(metrics[i, 0], metrics[i, 1]))

    # Create visualizations directory
    viz_dir = "./visualizations/%s" % args.model
    if not os.path.isdir(viz_dir):
        os.makedirs(viz_dir)
    
    viz_base_path = os.path.join(viz_dir, f'{log_file}_Reg_{best_model.wReg}')
    if video_predictions is not None:
        # visualize_prediction_video(all_pairs, viz_base_path + '_prediction_video.mp4', fps=1)
        visualize_prediction_images(0, all_pairs, viz_base_path + '_prediction_images')
        visualize_prediction_images(1, all_pairs, viz_base_path + '_actual_images')
    # # Visualize covariance matrices
    # for cov_name, cov_list in covariance_matrices.items():
    #     if cov_list:
    #         # Stack all covariance matrices from different prediction rounds
    #         cov_tensor = torch.stack(cov_list)  # [num_repeat, batch, time, features, features]
    #         # visualize_covariance_matrices(cov_tensor, viz_base_path, cov_name)
            
    #         # Plot Sigma_bat using the vis.py function
    #         if cov_name == 'sigma_D':
    #             # Take median across repeats and select a representative time step
    #             cov_median = torch.median(cov_tensor, dim=0)[0]  # [batch, time, features, features]
    #             # Select middle time step for visualization
    #             t_idx = cov_median.shape[1] // 2
    #             Sigma_bat = cov_median[:, t_idx, :, :]  # [batch, features, features]
    #             # Reshape if needed - Sigma_bat should be (D*B x D*B)
    #             if len(Sigma_bat.shape) == 3:
    #                 # Average across batch for a single representative matrix
    #                 Sigma_bat = Sigma_bat.mean(dim=0)  # [features, features]
    

    # Visualize prediction intervals
    # if all_predictions:
    #     predictions_tensor = torch.stack(all_predictions)  # [num_repeat, num_rolling, batch, time, features]
    #     visualize_prediction_intervals(predictions_tensor, actuals, viz_base_path, x_data=all_x_data)

    # ---- Curvature-specific manuscript visualizations ----
    if (args.loss == 'curvature' or args.loss == 'learnable') and hasattr(best_model, 'loss'):
        loss_mod = best_model.loss

        # Fig 1: κ → b_ij → W' → G_t pipeline
        try:
            visualize_curvature_precision_effectiveness(
                loss_module=loss_mod,
                save_path=viz_base_path + '_curvature_effectiveness.png',
            )
            print("Saved: curvature_effectiveness")
        except Exception as e:
            print(f"[vis] visualize_curvature_precision_effectiveness failed: {e}")

        # Fig 2: G_t ablation across model variants.
        # At single-checkpoint inference, loss_mod is passed for all three roles so
        # the figure still renders; swap in trained kernel/graph checkpoints for a
        # proper ablation comparison.
        try:
            visualize_ablation_Gt_vs_IR(
                loss_curvature=loss_mod,
                loss_kernel=loss_mod,   # replace with loaded BatchMGD_Kernel for true ablation
                loss_graph=loss_mod,    # replace with loaded BatchMGDGraph_Kernel for true ablation
                save_path=viz_base_path + '_ablation_Gt_vs_IR.png',
            )
            print("Saved: ablation_Gt_vs_IR")
        except Exception as e:
            print(f"[vis] visualize_ablation_Gt_vs_IR failed: {e}")

        # Fig 3: C_t ⊗ G_t Kronecker structure
        try:
            visualize_batch_covariance_Sigma(
                loss_module=loss_mod,
                sample_pred=None,
                save_path=viz_base_path + '_batch_covariance_Sigma.png',
            )
            print("Saved: batch_covariance_Sigma")
        except Exception as e:
            print(f"[vis] visualize_batch_covariance_Sigma failed: {e}")

        # Fig 4: Converged curvature parameter snapshot across test batches
        try:
            track_curvature_params_during_training(
                loss_module=loss_mod,
                optimizer=None,
                dataloader=test_dataloader,
                n_steps=min(50, len(test_dataloader)),
                save_path=viz_base_path + '_curvature_param_snapshot.png',
            )
            print("Saved: curvature_param_snapshot")
        except Exception as e:
            print(f"[vis] track_curvature_params_during_training failed: {e}")

        # Table 1: Quantitative effectiveness metrics
        try:
            metrics = compute_curvature_effectiveness_metrics(loss_mod)
            # Save metrics to JSON
            import json
            with open(viz_base_path + '_curvature_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            print("Saved: curvature_metrics.json")
        except Exception as e:
            print(f"[vis] compute_curvature_effectiveness_metrics failed: {e}")

    print(f"Visualizations saved to {viz_dir}")

    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    if args.batch_cov_horizon <= args.prediction_horizon:
        score = main()
