"""
Spatial Error Propagation Visualization across Graph Nodes
==========================================================

Visualizes how forecast errors propagate spatially through the traffic sensor
network.  Given model predictions you can either:

  (a) Load a trained BatchDeepAR / BatchGPT / BatchxLSTM checkpoint and
      generate predictions on the test set, or
  (b) Supply precomputed prediction / actual tensors (.pt files).

Outputs
-------
1. **Spatial error heatmap** – node-level MAE / RMSE on the graph layout
   at individual forecast horizons (animated or panel).
2. **Error correlation vs. graph distance** – Pearson ρ between node errors
   as a function of shortest-path hops.
3. **Multi-hop error propagation** – given a "seed" node with high error,
   show how error magnitude decays over k-hop neighbours.
4. **Lagged spatial cross-correlation** – cross-node error correlation at
   temporal lags Δ=0,1,2,…, conditioned on graph distance.

Usage (from repo root)
----------------------
    # From a checkpoint:
    python ./src/vis_spatial_error_propagation.py \\
        --checkpoint logs/gpt/pems03_flow_.../checkpoints/epoch=41-val_loss=86.80.ckpt \\
        --model gpt --dataset pems03_flow --n_samples 100

    # From saved tensors:
    python ./src/vis_spatial_error_propagation.py \\
        --preds_path saved_preds.pt --actuals_path saved_actuals.pt \\
        --dataset pems03_flow
"""

import warnings

from pytorch_forecasting.models.deepar import DeepAR
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
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────── CLI ────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Spatial error propagation visualisation on PEMS03 graph",
)
# --- data / model ---
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a BatchDeepAR/GPT/xLSTM checkpoint")
parser.add_argument("--model", type=str, default="gpt",
                    choices=["deepar", "gpt", "xLSTM"])
parser.add_argument("--loss", type=str, default="curvature",
                    choices=["kernel", "curvature", "ar", "learnable"])
parser.add_argument("--dataset", type=str, default="pems03_flow")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--num_repeat", type=int, default=1)
# --- pre-saved tensors (skip checkpoint) ---
parser.add_argument("--preds_path", type=str, default=None,
                    help=".pt file with predictions (B, Q, S)")
parser.add_argument("--actuals_path", type=str, default=None,
                    help=".pt file with actuals (B, Q)")
# --- graph ---
parser.add_argument("--graph_csv", type=str,
                    default="./datasets/PEMS03_graph.csv")
# --- vis options ---
parser.add_argument("--max_hops", type=int, default=6,
                    help="Max hops for propagation and distance analysis")
parser.add_argument("--seed_topk", type=int, default=5,
                    help="Number of seed nodes to show propagation for")
parser.add_argument("--horizons", type=str, default="0,3,6,11",
                    help="Comma-separated forecast-step indices for panel plots")
parser.add_argument("--out_dir", type=str,
                    default="visualizations/spatial_error_propagation")

args = parser.parse_args()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

os.makedirs(args.out_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Load & build graph
# ═══════════════════════════════════════════════════════════════════════════

def load_graph(csv_path: str):
    """
    Load PEMS03 graph CSV → adjacency matrix + NetworkX graph.

    Returns
    -------
    adj : np.ndarray (N, N)   weighted adjacency
    G   : networkx.Graph
    node_ids : list[int]       sorted original PEMS sensor IDs
    id2idx : dict              sensor-ID → 0-based index
    dist_mat : np.ndarray      shortest-path hop count (N, N)
    """
    df = pd.read_csv(csv_path, skip_blank_lines=True).dropna()
    # Strip whitespace from column names (Windows \r artefacts)
    df.columns = [c.strip() for c in df.columns]
    df["from"] = df["from"].astype(int)
    df["to"] = df["to"].astype(int)
    df["distance"] = df["distance"].astype(float)

    node_ids = sorted(set(df["from"].tolist() + df["to"].tolist()))
    N = len(node_ids)
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    adj = np.zeros((N, N), dtype=np.float32)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for _, row in df.iterrows():
        i, j = id2idx[int(row["from"])], id2idx[int(row["to"])]
        w = float(row["distance"])
        # Use inverse distance as weight (closer → stronger coupling)
        adj[i, j] = w
        adj[j, i] = w
        G.add_edge(i, j, weight=w)

    # Shortest-path in hops (unweighted)
    sp_binary = csr_matrix((adj > 0).astype(np.float32))
    dist_mat = shortest_path(sp_binary, directed=False, unweighted=True)
    dist_mat[np.isinf(dist_mat)] = N  # disconnected → large

    return adj, G, node_ids, id2idx, dist_mat


print("Loading graph …")
adj, G, node_ids, id2idx, dist_mat = load_graph(args.graph_csv)
N = len(node_ids)
print(f"  {N} nodes, {int((adj > 0).sum())} directed edges")


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Obtain predictions & actuals
# ═══════════════════════════════════════════════════════════════════════════

def load_predictions_from_checkpoint():
    """Run inference from a saved checkpoint – mirrors train_batch.py logic."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
    from batched_model import (BatchDeepAREstimator, BatchDeepARPredictor,
                               BatchGPTEstimator, BatchGPTPredictor,
                               BatchxLSTMEstimator, BatchxLSTMPredictor)

    with open("./datasets/pred_horizon_dict.pkl", "rb") as f:
        pred_horizon_dict = pickle.load(f)
    with open("./datasets/pred_rolling_dict.pkl", "rb") as f:
        pred_rolling_dict = pickle.load(f)
    with open("./datasets/dataset_freq.pkl", "rb") as f:
        dataset_freq_dict = pickle.load(f)

    default_rolling = {"B": 5, "30min": 56, "M": 1, "W": 3,
                       "5min": 56, "D": 5, "Q": 1, "H": 7, "Y": 1}
    freq = dataset_freq_dict[args.dataset]
    pred_horizon = pred_horizon_dict[args.dataset]
    num_rolling = pred_rolling_dict.get(args.dataset, default_rolling[freq])

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
    training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx", target="value",
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
        training, data, min_prediction_idx=validation_cutoff + 1,
    )
    test_loader = testing.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0,
        batch_sampler="synchronized",
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    # Determine model class from --model flag
    if args.model == "deepar":
        ModelClass = DeepAR
    elif args.model == "gpt":
        ModelClass = BatchGPTPredictor
    elif args.model == "xLSTM":
        ModelClass = BatchxLSTMPredictor
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = ModelClass.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).to(device)
    model.wReg = True
    model.eval()

    all_preds, all_actuals = [], []
    for _ in range(args.num_repeat):
        with torch.no_grad():
            (raw_predictions, x), _ = model.predict(
                test_loader, mode="raw", n_samples=args.n_samples,
                show_progress_bar=True, return_x=True,
            )
        # Handle different output formats from predict()
        # raw_predictions could be: dict, list, Output object, or tensor
        if isinstance(raw_predictions, dict):
            preds = raw_predictions["prediction"].cpu().float()     # (B_total, Q, S)
        elif isinstance(raw_predictions, list):
            preds = raw_predictions[0].cpu().float()                # (B_total, Q, S)
        elif hasattr(raw_predictions, 'prediction'):
            # Named tuple or Output object with 'prediction' attribute
            preds = raw_predictions.prediction.cpu().float()        # (B_total, Q, S)
        elif hasattr(raw_predictions, '__getitem__') and hasattr(raw_predictions, '_fields'):
            # Named tuple - try to get 'prediction' field
            preds = raw_predictions[raw_predictions._fields.index('prediction')].cpu().float()
        else:
            preds = raw_predictions.cpu().float()                   # (B_total, Q, S)

        actuals = x["decoder_target"].cpu().float()             # (B_total, Q)

        try:
            R = preds.shape[0] // N
            preds = preds.reshape(R, N, preds.shape[1], preds.shape[2])
            actuals = actuals.reshape(R, N, actuals.shape[1])
        except Exception:
            preds = preds.unsqueeze(0)
            actuals = actuals.unsqueeze(0)

        all_preds.append(preds)
        all_actuals.append(actuals)

    preds = torch.cat(all_preds, dim=0)      # (R_total, N, Q, S)
    actuals = torch.cat(all_actuals, dim=0)   # (R_total, N, Q)
    return preds, actuals, pred_horizon


def load_predictions_from_files():
    preds = torch.load(args.preds_path, map_location="cpu").float()
    actuals = torch.load(args.actuals_path, map_location="cpu").float()
    Q = preds.shape[-2] if preds.dim() >= 3 else preds.shape[-1]
    # Ensure (R, N, Q, S) shape
    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
        actuals = actuals.unsqueeze(0)
    return preds, actuals, Q


if args.preds_path and args.actuals_path:
    print("Loading precomputed predictions …")
    preds, actuals, Q = load_predictions_from_files()
elif args.checkpoint:
    preds, actuals, Q = load_predictions_from_checkpoint()
else:
    print("ERROR: Supply either --checkpoint or --preds_path / --actuals_path")
    sys.exit(1)

R = preds.shape[0]
assert preds.shape[1] == N, (
    f"Number of nodes in predictions ({preds.shape[1]}) ≠ graph ({N})")
Q = preds.shape[2]
S = preds.shape[3]
print(f"Predictions: R={R} rolling windows, N={N} nodes, Q={Q} horizons, S={S} samples")


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Compute error statistics
# ═══════════════════════════════════════════════════════════════════════════

# Prediction mean per node per horizon
pred_mean = preds.mean(dim=-1)          # (R, N, Q)
# Residuals: predicted mean - actual
residuals = pred_mean - actuals         # (R, N, Q)
abs_errors = residuals.abs()            # (R, N, Q)

# Per-node, per-horizon MAE (averaged over rolling windows)
node_mae = abs_errors.mean(dim=0).numpy()            # (N, Q)
# Per-node MAE across all horizons
node_mae_overall = node_mae.mean(axis=1)              # (N,)
# Per-node RMSE across rolling windows
node_rmse = (residuals ** 2).mean(dim=0).sqrt().numpy()  # (N, Q)

# Normalised node error for coloring (0–1 range per horizon)
def minmax(arr, axis=None):
    lo, hi = arr.min(axis=axis, keepdims=True), arr.max(axis=axis, keepdims=True)
    return np.where(hi - lo > 1e-9, (arr - lo) / (hi - lo), 0.5)

node_mae_norm = minmax(node_mae, axis=0)              # (N, Q)


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Graph layout (spring layout, cached per session)
# ═══════════════════════════════════════════════════════════════════════════

print("Computing graph layout …")
pos = nx.spring_layout(G, seed=42, weight="weight", iterations=200, k=1.5/np.sqrt(N))
node_xy = np.array([pos[i] for i in range(N)])   # (N, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  5.  FIGURE 1 – Spatial error heatmap at selected horizons
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_heatmap_panels(horizons, metric, metric_name, fname):
    """
    Draw the graph with nodes colored by forecast error at each horizon.
    """
    n_panels = len(horizons)
    fig = plt.figure(figsize=(5.5 * n_panels + 0.6, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(1, n_panels + 1, width_ratios=[1] * n_panels + [0.05], wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cax = fig.add_subplot(gs[0, -1])

    vmin, vmax = metric[:, horizons].min(), metric[:, horizons].max()
    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ax, h in zip(axes, horizons):
        vals = metric[:, h]

        # Draw edges
        edge_segments = []
        for u, v in G.edges():
            edge_segments.append([node_xy[u], node_xy[v]])
        lc = LineCollection(edge_segments, colors="0.85", linewidths=0.4, zorder=1)
        ax.add_collection(lc)

        # Draw nodes
        sc = ax.scatter(
            node_xy[:, 0], node_xy[:, 1],
            c=vals, cmap=cmap, norm=norm,
            s=28, edgecolors="0.3", linewidths=0.3, zorder=2,
        )
        ax.set_title(f"$t + {h+1}$", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Shared colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(metric_name, fontsize=11)

    fig.suptitle(
        f"Spatial {metric_name} at selected forecast horizons — {args.dataset}",
        fontsize=13,
    )
    fig.savefig(os.path.join(args.out_dir, fname), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


horizon_indices = [int(h) for h in args.horizons.split(",")]
horizon_indices = [h for h in horizon_indices if h < Q]

print("\n■ Figure 1: Spatial error heatmap panels")
plot_error_heatmap_panels(horizon_indices, node_mae, "MAE", f"{args.model}BASE__fig1_spatial_mae_panels.png")
plot_error_heatmap_panels(horizon_indices, node_rmse, "RMSE", f"{args.model}BASE__fig1_spatial_rmse_panels.png")


# ═══════════════════════════════════════════════════════════════════════════
#  6.  FIGURE 2 – Error correlation vs. graph distance (hop count)
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_corr_vs_distance(max_hops, fname):
    """
    For each hop distance d=1..max_hops, compute the average Pearson
    correlation of absolute-error time series between all node pairs at
    that distance.
    """
    # node error time series: (N, R*Q) flattened over rolling × horizon
    err_ts = abs_errors.permute(1, 0, 2).reshape(N, -1).numpy()  # (N, T_eff)

    hop_corrs_mean = []
    hop_corrs_std = []
    hop_counts = []

    for d in range(0, max_hops + 1):
        pairs = np.argwhere(dist_mat == d)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]  # upper triangle
        if len(pairs) == 0:
            hop_corrs_mean.append(np.nan)
            hop_corrs_std.append(np.nan)
            hop_counts.append(0)
            continue
        corrs = []
        for i, j in pairs:
            r, _ = pearsonr(err_ts[i], err_ts[j])
            if not np.isnan(r):
                corrs.append(r)
        hop_corrs_mean.append(np.mean(corrs) if corrs else np.nan)
        hop_corrs_std.append(np.std(corrs) if corrs else np.nan)
        hop_counts.append(len(corrs))

    hops = np.arange(0, max_hops + 1)
    means = np.array(hop_corrs_mean)
    stds = np.array(hop_corrs_std)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    valid = ~np.isnan(means)

    ax1.errorbar(hops[valid], means[valid], yerr=stds[valid],
                 fmt="o-", color="#d62728", capsize=4, markersize=6, linewidth=1.8,
                 label=r"Pearson $\rho$ (mean ± std)")
    ax1.axhline(0, color="0.6", ls="--", lw=0.8)
    ax1.set_xlabel("Shortest-path distance (hops)", fontsize=12)
    ax1.set_ylabel(r"Error correlation  $\rho$", fontsize=12, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    # Secondary axis: number of pairs
    ax2 = ax1.twinx()
    ax2.bar(hops[valid], np.array(hop_counts)[valid],
            alpha=0.2, color="#1f77b4", width=0.6, label="# pairs")
    ax2.set_ylabel("Number of node pairs", fontsize=11, color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title(
        f"Forecast error correlation vs graph distance — {args.dataset}",
        fontsize=12,
    )
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, fname), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


print("\n■ Figure 2: Error correlation vs graph distance")
plot_error_corr_vs_distance(args.max_hops, f"{args.model}BASE__fig2_error_corr_vs_hops.png")


# ═══════════════════════════════════════════════════════════════════════════
#  7.  FIGURE 3 – Multi-hop error propagation from seed nodes
# ═══════════════════════════════════════════════════════════════════════════

def plot_multihop_propagation(seed_topk, max_hops, fname):
    """
    For the top-k highest-error nodes ("seeds"), plot how the average
    error of their k-hop neighbours decays with hop distance, for each
    forecast horizon.
    """
    # Identify seed nodes (highest overall MAE)
    seed_nodes = np.argsort(node_mae_overall)[-seed_topk:][::-1]

    # For each horizon pick a few representative ones
    sel_horizons = [0, Q // 4, Q // 2, Q - 1]
    sel_horizons = sorted(set([min(h, Q - 1) for h in sel_horizons]))

    fig, axes = plt.subplots(1, len(sel_horizons), figsize=(5.5 * len(sel_horizons), 4.5))
    if len(sel_horizons) == 1:
        axes = [axes]
    cmap_seeds = cm.get_cmap("tab10")

    for ax, h_idx in zip(axes, sel_horizons):
        for rank, seed in enumerate(seed_nodes):
            hop_errors = []
            for d in range(0, max_hops + 1):
                nbrs = np.where(dist_mat[seed] == d)[0]
                if len(nbrs) == 0:
                    hop_errors.append(np.nan)
                else:
                    hop_errors.append(node_mae[nbrs, h_idx].mean())
            ax.plot(range(max_hops + 1), hop_errors,
                    "o-", color=cmap_seeds(rank), markersize=5, linewidth=1.5,
                    label=f"Seed {seed} (MAE={node_mae_overall[seed]:.2f})")

        ax.set_xlabel("Hop distance from seed", fontsize=11)
        ax.set_ylabel("Mean neighbour MAE", fontsize=11)
        ax.set_title(f"Horizon $t+{h_idx+1}$", fontsize=12)
        ax.set_xticks(range(max_hops + 1))
        if h_idx == sel_horizons[0]:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Error propagation from top-{seed_topk} seed nodes — {args.dataset}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, fname), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


print("\n■ Figure 3: Multi-hop error propagation from seed nodes")
plot_multihop_propagation(args.seed_topk, args.max_hops,
                          f"{args.model}BASE__fig3_multihop_propagation.png")


# ═══════════════════════════════════════════════════════════════════════════
#  8.  FIGURE 4 – Temporal-lag × spatial-distance error correlation
# ═══════════════════════════════════════════════════════════════════════════

def plot_lagged_spatial_correlation(max_lags=4, max_hops_plot=5, fname="fig4_lag_distance_corr.png"):
    """
    2-D heatmap: (temporal lag Δ) × (graph distance d) → mean cross-node
    error correlation.  Shows how error signal dissipates jointly in time
    and space.
    """
    # residuals shape: (R, N, Q)
    res = residuals.numpy()  # (R, N, Q)

    max_lags = min(max_lags, Q - 1)
    corr_grid = np.full((max_lags + 1, max_hops_plot + 1), np.nan)

    # Flatten rolling windows: (N, R*Q)
    for lag in range(0, max_lags + 1):
        # For each node pair, compute corr between node i at t and node j at t+lag
        # err_i: (R, N, Q-lag)  err_j: (R, N, Q-lag)
        if lag == 0:
            err_i = res[:, :, :]
            err_j = res[:, :, :]
        else:
            err_i = res[:, :, :-lag]
            err_j = res[:, :, lag:]

        # flatten to (N, T_eff)
        T_eff = err_i.shape[0] * err_i.shape[2]
        err_i_flat = err_i.transpose(1, 0, 2).reshape(N, T_eff)
        err_j_flat = err_j.transpose(1, 0, 2).reshape(N, T_eff)

        for d in range(0, max_hops_plot + 1):
            pairs = np.argwhere(dist_mat == d)
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]
            if len(pairs) == 0:
                continue
            # Sub-sample if too many pairs (for speed)
            if len(pairs) > 2000:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(len(pairs), 2000, replace=False)
                pairs = pairs[idx]
            corrs = []
            for i, j in pairs:
                r, _ = pearsonr(err_i_flat[i], err_j_flat[j])
                if not np.isnan(r):
                    corrs.append(r)
            if corrs:
                corr_grid[lag, d] = np.mean(corrs)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        corr_grid, cmap="RdBu_r", vmin=-0.3, vmax=0.8,
        origin="lower", aspect="auto", interpolation="nearest",
    )
    ax.set_xlabel("Graph distance (hops)", fontsize=12)
    ax.set_ylabel("Temporal lag Δ", fontsize=12)
    ax.set_xticks(range(max_hops_plot + 1))
    ax.set_yticks(range(max_lags + 1))
    ax.set_title(
        f"Error cross-correlation: temporal lag × spatial distance — {args.dataset}",
        fontsize=12,
    )

    # Annotate cells
    for lag in range(corr_grid.shape[0]):
        for d in range(corr_grid.shape[1]):
            val = corr_grid[lag, d]
            if not np.isnan(val):
                ax.text(d, lag, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="k" if abs(val) < 0.4 else "w")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    fig.colorbar(im, cax=cax, label=r"Pearson $\rho$")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, fname), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


print("\n■ Figure 4: Temporal-lag × spatial-distance error correlation")
plot_lagged_spatial_correlation(max_lags=min(4, Q - 1),
                                max_hops_plot=args.max_hops,
                                fname=f"{args.model}BASE__fig4_lag_distance_corr.png")


# ═══════════════════════════════════════════════════════════════════════════
#  9.  FIGURE 5 – Per-node error on graph with directed error-flow arrows
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_flow_graph(horizon_idx, fname):
    """
    For a single forecast horizon, draw the graph with:
      - node colour ∝ MAE
      - directed arrow from i→j whenever error at i is substantially larger
        than at j
        (shows the direction error "flows to" in the spatial sense)
    """
    vals = node_mae[:, horizon_idx]
    cmap = cm.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())

    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw edges with error-flow arrows
    threshold = np.median(vals) * 0.15     # arrow when difference > threshold
    for u, v in G.edges():
        diff = vals[u] - vals[v]
        if abs(diff) < threshold:
            # Undirected thin edge
            ax.plot(
                [node_xy[u, 0], node_xy[v, 0]],
                [node_xy[u, 1], node_xy[v, 1]],
                "-", color="0.85", lw=0.4, zorder=1,
            )
        else:
            # Arrow from high-error → low-error (propagation direction)
            src, dst = (u, v) if diff > 0 else (v, u)
            arrow_color = cmap(norm(max(vals[src], vals[dst])))
            ax.annotate(
                "", xy=node_xy[dst], xytext=node_xy[src],
                arrowprops=dict(
                    arrowstyle="->,head_width=0.15,head_length=0.1",
                    color=arrow_color, lw=0.8, alpha=0.7,
                ),
                zorder=1,
            )

    # Draw nodes
    sc = ax.scatter(
        node_xy[:, 0], node_xy[:, 1],
        c=vals, cmap=cmap, norm=norm,
        s=40, edgecolors="0.3", linewidths=0.4, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("MAE", fontsize=11)

    ax.set_title(
        f"Spatial error flow at horizon $t+{horizon_idx+1}$ — {args.dataset}",
        fontsize=13,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, fname), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


# Pick first and last available horizons
print("\n■ Figure 5: Error-flow directed graph")
for h in [horizon_indices[0], horizon_indices[-1]]:
    if h < Q:
        plot_error_flow_graph(h, f"{args.model}BASE__fig5_error_flow_h{h+1}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 10.  FIGURE 6 – Summary: node-level statistics table & histogram
# ═══════════════════════════════════════════════════════════════════════════

def plot_node_error_distribution(fname):
    """Histogram of per-node overall MAE with graph-distance overlay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram of node MAE
    ax1.hist(node_mae_overall, bins=30, color="#2ca02c", edgecolor="0.3",
             alpha=0.7)
    ax1.axvline(np.median(node_mae_overall), color="#d62728", ls="--", lw=1.5,
                label=f"Median = {np.median(node_mae_overall):.3f}")
    ax1.set_xlabel("Node-level MAE (avg over horizons)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of per-node MAE", fontsize=12)
    ax1.legend(fontsize=10)

    # Scatter: node degree vs MAE
    degrees = np.array([G.degree(n) for n in range(N)])
    ax2.scatter(degrees, node_mae_overall, s=18, alpha=0.6, c="#1f77b4",
                edgecolors="0.4", linewidths=0.3)
    # Trend line
    z = np.polyfit(degrees, node_mae_overall, 1)
    x_fit = np.linspace(degrees.min(), degrees.max(), 50)
    ax2.plot(x_fit, np.polyval(z, x_fit), "r--", lw=1.4,
             label=f"Linear fit (slope={z[0]:.4f})")
    ax2.set_xlabel("Node degree", fontsize=11)
    ax2.set_ylabel("Node MAE", fontsize=11)
    ax2.set_title("Node degree vs forecast error", fontsize=12)
    ax2.legend(fontsize=10)

    fig.suptitle(f"Node-level error statistics — {args.dataset}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, fname), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


print("\n■ Figure 6: Node error distribution & degree analysis")
plot_node_error_distribution(f"{args.model}BASE__fig6_node_error_stats.png")


# ═══════════════════════════════════════════════════════════════════════════
# 11.  Save raw arrays for downstream use
# ═══════════════════════════════════════════════════════════════════════════

np.save(os.path.join(args.out_dir, "node_mae.npy"), node_mae)
np.save(os.path.join(args.out_dir, "node_rmse.npy"), node_rmse)
np.save(os.path.join(args.out_dir, "dist_matrix_hops.npy"), dist_mat)
np.save(os.path.join(args.out_dir, "adj_matrix.npy"), adj)
np.save(os.path.join(args.out_dir, "node_positions.npy"), node_xy)
print(f"\nRaw arrays saved to {args.out_dir}/")
print("Done.")
