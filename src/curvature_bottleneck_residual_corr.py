"""
Empirical residual correlation at bottleneck vs. non-bottleneck edges.

Rebuttal Priority 2 / kgxJ W1: "Comparing residual correlation at bottleneck
vs. non-bottleneck edges would help support this claim." This script:

  1. Runs the trained checkpoint's backbone on the held-out test split to get
     real one-step-ahead residuals eta_i(t) = pred_mean_i(t) - actual_i(t)
     per sensor (NOT samples from the model's own covariance -- an
     independent empirical signal the curvature module never sees).
  2. Computes the empirical residual correlation matrix across sensors.
  3. Computes Balanced Forman curvature kappa_ij and the learned bottleneck
     score b_ij on the checkpoint's static graph (same functions used for
     the reviewer-facing graph diagnostics in curvature_graph_diagnostics.py).
  4. Splits edges into bottleneck vs. non-bottleneck by b_ij and tests
     whether |corr| differs, against a node-permutation null: permute which
     residual time series is attached to which graph node (holding the graph
     topology and the bottleneck labeling fixed), so the test does not rely
     on treating edges as independent draws -- it only asks whether the
     *specific* alignment between curvature-flagged edges and the observed
     correlation structure is unusual relative to random relabelings.

CAVEAT (stated explicitly, not hidden): the codebase reserves only
num_pred_rolling + prediction_horizon steps after the train/val cutoff for
test, so the one-step-ahead (h=1) residual series has only T = num_pred_rolling
real, non-overlapping calendar timestamps (T=24 for Brussels/xLSTM). That is
too short for a well-conditioned per-edge Pearson estimate individually, so we
report two versions of the test: h=1 only (T=24, temporally independent
across the 24 rolling origins) and all-horizons-pooled (T=24*12=288, temporally
correlated pseudo-samples, higher power). Both are reported side by side --
neither is cherry-picked after seeing the other.

REVISION (round 2, after the raw quartile profile came back Q0 > Q3, i.e. the
*least*-bottlenecked edges show the highest residual correlation -- exactly
backwards from lines 55-57's prediction of a monotone rise). Three additions
address the reviewer's diagnosis that this is very likely a distance confound
(Balanced Forman curvature on a proximity graph is close to a common-neighbour
count, so low bottleneck score ~ dense geographic cluster ~ physically close
sensors that correlate for reasons unrelated to bottlenecks):

  (a) Distance/degree control. Real sensor coordinates (datasets/locations.pkl)
      give haversine distance per edge; weighted endpoint degree from the
      static graph is the second covariate. We report both (i) a partial
      linear regression of |corr| on standardized [bottleneck score, distance,
      degree] with a node-permutation null on the bottleneck coefficient, and
      (ii) the bottleneck-quartile profile stratified within distance tertiles,
      so a reviewer can see directly whether Q0's dominance survives
      conditioning on distance.
  (b) Noise-floor correction. Under independence, E|r_hat| ~ sqrt(2/(pi*(T-1)))
      (T-1 degrees of freedom for a zero-mean bivariate-normal Pearson
      correlation). We report this floor per condition and the *excess* over
      it (mean_abs_corr - floor) alongside the raw quartile means, instead of
      presenting the raw means as if the whole magnitude were signal.
  (c) Permutation null on the quartile profile itself (not just the two-group
      diff-in-means test): shuffle bottleneck scores across edges (correlation
      values stay fixed per edge), recompute the quartile profile n_perm times,
      report the null band per bin -- exactly what kgxJ asked for.

Usage:
    python src/curvature_bottleneck_residual_corr.py \\
        --checkpoint logs/xLSTM/brussels_batch_curvature_.../checkpoints/epoch=17-val_loss=64.74.ckpt \\
        --dataset brussels --model xLSTM --n_samples 100 --n_perm 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np

if not hasattr(np, "float"):
    # pytorch_forecasting's GroupNormalizer still references the deprecated
    # np.float alias (removed in numpy>=1.20); restore it in-process only.
    np.float = float

import pandas as pd
import pickle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_graph_diagnostics import balanced_forman_curvature, bottleneck_indicator, load_curvature_state

DEFAULT_CHECKPOINT = (
    "logs/xLSTM/brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_"
    "LossLRw1.0_RegW2.5_TrainL_False/checkpoints/epoch=17-val_loss=64.74.ckpt"
)
DEFAULT_JSON_OUT = "metrics_stat/xLSTM/brussels_xlstm_bottleneck_residual_corr.json"
DEFAULT_FIG_OUT = "visualizations/brussels_xlstm_bottleneck_residual_corr.pdf"


# ─────────────────────────────────────────────────────────────────────────
# Data / model loading — mirrors train_batch.py's brussels/xLSTM construction
# exactly (dataset-specific frequency, lags, cutoffs), so the held-out split
# is identical to the one the checkpoint was actually validated against.
# ─────────────────────────────────────────────────────────────────────────

def build_test_dataloader(
    dataset: str,
    batch_size: int,
    min_prediction_idx: int = None,
    max_prediction_idx: int = None,
):
    """Held-out test dataloader for `dataset`.

    `min_prediction_idx` / `max_prediction_idx` are an additive extension used by
    curvature_bottleneck_residual_corr_timeresolved.py to slide the evaluation
    window over a longer (in-sample) span. Both default to None, which reproduces
    the original held-out-test behaviour byte for byte -- the `training`
    TimeSeriesDataSet (and therefore the normalizer and categorical encoders) is
    built from data <= training_cutoff regardless, so the model never sees a
    different preprocessing pipeline than the one it was trained with.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder

    with open("./datasets/pred_horizon_dict_v1.pkl", "rb") as f:
        pred_horizon_dict = pickle.load(f)
    with open("./datasets/pred_rolling_dict_v1.pkl", "rb") as f:
        pred_rolling_dict = pickle.load(f)
    with open("./datasets/dataset_freq_v1.pkl", "rb") as f:
        dataset_freq_dict = pickle.load(f)

    freq = dataset_freq_dict[dataset]
    prediction_horizon = pred_horizon_dict[dataset]
    num_pred_rolling = pred_rolling_dict[dataset]

    data = pd.read_csv(f"./datasets/{dataset}.csv")

    if dataset.startswith("pems"):
        # The PeMS CSVs ship with an integer `datetime` column (0, 1, 2, ...).
        # Passed through pd.to_datetime it becomes nanoseconds after the epoch,
        # which yields one unique `tod` category *per timestep* (26,138 for
        # pems03) and a single `dow`. The trained checkpoints do not have that
        # encoding -- their embeddings are tod:(288, 38) and dow:(7, 5) -- so
        # they were fitted with train_batch.py's pems branch, currently
        # commented out there, which overwrites datetime with a synthetic
        # 5-minute range over ROWS. Reproduced verbatim (same start, same
        # periods, same freq) so the model is fed the categorical encoding it
        # was actually trained with; anything else indexes its embedding tables
        # out of range and crashes in the CUDA index kernel.
        #
        # NOTE for interpretation: because the range runs over rows rather than
        # timesteps, these tod/dow features do NOT correspond to the real time
        # of day of a reading. The model therefore has no valid calendar
        # covariate on PeMS -- which means any time-of-day structure found in
        # its residuals cannot have been read off a time-of-day input feature.
        data["datetime"] = pd.date_range(start="2012-01-01", periods=len(data["datetime"]), freq="5min")

    if freq in ["30min", "5min", "H", "T"]:
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["tod"] = (
            data["datetime"].values - data["datetime"].values.astype("datetime64[D]")
        ) / np.timedelta64(1, "D")
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

    validation_cutoff = data["time_idx"].max() - prediction_horizon - num_pred_rolling + 1
    training_cutoff = validation_cutoff - (data["time_idx"].max() - validation_cutoff)

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
        min_encoder_length=prediction_horizon,
        max_encoder_length=prediction_horizon,
        min_prediction_length=prediction_horizon,
        max_prediction_length=prediction_horizon,
        allow_missing_timesteps=False,
    )
    if min_prediction_idx is None:
        min_prediction_idx = validation_cutoff + 1
    data_for_test = data if max_prediction_idx is None else data[lambda x: x.time_idx <= max_prediction_idx]
    testing = TimeSeriesDataSet.from_dataset(training, data_for_test, min_prediction_idx=min_prediction_idx)
    # Sorted alphabetically by NaNLabelEncoder -- verified to match the node
    # ordering baked into the checkpoint's static_adj (both are `sorted(sensor ids)`).
    node_order = list(NaNLabelEncoder().fit(data.sensor).classes_.keys())
    test_loader = testing.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized",
    )
    return test_loader, node_order, prediction_horizon


def load_checkpoint_model(checkpoint: str, model_name: str, device: str):
    """Restore the trained model exactly as it behaved at train time.

    Extracted verbatim from run_inference so that the time-resolved companion
    script (curvature_bottleneck_residual_corr_timeresolved.py) cannot drift
    from the checkpoint-restoration semantics used here -- in particular the
    `reweight_mode` back-fill below, which silently changes what the curvature
    module computes if it is applied in one script and not the other.
    """
    from batched_model import BatchxLSTMPredictor

    if model_name != "xLSTM":
        raise NotImplementedError("Only the xLSTM backbone checkpoint naming/loading is wired up here.")

    model = BatchxLSTMPredictor.load_from_checkpoint(checkpoint, map_location=device).to(device)
    model.wReg = True
    model.eval()

    # This checkpoint predates the ablation-ladder `reweight_mode` attribute
    # added to CurvatureAwareGraphPrecision_LearnP; Lightning restores the
    # pickled submodule's __dict__ as-is, so old checkpoints lack it.
    # "curvature" (learned end-to-end reweighting) was the only mode that
    # existed when this checkpoint was trained, so this restores the exact
    # behavior used at train time rather than changing it.
    curvature_module = model.loss.curvature_precision
    if not hasattr(curvature_module, "reweight_mode"):
        curvature_module.reweight_mode = "curvature"
    return model


def run_inference(
    checkpoint: str, model_name: str, dataset: str, device: str, batch_size: int, n_samples: int, seed: int,
):
    test_loader, node_order, Q = build_test_dataloader(dataset, batch_size)
    N = len(node_order)

    model = load_checkpoint_model(checkpoint, model_name, device)

    # `prediction` is a Monte-Carlo average over n_samples draws from the model's
    # own predictive distribution, i.e. a random quantity. Without seeding torch
    # nothing downstream reproduces: re-running this script unchanged moved the
    # h=1 Q0 bin from 0.2074 to 0.1923 and flipped the sign of the median-split
    # statistic (-0.0020 -> +0.0015). Seed before any draw is taken.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        (raw_predictions, x), _ = model.predict(
            test_loader, mode="raw", n_samples=n_samples, show_progress_bar=True, return_x=True,
        )
    preds = raw_predictions["prediction"].cpu().float() if isinstance(raw_predictions, dict) else raw_predictions.prediction.cpu().float()
    actuals = x["decoder_target"].cpu().float()

    assert preds.shape[0] % N == 0, f"predictions ({preds.shape[0]}) not a multiple of N={N}"
    R = preds.shape[0] // N
    preds = preds.reshape(R, N, preds.shape[1], preds.shape[2])
    actuals = actuals.reshape(R, N, actuals.shape[1])
    return preds, actuals, node_order


# ─────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────

def edge_arrays(W: np.ndarray, kappa_0: float, tau: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (edge_i, edge_j, b_edge, kappa_edge) for the upper-triangular edge set."""
    mask = (W > 0).astype(float)
    kappa = balanced_forman_curvature(W)
    b = bottleneck_indicator(kappa, mask, kappa_0, tau)
    edge_i, edge_j = np.where(np.triu(W > 0, k=1))
    return edge_i, edge_j, b[edge_i, edge_j], kappa[edge_i, edge_j]


# ─────────────────────────────────────────────────────────────────────────
# (a) Distance / degree covariates for confound control
# ─────────────────────────────────────────────────────────────────────────

def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Same formula used in exps/generate_datasets.ipynb to build the graph itself."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371.0


def edge_covariates(
    node_order: list, edge_i: np.ndarray, edge_j: np.ndarray, W: np.ndarray, locations_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (distance_km, endpoint_degree) per edge. distance_km is NaN for
    sensors missing from datasets/locations.pkl (a handful of the 195)."""
    with open(locations_path, "rb") as f:
        locations: Dict[str, list] = pickle.load(f)

    lonlat = np.full((len(node_order), 2), np.nan)
    for idx, sensor_id in enumerate(node_order):
        if sensor_id in locations:
            lonlat[idx] = locations[sensor_id]

    lon1, lat1 = lonlat[edge_i, 0], lonlat[edge_i, 1]
    lon2, lat2 = lonlat[edge_j, 0], lonlat[edge_j, 1]
    with np.errstate(invalid="ignore"):
        distance_km = haversine_km(lon1, lat1, lon2, lat2)

    deg = W.sum(axis=1)
    endpoint_degree = 0.5 * (deg[edge_i] + deg[edge_j])
    return distance_km, endpoint_degree


# ─────────────────────────────────────────────────────────────────────────
# (b) Noise floor under independence: E|r_hat| ~ sqrt(2/(pi*(T-1)))
# (standard result for the sample Pearson correlation of two independent
# zero-mean Gaussians; see e.g. Fisher 1915 for the exact distribution).
# ─────────────────────────────────────────────────────────────────────────

def independence_noise_floor(T: int) -> float:
    return float(np.sqrt(2.0 / (np.pi * (T - 1))))


def group_diff_stat(corr: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, bottleneck: np.ndarray) -> float:
    c = np.abs(corr[edge_i, edge_j])
    return float(c[bottleneck].mean() - c[~bottleneck].mean())


def permutation_test(
    corr: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, bottleneck: np.ndarray, n_perm: int, rng: np.random.Generator,
) -> Dict:
    n_nodes = corr.shape[0]
    observed = group_diff_stat(corr, edge_i, edge_j, bottleneck)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n_nodes)
        corr_perm = corr[np.ix_(perm, perm)]
        null[k] = group_diff_stat(corr_perm, edge_i, edge_j, bottleneck)
    p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
    return {
        "observed_diff": observed,
        "n_bottleneck_edges": int(bottleneck.sum()),
        "n_nonbottleneck_edges": int((~bottleneck).sum()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "z_score": float((observed - null.mean()) / null.std()) if null.std() > 0 else float("nan"),
        "p_value_two_sided": float(p_value),
        "n_perm": n_perm,
    }


def quantile_bin_edges(b_edge: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(b_edge, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def quantile_bin_summary(b_edge: np.ndarray, corr_abs_edge: np.ndarray, n_bins: int, floor: float = 0.0) -> list:
    edges = quantile_bin_edges(b_edge, n_bins)
    bin_idx = np.digitize(b_edge, edges[1:-1])
    rows = []
    for i in range(n_bins):
        sel = bin_idx == i
        if sel.sum() == 0:
            continue
        mean_abs_corr = float(corr_abs_edge[sel].mean())
        rows.append(
            {
                "bin": i,
                "b_lo": float(edges[i]),
                "b_hi": float(edges[i + 1]),
                "n_edges": int(sel.sum()),
                "mean_abs_corr": mean_abs_corr,
                "std_abs_corr": float(corr_abs_edge[sel].std()),
                "excess_over_floor": mean_abs_corr - floor,
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────
# (c) Permutation null on the quartile profile: shuffle bottleneck scores
# across edges (correlation values stay fixed per edge), recompute the
# profile, and report a null band per bin.
# ─────────────────────────────────────────────────────────────────────────

def quantile_profile_permutation_null(
    b_edge: np.ndarray, corr_abs_edge: np.ndarray, n_bins: int, n_perm: int, rng: np.random.Generator,
) -> list:
    edges = quantile_bin_edges(b_edge, n_bins)
    bin_idx_obs = np.digitize(b_edge, edges[1:-1])
    null_by_bin = [[] for _ in range(n_bins)]
    for _ in range(n_perm):
        b_perm = rng.permutation(b_edge)
        bin_idx_perm = np.digitize(b_perm, edges[1:-1])
        for i in range(n_bins):
            sel = bin_idx_perm == i
            if sel.sum() > 0:
                null_by_bin[i].append(corr_abs_edge[sel].mean())

    rows = []
    for i in range(n_bins):
        sel = bin_idx_obs == i
        if sel.sum() == 0 or len(null_by_bin[i]) == 0:
            continue
        null_arr = np.array(null_by_bin[i])
        observed = float(corr_abs_edge[sel].mean())
        p_value = (1 + np.sum(np.abs(null_arr - null_arr.mean()) >= abs(observed - null_arr.mean()))) / (len(null_arr) + 1)
        rows.append(
            {
                "bin": i,
                "observed_mean_abs_corr": observed,
                "null_mean": float(null_arr.mean()),
                "null_std": float(null_arr.std()),
                "null_p2_5": float(np.percentile(null_arr, 2.5)),
                "null_p97_5": float(np.percentile(null_arr, 97.5)),
                "p_value_two_sided": float(p_value),
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────
# (a) Partial-regression control for distance and endpoint degree, with a
# node-permutation null on the bottleneck-score coefficient. The design
# matrix X (bottleneck score, distance, degree, standardized) is fixed under
# the node permutation -- only the correlation values (y) change -- so the
# OLS solve reduces to a single fixed linear operator applied to each
# permuted y, letting all n_perm refits be done as one matrix multiply.
# ─────────────────────────────────────────────────────────────────────────

def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def distance_controlled_regression(
    corr: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    b_edge: np.ndarray,
    distance_edge: np.ndarray,
    degree_edge: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict:
    import statsmodels.api as sm

    valid = np.isfinite(distance_edge)
    ei, ej = edge_i[valid], edge_j[valid]
    X = np.column_stack([zscore(b_edge[valid]), zscore(distance_edge[valid]), zscore(degree_edge[valid])])
    X_design = sm.add_constant(X)  # (n_valid, 4): [const, b_z, dist_z, deg_z]

    y_obs = np.abs(corr[ei, ej])
    ols = sm.OLS(y_obs, X_design).fit(cov_type="HC3")

    # Fixed linear operator: beta = (X'X)^{-1} X' y, reused for every permuted y.
    w = np.linalg.pinv(X_design.T @ X_design) @ X_design.T  # (4, n_valid)
    n_nodes = corr.shape[0]
    null_coef_b = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n_nodes)
        corr_perm = corr[np.ix_(perm, perm)]
        y_perm = np.abs(corr_perm[ei, ej])
        null_coef_b[k] = (w @ y_perm)[1]  # index 1 = bottleneck-score coefficient

    coef_b_obs = float(ols.params[1])
    p_perm = (1 + np.sum(np.abs(null_coef_b) >= abs(coef_b_obs))) / (n_perm + 1)

    return {
        "n_edges_valid_distance": int(valid.sum()),
        "n_edges_excluded_missing_location": int((~valid).sum()),
        "ols_params": {"const": float(ols.params[0]), "bottleneck_z": coef_b_obs, "distance_z": float(ols.params[2]), "degree_z": float(ols.params[3])},
        "ols_pvalues_iid_assumption": {"bottleneck_z": float(ols.pvalues[1]), "distance_z": float(ols.pvalues[2]), "degree_z": float(ols.pvalues[3])},
        "ols_r_squared": float(ols.rsquared),
        "bottleneck_coef_permutation_null_mean": float(null_coef_b.mean()),
        "bottleneck_coef_permutation_null_std": float(null_coef_b.std()),
        "bottleneck_coef_p_value_node_permutation": float(p_perm),
        "n_perm": n_perm,
    }


def distance_stratified_profile(
    b_edge: np.ndarray, distance_edge: np.ndarray, corr_abs_edge: np.ndarray, n_dist_bands: int, n_bbins: int,
) -> list:
    valid = np.isfinite(distance_edge)
    b_v, d_v, c_v = b_edge[valid], distance_edge[valid], corr_abs_edge[valid]
    dist_edges = quantile_bin_edges(d_v, n_dist_bands)
    dist_bin_idx = np.digitize(d_v, dist_edges[1:-1])

    bands = []
    for band in range(n_dist_bands):
        sel = dist_bin_idx == band
        if sel.sum() == 0:
            continue
        bands.append(
            {
                "band": band,
                "distance_km_lo": float(dist_edges[band]),
                "distance_km_hi": float(dist_edges[band + 1]),
                "n_edges": int(sel.sum()),
                "quantile_bins": quantile_bin_summary(b_v[sel], c_v[sel], n_bbins),
            }
        )
    return bands


def analyze_condition(
    label: str,
    corr: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    b_edge: np.ndarray,
    distance_edge: np.ndarray,
    degree_edge: np.ndarray,
    T: int,
    n_perm: int,
    n_profile_perm: int,
    seed: int,
    n_qbins: int,
    n_dist_bands: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    median_split = b_edge > np.median(b_edge)
    q25, q75 = np.quantile(b_edge, [0.25, 0.75])
    top_q = b_edge >= q75
    bot_q = b_edge <= q25
    extreme_mask = top_q | bot_q
    extreme_bottleneck = top_q[extreme_mask]

    corr_abs_edge = np.abs(corr[edge_i, edge_j])
    floor = independence_noise_floor(T)

    result = {
        "label": label,
        "T": T,
        "independence_noise_floor": floor,
        "median_split": permutation_test(corr, edge_i, edge_j, median_split, n_perm, rng),
        "extreme_quartile_split": permutation_test(
            corr, edge_i[extreme_mask], edge_j[extreme_mask], extreme_bottleneck, n_perm, rng,
        ),
        "quantile_bins": quantile_bin_summary(b_edge, corr_abs_edge, n_qbins, floor=floor),
        "quantile_bins_permutation_null": quantile_profile_permutation_null(
            b_edge, corr_abs_edge, n_qbins, n_profile_perm, rng,
        ),
        "distance_controlled_regression": distance_controlled_regression(
            corr, edge_i, edge_j, b_edge, distance_edge, degree_edge, n_perm, rng,
        ),
        "distance_stratified_profile": distance_stratified_profile(
            b_edge, distance_edge, corr_abs_edge, n_dist_bands, n_qbins,
        ),
    }
    return result


def make_figure(results: Dict, fig_out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))

    # Row 1: raw quartile profile with permutation null band + noise floor.
    for ax, key, title in [(axes[0, 0], "h1", "h = 1 (T=24)"), (axes[0, 1], "pooled", "all horizons pooled (T=288)")]:
        r = results[key]
        rows = r["quantile_bins"]
        null_rows = {row["bin"]: row for row in r["quantile_bins_permutation_null"]}
        x = [row["bin"] for row in rows]
        y = [row["mean_abs_corr"] for row in rows]
        yerr = [row["std_abs_corr"] / max(row["n_edges"], 1) ** 0.5 for row in rows]
        null_lo = [null_rows[b]["null_p2_5"] for b in x]
        null_hi = [null_rows[b]["null_p97_5"] for b in x]
        ax.fill_between(x, null_lo, null_hi, color="gray", alpha=0.3, label="permutation null (95%)")
        ax.errorbar(x, y, yerr=yerr, marker="o", color="darkorange", label="observed")
        ax.axhline(r["independence_noise_floor"], color="black", linestyle="--", linewidth=1, label="independence floor")
        ax.set_xlabel("Bottleneck-score quartile (0=least bottlenecked)")
        ax.set_ylabel(r"mean $|\rho_{ij}|$ (residuals)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Row 2: distance-stratified profile (within-band quartile means), h=1 and pooled.
    for ax, key, title in [(axes[1, 0], "h1", "h = 1, within distance tertiles"), (axes[1, 1], "pooled", "pooled, within distance tertiles")]:
        bands = results[key]["distance_stratified_profile"]
        cmap = plt.get_cmap("viridis")
        for band in bands:
            rows = band["quantile_bins"]
            x = [row["bin"] for row in rows]
            y = [row["mean_abs_corr"] for row in rows]
            ax.plot(
                x, y, marker="o",
                color=cmap(band["band"] / max(len(bands) - 1, 1)),
                label=f"{band['distance_km_lo']:.1f}-{band['distance_km_hi']:.1f} km",
            )
        ax.set_xlabel("Bottleneck-score quartile (within band)")
        ax.set_ylabel(r"mean $|\rho_{ij}|$ (residuals)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, title="distance band")
        ax.grid(alpha=0.3)

    fig.suptitle("Residual correlation vs. curvature bottleneck score, with distance control — Brussels/xLSTM")
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", default="brussels")
    parser.add_argument("--model", default="xLSTM")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=195)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--n_profile_perm", type=int, default=1000, help="perms for the quartile-profile null band (kgxJ's request)")
    parser.add_argument("--n_qbins", type=int, default=4)
    parser.add_argument("--n_dist_bands", type=int, default=3)
    parser.add_argument("--locations-path", default="./datasets/locations.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--fig-out", default=DEFAULT_FIG_OUT)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print("Running inference to collect held-out residuals ...")
    preds, actuals, node_order = run_inference(
        args.checkpoint, args.model, args.dataset, device, args.batch_size, args.n_samples, args.seed,
    )
    R, N, Q, S = preds.shape
    print(f"preds: R={R} rolling origins, N={N} nodes, Q={Q} horizons, S={S} MC samples")

    pred_mean = preds.mean(dim=-1)          # (R, N, Q)
    residuals = (pred_mean - actuals).numpy()  # (R, N, Q)

    X_h1 = residuals[:, :, 0]               # (R, N) -- T=R independent calendar origins
    X_pooled = residuals.transpose(0, 2, 1).reshape(R * Q, N)  # (R*Q, N) -- pooled, NOT independent

    corr_h1 = np.corrcoef(X_h1, rowvar=False)
    corr_pooled = np.corrcoef(X_pooled, rowvar=False)
    assert corr_h1.shape == (N, N)

    print("Loading curvature module state from checkpoint ...")
    W, params = load_curvature_state(Path(args.checkpoint))
    assert W.shape == (N, N), f"graph size {W.shape} != number of sensors {N}"
    edge_i, edge_j, b_edge, kappa_edge = edge_arrays(W, params["kappa_0"], params["tau"])
    print(f"{len(edge_i)} edges; bottleneck score b in [{b_edge.min():.3f}, {b_edge.max():.3f}], median={np.median(b_edge):.3f}")

    print("Loading sensor coordinates for distance control ...")
    distance_edge, degree_edge = edge_covariates(node_order, edge_i, edge_j, W, args.locations_path)
    n_missing_loc = int((~np.isfinite(distance_edge)).sum())
    print(f"distance range {np.nanmin(distance_edge):.3f}-{np.nanmax(distance_edge):.3f} km; "
          f"{n_missing_loc}/{len(edge_i)} edges excluded from distance analyses (missing sensor coordinates)")

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n_nodes": N,
        "n_rolling_origins": R,
        "n_horizons": Q,
        "n_edges": int(len(edge_i)),
        "n_edges_missing_location": n_missing_loc,
        "curvature_params": params,
        "h1": analyze_condition(
            "h=1 (T=24, independent)", corr_h1, edge_i, edge_j, b_edge, distance_edge, degree_edge,
            R, args.n_perm, args.n_profile_perm, args.seed, args.n_qbins, args.n_dist_bands,
        ),
        "pooled": analyze_condition(
            "all horizons pooled (T=288, not independent)", corr_pooled, edge_i, edge_j, b_edge, distance_edge, degree_edge,
            R * Q, args.n_perm, args.n_profile_perm, args.seed + 1, args.n_qbins, args.n_dist_bands,
        ),
        "caveat": (
            "h=1 uses T=24 truly independent calendar-time residual snapshots "
            "(the entire held-out tail reserved by this codebase's train/val/test split); "
            "pooled uses T=24*12=288 pseudo-samples across forecast horizons that are NOT "
            "temporally independent (nearby real timestamps are re-used across origins/horizons) "
            "and should be read as a higher-power robustness check, not a stronger result. "
            "Independence noise floor E|r_hat| ~ sqrt(2/(pi*(T-1))) is reported per condition; "
            "raw quartile means include this floor, 'excess_over_floor' subtracts it out."
        ),
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2) + "\n")
    make_figure(results, Path(args.fig_out))

    print(f"\nWrote {json_out}")
    print(f"Wrote {args.fig_out}")
    for key in ["h1", "pooled"]:
        r = results[key]
        print(f"\n--- {r['label']} ---")
        print(f"  independence noise floor (T={r['T']}): {r['independence_noise_floor']:.4f}")
        for split in ["median_split", "extreme_quartile_split"]:
            s = r[split]
            print(
                f"  {split}: diff={s['observed_diff']:+.4f} (bottleneck n={s['n_bottleneck_edges']}, "
                f"non-bottleneck n={s['n_nonbottleneck_edges']}), z={s['z_score']:.2f}, p={s['p_value_two_sided']:.4f}"
            )
        print("  quartile profile (raw / excess-over-floor / null p-value):")
        null_rows = {row["bin"]: row for row in r["quantile_bins_permutation_null"]}
        for row in r["quantile_bins"]:
            p = null_rows[row["bin"]]["p_value_two_sided"]
            print(f"    Q{row['bin']}: raw={row['mean_abs_corr']:.4f}  excess={row['excess_over_floor']:+.4f}  p_null={p:.4f}")
        reg = r["distance_controlled_regression"]
        print(
            f"  distance-controlled regression (standardized coefs, n={reg['n_edges_valid_distance']}): "
            f"bottleneck={reg['ols_params']['bottleneck_z']:+.4f} (p_perm={reg['bottleneck_coef_p_value_node_permutation']:.4f}), "
            f"distance={reg['ols_params']['distance_z']:+.4f}, degree={reg['ols_params']['degree_z']:+.4f}, "
            f"R^2={reg['ols_r_squared']:.4f}"
        )


if __name__ == "__main__":
    main()
