"""
Time-resolved residual correlation vs. curvature bottleneck score (kgxJ W1).

WHY THIS SCRIPT EXISTS
──────────────────────
curvature_bottleneck_residual_corr.py answered kgxJ's W1 request on the
held-out test split and came back null / mildly backwards (Brussels/xLSTM
quartile profile Q0=0.207, Q1=0.202, Q2=0.195, Q3=0.210 -- non-monotone,
distance-controlled bottleneck coefficient +0.0057 with node-permutation
p=0.101 at h=1 and p=0.908 pooled).

That test had no ability to detect the effect the manuscript actually claims.
This codebase reserves `prediction_horizon + num_pred_rolling` steps after the
validation cutoff for test, which for every 5-min traffic dataset (Brussels,
PeMS03/04/07: H=12, R=24) is 35 steps = 2.9 hours of calendar time on a single
weekday. Brussels' test decoder span is 2024-01-10 09:20-12:10; all 24 rolling
origins fall between 09:20 and 11:15. The manuscript claim under test is

    "...this propagation is both time-varying and influenced by latent system
     states ... the same bottlenecks that distort message passing also distort
     the spatial covariance of residuals---concentrating dependence along thin
     chains and underestimating cross-region correlation."

A 2.9-hour single-regime window cannot speak to a *time-varying*, *latent-state
dependent* claim. Sub-windowing the existing 24 origins is not an option
either: at T<24 the independence noise floor sqrt(2/(pi(T-1))) exceeds 0.24,
which is larger than the entire Q0-Q3 spread being interpreted (0.195-0.210),
so any sub-window profile would be estimation noise.

This script therefore evaluates the same checkpoint over a multi-week span and
tests the claim where it is actually falsifiable: across traffic regimes.

═══════════════════════════════════════════════════════════════════════════
PRE-SPECIFICATION  (fixed before any window-resolved statistic was computed.
Commit this file before the first run to make that auditable.)
═══════════════════════════════════════════════════════════════════════════

HYPOTHESIS (from traffic physics, not from the data). A graph bottleneck is a
capacity constraint. Under free flow, a low-curvature edge carries no more
coupling than any other edge: vehicles are not queueing, so a disturbance at
one end does not propagate to the other. Under congestion, the same edge is
the only channel between two regions and queue spillback forces their errors
to move together. Therefore:

    H1 (directional): the association between an edge's Balanced Forman
        bottleneck score b_ij and the absolute residual correlation |rho_ij|
        is STRONGER during congested periods (AM/PM peak, weekdays) than
        during free-flow periods (night, weekdays).

    H0: the association is constant across time of day.

H1 predicts a *null-to-positive* gradient in the pooled analysis (which mixes
regimes and is therefore diluted) -- i.e. the existing null result is what H1
predicts, not evidence against it. H1 is falsified if the peak-vs-night
contrast is <= 0, or if it is no larger on weekdays than on weekends.

WINDOWS (local clock, fixed a priori from standard traffic-engineering peak
definitions; NOT selected by looking at outcomes):
    night        22:00-05:00     free flow          (H1 reference level)
    am_shoulder  05:00-07:00     onset
    am_peak      07:00-10:00     congested          (H1 elevated)
    midday       10:00-16:00     intermediate
    pm_peak      16:00-19:00     congested          (H1 elevated)
    pm_shoulder  19:00-22:00     decay
Each is evaluated separately on weekdays (Mon-Fri) and weekends (Sat-Sun).

PRIMARY TEST (one, pre-specified, no correction needed):
    Delta = beta_b(peak_weekday) - beta_b(night_weekday),  one-sided H1: Delta > 0
    where peak_weekday = am_peak U pm_peak on weekdays, and beta_b is the
    distance- and degree-adjusted coefficient defined below. The null is the
    node permutation, applied with a SHARED permutation across both windows so
    the paired structure is preserved.

CONTROL TEST (pre-specified falsifier):
    the same Delta computed on weekends, where the peak regime does not exist.
    H1 predicts Delta_weekend < Delta_weekday.

STATISTICS:
    beta_b (primary)  -- coefficient on standardized bottleneck score in
        OLS  |rho_ij| ~ 1 + z(b_ij) + z(distance_ij) + z(endpoint degree_ij).
        Adjusted rather than raw because Balanced Forman curvature on a
        proximity graph is close to a common-neighbour count, so raw b is
        confounded with geographic distance (this is the confound that the
        round-2 revision of the companion script was written to address).
        Reuses curvature_bottleneck_residual_corr.distance_controlled_regression's
        design so the two scripts cannot diverge.
    rho_S (secondary) -- Spearman rank correlation between b_ij and |rho_ij|.
        This is the direct "is the profile monotone?" statistic the reviewer's
        plot is about, unadjusted.
    rho_S^adj (secondary) -- Spearman between b_ij and the part of |rho_ij| left
        after regressing out distance and degree. beta_b is adjusted but linear;
        rho_S is monotone but unadjusted; this is the adjusted monotone version,
        i.e. the statistic that literally asks whether the reviewer's profile is
        monotone once the distance confound is removed.
    All three are reported per window with the quartile profile.

NULL: node permutation. Permute which residual time series is attached to
which graph node, holding graph topology, b, distance and degree fixed. This
null is valid under temporal autocorrelation of the residuals (it permutes the
node axis, never the time axis, so each series keeps its autocorrelation
intact), which matters here because 5-min origins are strongly autocorrelated.

SPAN SELECTION (calendar-based, decided without reference to any outcome):
    The evaluation span must consist of ordinary term-time weeks. Public and
    school holiday periods are excluded, because the hypothesis is about
    commute-driven congestion and holidays have no commute peak -- including
    them would dilute the very contrast under test in a direction that flatters
    H0, not H1. Days with obviously broken instrumentation (network mean flow
    near zero) are also excluded. Both exclusions are made from the calendar and
    from the marginal flow series, never from the bottleneck statistic.
    Applied here (21 days each, i.e. 15 weekdays + 6 weekend days per dataset):
      brussels     2023-11-20 -> 2023-12-10. Starts after 2023-11-19 (network
                   mean flow 0.15, i.e. a broken partial first day) and ends
                   well before 2023-12-23, when the Belgian Christmas/New Year
                   school holidays begin. Christmas Day and New Year's Day both
                   fell on a Monday in this file, which is why the full-file
                   weekly minimum is Sunday+Monday rather than the weekend; the
                   2024-01-01..10 tail additionally shows implausible values
                   (New Year's Day above a normal Monday) and is excluded.
                   No Belgian public holiday falls inside the chosen span.
      pems03_flow  2018-09-04 -> 2018-09-24. Starts after US Labor Day
                   (2018-09-03) and ends before Columbus Day (2018-10-08);
                   no US federal holiday falls inside the chosen span.

MULTIPLICITY: the 12 window-level tests (6 windows x {weekday, weekend}) are
secondary and Holm-corrected. The 24 hour-of-day bins are DESCRIPTIVE ONLY --
plotted, never used to support a claim, p-values reported uncorrected and
labelled as such.

═══════════════════════════════════════════════════════════════════════════
CAVEAT -- THESE RESIDUALS ARE IN-SAMPLE  (stated up front, not buried)
═══════════════════════════════════════════════════════════════════════════
The evaluation span lies inside the training period, because the held-out tail
is only 2.9 hours (above). The point forecaster has fitted this data, so
residual *magnitudes* are shrunk relative to true out-of-sample error. Two
things limit, but do not eliminate, the damage:

  (i) The quantity analysed is the *spatial correlation* of residuals across
      sensors, not their magnitude. Fitting the conditional mean in-sample
      shrinks residuals but is not an objective that removes cross-sensor
      correlation structure.
 (ii) The comparison is *between time windows of the same fitted model*, so
      any in-sample optimism common to all windows differences out of the
      primary contrast.

What it does NOT rule out: the covariance head was trained on this span with a
curvature-derived prior, so the model has had the opportunity to fit
curvature-aligned structure here. This analysis is therefore a diagnostic of
the residual covariance structure, NOT out-of-sample confirmation. A clean
out-of-sample version requires retraining with the split shifted back ~14 days.
Report it as such.

Usage:
    python src/curvature_bottleneck_residual_corr_timeresolved.py \\
        --dataset brussels --model xLSTM --span_start 2023-11-20 --span_end 2023-12-10 --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if not hasattr(np, "float"):
    # pytorch_forecasting's GroupNormalizer still references the deprecated
    # np.float alias (removed in numpy>=1.20); restore it in-process only.
    np.float = float

import pandas as pd
import pickle
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curvature_graph_diagnostics import load_curvature_state
from curvature_bottleneck_residual_corr import (
    build_test_dataloader,
    edge_arrays,
    edge_covariates,
    independence_noise_floor,
    load_checkpoint_model,
    quantile_bin_summary,
    zscore,
)

# ─────────────────────────────────────────────────────────────────────────
# Pre-specified windows. (label, [(hour_lo, hour_hi), ...], regime)
# Intervals are half-open [lo, hi) on the local 24-hour clock.
# ─────────────────────────────────────────────────────────────────────────
PRESPECIFIED_WINDOWS: List[Tuple[str, List[Tuple[int, int]], str]] = [
    ("night", [(22, 24), (0, 5)], "free_flow"),
    ("am_shoulder", [(5, 7)], "onset"),
    ("am_peak", [(7, 10)], "congested"),
    ("midday", [(10, 16)], "intermediate"),
    ("pm_peak", [(16, 19)], "congested"),
    ("pm_shoulder", [(19, 22)], "decay"),
]
PEAK_WINDOWS = ("am_peak", "pm_peak")
REFERENCE_WINDOW = "night"

# PeMS03 ships in this repo without wall-clock timestamps (the `datetime`
# column is 0,1,2,...). The canonical PEMS03 release starts 2018-09-01 00:00 at
# 5-minute resolution; both facts implied by that anchor are asserted against
# the data in validate_time_base() rather than trusted.
PEMS_TIME_ANCHORS = {
    "pems03_flow": ("2018-09-01 00:00", "5min"),
    "pems04_flow": ("2018-01-01 00:00", "5min"),
    "pems07_flow": ("2017-05-01 00:00", "5min"),
    "pems08_flow": ("2016-07-01 00:00", "5min"),
}


# ─────────────────────────────────────────────────────────────────────────
# Time base
# ─────────────────────────────────────────────────────────────────────────

def build_time_index(dataset: str, data: pd.DataFrame) -> pd.Series:
    """time_idx -> wall-clock Timestamp, as a Series indexed by time_idx."""
    if dataset in PEMS_TIME_ANCHORS:
        anchor, freq = PEMS_TIME_ANCHORS[dataset]
        n_steps = int(data["time_idx"].max()) + 1
        stamps = pd.date_range(start=anchor, periods=n_steps, freq=freq)
        return pd.Series(stamps, index=np.arange(n_steps), name="datetime")

    stamps = data.drop_duplicates("time_idx").set_index("time_idx")["datetime"]
    stamps = pd.to_datetime(stamps).sort_index()
    assert stamps.index.equals(pd.RangeIndex(len(stamps))), "time_idx is not a contiguous 0..T-1 range"
    return stamps


def validate_time_base(
    dataset: str, data: pd.DataFrame, time_index: pd.Series, span_time_idx: np.ndarray,
) -> Dict:
    """Fail loudly if the assumed clock does not reproduce known traffic physics.

    Both checks run on the *analysis span* (not the whole file) and on the
    target series rather than on metadata, so they validate the assumed anchor
    for PeMS and cross-check the real timestamps for Brussels:

      1. Diurnal trough. Network mean flow must bottom out in the small hours
         (02:00-05:00). A clock misaligned by more than ~2 hours fails this.
      2. Weekly trough. Sunday must be the single lowest-flow day of the week,
         and mean weekend flow must be below mean weekday flow. A wrong weekday
         phase fails this.

    Check 2 is deliberately phrased as "Sunday is the minimum" rather than "Sat
    and Sun are the two lowest": on the full Brussels file the two lowest days
    are Sunday and *Monday*, because the file spans Christmas and New Year, both
    of which fell on a Monday in 2023-24 and drag the Monday mean below
    Saturday's. That is a holiday artefact, not a phase error -- and it is
    exactly why the analysis span must exclude holiday periods (see
    --exclude_dates), since a commute-peak hypothesis is untestable on days that
    have no commute peak.
    """
    per_step = data.groupby("time_idx")["value"].mean()
    per_step = per_step.loc[per_step.index.isin(span_time_idx)]
    stamps = time_index.loc[per_step.index]

    hourly = pd.Series(per_step.values).groupby(stamps.dt.hour.values).mean()
    trough_hour = int(hourly.idxmin())
    assert 2 <= trough_hour <= 5, (
        f"{dataset}: diurnal flow trough at {trough_hour:02d}:00, expected 02:00-05:00. "
        "The assumed time base is misaligned -- refusing to run a time-of-day analysis."
    )

    dow = stamps.dt.weekday.values
    daily = pd.Series(per_step.values).groupby(dow).mean()
    assert set(daily.index) == set(range(7)), (
        f"{dataset}: analysis span covers only weekdays {sorted(daily.index)}; all 7 are required "
        "because the pre-specified control test contrasts weekday against weekend windows."
    )
    assert int(daily.idxmin()) == 6, (
        f"{dataset}: lowest-flow day of week is index {int(daily.idxmin())}, expected 6 (Sunday). "
        "The assumed weekday phase is wrong -- refusing to run a weekday/weekend analysis."
    )
    weekday_mean = float(daily.loc[daily.index <= 4].mean())
    weekend_mean = float(daily.loc[daily.index >= 5].mean())
    assert weekend_mean < weekday_mean, (
        f"{dataset}: weekend mean flow ({weekend_mean:.2f}) is not below weekday mean "
        f"({weekday_mean:.2f}) over the analysis span -- the span does not show a normal "
        "commute pattern, so a peak-vs-free-flow contrast is not interpretable on it."
    )

    by_date = pd.Series(per_step.values).groupby(stamps.dt.date.values).mean()
    return {
        "source": "anchored" if dataset in PEMS_TIME_ANCHORS else "csv_datetime_column",
        "anchor": PEMS_TIME_ANCHORS.get(dataset, (None,))[0],
        "validated_on": "analysis span only",
        "diurnal_trough_hour": trough_hour,
        "lowest_flow_day_of_week": int(daily.idxmin()),
        "weekday_mean_flow": weekday_mean,
        "weekend_mean_flow": weekend_mean,
        "mean_flow_by_hour": {int(h): float(v) for h, v in hourly.items()},
        "mean_flow_by_day_of_week": {int(d): float(v) for d, v in daily.items()},
        "mean_flow_by_date": {str(d): float(v) for d, v in by_date.items()},
    }


# ─────────────────────────────────────────────────────────────────────────
# Distance covariate. Brussels has sensor coordinates; PeMS ships road-network
# edge distances instead. Both give a per-edge length in km.
# ─────────────────────────────────────────────────────────────────────────

def pems_edge_distance(
    graph_csv: str, edge_i: np.ndarray, edge_j: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Road-network distance per edge, indexed with the SAME node ordering that
    dynamic_graph.load_static_graph uses to build the adjacency the checkpoint
    was trained on (sorted unique ids over the `from`/`to` columns)."""
    df = pd.read_csv(graph_csv)
    all_nodes = pd.concat([df["from"], df["to"]]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}

    dist = np.full((n_nodes, n_nodes), np.nan)
    for _, row in df.iterrows():
        a, b = node_to_idx.get(row["from"]), node_to_idx.get(row["to"])
        if a is None or b is None or a >= n_nodes or b >= n_nodes:
            continue
        dist[a, b] = dist[b, a] = float(row["distance"])
    return dist[edge_i, edge_j]


def build_edge_covariates(
    dataset: str, node_order: list, edge_i: np.ndarray, edge_j: np.ndarray, W: np.ndarray, locations_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    deg = W.sum(axis=1)
    endpoint_degree = 0.5 * (deg[edge_i] + deg[edge_j])
    if dataset in PEMS_TIME_ANCHORS:
        graph_csv = f"./datasets/{dataset.split('_')[0].upper()}_graph.csv"
        return pems_edge_distance(graph_csv, edge_i, edge_j, W.shape[0]), endpoint_degree
    return edge_covariates(node_order, edge_i, edge_j, W, locations_path)


# ─────────────────────────────────────────────────────────────────────────
# Long-span in-sample inference, chunked over rolling origins
# ─────────────────────────────────────────────────────────────────────────

def assert_categorical_vocab_matches(model, loader) -> None:
    """Verify the dataloader's categorical vocabularies match the checkpoint's
    embedding tables before any forward pass.

    A mismatch means the preprocessing being applied now is not the
    preprocessing the model was fitted with. Left unchecked it surfaces as an
    out-of-bounds index inside the CUDA embedding kernel, which corrupts the
    context and then reports a nonsense downstream error (a garbage
    `decoder_length` of 4295000061 and a zero-length encoder), sending you
    debugging the wrong function entirely. Check it up front instead.
    """
    dataset = loader.dataset
    embeddings = model.embeddings.embeddings
    for name in dataset.categoricals:
        if name not in embeddings:
            continue
        n_classes = len(dataset.categorical_encoders[name].classes_)
        n_rows = embeddings[name].weight.shape[0]
        assert n_classes == n_rows, (
            f"categorical '{name}': the dataloader encodes {n_classes} classes but the checkpoint's "
            f"embedding table has {n_rows} rows. The preprocessing in build_test_dataloader does not "
            f"reproduce the one this checkpoint was trained with -- fix the preprocessing rather than "
            f"the embedding, or every residual computed here is from a model fed inputs it never saw."
        )


def collect_residuals(
    checkpoint: str,
    model_name: str,
    dataset: str,
    device: str,
    batch_size: int,
    n_samples: int,
    first_origin: int,
    last_origin: int,
    chunk_origins: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Residuals over origins [first_origin, last_origin], evaluated in chunks.

    Returns (residuals (R, N, Q), origin_time_idx (R,), node_order).
    residual = predictive mean over MC samples - actual, i.e. the point-forecast
    error, matching curvature_bottleneck_residual_corr.py exactly.
    """
    model = load_checkpoint_model(checkpoint, model_name, device)

    # The predictive mean is a Monte-Carlo average over `n_samples` draws from
    # the model's own predictive distribution, so it is a random quantity.
    # Leaving torch unseeded makes every number downstream irreproducible: two
    # runs of the companion held-out script differ in mean |rho| by ~0.015 and
    # flip the sign of its median-split statistic. Seed once, before any draw.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    residual_chunks, origin_chunks, node_order = [], [], None
    starts = list(range(first_origin, last_origin + 1, chunk_origins))
    for chunk_no, start in enumerate(starts, 1):
        stop = min(start + chunk_origins - 1, last_origin)
        loader, node_order_chunk, Q = build_test_dataloader(
            dataset, batch_size, min_prediction_idx=start, max_prediction_idx=stop + 12 - 1,
        )
        if node_order is None:
            node_order = node_order_chunk
            assert_categorical_vocab_matches(model, loader)
        assert node_order_chunk == node_order, "node ordering changed between chunks"
        N = len(node_order)

        with torch.no_grad():
            (raw_predictions, x), _ = model.predict(
                loader, mode="raw", n_samples=n_samples, show_progress_bar=False, return_x=True,
            )
        preds = raw_predictions["prediction"].cpu().float()
        actuals = x["decoder_target"].cpu().float()
        dec_time = x["decoder_time_idx"].cpu().numpy()
        groups = x["groups"].cpu().numpy().reshape(-1)

        assert preds.shape[0] % N == 0, f"predictions ({preds.shape[0]}) not a multiple of N={N}"
        R = preds.shape[0] // N
        preds = preds.reshape(R, N, preds.shape[1], preds.shape[2])
        actuals = actuals.reshape(R, N, actuals.shape[1])
        dec_time = dec_time.reshape(R, N, -1)
        groups = groups.reshape(R, N)

        # The (R, N) reshape above assumes origin-major / node-minor ordering.
        # Verify it rather than trust it: every node in a row must share the
        # same decoder timestamps, and every row must list nodes in encoder
        # order (which is the order W's rows are in).
        assert (dec_time == dec_time[:, :1, :]).all(), "reshape mismatch: nodes in a row disagree on time"
        assert (groups == np.arange(N)[None, :]).all(), "reshape mismatch: node axis is not in encoder order"

        residual_chunks.append((preds.mean(dim=-1) - actuals).numpy())
        origin_chunks.append(dec_time[:, 0, 0])
        print(f"  chunk {chunk_no}/{len(starts)}: origins {start}-{stop} -> {R} windows", flush=True)

    residuals = np.concatenate(residual_chunks, axis=0)
    origin_time_idx = np.concatenate(origin_chunks, axis=0)
    assert np.all(np.diff(origin_time_idx) > 0), "rolling origins are not strictly increasing"
    return residuals, origin_time_idx, node_order


# ─────────────────────────────────────────────────────────────────────────
# Per-window statistics
# ─────────────────────────────────────────────────────────────────────────

def window_mask(stamps: pd.DatetimeIndex, hour_ranges: List[Tuple[int, int]], daytype: str) -> np.ndarray:
    hour = stamps.hour.values + stamps.minute.values / 60.0
    in_hours = np.zeros(len(stamps), dtype=bool)
    for lo, hi in hour_ranges:
        in_hours |= (hour >= lo) & (hour < hi)
    is_weekend = stamps.weekday.values >= 5
    if daytype == "weekday":
        return in_hours & ~is_weekend
    if daytype == "weekend":
        return in_hours & is_weekend
    raise ValueError(f"unknown daytype {daytype!r}")


def spearman_from_ranks(rank_x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks. y is continuous (absolute Pearson
    correlations), so ordinal ranks equal average ranks; asserted by the caller."""
    rank_y = np.empty(len(y))
    rank_y[np.argsort(y, kind="stable")] = np.arange(1, len(y) + 1)
    xc = rank_x - rank_x.mean()
    yc = rank_y - rank_y.mean()
    return float(xc @ yc / np.sqrt((xc @ xc) * (yc @ yc)))


def effective_sample_size(X: np.ndarray, origin_time_idx: np.ndarray) -> float:
    """AR(1)-adjusted effective number of independent origins, T*(1-phi)/(1+phi).

    phi is the lag-1 autocorrelation estimated only from consecutive-in-time
    origin pairs (windows are broken by nightly gaps, so non-adjacent pairs must
    not be treated as lag-1). Descriptive only -- the inferential null is the
    node permutation, which needs no independence assumption over time.
    """
    T = X.shape[0]
    adjacent = np.diff(origin_time_idx) == 1
    if adjacent.sum() < 2:
        return float(T)
    Xc = X - X.mean(axis=0, keepdims=True)
    a, b = Xc[:-1][adjacent], Xc[1:][adjacent]
    phi = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    phi = min(max(phi, 0.0), 0.99)
    return float(T * (1 - phi) / (1 + phi))


def holm(p_values: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(p_values, key=lambda k: p_values[k])
    m, adjusted, running = len(keys), {}, 0.0
    for rank, key in enumerate(keys):
        running = max(running, (m - rank) * p_values[key])
        adjusted[key] = float(min(running, 1.0))
    return adjusted


def analyze_windows(
    residuals_h1: np.ndarray,
    origin_time_idx: np.ndarray,
    stamps: pd.DatetimeIndex,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    b_edge: np.ndarray,
    distance_edge: np.ndarray,
    degree_edge: np.ndarray,
    n_perm: int,
    n_qbins: int,
    min_origins: int,
    seed: int,
) -> Dict:
    """All windows share one node-permutation stream so that between-window
    contrasts are paired (the same relabeling is applied to every window)."""
    import statsmodels.api as sm

    N = residuals_h1.shape[1]
    valid = np.isfinite(distance_edge)
    n_dropped = int((~valid).sum())
    ei, ej = edge_i[valid], edge_j[valid]
    b_v, dist_v, deg_v = b_edge[valid], distance_edge[valid], degree_edge[valid]

    X_design = sm.add_constant(np.column_stack([zscore(b_v), zscore(dist_v), zscore(deg_v)]))
    ols_operator = np.linalg.pinv(X_design.T @ X_design) @ X_design.T  # (4, n_valid)
    rank_b = rankdata(b_v)

    # Nuisance-only design (distance + degree, no bottleneck score): used to
    # strip the distance confound out of |rho| before taking the rank
    # correlation, giving an adjusted *monotone* statistic to sit alongside the
    # adjusted *linear* one (beta_b) and the unadjusted monotone one (rho_S).
    X_nuis = sm.add_constant(np.column_stack([zscore(dist_v), zscore(deg_v)]))
    nuis_operator = np.linalg.pinv(X_nuis.T @ X_nuis) @ X_nuis.T  # (3, n_valid)

    def adjusted_spearman(y: np.ndarray) -> float:
        return spearman_from_ranks(rank_b, y - X_nuis @ (nuis_operator @ y))

    # Enumerate every window we will evaluate: pre-specified (confirmatory,
    # Holm-corrected), the peak union used by the primary contrast, and the
    # 24 hour-of-day bins (descriptive only).
    specs = []
    for daytype in ("weekday", "weekend"):
        for label, hours, regime in PRESPECIFIED_WINDOWS:
            specs.append((f"{label}__{daytype}", hours, daytype, regime, "prespecified"))
        peak_hours = [h for lbl, hs, _ in PRESPECIFIED_WINDOWS if lbl in PEAK_WINDOWS for h in hs]
        specs.append((f"peak_union__{daytype}", peak_hours, daytype, "congested", "primary_contrast"))
    for hour in range(24):
        specs.append((f"hour{hour:02d}__weekday", [(hour, hour + 1)], "weekday", "n/a", "descriptive"))

    windows, obs_beta, obs_rho, obs_rho_adj = {}, {}, {}, {}
    for name, hours, daytype, regime, role in specs:
        mask = window_mask(stamps, hours, daytype)
        n_origins = int(mask.sum())
        if n_origins < min_origins:
            windows[name] = {
                "role": role, "regime": regime, "daytype": daytype, "hour_ranges": hours,
                "n_origins": n_origins, "skipped_reason": f"fewer than min_origins={min_origins}",
            }
            continue

        Xw = residuals_h1[mask]                       # (T_w, N)
        corr = np.corrcoef(Xw, rowvar=False)
        assert corr.shape == (N, N)
        y = np.abs(corr[ei, ej])
        assert np.isfinite(y).all(), f"window {name}: non-finite residual correlations"
        n_ties = len(y) - len(np.unique(y))
        assert n_ties / len(y) < 1e-3, f"window {name}: {n_ties} tied |rho| values, ordinal ranks unsafe"

        beta = float((ols_operator @ y)[1])
        rho_s = spearman_from_ranks(rank_b, y)
        rho_s_adj = adjusted_spearman(y)
        T_eff = effective_sample_size(Xw, origin_time_idx[mask])

        obs_beta[name], obs_rho[name], obs_rho_adj[name] = beta, rho_s, rho_s_adj
        windows[name] = {
            "role": role, "regime": regime, "daytype": daytype, "hour_ranges": hours,
            "n_origins": n_origins,
            "n_day_blocks": int(np.sum(np.diff(origin_time_idx[mask]) != 1) + 1),
            "effective_n_origins_ar1": T_eff,
            "independence_noise_floor_at_T_eff": independence_noise_floor(max(T_eff, 2.0)),
            "mean_abs_corr": float(y.mean()),
            "beta_bottleneck_adjusted": beta,
            "spearman_rho_unadjusted": rho_s,
            "spearman_rho_adjusted": rho_s_adj,
            "quantile_bins": quantile_bin_summary(b_v, y, n_qbins, floor=independence_noise_floor(max(T_eff, 2.0))),
        }

    # ── Node-permutation null, shared stream across windows ──────────────
    rng = np.random.default_rng(seed)
    live = [name for name in obs_beta]
    null_beta = {name: np.empty(n_perm) for name in live}
    null_rho = {name: np.empty(n_perm) for name in live}
    null_rho_adj = {name: np.empty(n_perm) for name in live}
    corr_cache = {
        name: np.corrcoef(residuals_h1[window_mask(stamps, windows[name]["hour_ranges"], windows[name]["daytype"])], rowvar=False)
        for name in live
    }
    for k in range(n_perm):
        perm = rng.permutation(N)
        pi, pj = perm[ei], perm[ej]
        for name in live:
            y_perm = np.abs(corr_cache[name][pi, pj])
            null_beta[name][k] = (ols_operator @ y_perm)[1]
            null_rho[name][k] = spearman_from_ranks(rank_b, y_perm)
            null_rho_adj[name][k] = adjusted_spearman(y_perm)

    for name in live:
        for stat, obs, null in (("beta_bottleneck_adjusted", obs_beta[name], null_beta[name]),
                                ("spearman_rho_unadjusted", obs_rho[name], null_rho[name]),
                                ("spearman_rho_adjusted", obs_rho_adj[name], null_rho_adj[name])):
            p_two = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (n_perm + 1)
            p_one = (1 + np.sum(null >= obs)) / (n_perm + 1)
            windows[name][f"{stat}_null"] = {
                "null_mean": float(null.mean()), "null_std": float(null.std()),
                "z_score": float((obs - null.mean()) / null.std()) if null.std() > 0 else float("nan"),
                "p_value_two_sided": float(p_two), "p_value_one_sided_greater": float(p_one),
            }

    prespecified = [n for n in live if windows[n]["role"] == "prespecified"]
    holm_beta = holm({n: windows[n]["beta_bottleneck_adjusted_null"]["p_value_two_sided"] for n in prespecified})
    for name, p_adj in holm_beta.items():
        windows[name]["beta_bottleneck_adjusted_null"]["p_value_holm"] = p_adj

    # ── Primary + control contrasts (paired: same permutation both windows) ──
    contrasts = {}
    for daytype in ("weekday", "weekend"):
        peak, ref = f"peak_union__{daytype}", f"{REFERENCE_WINDOW}__{daytype}"
        if peak not in live or ref not in live:
            contrasts[daytype] = {"skipped_reason": "one or both windows had too few origins"}
            continue
        delta_obs = obs_beta[peak] - obs_beta[ref]
        delta_null = null_beta[peak] - null_beta[ref]
        contrasts[daytype] = {
            "definition": f"beta_b({peak}) - beta_b({ref})",
            "role": "PRIMARY (pre-specified, one-sided H1: Delta > 0)" if daytype == "weekday"
                    else "CONTROL (pre-specified falsifier; H1 predicts smaller than weekday)",
            "beta_peak": obs_beta[peak], "beta_reference": obs_beta[ref], "delta": float(delta_obs),
            "null_mean": float(delta_null.mean()), "null_std": float(delta_null.std()),
            "z_score": float((delta_obs - delta_null.mean()) / delta_null.std()) if delta_null.std() > 0 else float("nan"),
            "p_value_one_sided_greater": float((1 + np.sum(delta_null >= delta_obs)) / (n_perm + 1)),
            "p_value_two_sided": float((1 + np.sum(np.abs(delta_null - delta_null.mean()) >= abs(delta_obs - delta_null.mean()))) / (n_perm + 1)),
        }
    if "delta" in contrasts.get("weekday", {}) and "delta" in contrasts.get("weekend", {}):
        contrasts["weekday_minus_weekend"] = {
            "role": "H1 requires this to be positive",
            "delta_of_deltas": contrasts["weekday"]["delta"] - contrasts["weekend"]["delta"],
        }

    return {
        "n_edges_used": int(valid.sum()),
        "n_edges_dropped_missing_distance": n_dropped,
        "n_perm": n_perm,
        "windows": windows,
        "contrasts": contrasts,
    }


# ─────────────────────────────────────────────────────────────────────────

def make_figure(results: Dict, fig_out: Path, dataset: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = results["analysis"]["windows"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    # (1) hour-of-day profile of the adjusted coefficient, with the null band.
    hours, beta, lo, hi = [], [], [], []
    for h in range(24):
        w = windows.get(f"hour{h:02d}__weekday", {})
        if "beta_bottleneck_adjusted" not in w:
            continue
        null = w["beta_bottleneck_adjusted_null"]
        hours.append(h)
        beta.append(w["beta_bottleneck_adjusted"])
        lo.append(null["null_mean"] - 1.96 * null["null_std"])
        hi.append(null["null_mean"] + 1.96 * null["null_std"])
    ax = axes[0]
    ax.fill_between(hours, lo, hi, color="gray", alpha=0.3, label="node-permutation null (95%)")
    ax.plot(hours, beta, marker="o", color="darkorange", label=r"observed $\beta_b$")
    ax.axhline(0, color="black", lw=0.8)
    for lbl, hs, _ in PRESPECIFIED_WINDOWS:
        if lbl in PEAK_WINDOWS:
            for l, r in hs:
                ax.axvspan(l, r, color="tab:red", alpha=0.10)
    ax.set_xlabel("hour of day (weekdays)")
    ax.set_ylabel(r"distance-adjusted $\beta_b$")
    ax.set_title("Bottleneck$\\to$residual-correlation coupling\nby hour (shaded = pre-specified peaks)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (2) pre-specified windows, weekday vs weekend.
    ax = axes[1]
    labels = [lbl for lbl, _, _ in PRESPECIFIED_WINDOWS]
    xs = np.arange(len(labels))
    for offset, daytype, color in ((-0.17, "weekday", "tab:blue"), (0.17, "weekend", "tab:gray")):
        vals, errs = [], []
        for lbl in labels:
            w = windows.get(f"{lbl}__{daytype}", {})
            vals.append(w.get("beta_bottleneck_adjusted", np.nan))
            errs.append(w.get("beta_bottleneck_adjusted_null", {}).get("null_std", np.nan))
        ax.errorbar(xs + offset, vals, yerr=1.96 * np.array(errs, dtype=float), marker="s", ls="none",
                    capsize=3, color=color, label=daytype)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"distance-adjusted $\beta_b$")
    ax.set_title("Pre-specified regimes\n(bars = 95% node-permutation null)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (3) quartile profile: peak vs night -- the monotonicity the reviewer asked about.
    ax = axes[2]
    for name, color, style in ((f"peak_union__weekday", "tab:red", "-"), (f"{REFERENCE_WINDOW}__weekday", "tab:blue", "--")):
        w = windows.get(name, {})
        rows = w.get("quantile_bins", [])
        if not rows:
            continue
        ax.plot([r["bin"] for r in rows], [r["mean_abs_corr"] for r in rows],
                marker="o", ls=style, color=color,
                label=f"{name.split('__')[0]} ($\\rho_S$={w['spearman_rho_unadjusted']:+.3f})")
    ax.set_xlabel("bottleneck-score quartile (0 = least bottlenecked)")
    ax.set_ylabel(r"mean $|\rho_{ij}|$ (residuals)")
    ax.set_title("Quartile profile, congested vs free-flow", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Time-resolved residual correlation vs. curvature bottleneck score — {dataset}/xLSTM "
        f"(in-sample span {results['span']['first_timestamp']} → {results['span']['last_timestamp']})",
        fontsize=10,
    )
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="xLSTM")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None, help="defaults to N (one rolling origin per batch)")
    parser.add_argument("--n_samples", type=int, default=100, help="MC samples for the predictive mean; 100 matches the held-out-test run")
    parser.add_argument("--span_start", required=True, help="first date of the evaluation span, YYYY-MM-DD (inclusive)")
    parser.add_argument("--span_end", required=True, help="last date of the evaluation span, YYYY-MM-DD (inclusive)")
    parser.add_argument("--exclude_dates", default="", help="comma-separated YYYY-MM-DD holidays to drop from the span")
    parser.add_argument("--origin_stride", type=int, default=1, help="keep every k-th rolling origin")
    parser.add_argument("--chunk_origins", type=int, default=288)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--n_qbins", type=int, default=4)
    parser.add_argument("--min_origins", type=int, default=30)
    parser.add_argument("--locations-path", default="./datasets/locations.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--fig-out", required=True)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    with open("./datasets/pred_horizon_dict_v1.pkl", "rb") as f:
        prediction_horizon = pickle.load(f)[args.dataset]
    with open("./datasets/pred_rolling_dict_v1.pkl", "rb") as f:
        num_pred_rolling = pickle.load(f)[args.dataset]
    with open("./datasets/dataset_freq_v1.pkl", "rb") as f:
        freq = pickle.load(f)[args.dataset]
    assert freq == "5min", f"time-of-day windows are defined for 5-min traffic data; {args.dataset} is {freq}"

    data = pd.read_csv(f"./datasets/{args.dataset}.csv")
    time_index = build_time_index(args.dataset, data)

    # Evaluation span, chosen on calendar grounds only (see SPAN SELECTION in
    # the module docstring). The final origin is clipped so its whole decoder
    # window stays at or before training_cutoff, i.e. strictly inside the data
    # the model was fitted on -- in-sample by construction, see the caveat.
    max_idx = int(data["time_idx"].max())
    validation_cutoff = max_idx - prediction_horizon - num_pred_rolling + 1
    training_cutoff = validation_cutoff - (max_idx - validation_cutoff)

    span_start = pd.Timestamp(args.span_start)
    span_end = pd.Timestamp(args.span_end) + pd.Timedelta(days=1)  # inclusive of the end date
    in_span = (time_index >= span_start) & (time_index < span_end)
    span_idx = time_index.index[in_span].to_numpy()
    assert len(span_idx) > 0, f"no timesteps in [{args.span_start}, {args.span_end}]"

    first_origin = int(span_idx.min())
    last_origin = min(int(span_idx.max()) - prediction_horizon + 1, training_cutoff - prediction_horizon + 1)
    assert last_origin > first_origin, "evaluation span is shorter than one decoder window"
    assert last_origin + prediction_horizon - 1 <= training_cutoff, "span leaks past the training cutoff"

    time_base = validate_time_base(args.dataset, data, time_index, span_idx)
    print(f"Time base validated on span: diurnal trough {time_base['diurnal_trough_hour']:02d}:00, "
          f"lowest day-of-week {time_base['lowest_flow_day_of_week']} (Sunday), "
          f"weekend {time_base['weekend_mean_flow']:.2f} < weekday {time_base['weekday_mean_flow']:.2f}")
    print(f"Evaluation span: origins {first_origin}-{last_origin} "
          f"({time_index.loc[first_origin]} -> {time_index.loc[last_origin + prediction_horizon - 1]}), "
          f"training_cutoff={training_cutoff}")

    residuals, origin_time_idx, node_order = collect_residuals(
        args.checkpoint, args.model, args.dataset, device,
        args.batch_size or int(data["sensor"].nunique()),
        args.n_samples, first_origin, last_origin, args.chunk_origins, args.seed,
    )
    R, N, Q = residuals.shape
    print(f"residuals: R={R} origins, N={N} nodes, Q={Q} horizons")

    if args.origin_stride > 1:
        keep = np.arange(0, R, args.origin_stride)
        residuals, origin_time_idx = residuals[keep], origin_time_idx[keep]
        print(f"strided to {len(keep)} origins (every {args.origin_stride})")

    stamps = pd.DatetimeIndex(time_index.loc[origin_time_idx].values)
    excluded_dates = [d.strip() for d in args.exclude_dates.split(",") if d.strip()]
    if excluded_dates:
        drop = np.isin(stamps.date.astype(str), np.array(excluded_dates))
        assert drop.sum() > 0, f"none of --exclude_dates {excluded_dates} fall inside the span"
        residuals, origin_time_idx, stamps = residuals[~drop], origin_time_idx[~drop], stamps[~drop]
        print(f"excluded {int(drop.sum())} origins on holidays {excluded_dates} -> {len(stamps)} origins")

    residuals_h1 = residuals[:, :, 0]

    W, params = load_curvature_state(Path(args.checkpoint))
    assert W.shape == (N, N), f"graph size {W.shape} != number of sensors {N}"
    edge_i, edge_j, b_edge, _ = edge_arrays(W, params["kappa_0"], params["tau"])
    distance_edge, degree_edge = build_edge_covariates(
        args.dataset, node_order, edge_i, edge_j, W, args.locations_path,
    )
    print(f"{len(edge_i)} edges; distance available for {int(np.isfinite(distance_edge).sum())}")

    analysis = analyze_windows(
        residuals_h1, origin_time_idx, stamps, edge_i, edge_j, b_edge, distance_edge, degree_edge,
        args.n_perm, args.n_qbins, args.min_origins, args.seed,
    )

    results = {
        "script": "curvature_bottleneck_residual_corr_timeresolved.py",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "model": args.model,
        "n_nodes": N,
        "n_edges": int(len(edge_i)),
        "curvature_params": params,
        "n_samples_mc": args.n_samples,
        "seed": args.seed,
        "sample_status": "IN-SAMPLE (evaluation span lies inside the training period; see module docstring)",
        "span": {
            "first_origin_time_idx": first_origin, "last_origin_time_idx": last_origin,
            "first_timestamp": str(time_index.loc[first_origin]),
            "last_timestamp": str(time_index.loc[last_origin + prediction_horizon - 1]),
            "span_start": args.span_start, "span_end": args.span_end,
            "excluded_dates": excluded_dates, "origin_stride": args.origin_stride,
            "n_origins_used": int(len(origin_time_idx)),
            "training_cutoff_time_idx": training_cutoff,
            "held_out_test_span_hours": (max_idx - validation_cutoff) * 5 / 60.0,
        },
        "time_base": time_base,
        "prespecification": {
            "hypothesis": "beta_b is larger in congested (peak) than free-flow (night) windows on weekdays",
            "primary_test": "one-sided node-permutation test on beta_b(peak_union__weekday) - beta_b(night__weekday)",
            "control_test": "same contrast on weekends; H1 predicts it to be smaller",
            "windows": {lbl: {"hour_ranges": hrs, "regime": reg} for lbl, hrs, reg in PRESPECIFIED_WINDOWS},
            "multiplicity": "Holm across the 12 pre-specified window tests; 24 hour-of-day bins are descriptive only",
        },
        "analysis": analysis,
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    make_figure(results, Path(args.fig_out), args.dataset)

    print(f"\nWrote {json_out}\nWrote {args.fig_out}")
    print("\n--- pre-specified windows (weekday) ---")
    for lbl, _, regime in PRESPECIFIED_WINDOWS:
        w = analysis["windows"].get(f"{lbl}__weekday", {})
        if "beta_bottleneck_adjusted" not in w:
            print(f"  {lbl:14s} skipped ({w.get('skipped_reason')})")
            continue
        nb = w["beta_bottleneck_adjusted_null"]
        print(f"  {lbl:14s} [{regime:12s}] T={w['n_origins']:5d} (T_eff={w['effective_n_origins_ar1']:6.1f})  "
              f"beta_b={w['beta_bottleneck_adjusted']:+.5f}  z={nb['z_score']:+.2f}  "
              f"p={nb['p_value_two_sided']:.4f}  p_holm={nb['p_value_holm']:.4f}  "
              f"rho_S={w['spearman_rho_unadjusted']:+.4f}")
    print("\n--- contrasts ---")
    for key, c in analysis["contrasts"].items():
        print(f"  {key}: {json.dumps(c, default=float)}")


if __name__ == "__main__":
    main()
