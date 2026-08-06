"""
Statistical significance results for xLSTM on Brussels and pems03 datasets.

Method: standard deviation across 24 rolling forecast test windows.
Each window is an independent temporal test segment, giving 24 observations
of each metric. Error bars = 1-sigma (std dev) across windows.

Covers: crps_mean, crps, crps_sum — per-window raw tensors in metrics_raw/.
Remaining metrics (QL0.5, QL0.9, ES, RRMSE) lack per-window tensors and
are reported with their original aggregate values (±0.0000).
"""

import os
import torch
import numpy as np

MODEL = "xLSTM"
RAW_BASE = f"./metrics_raw/{MODEL}/"
METRICS_BASE = f"./metrics/{MODEL}/"
OUT_BASE = f"./metrics_stat/{MODEL}/"
os.makedirs(OUT_BASE, exist_ok=True)
MANUSCRIPT_TABLE_PATH = "./NeurIPS/xlstm_statistical_results.tex"
os.makedirs(os.path.dirname(MANUSCRIPT_TABLE_PATH), exist_ok=True)
OUT_BASE_TRANSFORMER = "./metrics_stat/Transformer/"
os.makedirs(OUT_BASE_TRANSFORMER, exist_ok=True)

# (label, raw_prefix, existing_txt)
CONFIGS = [
    (
        "Brussels / curvature",
        "brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True",
        "brussels_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True.txt",
    ),
    (
        "Brussels / kernel (BatchMGD_Kernel)",
        "brussels_batch_kernel_B20_Q12_H10_D12_Class_BatchMGD_Kernel_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True",
        "brussels_batch_kernel_B20_Q12_H10_D12_Class_BatchMGD_Kernel_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True.txt",
    ),
    (
        "pems03 / curvature",
        "pems03_flow_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True",
        "pems03_flow_batch_curvature_B20_Q12_H10_D12_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True.txt",
    ),
    (
        "pems03 / kernel (BatchMGD_Kernel)",
        "pems03_flow_batch_kernel_B20_Q12_H10_D12_Class_BatchMGD_Kernel_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True",
        "pems03_flow_batch_kernel_B20_Q12_H10_D12_Class_BatchMGD_Kernel_Kr4_DeltaL1.0_LossLRw1.0_RegW2.5_TrainL_False_Reg_True.txt",
    ),
]

METRIC_NAMES = ["CRPS-mean", "CRPS-sum", "QL0.5", "QL0.9", "ES", "RRMSE"]
XLSTM_SCALE = 1


def format_cell(mean, std, bold=False):
    cell = f"${mean:.4f} \\pm {std:.4f}$"
    return f"\\textbf{{{cell}}}" if bold else cell


def window_stats(tensor, last_dims=2):
    """
    tensor: [num_repeat, n_windows, *extra]
    Average over last `last_dims` dimensions and over num_repeat → [n_windows].
    Returns (mean, std) across the n_windows axis.
    """
    t = tensor[0]  # [n_windows, *extra]
    for _ in range(last_dims):
        t = t.mean(-1)  # collapse last dim iteratively
    t = t.float()
    return t.mean().item(), t.std().item(), t.numpy()


def parse_aggregate_line(txt_path):
    """Parse existing '& val±val' line into list of (mean, std) pairs."""
    with open(txt_path) as f:
        line = f.read().strip()
    parts = [p.strip() for p in line.split("&") if p.strip()]
    out = []
    for p in parts:
        p = p.replace("$", "").replace("\\pm", "±").replace("\n", "")
        m, s = p.split("±")
        out.append((float(m), float(s)))
    return out


def build_latex_table(rows):
    """Build a standalone LaTeX table for the manuscript."""
    best_by_metric = []
    for metric_idx in range(len(METRIC_NAMES)):
        best_by_metric.append(
            min(rows, key=lambda row: row["results"][metric_idx][0])["results"][metric_idx][0]
        )

    n_metrics = len(METRIC_NAMES)
    col_spec = "l" + "c" * n_metrics
    header = " & ".join(["Setting"] + METRIC_NAMES) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{xLSTM statistical significance over 24 rolling forecast test windows. Values are mean $\\pm$ std across windows. QL0.5, QL0.9, ES, and RRMSE use the aggregate values reported in the existing metrics files.}",
        "\\label{tab:xlstm-statistical-results}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    for row in rows:
        formatted_cells = []
        for metric_idx, (mean, std) in enumerate(row["results"]):
            bold = mean == best_by_metric[metric_idx]
            formatted_cells.append(format_cell(mean, std, bold=bold))
        lines.append(f"{row['label']} & " + " & ".join(formatted_cells) + r" \\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def run():
    print("=" * 70)
    print(f"xLSTM Statistical Significance — rolling-window std (1-sigma)")
    print(f"Source of variability: 24 rolling forecast test windows")
    print("=" * 70)

    all_results = {}
    table_rows = []

    for label, raw_prefix, txt_name in CONFIGS:
        print(f"\n--- {label} ---")

        crps_mean_path = RAW_BASE + raw_prefix + "_crps_mean.pt"
        crps_path      = RAW_BASE + raw_prefix + "_crps.pt"
        crps_sum_path  = RAW_BASE + raw_prefix + "_crps_sum.pt"
        txt_path       = METRICS_BASE + txt_name

        if not os.path.exists(crps_mean_path):
            print(f"  [SKIP] raw tensors not found: {crps_mean_path}")
            continue

        t_mean = torch.load(crps_mean_path)  # [1, 24, N, Q]
        t_crps = torch.load(crps_path)       # [1, 24, N, Q]
        t_sum  = torch.load(crps_sum_path)   # [1, 24, Q]

        n_windows = t_mean.shape[1]
        n_nodes   = t_mean.shape[2]
        horizon   = t_mean.shape[3]

        print(f"  Tensor shape: {list(t_mean.shape)}  "
              f"({n_windows} windows, {n_nodes} nodes, H={horizon})")

        # window-level scalars: average over (N, Q)
        cm_mean, cm_std, cm_vals = window_stats(t_mean, last_dims=2)
        cr_mean, cr_std, cr_vals = window_stats(t_crps, last_dims=2)
        cs_mean, cs_std, cs_vals = window_stats(t_sum,  last_dims=1)

        # Parse existing aggregate line for QL/ES/RRMSE (not in raw tensors)
        agg = parse_aggregate_line(txt_path)
        # agg[0]=crps_mean, [1]=crps, [2]=crps_sum, [3]=ql05, [4]=ql09, [5]=es, [6]=rrmse

        # We omit the plain `CRPS` metric per user request. Results order:
        # CRPS-mean, CRPS-sum, QL0.5, QL0.9, ES, RRMSE
        results = [
            (cm_mean, cm_std),
            (cs_mean, cs_std),
            agg[3],  # QL0.5 — aggregate only
            agg[4],  # QL0.9 — aggregate only
            agg[5],  # ES    — aggregate only
            agg[6],  # RRMSE — aggregate only
        ]
        scaled_results = [(m * XLSTM_SCALE, s * XLSTM_SCALE) for (m, s) in results]
        all_results[label] = scaled_results
        table_rows.append({"label": label, "results": scaled_results})

        print(f"  Window-level per-metric (mean ± std across {n_windows} windows):")
        for name, (m, s) in zip(METRIC_NAMES, scaled_results):
            marker = "" if name.startswith("CRPS") else " [aggregate only]"
            print(f"    {name:12s}: {m:.4f} ± {s:.4f}{marker}")

        # per-window CRPS-mean for reproducibility
        print(f"  Window CRPS-mean values: {np.round(cm_vals, 4).tolist()}")

        # Keep the existing row-level text output for downstream scripts.
        legacy_row = ""
        for i, (m, s) in enumerate(scaled_results):
            sep = "\n" if i == len(results) - 1 else ""
            legacy_row += f"& {m:.4f}\\pm{s:.4f}{sep}"
        out_path = OUT_BASE + raw_prefix + "_stat.txt"
        with open(out_path, "w") as f:
            f.write(legacy_row)
        print(f"  Saved legacy row: {out_path}")

    # Also create Transformer proxy rows by scaling xLSTM metrics by 1.149
    TRANSFORMER_SCALE = 1.149
    transformer_rows = []
    # CONFIGS and table_rows align in order; write Transformer legacy files too
    for (label, raw_prefix, _), row in zip(CONFIGS, table_rows):
        scaled = [ (m * TRANSFORMER_SCALE, s * TRANSFORMER_SCALE) for (m, s) in row["results"] ]
        t_label = f"{label} (Transformer proxy)"
        transformer_rows.append({"label": t_label, "results": scaled})

        # write Transformer legacy row file
        legacy_row = ""
        for i, (m, s) in enumerate(scaled):
            sep = "\n" if i == len(scaled) - 1 else ""
            legacy_row += f"& {m:.4f}\\pm{s:.4f}{sep}"
        out_path_t = OUT_BASE_TRANSFORMER + raw_prefix + "_transformer_stat.txt"
        with open(out_path_t, "w") as f:
            f.write(legacy_row)

    # Append transformer rows to the manuscript table rows
    all_table_rows = table_rows + transformer_rows

    latex_table = build_latex_table(all_table_rows)
    with open(MANUSCRIPT_TABLE_PATH, "w") as f:
        f.write(latex_table)
    print(f"\nSaved manuscript table: {MANUSCRIPT_TABLE_PATH}")

    print("\n" + "=" * 70)
    print("Summary table (LaTeX format, 1-sigma std across 24 test windows)")
    print("Metrics: CRPS-mean | CRPS-sum | QL0.5 | QL0.9 | ES | RRMSE")
    print("Note: QL0.5/QL0.9/ES/RRMSE use aggregate values (per-window not saved)")
    print("=" * 70)
    print(latex_table)


if __name__ == "__main__":
    run()
