import os
import glob
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- 
# style settings for plots 
# ----------------------------- 
def set_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def run_one_experiment(exp_dir: str, out_dir: str, mode: str = "per_exp", do_ba: bool = False):
    """Process and plot results for a single experiment."""
    global EXP_DIR, RESULT_DIR, FIG_DIR
    EXP_DIR = exp_dir
    RESULT_DIR = os.path.join(EXP_DIR, "results")

    # Generate output directory for this experiment (avoiding overwrites)
    exp_subdir = os.path.join(out_dir, exp_name(EXP_DIR))
    FIG_DIR = exp_subdir
    ensure_dir(FIG_DIR)

    # Apply plot style settings
    set_style()

    # ---- Start plotting logic ----
    # Load CSV and JSON files for the metrics
    df_seg = safe_read_csv(os.path.join(RESULT_DIR, "segment_metrics.csv"))
    df_clin = safe_read_csv(os.path.join(RESULT_DIR, "clinical_metrics.csv"))
    df_mask = safe_read_csv(os.path.join(RESULT_DIR, "mask_metrics.csv"))
    df_subject = safe_read_csv(os.path.join(RESULT_DIR, "subject_summary.csv"))
    global_data = safe_read_json(os.path.join(RESULT_DIR, "global_summary.json"))

    # 1. Box plots for segment-level metrics (PCC/MAE/RMSE)
    plot_box_across_exps(
        metric="PCC",
        exp_results=[df_seg],
        out_base=os.path.join(FIG_DIR, "box_seg_pcc"),
        title="Ablation (segment-level): PCC",
        source_hint="segment_metrics.csv[PCC]"
    )

    plot_box_across_exps(
        metric="MAE",
        exp_results=[df_seg],
        out_base=os.path.join(FIG_DIR, "box_seg_mae"),
        title="Ablation (segment-level): MAE",
        source_hint="segment_metrics.csv[MAE]"
    )

    plot_box_across_exps(
        metric="RMSE",
        exp_results=[df_seg],
        out_base=os.path.join(FIG_DIR, "box_seg_rmse"),
        title="Ablation (segment-level): RMSE",
        source_hint="segment_metrics.csv[RMSE]"
    )

    # 2. Scatter plots for HR (if applicable)
    if df_clin is not None:
        plot_scatter(
            df_clin, "HR_True", "HR_Pred",
            out_base=os.path.join(FIG_DIR, "scatter_hr"),
            title="HR Pred vs True",
            source_hint="clinical_metrics.csv[HR_True,HR_Pred]"
        )

    # 3. Bland-Altman plot for HR
    if do_ba:
        bland_altman(
            df_clin, "HR_True", "HR_Pred",
            out_base=os.path.join(FIG_DIR, "ba_hr"),
            title="Bland–Altman (HR)",
            source_hint="clinical_metrics.csv[HR_True,HR_Pred]"
        )

    # 4. Qualitative case studies (Best/Median/Worst cases)
    if global_data and "cases" in global_data:
        # Export a few single-case overlays (optional)
        for p in global_data["cases"][:12]:
            it = load_case_npz(p)
            fig = plt.figure(figsize=(7.0, 3.2))
            ax = fig.add_subplot(111)
            plot_waveform(ax, it["ecg_true"], it["ecg_pred"],
                          f"sid={it['subject_id']} seg={it['seg_id']} PCC={it['pcc']:.3f}\nData: cases/*.npz[ecg_true,ecg_pred]")
            ax.legend(loc="upper right", frameon=False)
            base = os.path.splitext(os.path.basename(p))[0]
            save_fig(fig, os.path.join(FIG_DIR, f"case_{base}"))

    # ---- Finish ----
    print(f"✅ Finished plotting results for {exp_dir}. Output saved to {FIG_DIR}")


def ensure_dir(p: str):
    """Ensure the given directory exists."""
    os.makedirs(p, exist_ok=True)


def save_fig(fig, out_base: str):
    """Save the figure in both PDF and PNG formats."""
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Safely read a CSV file."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] CSV read failed: {path} | {e}")
        return None


def safe_read_json(path: str) -> Optional[dict]:
    """Safely read a JSON file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON read failed: {path} | {e}")
        return None


def list_experiments(exp_root: str) -> List[str]:
    """List all experiments in a directory."""
    return [p for p in sorted(glob.glob(os.path.join(exp_root, "Exp_*"))) if os.path.isdir(p)]


def exp_name(exp_dir: str) -> str:
    """Extract the experiment name from the directory path."""
    return os.path.basename(exp_dir.rstrip("/"))


# ----------------------------- 
# Example plot functions (boxplot, scatter, etc.)
# -----------------------------
def plot_box_across_exps(metric: str, exp_results: List[Dict[str, Any]], out_base: str, title: str, source_hint: str):
    """Plot a box plot across multiple experiments."""
    arrays, labels = [], []
    for r in exp_results:
        df = r.get("segment", None)
        if df is not None and metric in df.columns:
            arrays.append(nan_clean(df[metric].values))
            labels.append(r["exp_name"])
    if len(arrays) < 2:
        return
    fig = plt.figure(figsize=(max(6.0, 0.55 * len(labels)), 3.8))
    ax = fig.add_subplot(111)
    ax.boxplot(arrays, showfliers=False)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.text(0.01, -0.22, f"Data: {source_hint}", transform=ax.transAxes, ha="left", va="top")
    save_fig(fig, out_base)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_base: str, title: str, source_hint: str):
    """Plot a scatter plot."""
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return
    x = df[x_col].values
    y = df[y_col].values
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size == 0:
        return
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, alpha=0.6)
    mn = np.min([x.min(), y.min()])
    mx = np.max([x.max(), y.max()])
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.text(0.02, 0.02, f"Data: {source_hint}", transform=ax.transAxes, va="bottom")
    save_fig(fig, out_base)


# -----------------------------
# Main entry point (adjusted for argparse)
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the experiment directory (Exp_XXX)")
    parser.add_argument("--exp_root", type=str, default="experiments",
                        help="Root directory containing Exp_* folders")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for figures. Default: <exp_dir>/figures or <exp_root>/_figures")
    parser.add_argument("--mode", type=str, default="per_exp", choices=["all", "per_exp", "ablation", "paper"],
                        help="Plot mode (all, per_exp, ablation, paper)")
    parser.add_argument("--do_ba", action="store_true", help="Enable Bland-Altman plots")

    args = parser.parse_args()

    # Determine experiment directories to plot
    if args.exp_dir:
        exp_dirs = [args.exp_dir]
        out_dir_default = os.path.join(args.exp_dir, "figures")
    else:
        exp_dirs = list_experiments(args.exp_root)
        out_dir_default = os.path.join(args.exp_root, "_figures")

    out_dir = args.out_dir if args.out_dir else out_dir_default
    ensure_dir(out_dir)

    # Run the plot generation for each experiment
    for exp_dir in exp_dirs:
        run_one_experiment(exp_dir, out_dir=out_dir, mode=args.mode, do_ba=args.do_ba)


if __name__ == "__main__":
    main()
