"""
plot_figures.py (Publication-oriented)
=====================================
This script generates paper-ready figures from test.py outputs.

Expected per-experiment result files under:
  experiments/Exp_*/results/
    - segment_metrics.csv     # seg_id, Subject_ID, PCC, MAE, RMSE
    - clinical_metrics.csv    # seg_id, Subject_ID, HR_True, HR_Pred, ..., HR_Error, RR_Error, ...
    - mask_metrics.csv        # seg_id, Subject_ID, Mask_F1, Mask_Precision, Mask_Recall, ...
    - subject_summary.csv     # Subject_ID + subject-wise mean metrics
    - global_summary.json     # global stats (mean/std/quantiles)

Optional (recommended) cases exported by test.py:
  experiments/Exp_*/results/cases/*.npz
    keys: radar, ecg_true, ecg_pred, mask_true, mask_prob, pcc, seg_id, subject_id

Figures will be saved to:
  <out_dir>/<ExpName>/*.pdf and *.png
  <out_dir>/Ablation_Comparison/*.pdf and *.png
"""

import os
import re
import glob
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Basic I/O utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read csv: {path} | {e}")
        return None


def safe_read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] failed to read json: {path} | {e}")
        return None


def list_experiments(exp_root: str) -> List[str]:
    # Only directories starting with Exp_
    dirs = []
    for p in sorted(glob.glob(os.path.join(exp_root, "Exp_*"))):
        if os.path.isdir(p):
            dirs.append(p)
    return dirs


def exp_name_from_path(p: str) -> str:
    return os.path.basename(p.rstrip("/"))


# -----------------------------
# Plot style (journal-like)
# -----------------------------
def set_matplotlib_style():
    # Keep it clean; do not set colors explicitly (journal-friendly).
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
        "pdf.fonttype": 42,   # editable text in pdf
        "ps.fonttype": 42,
    })


def save_fig(fig, out_base: str):
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)


# -----------------------------
# Metrics helpers
# -----------------------------
def nan_clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    diff = y_pred - y_true
    mean = (y_pred + y_true) / 2.0
    md = np.mean(diff) if diff.size else np.nan
    sd = np.std(diff) if diff.size else np.nan
    loa_low = md - 1.96 * sd if np.isfinite(md) and np.isfinite(sd) else np.nan
    loa_high = md + 1.96 * sd if np.isfinite(md) and np.isfinite(sd) else np.nan
    return mean, diff, md, loa_low, loa_high


# -----------------------------
# Figure generators (single exp)
# -----------------------------
def plot_scatter_pred_true(df_clin: pd.DataFrame, exp_out: str, var: str):
    """
    Scatter: Pred vs True
    Data source:
      clinical_metrics.csv fields:
        - <var>_True, <var>_Pred
      e.g., HR_True/HR_Pred, RR_True/RR_Pred
    """
    x_col = f"{var}_True"
    y_col = f"{var}_Pred"
    if x_col not in df_clin.columns or y_col not in df_clin.columns:
        return

    x = df_clin[x_col].values
    y = df_clin[y_col].values
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size == 0:
        return

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, alpha=0.6)

    # y=x reference
    mn = np.min([x.min(), y.min()])
    mx = np.max([x.max(), y.max()])
    ax.plot([mn, mx], [mn, mx], linestyle="--")

    ax.set_xlabel(f"{var} True")
    ax.set_ylabel(f"{var} Pred")
    ax.set_title(f"{var} Pred vs True")

    out_base = os.path.join(exp_out, f"scatter_{var.lower()}")
    save_fig(fig, out_base)


def plot_bland_altman(df_clin: pd.DataFrame, exp_out: str, var: str):
    """
    Bland–Altman plot
    Data source:
      clinical_metrics.csv fields:
        - <var>_True, <var>_Pred
    """
    x_col = f"{var}_True"
    y_col = f"{var}_Pred"
    if x_col not in df_clin.columns or y_col not in df_clin.columns:
        return

    mean, diff, md, loa_low, loa_high = bland_altman_stats(df_clin[x_col].values, df_clin[y_col].values)
    if diff.size == 0:
        return

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.scatter(mean, diff, s=8, alpha=0.6)

    # mean + LoA lines
    ax.axhline(md, linestyle="--")
    ax.axhline(loa_low, linestyle="--")
    ax.axhline(loa_high, linestyle="--")

    ax.set_xlabel(f"Mean of {var} (True & Pred)")
    ax.set_ylabel(f"Pred - True ({var})")
    ax.set_title(f"Bland–Altman: {var}")

    # annotate numbers (journal-friendly)
    ax.text(0.02, 0.98, f"Mean diff={md:.3f}\nLoA=[{loa_low:.3f}, {loa_high:.3f}]",
            transform=ax.transAxes, va="top")

    out_base = os.path.join(exp_out, f"bland_altman_{var.lower()}")
    save_fig(fig, out_base)


def plot_error_hist(df_clin: pd.DataFrame, exp_out: str, err_col: str):
    """
    Error histogram
    Data source:
      clinical_metrics.csv fields:
        - HR_Error, RR_Error, QRS_Error, QT_Error (if available)
    """
    if err_col not in df_clin.columns:
        return
    x = nan_clean(df_clin[err_col].values)
    if x.size == 0:
        return

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.hist(x, bins=30, alpha=0.85)

    ax.set_xlabel(err_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution: {err_col}")

    out_base = os.path.join(exp_out, f"hist_{err_col.lower()}")
    save_fig(fig, out_base)


def plot_case_waveform(npz_path: str, exp_out: str):
    """
    Waveform overlay for one case
    Data source:
      results/cases/*.npz keys:
        - ecg_true, ecg_pred, (optional) mask_true, mask_prob, pcc, seg_id, subject_id
    """
    try:
        d = np.load(npz_path, allow_pickle=True)
        t = d["ecg_true"].reshape(-1)
        p = d["ecg_pred"].reshape(-1)
        pcc = float(d["pcc"].reshape(-1)[0]) if "pcc" in d else np.nan
        seg_id = int(d["seg_id"].reshape(-1)[0]) if "seg_id" in d else -1
        sid = int(d["subject_id"].reshape(-1)[0]) if "subject_id" in d else -1
    except Exception as e:
        print(f"[WARN] failed to read case npz: {npz_path} | {e}")
        return

    fig = plt.figure(figsize=(7.0, 3.2))
    ax = fig.add_subplot(111)
    ax.plot(t, label="ECG True", alpha=0.9)
    ax.plot(p, label="ECG Pred", alpha=0.9)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title(f"Case waveform | sid={sid} seg={seg_id} | PCC={pcc:.3f}")
    ax.legend(loc="upper right")

    base = os.path.splitext(os.path.basename(npz_path))[0]
    out_base = os.path.join(exp_out, f"case_{base}")
    save_fig(fig, out_base)


def export_cases_figures(exp_dir: str, exp_out: str, max_cases: int = 12):
    """
    Draw waveform overlays for representative cases.
    Data source:
      experiments/Exp_*/results/cases/*.npz
    """
    cases_dir = os.path.join(exp_dir, "results", "cases")
    if not os.path.isdir(cases_dir):
        return
    paths = sorted(glob.glob(os.path.join(cases_dir, "*.npz")))
    if len(paths) == 0:
        return
    # keep a reasonable amount
    for p in paths[:max_cases]:
        plot_case_waveform(p, exp_out)


# -----------------------------
# Cross-experiment: ablation figures
# -----------------------------
def collect_metric_across_exps(exp_dirs: List[str]) -> Dict[str, dict]:
    """
    Collect key arrays across experiments for ablation comparison.

    Uses:
      - segment_metrics.csv: PCC/MAE/RMSE
      - mask_metrics.csv: Mask_F1
      - clinical_metrics.csv: HR_Error, RR_Error
      - subject_summary.csv: subject-wise metrics if needed
    """
    data = {}
    for ed in exp_dirs:
        name = exp_name_from_path(ed)
        res = os.path.join(ed, "results")

        df_seg = safe_read_csv(os.path.join(res, "segment_metrics.csv"))
        df_clin = safe_read_csv(os.path.join(res, "clinical_metrics.csv"))
        df_mask = safe_read_csv(os.path.join(res, "mask_metrics.csv"))
        df_subj = safe_read_csv(os.path.join(res, "subject_summary.csv"))

        data[name] = {
            "df_seg": df_seg,
            "df_clin": df_clin,
            "df_mask": df_mask,
            "df_subj": df_subj,
        }
    return data


def plot_ablation_boxplot(metric_name: str, arrays: List[np.ndarray], labels: List[str], out_base: str, title: str):
    fig = plt.figure(figsize=(max(6.0, 0.55 * len(labels)), 3.8))
    ax = fig.add_subplot(111)

    ax.boxplot(arrays, showfliers=False)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title)

    save_fig(fig, out_base)


def ablation_comparison(exp_data: Dict[str, dict], out_dir: str):
    ensure_dir(out_dir)

    names = list(exp_data.keys())

    # 1) Segment metrics across exps
    # Data source: segment_metrics.csv fields PCC/MAE/RMSE
    for met in ["PCC", "MAE", "RMSE"]:
        arrays = []
        labels = []
        for n in names:
            df = exp_data[n]["df_seg"]
            if df is None or met not in df.columns:
                continue
            arrays.append(nan_clean(df[met].values))
            labels.append(n)
        if len(arrays) >= 2:
            out_base = os.path.join(out_dir, f"box_segment_{met.lower()}")
            plot_ablation_boxplot(met, arrays, labels, out_base, f"Ablation Comparison (segment-level): {met}")

    # 2) Clinical errors across exps
    # Data source: clinical_metrics.csv fields HR_Error/RR_Error
    for met in ["HR_Error", "RR_Error"]:
        arrays = []
        labels = []
        for n in names:
            df = exp_data[n]["df_clin"]
            if df is None or met not in df.columns:
                continue
            arrays.append(nan_clean(df[met].values))
            labels.append(n)
        if len(arrays) >= 2:
            out_base = os.path.join(out_dir, f"box_clin_{met.lower()}")
            plot_ablation_boxplot(met, arrays, labels, out_base, f"Ablation Comparison (segment-level): {met}")

    # 3) Mask F1 across exps
    # Data source: mask_metrics.csv field Mask_F1
    met = "Mask_F1"
    arrays = []
    labels = []
    for n in names:
        df = exp_data[n]["df_mask"]
        if df is None or met not in df.columns:
            continue
        arrays.append(nan_clean(df[met].values))
        labels.append(n)
    if len(arrays) >= 2:
        out_base = os.path.join(out_dir, "box_mask_f1")
        plot_ablation_boxplot(met, arrays, labels, out_base, "Ablation Comparison (segment-level): Mask_F1")


# -----------------------------
# Main: per-exp + cross-exp
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, default="experiments", help="root folder that contains Exp_*")
    parser.add_argument("--out_dir", type=str, default="Figures", help="where to save figures")
    parser.add_argument("--do_ba", action="store_true", help="also generate Bland–Altman plots")
    parser.add_argument("--do_rr", action="store_true", help="also include RR plots (scatter/BA/hist)")
    parser.add_argument("--max_case_figs", type=int, default=12, help="max case waveforms per experiment")
    args = parser.parse_args()

    set_matplotlib_style()
    ensure_dir(args.out_dir)

    exp_dirs = list_experiments(args.exp_root)
    if len(exp_dirs) == 0:
        raise RuntimeError(f"No experiments found under: {args.exp_root}/Exp_*")

    # Cross-experiment ablation comparison
    exp_data = collect_metric_across_exps(exp_dirs)
    ablation_dir = os.path.join(args.out_dir, "Ablation_Comparison")
    ablation_comparison(exp_data, ablation_dir)

    # Per-experiment figures
    for ed in exp_dirs:
        name = exp_name_from_path(ed)
        res = os.path.join(ed, "results")
        if not os.path.isdir(res):
            continue

        exp_out = os.path.join(args.out_dir, name)
        ensure_dir(exp_out)

        df_clin = safe_read_csv(os.path.join(res, "clinical_metrics.csv"))
        if df_clin is not None:
            # Scatter: HR (clinical_metrics.csv: HR_True/HR_Pred)
            plot_scatter_pred_true(df_clin, exp_out, var="HR")

            # Optional: RR (clinical_metrics.csv: RR_True/RR_Pred)
            if args.do_rr:
                plot_scatter_pred_true(df_clin, exp_out, var="RR")

            # Hist: HR_Error, RR_Error (clinical_metrics.csv: HR_Error/RR_Error)
            plot_error_hist(df_clin, exp_out, "HR_Error")
            if args.do_rr:
                plot_error_hist(df_clin, exp_out, "RR_Error")

            # Bland–Altman (optional)
            if args.do_ba:
                plot_bland_altman(df_clin, exp_out, var="HR")
                if args.do_rr:
                    plot_bland_altman(df_clin, exp_out, var="RR")

        # Cases waveform overlays (results/cases/*.npz)
        export_cases_figures(ed, exp_out, max_cases=args.max_case_figs)

    print(f"\n✅ Done. Figures saved to: {args.out_dir}")
    print(f"   - Cross-exp: {ablation_dir}")
    print(f"   - Per-exp  : {os.path.join(args.out_dir, 'Exp_*')}")


if __name__ == "__main__":
    main()
