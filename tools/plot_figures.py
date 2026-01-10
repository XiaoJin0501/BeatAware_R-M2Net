# tools/plot_figures.py
import os
import glob
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# style
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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_fig(fig, out_base: str):
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] CSV read failed: {path} | {e}")
        return None

def safe_read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON read failed: {path} | {e}")
        return None

def list_experiments(exp_root: str) -> List[str]:
    return [p for p in sorted(glob.glob(os.path.join(exp_root, "Exp_*"))) if os.path.isdir(p)]

def exp_name(exp_dir: str) -> str:
    return os.path.basename(exp_dir.rstrip("/"))

def nan_clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]

# -----------------------------
# unified loader
# -----------------------------
def load_exp_results(exp_dir: str) -> Dict[str, Any]:
    res_dir = os.path.join(exp_dir, "results")
    out = {
        "exp_dir": exp_dir,
        "exp_name": exp_name(exp_dir),
        "res_dir": res_dir,
        "segment": safe_read_csv(os.path.join(res_dir, "segment_metrics.csv")),
        "clinical": safe_read_csv(os.path.join(res_dir, "clinical_metrics.csv")),
        "mask": safe_read_csv(os.path.join(res_dir, "mask_metrics.csv")),
        "subject": safe_read_csv(os.path.join(res_dir, "subject_summary.csv")),
        "global": safe_read_json(os.path.join(res_dir, "global_summary.json")),
        "cases": sorted(glob.glob(os.path.join(res_dir, "cases", "*.npz"))),
    }
    return out

# -----------------------------
# core plots (single)
# -----------------------------
def plot_box_across_exps(metric: str, exp_results: List[Dict[str, Any]], out_base: str, title: str,
                         source_hint: str):
    arrays, labels = [], []
    for r in exp_results:
        df = r.get("segment", None)
        # clinical/mask can be switched outside
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

def bland_altman(df: pd.DataFrame, true_col: str, pred_col: str, out_base: str, title: str, source_hint: str):
    if df is None or true_col not in df.columns or pred_col not in df.columns:
        return
    y_true = df[true_col].values.astype(np.float64)
    y_pred = df[pred_col].values.astype(np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]; y_pred = y_pred[ok]
    if y_true.size == 0:
        return
    diff = y_pred - y_true
    mean = (y_pred + y_true) / 2.0
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    loa_low = md - 1.96 * sd
    loa_high = md + 1.96 * sd

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.scatter(mean, diff, s=8, alpha=0.6)
    ax.axhline(md, linestyle="--")
    ax.axhline(loa_low, linestyle="--")
    ax.axhline(loa_high, linestyle="--")
    ax.set_xlabel(f"Mean({true_col},{pred_col})")
    ax.set_ylabel(f"{pred_col} - {true_col}")
    ax.set_title(title)
    ax.text(0.02, 0.98, f"Mean diff={md:.3f}\nLoA=[{loa_low:.3f},{loa_high:.3f}]",
            transform=ax.transAxes, va="top")
    ax.text(0.02, 0.02, f"Data: {source_hint}", transform=ax.transAxes, va="bottom")
    save_fig(fig, out_base)

def plot_hist(df: pd.DataFrame, col: str, out_base: str, title: str, source_hint: str):
    if df is None or col not in df.columns:
        return
    x = nan_clean(df[col].values)
    if x.size == 0:
        return
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.hist(x, bins=30, alpha=0.85)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.text(0.02, 0.02, f"Data: {source_hint}", transform=ax.transAxes, va="bottom")
    save_fig(fig, out_base)

# -----------------------------
# cases
# -----------------------------
def load_case_npz(npz_path: str) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    out = {
        "ecg_true": d["ecg_true"].reshape(-1),
        "ecg_pred": d["ecg_pred"].reshape(-1),
        "pcc": float(d["pcc"].reshape(-1)[0]) if "pcc" in d else np.nan,
        "seg_id": int(d["seg_id"].reshape(-1)[0]) if "seg_id" in d else -1,
        "subject_id": int(d["subject_id"].reshape(-1)[0]) if "subject_id" in d else -1,
        "path": npz_path,
    }
    return out

def plot_waveform(ax, t: np.ndarray, p: np.ndarray, title: str):
    ax.plot(t, label="ECG True", alpha=0.9)
    ax.plot(p, label="ECG Pred", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

def pick_cases_by_name(case_paths: List[str]) -> Dict[str, Optional[str]]:
    """
    Prefer files named like best_*, worst_*, median_*, random_* (from your test.py naming).
    Fallback to first few if naming not present.
    """
    out = {"best": None, "worst": None, "median": None}
    for p in case_paths:
        base = os.path.basename(p).lower()
        if out["best"] is None and base.startswith("best"):
            out["best"] = p
        if out["worst"] is None and base.startswith("worst"):
            out["worst"] = p
        if out["median"] is None and base.startswith("median"):
            out["median"] = p
    # fallback
    if out["best"] is None and len(case_paths) > 0:
        out["best"] = case_paths[0]
    if out["worst"] is None and len(case_paths) > 1:
        out["worst"] = case_paths[1]
    if out["median"] is None and len(case_paths) > 2:
        out["median"] = case_paths[2]
    return out

# -----------------------------
# multi-panel (paper figures)
# -----------------------------
def paper_fig_waveform_triplet(r: Dict[str, Any], out_dir: str):
    """
    Paper Fig (Qualitative):
      Three panels: best / median / worst waveform overlays.
    Data:
      results/cases/*.npz -> ecg_true, ecg_pred, pcc, subject_id, seg_id
    """
    if not r["cases"]:
        return
    picks = pick_cases_by_name(r["cases"])
    paths = [picks["best"], picks["median"], picks["worst"]]
    if any(p is None for p in paths):
        return

    items = [load_case_npz(p) for p in paths]

    fig = plt.figure(figsize=(10.5, 3.2))
    axes = [fig.add_subplot(1, 3, i+1) for i in range(3)]
    labels = ["Best", "Median", "Worst"]

    for ax, it, lab in zip(axes, items, labels):
        title = f"{lab}\nsid={it['subject_id']} seg={it['seg_id']} PCC={it['pcc']:.3f}"
        plot_waveform(ax, it["ecg_true"], it["ecg_pred"], title)

    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(f"Qualitative ECG reconstruction examples ({r['exp_name']})")

    out_base = os.path.join(out_dir, f"Figure_Qualitative_{r['exp_name']}")
    save_fig(fig, out_base)

def paper_fig_hr_clinical_triplet(r: Dict[str, Any], out_dir: str):
    """
    Paper Fig (Clinical HR):
      Panel A: HR scatter (HR_True vs HR_Pred)
      Panel B: HR Bland–Altman
      Panel C: HR_Error histogram
    Data:
      clinical_metrics.csv -> HR_True, HR_Pred, HR_Error
    """
    df = r.get("clinical", None)
    if df is None:
        return
    needed = ["HR_True", "HR_Pred", "HR_Error"]
    if any(c not in df.columns for c in needed):
        return

    # clean
    x = df["HR_True"].values.astype(np.float64)
    y = df["HR_Pred"].values.astype(np.float64)
    e = df["HR_Error"].values.astype(np.float64)
    ok_xy = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[ok_xy], y[ok_xy]
    e2 = nan_clean(e)

    fig = plt.figure(figsize=(10.5, 3.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # A scatter
    ax1.scatter(x2, y2, s=8, alpha=0.6)
    if x2.size:
        mn = np.min([x2.min(), y2.min()])
        mx = np.max([x2.max(), y2.max()])
        ax1.plot([mn, mx], [mn, mx], linestyle="--")
    ax1.set_xlabel("HR_True")
    ax1.set_ylabel("HR_Pred")
    ax1.set_title("A. Pred vs True (HR)\nData: clinical_metrics.csv[HR_True,HR_Pred]")

    # B BA
    if x2.size:
        diff = y2 - x2
        mean = (y2 + x2) / 2.0
        md = float(np.mean(diff))
        sd = float(np.std(diff))
        loa_low = md - 1.96 * sd
        loa_high = md + 1.96 * sd
        ax2.scatter(mean, diff, s=8, alpha=0.6)
        ax2.axhline(md, linestyle="--")
        ax2.axhline(loa_low, linestyle="--")
        ax2.axhline(loa_high, linestyle="--")
        ax2.text(0.02, 0.98, f"Mean diff={md:.2f}\nLoA=[{loa_low:.2f},{loa_high:.2f}]",
                 transform=ax2.transAxes, va="top")
    ax2.set_xlabel("Mean(HR)")
    ax2.set_ylabel("Pred-True")
    ax2.set_title("B. Bland–Altman (HR)\nData: clinical_metrics.csv[HR_True,HR_Pred]")

    # C hist
    if e2.size:
        ax3.hist(e2, bins=30, alpha=0.85)
    ax3.set_xlabel("HR_Error")
    ax3.set_ylabel("Count")
    ax3.set_title("C. Error distribution\nData: clinical_metrics.csv[HR_Error]")

    fig.suptitle(f"Clinical HR evaluation ({r['exp_name']})")
    out_base = os.path.join(out_dir, f"Figure_ClinicalHR_{r['exp_name']}")
    save_fig(fig, out_base)

# -----------------------------
# Ablation table
# -----------------------------
def build_ablation_table(exp_results: List[Dict[str, Any]], out_csv: str):
    """
    Build ablation table from global_summary.json (preferred).
    Data:
      global_summary.json -> segment_level[PCC/MAE/RMSE], subject_level[PCC/MAE/RMSE/HR_Error/Mask_F1]
    """
    rows = []
    for r in exp_results:
        g = r.get("global", None)
        if not g:
            continue

        def pick(path: Tuple[str, str, str], default=np.nan):
            # (level, metric, stat)
            level, metric, stat = path
            try:
                return g[level][metric][stat]
            except Exception:
                return default

        rows.append({
            "exp_name": r["exp_name"],
            # segment-level
            "seg_PCC_mean": pick(("segment_level", "PCC", "mean")),
            "seg_PCC_std": pick(("segment_level", "PCC", "std")),
            "seg_MAE_mean": pick(("segment_level", "MAE", "mean")),
            "seg_MAE_std": pick(("segment_level", "MAE", "std")),
            "seg_RMSE_mean": pick(("segment_level", "RMSE", "mean")),
            "seg_RMSE_std": pick(("segment_level", "RMSE", "std")),
            # subject-level
            "subj_PCC_mean": pick(("subject_level", "PCC", "mean")),
            "subj_PCC_std": pick(("subject_level", "PCC", "std")),
            "subj_MAE_mean": pick(("subject_level", "MAE", "mean")),
            "subj_MAE_std": pick(("subject_level", "MAE", "std")),
            "subj_RMSE_mean": pick(("subject_level", "RMSE", "mean")),
            "subj_RMSE_std": pick(("subject_level", "RMSE", "std")),
            "subj_HRerr_mean": pick(("subject_level", "HR_Error", "mean")),
            "subj_HRerr_std": pick(("subject_level", "HR_Error", "std")),
            "subj_MaskF1_mean": pick(("subject_level", "Mask_F1", "mean")),
            "subj_MaskF1_std": pick(("subject_level", "Mask_F1", "std")),
        })

    if len(rows) == 0:
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Ablation table saved: {out_csv}")

# -----------------------------
# runners
# -----------------------------
def run_ablation(exp_results: List[Dict[str, Any]], out_dir: str):
    ensure_dir(out_dir)

    # Ablation table CSV
    build_ablation_table(exp_results, os.path.join(out_dir, "ablation_table.csv"))

    # Boxplots (segment-level)
    plot_box_across_exps(
        metric="PCC",
        exp_results=exp_results,
        out_base=os.path.join(out_dir, "box_seg_pcc"),
        title="Ablation (segment-level): PCC",
        source_hint="segment_metrics.csv[PCC]"
    )
    plot_box_across_exps(
        metric="MAE",
        exp_results=exp_results,
        out_base=os.path.join(out_dir, "box_seg_mae"),
        title="Ablation (segment-level): MAE",
        source_hint="segment_metrics.csv[MAE]"
    )
    plot_box_across_exps(
        metric="RMSE",
        exp_results=exp_results,
        out_base=os.path.join(out_dir, "box_seg_rmse"),
        title="Ablation (segment-level): RMSE",
        source_hint="segment_metrics.csv[RMSE]"
    )

    # Mask F1 (mask_metrics.csv)
    arrays, labels = [], []
    for r in exp_results:
        df = r.get("mask", None)
        if df is None or "Mask_F1" not in df.columns:
            continue
        arrays.append(nan_clean(df["Mask_F1"].values))
        labels.append(r["exp_name"])
    if len(arrays) >= 2:
        fig = plt.figure(figsize=(max(6.0, 0.55 * len(labels)), 3.8))
        ax = fig.add_subplot(111)
        ax.boxplot(arrays, showfliers=False)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Mask_F1")
        ax.set_title("Ablation (segment-level): Mask_F1")
        ax.text(0.01, -0.22, "Data: mask_metrics.csv[Mask_F1]", transform=ax.transAxes, ha="left", va="top")
        save_fig(fig, os.path.join(out_dir, "box_mask_f1"))

def run_per_exp(exp_results: List[Dict[str, Any]], out_dir: str, do_ba: bool):
    for r in exp_results:
        if not os.path.isdir(r["res_dir"]):
            continue
        exp_out = os.path.join(out_dir, r["exp_name"])
        ensure_dir(exp_out)

        dfc = r.get("clinical", None)
        if dfc is not None:
            plot_scatter(
                dfc, "HR_True", "HR_Pred",
                out_base=os.path.join(exp_out, "scatter_hr"),
                title="HR Pred vs True",
                source_hint="clinical_metrics.csv[HR_True,HR_Pred]"
            )
            plot_hist(
                dfc, "HR_Error",
                out_base=os.path.join(exp_out, "hist_hr_error"),
                title="HR Error Distribution",
                source_hint="clinical_metrics.csv[HR_Error]"
            )
            if do_ba:
                bland_altman(
                    dfc, "HR_True", "HR_Pred",
                    out_base=os.path.join(exp_out, "ba_hr"),
                    title="Bland–Altman (HR)",
                    source_hint="clinical_metrics.csv[HR_True,HR_Pred]"
                )

        # qualitative cases
        if r["cases"]:
            # export a few single-case overlays (optional)
            for p in r["cases"][:12]:
                it = load_case_npz(p)
                fig = plt.figure(figsize=(7.0, 3.2))
                ax = fig.add_subplot(111)
                plot_waveform(ax, it["ecg_true"], it["ecg_pred"],
                              f"sid={it['subject_id']} seg={it['seg_id']} PCC={it['pcc']:.3f}\nData: cases/*.npz[ecg_true,ecg_pred]")
                ax.legend(loc="upper right", frameon=False)
                base = os.path.splitext(os.path.basename(p))[0]
                save_fig(fig, os.path.join(exp_out, f"case_{base}"))

def run_paper(exp_results: List[Dict[str, Any]], out_dir: str):
    ensure_dir(out_dir)
    # 你一般只对“主方法”那个实验出 paper figures，避免太多
    # 这里默认：对所有实验都出；你也可以后续加 --paper_only_exp 过滤
    for r in exp_results:
        paper_fig_waveform_triplet(r, out_dir)
        paper_fig_hr_clinical_triplet(r, out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, default="experiments")
    parser.add_argument("--out_dir", type=str, default="Figures")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "per_exp", "ablation", "paper"])
    parser.add_argument("--do_ba", action="store_true", help="also export Bland–Altman for per-exp")
    args = parser.parse_args()

    set_style()
    ensure_dir(args.out_dir)

    exp_dirs = list_experiments(args.exp_root)
    if len(exp_dirs) == 0:
        raise RuntimeError(f"No experiments found under {args.exp_root}/Exp_*")

    exp_results = [load_exp_results(ed) for ed in exp_dirs]

    if args.mode in ["all", "ablation"]:
        run_ablation(exp_results, os.path.join(args.out_dir, "Ablation_Comparison"))

    if args.mode in ["all", "per_exp"]:
        run_per_exp(exp_results, args.out_dir, do_ba=args.do_ba)

    if args.mode in ["all", "paper"]:
        run_paper(exp_results, os.path.join(args.out_dir, "Paper"))

    print(f"\n✅ Done. Output: {args.out_dir}")

if __name__ == "__main__":
    main()
