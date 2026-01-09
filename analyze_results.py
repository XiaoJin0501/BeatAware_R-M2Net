#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_results.py
------------------
Aggregate all experiments under experiments/Exp_* and produce:
  1) ablation_table.csv (one row per experiment, from global_summary.json + meta.json)
  2) all_subject_summary.csv (concat subject_summary.csv across experiments)
  3) all_segment_metrics.csv (concat segment_metrics.csv across experiments)
  4) all_clinical_metrics.csv (concat clinical_metrics.csv across experiments)
  5) all_mask_metrics.csv (concat mask_metrics.csv across experiments)
  6) all_bland_altman_pairs.csv (concat bland_altman_pairs.csv across experiments)
  7) index.json (audit trail of included experiments and missing files)

Usage:
  python analyze_results.py \
      --experiments_dir /root/Projects/BeatAware_R-M2Net/experiments \
      --pattern "Exp_*"

Recommended:
  Run after all ablations finished and each experiment has test.py outputs.
"""

import os
import json
import glob
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], keys: List[str], default=np.nan):
    """Safely access nested dict by keys list."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _flatten_summary(prefix: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a dict like:
      {"mean":..., "std":..., "median":..., "q05":..., ...}
    into:
      {f"{prefix}_mean":..., f"{prefix}_std":..., ...}
    """
    if not isinstance(summary, dict):
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    out = {}
    for k, v in summary.items():
        out[f"{prefix}_{k}"] = v
    return out


def _inject_exp_cols(df: pd.DataFrame, exp_info: Dict[str, Any]) -> pd.DataFrame:
    """Add experiment identifiers to each row."""
    for k, v in exp_info.items():
        if k not in df.columns:
            df[k] = v
    # Put identifiers in front (nice for analysis)
    front_cols = ["exp_name", "exp_tag", "alpha", "beta", "gamma"]
    cols = front_cols + [c for c in df.columns if c not in front_cols]
    return df[cols]


def _list_experiments(experiments_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(experiments_dir, pattern)))
    # keep only directories
    return [p for p in paths if os.path.isdir(p)]


# -------------------------
# Main aggregation
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate BeatAware_R-M2Net experiment results.")
    parser.add_argument("--experiments_dir", type=str, default="experiments",
                        help="Path to experiments directory (contains Exp_* subfolders).")
    parser.add_argument("--pattern", type=str, default="Exp_*", help="Experiment folder glob pattern.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: <experiments_dir>/_analysis")
    parser.add_argument("--strict", action="store_true",
                        help="If set, raise error when a required file is missing (not recommended).")
    args = parser.parse_args()

    experiments_dir = os.path.abspath(args.experiments_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(experiments_dir, "_analysis")
    os.makedirs(out_dir, exist_ok=True)

    exp_paths = _list_experiments(experiments_dir, args.pattern)
    if len(exp_paths) == 0:
        raise RuntimeError(f"No experiment folders found under: {experiments_dir} with pattern={args.pattern}")

    # Aggregation containers
    ablation_rows: List[Dict[str, Any]] = []
    all_subject_dfs: List[pd.DataFrame] = []
    all_segment_dfs: List[pd.DataFrame] = []
    all_clinical_dfs: List[pd.DataFrame] = []
    all_mask_dfs: List[pd.DataFrame] = []
    all_ba_dfs: List[pd.DataFrame] = []

    audit = {
        "experiments_dir": experiments_dir,
        "pattern": args.pattern,
        "out_dir": out_dir,
        "n_found": len(exp_paths),
        "included": [],
        "skipped": [],
    }

    # Iterate experiments
    for exp_dir in exp_paths:
        exp_name = os.path.basename(exp_dir)
        results_dir = os.path.join(exp_dir, "results")

        meta_path = os.path.join(results_dir, "meta.json")
        global_path = os.path.join(results_dir, "global_summary.json")
        subj_path = os.path.join(results_dir, "subject_summary.csv")
        seg_path = os.path.join(results_dir, "segment_metrics.csv")
        clin_path = os.path.join(results_dir, "clinical_metrics.csv")
        mask_path = os.path.join(results_dir, "mask_metrics.csv")
        ba_path = os.path.join(results_dir, "bland_altman_pairs.csv")

        meta = _read_json(meta_path) or {}
        global_sum = _read_json(global_path)

        # Basic exp identifiers
        exp_info = {
            "exp_name": exp_name,
            "exp_tag": meta.get("exp_tag", ""),
            "alpha": meta.get("alpha", np.nan),
            "beta": meta.get("beta", np.nan),
            "gamma": meta.get("gamma", np.nan),
        }

        # Require at least global_summary.json for ablation table
        if global_sum is None:
            msg = f"[SKIP] {exp_name}: missing global_summary.json at {global_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            audit["skipped"].append({"exp_name": exp_name, "reason": msg})
            continue

        # -------------------------
        # Build ablation table row
        # -------------------------
        row = {}
        row.update(exp_info)

        # Meta fields (keep as columns if present)
        row.update({
            "seed": meta.get("seed", np.nan),
            "device": meta.get("device", ""),
            "fs": meta.get("fs", np.nan),
            "stft_fmin": meta.get("stft_fmin", np.nan),
            "stft_fmax": meta.get("stft_fmax", np.nan),
            "stft_use_band": meta.get("stft_use_band", np.nan),
            "anchor_from_logits": meta.get("anchor_from_logits", np.nan),
            "anchor_pos_weight": meta.get("anchor_pos_weight", np.nan),
            "n_segments": global_sum.get("n_segments", np.nan),
            "n_subjects": global_sum.get("n_subjects", np.nan),
        })

        # Segment-level summaries
        for metric in ["PCC", "MAE", "RMSE", "Mask_F1"]:
            summ = _safe_get(global_sum, ["segment_level", metric], default={})
            row.update(_flatten_summary(f"seg_{metric}", summ))

        # Subject-level summaries
        for metric in ["PCC", "MAE", "RMSE", "HR_Error", "RR_Error", "QRS_Error", "QT_Error", "Mask_F1"]:
            summ = _safe_get(global_sum, ["subject_level", metric], default={})
            row.update(_flatten_summary(f"subj_{metric}", summ))

        ablation_rows.append(row)

        # -------------------------
        # Load & concat CSVs
        # -------------------------
        missing_files = []

        def _try_read_csv(path: str) -> Optional[pd.DataFrame]:
            if not os.path.exists(path):
                return None
            try:
                return pd.read_csv(path)
            except Exception:
                return None

        df_subj = _try_read_csv(subj_path)
        if df_subj is None:
            missing_files.append("subject_summary.csv")
        else:
            df_subj = _inject_exp_cols(df_subj, exp_info)
            all_subject_dfs.append(df_subj)

        df_seg = _try_read_csv(seg_path)
        if df_seg is None:
            missing_files.append("segment_metrics.csv")
        else:
            df_seg = _inject_exp_cols(df_seg, exp_info)
            all_segment_dfs.append(df_seg)

        df_clin = _try_read_csv(clin_path)
        if df_clin is None:
            missing_files.append("clinical_metrics.csv")
        else:
            df_clin = _inject_exp_cols(df_clin, exp_info)
            all_clinical_dfs.append(df_clin)

        df_mask = _try_read_csv(mask_path)
        if df_mask is None:
            missing_files.append("mask_metrics.csv")
        else:
            df_mask = _inject_exp_cols(df_mask, exp_info)
            all_mask_dfs.append(df_mask)

        df_ba = _try_read_csv(ba_path)
        if df_ba is None:
            missing_files.append("bland_altman_pairs.csv")
        else:
            df_ba = _inject_exp_cols(df_ba, exp_info)
            all_ba_dfs.append(df_ba)

        audit["included"].append({
            "exp_name": exp_name,
            "exp_dir": exp_dir,
            "results_dir": results_dir,
            "missing_files": missing_files,
        })

    # -------------------------
    # Write outputs
    # -------------------------
    # Ablation table
    df_ablation = pd.DataFrame(ablation_rows)

    # A standard, publication-friendly column ordering (keep the rest afterwards)
    preferred = [
        "exp_name", "exp_tag", "alpha", "beta", "gamma",
        "n_subjects", "n_segments", "fs",
        "subj_PCC_mean", "subj_PCC_std",
        "subj_MAE_mean", "subj_MAE_std",
        "subj_RMSE_mean", "subj_RMSE_std",
        "subj_HR_Error_mean", "subj_HR_Error_std",
        "subj_Mask_F1_mean", "subj_Mask_F1_std",
        "seg_PCC_mean", "seg_PCC_std",
        "seg_MAE_mean", "seg_MAE_std",
        "seg_RMSE_mean", "seg_RMSE_std",
    ]
    cols = preferred + [c for c in df_ablation.columns if c not in preferred]
    df_ablation = df_ablation[cols]

    ablation_csv = os.path.join(out_dir, "ablation_table.csv")
    df_ablation.to_csv(ablation_csv, index=False)

    # Concats
    def _concat_or_empty(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if len(dfs) == 0:
            return pd.DataFrame()
        return pd.concat(dfs, axis=0, ignore_index=True)

    df_all_subject = _concat_or_empty(all_subject_dfs)
    df_all_segment = _concat_or_empty(all_segment_dfs)
    df_all_clinical = _concat_or_empty(all_clinical_dfs)
    df_all_mask = _concat_or_empty(all_mask_dfs)
    df_all_ba = _concat_or_empty(all_ba_dfs)

    # Save concatenated CSVs
    if not df_all_subject.empty:
        df_all_subject.to_csv(os.path.join(out_dir, "all_subject_summary.csv"), index=False)
    if not df_all_segment.empty:
        df_all_segment.to_csv(os.path.join(out_dir, "all_segment_metrics.csv"), index=False)
    if not df_all_clinical.empty:
        df_all_clinical.to_csv(os.path.join(out_dir, "all_clinical_metrics.csv"), index=False)
    if not df_all_mask.empty:
        df_all_mask.to_csv(os.path.join(out_dir, "all_mask_metrics.csv"), index=False)
    if not df_all_ba.empty:
        df_all_ba.to_csv(os.path.join(out_dir, "all_bland_altman_pairs.csv"), index=False)

    # Audit trail
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # Print summary
    print("\n✅ Aggregation finished.")
    print(f"  - Experiments found : {audit['n_found']}")
    print(f"  - Experiments used  : {len(audit['included'])}")
    print(f"  - Experiments skipped: {len(audit['skipped'])}")
    print(f"\n📦 Outputs saved to: {out_dir}")
    print(f"  - {ablation_csv}")
    if not df_all_subject.empty:
        print(f"  - {os.path.join(out_dir, 'all_subject_summary.csv')}")
    if not df_all_segment.empty:
        print(f"  - {os.path.join(out_dir, 'all_segment_metrics.csv')}")
    if not df_all_clinical.empty:
        print(f"  - {os.path.join(out_dir, 'all_clinical_metrics.csv')}")
    if not df_all_mask.empty:
        print(f"  - {os.path.join(out_dir, 'all_mask_metrics.csv')}")
    if not df_all_ba.empty:
        print(f"  - {os.path.join(out_dir, 'all_bland_altman_pairs.csv')}")
    print(f"  - {os.path.join(out_dir, 'index.json')}\n")


if __name__ == "__main__":
    main()
