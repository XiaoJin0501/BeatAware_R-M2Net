#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _find_latest_global_json(exp_dir: str) -> Optional[str]:
    """
    Find the newest global_summary.json under exp_dir.
    We search recursively because different projects store results differently.
    """
    cand = glob.glob(os.path.join(exp_dir, "**", "global_summary.json"), recursive=True)
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def _parse_variant_from_tag(exp_tag: str) -> str:
    """
    Expect exp tags like:
      <prefix>_L0_TimeOnly
      <prefix>_L1_TimeSpectral
      <prefix>_L2_TimeAnchor
      <prefix>_L3_Full
    Return L0/L1/L2/L3 (fallback 'UNK').
    """
    m = re.search(r"(_L[0-3]_[A-Za-z0-9]+)$", exp_tag)
    if not m:
        m2 = re.search(r"(L[0-3])", exp_tag)
        return m2.group(1) if m2 else "UNK"
    return m.group(1).split("_")[1]  # "L0", "L1", ...


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None:
            return ""
        v = float(x)
        if pd.isna(v):
            return ""
        return f"{v:.{nd}f}"
    except Exception:
        return ""


def _extract_metrics(g: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected structure from our protocolized test.py summarize_metrics():
      segment_level: { PCC: {median,...}, MRE:{median,...}, ... }
      beat_level: { RR_err_ms:{median,...}, QRS_err_ms:{median,...}, QT_err_ms:{median,...} }
      counts: { n_segments, n_beats_total, n_beats_valid, drop_rate, drop_reasons }
    """
    out: Dict[str, Any] = {}

    out["PCC_median"] = _safe_get(g, ["segment_level", "PCC", "median"], None)
    out["MRE_median"] = _safe_get(g, ["segment_level", "MRE", "median"], None)

    out["RR_err_median_ms"] = _safe_get(g, ["beat_level", "RR_err_ms", "median"], None)
    out["QRS_err_median_ms"] = _safe_get(g, ["beat_level", "QRS_err_ms", "median"], None)
    out["QT_err_median_ms"] = _safe_get(g, ["beat_level", "QT_err_ms", "median"], None)

    out["n_segments"] = _safe_get(g, ["counts", "n_segments"], None)
    out["n_beats_total"] = _safe_get(g, ["counts", "n_beats_total"], None)
    out["n_beats_valid"] = _safe_get(g, ["counts", "n_beats_valid"], None)
    out["drop_rate"] = _safe_get(g, ["counts", "drop_rate"], None)

    # optional: keep path-related meta if present
    out["exp_name"] = g.get("exp_name", "")
    return out


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Minimal, clean LaTeX table.
    """
    cols = [
        "Variant",
        "Waveform(L1)",
        "Spectral(MR-STFT)",
        "Anchor(BCE)",
        "PCC↑ (median)",
        "MRE↓ (median)",
        "RR_err↓ (ms, median)",
        "QRS_err↓ (ms, median)",
        "QT_err↓ (ms, median)",
        "Beats(valid/total)",
        "DropRate",
    ]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{lccccccccc}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        beats = ""
        if pd.notna(r.get("n_beats_valid")) and pd.notna(r.get("n_beats_total")):
            beats = f"{int(r['n_beats_valid'])}/{int(r['n_beats_total'])}"
        dr = ""
        if pd.notna(r.get("drop_rate")):
            dr = f"{float(r['drop_rate']):.3f}"

        row = [
            str(r.get("Variant", "")),
            str(r.get("Waveform(L1)", "")),
            str(r.get("Spectral(MR-STFT)", "")),
            str(r.get("Anchor(BCE)", "")),
            _fmt(r.get("PCC_median"), 4),
            _fmt(r.get("MRE_median"), 4),
            _fmt(r.get("RR_err_median_ms"), 2),
            _fmt(r.get("QRS_err_median_ms"), 2),
            _fmt(r.get("QT_err_median_ms"), 2),
            beats,
            dr,
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect loss ablation results from experiments/* into a summary table.")
    parser.add_argument("--tag_prefix", type=str, required=True,
                        help="Prefix used in run_pipeline.sh, e.g., PaperAblation_v1")
    parser.add_argument("--experiments_dir", type=str, default="experiments",
                        help="Path to experiments directory (default: experiments)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: experiments/<tag_prefix>_COLLECTED")
    args = parser.parse_args()

    exp_root = os.path.abspath(args.experiments_dir)
    if not os.path.isdir(exp_root):
        raise FileNotFoundError(f"experiments_dir not found: {exp_root}")

    prefix = args.tag_prefix

    # output directory
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(exp_root, f"{prefix}_COLLECTED")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Find candidate experiment folders
    cand = sorted(glob.glob(os.path.join(exp_root, f"{prefix}_L*_*/")))
    if not cand:
        # also allow folders without trailing slash
        cand = sorted(glob.glob(os.path.join(exp_root, f"{prefix}_L*_*")))
    if not cand:
        raise FileNotFoundError(
            f"No experiment dirs found with prefix={prefix} under {exp_root}\n"
            f"Expected patterns like: {prefix}_L0_TimeOnly, {prefix}_L3_Full"
        )

    rows = []
    for d in cand:
        exp_dir = d.rstrip("/")

        exp_tag = os.path.basename(exp_dir)
        variant = _parse_variant_from_tag(exp_tag)

        gpath = _find_latest_global_json(exp_dir)
        if gpath is None:
            print(f"[WARN] global_summary.json not found under: {exp_dir}")
            continue

        g = _load_json(gpath)
        m = _extract_metrics(g)

        # infer loss switches from variant name (protocol definition)
        # L0: L1 only
        # L1: L1 + spectral
        # L2: L1 + anchor
        # L3: full
        if variant == "L0":
            sw = ("✓", "✗", "✗")
        elif variant == "L1":
            sw = ("✓", "✓", "✗")
        elif variant == "L2":
            sw = ("✓", "✗", "✓")
        elif variant == "L3":
            sw = ("✓", "✓", "✓")
        else:
            sw = ("", "", "")

        rows.append({
            "ExpTag": exp_tag,
            "Variant": variant,
            "Waveform(L1)": sw[0],
            "Spectral(MR-STFT)": sw[1],
            "Anchor(BCE)": sw[2],
            **m,
            "global_json_path": os.path.relpath(gpath, exp_root),
            "mtime": datetime.fromtimestamp(os.path.getmtime(gpath)).strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not rows:
        raise RuntimeError("Found experiment dirs but none had usable global_summary.json.")

    df = pd.DataFrame(rows)

    # enforce order L0->L3
    order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
    df["__ord"] = df["Variant"].map(lambda x: order.get(x, 99))
    df = df.sort_values(["__ord", "ExpTag"]).drop(columns=["__ord"]).reset_index(drop=True)

    # save CSV
    out_csv = os.path.join(out_dir, "ablation_summary.csv")
    df.to_csv(out_csv, index=False)

    # also save a clean "paper view" csv
    paper_cols = [
        "Variant", "Waveform(L1)", "Spectral(MR-STFT)", "Anchor(BCE)",
        "PCC_median", "MRE_median",
        "RR_err_median_ms", "QRS_err_median_ms", "QT_err_median_ms",
        "n_segments", "n_beats_total", "n_beats_valid", "drop_rate",
        "ExpTag",
    ]
    out_csv_paper = os.path.join(out_dir, "ablation_summary_paper.csv")
    df[paper_cols].to_csv(out_csv_paper, index=False)

    # save LaTeX (Table S2 style)
    caption = "Loss Function Design Ablation (median-first, protocol-aligned evaluation)."
    label = "tab:loss_ablation"
    out_tex = os.path.join(out_dir, "ablation_summary.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(_to_latex(df, caption=caption, label=label))

    print("\n✅ Collected ablation results:")
    print(f"  - {out_csv}")
    print(f"  - {out_csv_paper}")
    print(f"  - {out_tex}\n")
    print("Tip: Use ablation_summary_paper.csv for quick checking; use .tex for direct LaTeX insertion.")


if __name__ == "__main__":
    main()
