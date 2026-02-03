# test.py
import os
import json
import time
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.seeding import seed_everything

# ✅ protocol-aligned metrics
from utils.metrics import (
    compute_segment_metrics,
    compute_beat_metrics_nk,
    summarize_metrics,
)


# -------------------------
# I/O helpers
# -------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _to_1d_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _save_npz_case(path: str, radar_1d: np.ndarray, gt_1d: np.ndarray, pred_1d: np.ndarray,
                   meta: Dict[str, Any]):
    # keep it minimal and stable for plotting scripts
    np.savez(
        path,
        radar=radar_1d.astype(np.float32),
        ecg_true=gt_1d.astype(np.float32),
        ecg_pred=pred_1d.astype(np.float32),
        **{k: np.array([v]) if np.isscalar(v) else v for k, v in meta.items()}
    )


def _median_iqr_series(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float).values
    if x.size == 0:
        return {"median": np.nan, "p25": np.nan, "p75": np.nan}
    return {
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
    }


def _choose_fig2_sample(df_seg: pd.DataFrame,
                        prefer_subject: Optional[int] = None) -> Tuple[int, int]:
    """
    Protocol: choose one subject + one median-performing sample.
    If prefer_subject is None, pick the smallest subject id in test set.
    Returns (subject_id, seg_id).
    """
    if df_seg.empty:
        return -1, -1

    if prefer_subject is None:
        sid = int(df_seg["subject_id"].min())
    else:
        sid = int(prefer_subject)
        if sid not in set(df_seg["subject_id"].tolist()):
            sid = int(df_seg["subject_id"].min())

    df_s = df_seg[df_seg["subject_id"] == sid].sort_values("pcc").reset_index(drop=True)
    if len(df_s) == 0:
        # fallback to global median
        df_all = df_seg.sort_values("pcc").reset_index(drop=True)
        mid = int(len(df_all) // 2)
        return int(df_all.loc[mid, "subject_id"]), int(df_all.loc[mid, "seg_id"])

    mid = int(len(df_s) // 2)
    return int(df_s.loc[mid, "subject_id"]), int(df_s.loc[mid, "seg_id"])


@dataclass
class MetaInfo:
    exp_name: str
    exp_tag: str
    seed: int
    device: str
    fs: int
    ckpt_path: str
    ckpt_mode: str
    timestamp: str

    # Protocol pointers
    fig2_subject_id: int
    fig2_seg_id: int

    # Optional notes
    test_h5: str
    test_bad_indices_path: Optional[str] = None
    nk_method: str = "peak"
    rpeak_match_ms: float = 150.0


def test():
    parser = argparse.ArgumentParser(
        description="Test BeatAware R-M2Net (protocol-aligned exports: Fig.2–Fig.4 + Table III)"
    )

    parser.add_argument("--exp_tag", type=str, default="Default")

    # checkpoint control
    parser.add_argument("--ckpt", type=str, default="best",
                        help="best | last | /abs/path/to.pth or relative path")

    # neurokit delineation
    parser.add_argument("--nk_method", type=str, default="peak", choices=["peak", "dwt", "cwt"],
                        help="NeuroKit2 delineation method (peak is most stable).")
    parser.add_argument("--rpeak_match_ms", type=float, default=150.0,
                        help="Max R-peak matching distance in ms for beat-level metrics.")

    # protocol exports
    parser.add_argument("--prefer_fig2_subject", type=int, default=None,
                        help="Optional: force Fig.2 to use this subject id if available.")
    parser.add_argument("--save_cases", action="store_true",
                        help="Export per-subject median cases (Fig.1) + fixed Fig.2 sample.")
    parser.add_argument("--export_debug_clinical", action="store_true",
                        help="(Debug only) export segment-level HR/RR/QRS/QT mean features. Not paper mainline.")
    parser.add_argument("--export_debug_mask", action="store_true",
                        help="(Debug only) export mask metrics if model outputs mask logits.")

    args = parser.parse_args()

    # --------------------------
    # 1) Setup experiment paths
    # --------------------------
    Config.update_paths(f"{args.exp_tag}")
    seed_everything(Config.SEED)
    device = Config.DEVICE
    fs = int(getattr(Config, "FS_TARGET", getattr(Config, "FS", 200)))

    result_dir = Config.RESULT_DIR
    _ensure_dir(result_dir)
    cases_dir = os.path.join(result_dir, "cases")
    _ensure_dir(cases_dir)

    # QC (test bad indices optional)
    bad_path = getattr(Config, "TEST_BAD_INDICES_PATH", None)
    if bad_path is not None and (not os.path.exists(bad_path)):
        print(f"[QC] TEST_BAD_INDICES_PATH not found, disable: {bad_path}")
        bad_path = None
    print(f"[QC] TEST_BAD_INDICES_PATH = {bad_path}")

    # checkpoint resolve
    if args.ckpt.lower() == "best":
        ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")
        ckpt_mode = "best"
    elif args.ckpt.lower() == "last":
        ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_last.pth")
        ckpt_mode = "last"
    else:
        ckpt_path = args.ckpt
        ckpt_mode = "path"
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(Config.ROOT_DIR, ckpt_path)

    print(f"🚀 Starting Test: {Config.EXP_NAME}")
    print(f"   Test data : {Config.TEST_H5}")
    print(f"   Results   : {result_dir}")
    print(f"   CKPT mode : {ckpt_mode}")
    print(f"   CKPT path : {ckpt_path}")

    # --------------------------
    # 2) Dataset + loader
    # --------------------------
    test_set = RadarDataset(Config.TEST_H5, bad_indices_path=bad_path)
    # You can increase batch_size later; keep 1 for simplest reproducibility
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # --------------------------
    # 3) Model + weights
    # --------------------------
    model = BeatAwareRM2Net(
        in_channels=Config.IN_CHANNELS,
        base_channels=Config.BASE_CHANNELS
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[ERROR] Checkpoint not found: {ckpt_path}\n"
            f"Tips: train this EXP_NAME={Config.EXP_NAME}, or pass --ckpt /path/to.pth"
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("✅ Model weights loaded.")

    # --------------------------
    # 4) Inference + protocol metrics
    # --------------------------
    segment_rows: List[Dict[str, Any]] = []
    beat_rows_all: List[Dict[str, Any]] = []

    # counts for beat validity reporting
    beats_total_sum = 0
    beats_valid_sum = 0
    drop_reason_counter: Dict[str, int] = {}

    # minimal cache for case exports
    # (test set is small; safe to cache waveforms for median-case export)
    cache: Dict[int, Dict[str, Any]] = {}  # seg_id -> payload

    # optional debug exports
    clinical_rows: List[Dict[str, Any]] = []
    mask_rows: List[Dict[str, Any]] = []

    print(f"🚀 Running inference on {len(test_set)} segments ...")

    with torch.no_grad():
        for seg_id, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # expected from RadarDataset: radar, ecg, mask, subject_id
            radar, ecg, mask_true, subject_id = batch

            radar = radar.to(device)
            ecg = ecg.to(device)

            sid = int(subject_id.item())

            # forward: allow either (pred_ecg, pred_mask_logits) or pred_ecg only
            out = model(radar)
            if isinstance(out, (tuple, list)) and len(out) >= 1:
                pred_ecg = out[0]
                pred_mask_logits = out[1] if (len(out) > 1) else None
            else:
                pred_ecg = out
                pred_mask_logits = None

            p_np = _to_1d_np(pred_ecg)
            g_np = _to_1d_np(ecg)
            r_np = _to_1d_np(radar)

            # ---- segment-level (PCC/MRE) ----
            seg_m = compute_segment_metrics(p_np, g_np)
            segment_rows.append({
                "subject_id": sid,
                "seg_id": int(seg_id),
                "pcc": float(seg_m["pcc"]),
                "mre": float(seg_m["mre"]),
                # keep debug-friendly columns (not necessarily used in paper)
                "mae": float(seg_m["mae"]),
                "rmse": float(seg_m["rmse"]),
            })

            # ---- beat-level (RR/QRS/QT) ----
            beats, beat_meta = compute_beat_metrics_nk(
                p_np, g_np,
                fs=fs,
                method=str(args.nk_method),
                max_rpeak_match_ms=float(args.rpeak_match_ms),
            )

            # aggregate beat counts
            beats_total_sum += int(beat_meta.get("n_beats_total", 0))
            beats_valid_sum += int(beat_meta.get("n_beats_valid", 0))
            for k, v in (beat_meta.get("drop_reasons", {}) or {}).items():
                drop_reason_counter[k] = drop_reason_counter.get(k, 0) + int(v)

            # attach identifiers
            for b in beats:
                b.update({"subject_id": sid, "seg_id": int(seg_id)})
            beat_rows_all.extend(beats)

            # ---- cache for cases ----
            if args.save_cases:
                cache[int(seg_id)] = {
                    "subject_id": sid,
                    "seg_id": int(seg_id),
                    "pcc": float(seg_m["pcc"]),
                    "radar": r_np.astype(np.float32),
                    "ecg_true": g_np.astype(np.float32),
                    "ecg_pred": p_np.astype(np.float32),
                }

            # ---- optional debug: clinical features / mask ----
            if args.export_debug_clinical:
                # segment-mean clinical features (NOT paper mainline)
                # kept optional; you can remove entirely if you want.
                import neurokit2 as nk
                row = {"subject_id": sid, "seg_id": int(seg_id)}
                for name, sig in [("gt", g_np), ("pred", p_np)]:
                    try:
                        _, info = nk.ecg_peaks(sig, sampling_rate=fs)
                        rpk = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
                        if rpk.size >= 2:
                            rr_ms = np.diff(rpk) * (1000.0 / fs)
                            rr = float(np.nanmean(rr_ms))
                            hr = float(60000.0 / rr) if rr > 1e-6 else np.nan
                        else:
                            rr, hr = np.nan, np.nan
                    except Exception:
                        rr, hr = np.nan, np.nan
                    row[f"HR_{name}"] = hr
                    row[f"RR_{name}"] = rr
                clinical_rows.append(row)

            if args.export_debug_mask and pred_mask_logits is not None:
                # mask metrics (NOT paper mainline)
                import torch.nn.functional as F
                mt = mask_true.to(device)
                prob = torch.sigmoid(pred_mask_logits).reshape(-1)
                gt = mt.reshape(-1)
                # binarize
                pb = (prob >= 0.5).to(torch.int32)
                gb = (gt >= 0.5).to(torch.int32)
                tp = int(((pb == 1) & (gb == 1)).sum().item())
                fp = int(((pb == 1) & (gb == 0)).sum().item())
                fn = int(((pb == 0) & (gb == 1)).sum().item())
                prec = tp / (tp + fp + 1e-12)
                rec = tp / (tp + fn + 1e-12)
                f1 = 2 * prec * rec / (prec + rec + 1e-12)
                mask_rows.append({
                    "subject_id": sid,
                    "seg_id": int(seg_id),
                    "mask_f1": float(f1),
                    "mask_precision": float(prec),
                    "mask_recall": float(rec),
                    "mask_prob_mean": float(prob.mean().item()),
                })

    # --------------------------
    # 5) Save protocol CSVs
    # --------------------------
    df_seg = pd.DataFrame(segment_rows)
    df_beat = pd.DataFrame(beat_rows_all)

    seg_csv = os.path.join(result_dir, "segment_metrics.csv")
    beat_csv = os.path.join(result_dir, "beat_metrics.csv")
    df_seg.to_csv(seg_csv, index=False)
    df_beat.to_csv(beat_csv, index=False)

    print("\n✅ Saved protocol data:")
    print(f"   - {seg_csv}")
    print(f"   - {beat_csv}")

    # optional debug exports
    if args.export_debug_clinical and len(clinical_rows) > 0:
        clin_csv = os.path.join(result_dir, "clinical_metrics.csv")
        pd.DataFrame(clinical_rows).to_csv(clin_csv, index=False)
        print(f"   - {clin_csv}  (debug)")

    if args.export_debug_mask and len(mask_rows) > 0:
        mask_csv = os.path.join(result_dir, "mask_metrics.csv")
        pd.DataFrame(mask_rows).to_csv(mask_csv, index=False)
        print(f"   - {mask_csv}  (debug)")

    # --------------------------
    # 6) Subject-wise summary (Fig.3) — median + IQR
    # --------------------------
    # Segment-level (PCC/MRE) per subject
    subj_rows = []
    for sid, g in df_seg.groupby("subject_id"):
        pcc_stats = _median_iqr_series(g["pcc"])
        mre_stats = _median_iqr_series(g["mre"])
        subj_rows.append({
            "subject_id": int(sid),
            "pcc_median": pcc_stats["median"],
            "pcc_p25": pcc_stats["p25"],
            "pcc_p75": pcc_stats["p75"],
            "mre_median": mre_stats["median"],
        })

    df_subject = pd.DataFrame(subj_rows).sort_values("subject_id").reset_index(drop=True)

    # Beat-level errors aggregated per subject (median of beat errors)
    if not df_beat.empty:
        beat_agg = df_beat.groupby("subject_id").agg(
            rr_err_median_ms=("rr_err_ms", "median"),
            qrs_err_median_ms=("qrs_err_ms", "median"),
            qt_err_median_ms=("qt_err_ms", "median"),
        ).reset_index()
        df_subject = df_subject.merge(beat_agg, on="subject_id", how="left")

    subject_csv = os.path.join(result_dir, "subject_summary.csv")
    df_subject.to_csv(subject_csv, index=False)
    print(f"   - {subject_csv}")

    # --------------------------
    # 7) Global summary JSON (Table III) — median-first + beats validity
    # --------------------------
    summary = summarize_metrics(segment_rows, beat_rows_all)
    summary["exp_name"] = Config.EXP_NAME
    summary["n_subjects"] = int(df_subject.shape[0])

    # beats validity info (explicitly required by protocol)
    summary["counts"]["n_beats_total"] = int(beats_total_sum)
    summary["counts"]["n_beats_valid"] = int(beats_valid_sum)
    summary["counts"]["drop_rate"] = float(
        1.0 - (beats_valid_sum / max(beats_total_sum, 1))
    )
    summary["counts"]["drop_reasons"] = drop_reason_counter

    global_json = os.path.join(result_dir, "global_summary.json")
    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   - {global_json}")

    # --------------------------
    # 8) Fixed Fig.2 sample pointer + case exports (Fig.1 optional)
    # --------------------------
    fig2_sid, fig2_segid = _choose_fig2_sample(df_seg, prefer_subject=args.prefer_fig2_subject)

    meta = MetaInfo(
        exp_name=Config.EXP_NAME,
        exp_tag=str(args.exp_tag),
        seed=int(Config.SEED),
        device=str(device),
        fs=int(fs),
        ckpt_path=str(ckpt_path),
        ckpt_mode=str(ckpt_mode),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        fig2_subject_id=int(fig2_sid),
        fig2_seg_id=int(fig2_segid),
        test_h5=str(Config.TEST_H5),
        test_bad_indices_path=bad_path,
        nk_method=str(args.nk_method),
        rpeak_match_ms=float(args.rpeak_match_ms),
    )
    meta_json = os.path.join(result_dir, "meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)
    print(f"   - {meta_json}")

    if args.save_cases:
        print("\n🧪 Exporting protocol cases ...")

        # Fig.2 sample
        if fig2_segid in cache:
            item = cache[fig2_segid]
            fig2_path = os.path.join(cases_dir, f"fig2_sample_sid{fig2_sid}_seg{fig2_segid}.npz")
            _save_npz_case(
                fig2_path,
                radar_1d=item["radar"],
                gt_1d=item["ecg_true"],
                pred_1d=item["ecg_pred"],
                meta={"subject_id": int(fig2_sid), "seg_id": int(fig2_segid), "pcc": float(item["pcc"])}
            )
            print(f"   - Fig.2 sample: {fig2_path}")
        else:
            print("   [WARN] Fig.2 sample waveform not cached (unexpected).")

        # Fig.1 per-subject median cases:
        # For each subject: sort by PCC and take median segment
        df_seg_sorted = df_seg.sort_values(["subject_id", "pcc"]).reset_index(drop=True)
        for sid in sorted(df_seg["subject_id"].unique().tolist()):
            df_s = df_seg_sorted[df_seg_sorted["subject_id"] == sid].reset_index(drop=True)
            if len(df_s) == 0:
                continue
            mid = int(len(df_s) // 2)
            segid = int(df_s.loc[mid, "seg_id"])
            pccv = float(df_s.loc[mid, "pcc"])

            if segid not in cache:
                continue
            item = cache[segid]
            out_path = os.path.join(cases_dir, f"subject_{sid}_median_seg{segid}_pcc{pccv:.3f}.npz")
            _save_npz_case(
                out_path,
                radar_1d=item["radar"],
                gt_1d=item["ecg_true"],
                pred_1d=item["ecg_pred"],
                meta={"subject_id": int(sid), "seg_id": int(segid), "pcc": float(pccv)}
            )

        print(f"   - Cases saved under: {cases_dir}")

    print("\n✅ Test finished successfully.")
    print(f"   Results directory: {result_dir}")


if __name__ == "__main__":
    test()
