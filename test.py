import os
import json
import time
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- project imports ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.metrics import calculate_metrics, extract_clinical_features_nk
from utils.seeding import seed_everything


# =========================
# Utilities (safe + robust)
# =========================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_feat(x_1d: np.ndarray, fs: int) -> Dict[str, float]:
    """Return clinical features dict with float values; missing -> NaN."""
    try:
        feat = extract_clinical_features_nk(x_1d, fs=fs) or {}
        out = {}
        for k in ["HR", "RR", "QRS", "QT"]:
            v = feat.get(k, np.nan)
            try:
                out[k] = float(v)
            except Exception:
                out[k] = np.nan
        return out
    except Exception:
        return {"HR": np.nan, "RR": np.nan, "QRS": np.nan, "QT": np.nan}


def _nan_summary(x: np.ndarray) -> Dict[str, float]:
    """Return summary stats robust to NaN."""
    x = np.asarray(x, dtype=np.float64)
    x_f = x[np.isfinite(x)]
    if x_f.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "q05": np.nan, "q25": np.nan, "q75": np.nan, "q95": np.nan,
            "min": np.nan, "max": np.nan, "n": 0
        }
    return {
        "mean": float(np.mean(x_f)),
        "std": float(np.std(x_f)),
        "median": float(np.median(x_f)),
        "q05": float(np.quantile(x_f, 0.05)),
        "q25": float(np.quantile(x_f, 0.25)),
        "q75": float(np.quantile(x_f, 0.75)),
        "q95": float(np.quantile(x_f, 0.95)),
        "min": float(np.min(x_f)),
        "max": float(np.max(x_f)),
        "n": int(x_f.size),
    }


def _to_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _sigmoid_np(x: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(x).detach().cpu().numpy().astype(np.float64)


def _threshold_mask(prob_1d: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (prob_1d >= thr).astype(np.int32)


def _f1_precision_recall(mask_pred_bin: np.ndarray, mask_true_bin: np.ndarray) -> Tuple[float, float, float]:
    """Pixel-wise F1 on 0/1 mask."""
    mp = mask_pred_bin.astype(np.int32)
    mt = mask_true_bin.astype(np.int32)
    tp = int(np.sum((mp == 1) & (mt == 1)))
    fp = int(np.sum((mp == 1) & (mt == 0)))
    fn = int(np.sum((mp == 0) & (mt == 1)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(f1), float(prec), float(rec)


def _topk_case_indices(pcc_list: List[float], k: int = 5) -> Dict[str, List[int]]:
    """Return indices for best/worst/median/random cases."""
    pcc = np.asarray(pcc_list, dtype=np.float64)
    n = len(pcc_list)
    order = np.argsort(pcc)  # ascending
    best = order[-k:].tolist() if n >= k else order.tolist()
    worst = order[:k].tolist() if n >= k else order.tolist()
    median = [int(order[n // 2])] if n > 0 else []
    rng = np.random.RandomState(0)
    rand = rng.choice(np.arange(n), size=min(k, n), replace=False).tolist() if n > 0 else []
    return {"best": best, "worst": worst, "median": median, "random": rand}


@dataclass
class MetaInfo:
    exp_name: str
    alpha: float
    beta: float
    gamma: float
    exp_tag: str
    seed: int
    device: str
    fs: int
    qc_test_bad_indices_path: Optional[str]
    ckpt_path: str
    timestamp: str

    # Optional loss-related config if you added them in Config
    stft_fmin: Optional[float] = None
    stft_fmax: Optional[float] = None
    stft_use_band: Optional[bool] = None
    anchor_from_logits: Optional[bool] = None
    anchor_pos_weight: Optional[float] = None


def test():
    # --------------------------
    # 0) CLI
    # --------------------------
    parser = argparse.ArgumentParser(description="Test BeatAware R-M2Net (publication-grade)")
    parser.add_argument('--alpha', type=float, default=0.5, help='STFT loss weight used in training')
    parser.add_argument('--beta', type=float, default=1.0, help='Anchor loss weight used in training')
    parser.add_argument('--gamma', type=float, default=0.1, help='Smooth loss weight used in training')
    parser.add_argument('--exp_tag', type=str, default="Default", help='Tag used for this experiment')

    # optional knobs for analysis only (won't break your structure)
    parser.add_argument('--mask_thr', type=float, default=0.5, help='threshold for predicted mask (sigmoid probs)')
    parser.add_argument('--save_cases_k', type=int, default=5, help='number of best/worst/random cases to export')
    parser.add_argument('--export_full_npz', action='store_true', help='export per-segment NPZ (large), off by default')
    args = parser.parse_args()

    # --------------------------
    # 1) Path & env setup
    # --------------------------
    new_exp_name = f"Exp_a{args.alpha}_b{args.beta}_g{args.gamma}_{args.exp_tag}"
    Config.update_paths(new_exp_name)

    seed_everything(Config.SEED)
    device = Config.DEVICE
    fs = int(getattr(Config, "FS", 200))

    ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")
    result_dir = Config.RESULT_DIR
    _ensure_dir(result_dir)
    cases_dir = os.path.join(result_dir, "cases")
    _ensure_dir(cases_dir)

    print(f"🚀 Starting Test for Experiment: {Config.EXP_NAME}")
    print(f"   Reading test data from: {Config.TEST_H5}")
    print(f"   Writing results to     : {result_dir}")

    # QC (test)
    bad_path = getattr(Config, "TEST_BAD_INDICES_PATH", None)
    if bad_path is not None and (not os.path.exists(bad_path)):
        print(f"[QC] TEST_BAD_INDICES_PATH not found, disable: {bad_path}")
        bad_path = None
    print(f"[QC] TEST_BAD_INDICES_PATH = {bad_path}")

    # --------------------------
    # 2) Dataset
    # --------------------------
    test_set = RadarDataset(Config.TEST_H5, bad_indices_path=bad_path)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # --------------------------
    # 3) Model
    # --------------------------
    model = BeatAwareRM2Net(
        in_channels=Config.IN_CHANNELS,
        base_channels=Config.BASE_CHANNELS
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[ERROR] Checkpoint not found!\n"
            f"  Expected path: {ckpt_path}\n"
            f"  Please check:\n"
            f"   1) EXP_NAME consistency between train.py and test.py\n"
            f"   2) alpha/beta/gamma/exp_tag arguments\n"
            f"   3) Whether training finished and saved best.pth"
        )

    print(f"✅ Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # tolerant load
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    # --------------------------
    # 4) Meta export
    # --------------------------
    meta = MetaInfo(
        exp_name=Config.EXP_NAME,
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        exp_tag=str(args.exp_tag),
        seed=int(Config.SEED),
        device=str(device),
        fs=int(fs),
        qc_test_bad_indices_path=bad_path,
        ckpt_path=ckpt_path,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        stft_fmin=getattr(Config, "STFT_FMIN", None),
        stft_fmax=getattr(Config, "STFT_FMAX", None),
        stft_use_band=getattr(Config, "STFT_USE_BAND", None),
        anchor_from_logits=getattr(Config, "ANCHOR_FROM_LOGITS", None),
        anchor_pos_weight=getattr(Config, "ANCHOR_POS_WEIGHT", None),
    )
    with open(os.path.join(result_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    # --------------------------
    # 5) Inference + metrics
    # --------------------------
    # segment-level records (each test window)
    seg_rows: List[Dict[str, Any]] = []
    clinical_rows: List[Dict[str, Any]] = []
    mask_rows: List[Dict[str, Any]] = []

    # for case export
    pcc_list: List[float] = []
    pool_for_cases: List[Dict[str, Any]] = []

    # optional full dump (can be huge)
    full_npz_buffers = []

    print(f"🚀 Running inference on {len(test_set)} segments ...")
    with torch.no_grad():
        for seg_id, (radar, ecg, mask_true, subject_id) in enumerate(tqdm(test_loader, desc="Testing")):
            radar = radar.to(device)
            ecg = ecg.to(device)
            mask_true = mask_true.to(device)

            pred_ecg, pred_mask_logits = model(radar)  # pred_mask is logits (per your latest BA_M2Net)
            # NOTE: if your model outputs sigmoid already, then change sigmoid usage below accordingly.

            # ---- waveform metrics (batch=1) ----
            wave_m = calculate_metrics(pred_ecg, ecg)
            pcc_val = float(wave_m.get("Pearson", np.nan))
            mae_val = float(wave_m.get("MAE", np.nan))
            rmse_val = float(wave_m.get("RMSE", np.nan))

            # ---- to numpy vectors ----
            p_np = _to_1d(pred_ecg)
            t_np = _to_1d(ecg)
            r_np = _to_1d(radar)
            pm_prob = _sigmoid_np(pred_mask_logits).reshape(-1)
            tm_np = _to_1d(mask_true)

            # ---- clinical (HR/RR/QRS/QT) ----
            p_feat = _safe_feat(p_np, fs=fs)
            t_feat = _safe_feat(t_np, fs=fs)

            # ---- mask quality (pixel-wise) ----
            tm_bin = (tm_np >= 0.5).astype(np.int32)
            pm_bin = _threshold_mask(pm_prob, thr=float(args.mask_thr))
            f1, prec, rec = _f1_precision_recall(pm_bin, tm_bin)

            # ---- record: segment metrics ----
            sid = int(subject_id.item())
            seg_rows.append({
                "seg_id": int(seg_id),
                "Subject_ID": sid,
                "PCC": pcc_val,
                "MAE": mae_val,
                "RMSE": rmse_val,
            })

            # ---- record: clinical metrics ----
            row_c = {
                "seg_id": int(seg_id),
                "Subject_ID": sid,
                "HR_True": t_feat["HR"], "HR_Pred": p_feat["HR"],
                "RR_True": t_feat["RR"], "RR_Pred": p_feat["RR"],
                "QRS_True": t_feat["QRS"], "QRS_Pred": p_feat["QRS"],
                "QT_True": t_feat["QT"], "QT_Pred": p_feat["QT"],
            }
            row_c["HR_Error"] = abs(row_c["HR_Pred"] - row_c["HR_True"]) if np.isfinite(row_c["HR_Pred"]) and np.isfinite(row_c["HR_True"]) else np.nan
            row_c["RR_Error"] = abs(row_c["RR_Pred"] - row_c["RR_True"]) if np.isfinite(row_c["RR_Pred"]) and np.isfinite(row_c["RR_True"]) else np.nan
            row_c["QRS_Error"] = abs(row_c["QRS_Pred"] - row_c["QRS_True"]) if np.isfinite(row_c["QRS_Pred"]) and np.isfinite(row_c["QRS_True"]) else np.nan
            row_c["QT_Error"]  = abs(row_c["QT_Pred"] - row_c["QT_True"]) if np.isfinite(row_c["QT_Pred"]) and np.isfinite(row_c["QT_True"]) else np.nan
            clinical_rows.append(row_c)

            # ---- record: mask metrics ----
            mask_rows.append({
                "seg_id": int(seg_id),
                "Subject_ID": sid,
                "Mask_F1": float(f1),
                "Mask_Precision": float(prec),
                "Mask_Recall": float(rec),
                "Mask_Prob_Mean": float(np.mean(pm_prob)),
                "Mask_Prob_Max": float(np.max(pm_prob)),
                "Mask_True_Sparsity": float(np.mean(tm_bin)),  # fraction of ones
                "Mask_Pred_Sparsity": float(np.mean(pm_bin)),
            })

            pcc_list.append(pcc_val)

            # pool for later case selection (keep minimal but sufficient)
            pool_for_cases.append({
                "seg_id": int(seg_id),
                "Subject_ID": sid,
                "pcc": pcc_val,
                "radar": r_np.astype(np.float32),
                "ecg_true": t_np.astype(np.float32),
                "ecg_pred": p_np.astype(np.float32),
                "mask_true": tm_bin.astype(np.int16),
                "mask_prob": pm_prob.astype(np.float32),
            })

            # optional: full segment npz (can be huge!)
            if args.export_full_npz:
                full_npz_buffers.append({
                    "seg_id": int(seg_id),
                    "Subject_ID": sid,
                    "radar": r_np.astype(np.float32),
                    "ecg_true": t_np.astype(np.float32),
                    "ecg_pred": p_np.astype(np.float32),
                    "mask_true": tm_bin.astype(np.int16),
                    "mask_prob": pm_prob.astype(np.float32),
                })

    # --------------------------
    # 6) Save CSVs (segment-level)
    # --------------------------
    df_seg = pd.DataFrame(seg_rows)
    df_clin = pd.DataFrame(clinical_rows)
    df_mask = pd.DataFrame(mask_rows)

    seg_csv = os.path.join(result_dir, "segment_metrics.csv")
    clin_csv = os.path.join(result_dir, "clinical_metrics.csv")
    mask_csv = os.path.join(result_dir, "mask_metrics.csv")

    df_seg.to_csv(seg_csv, index=False)
    df_clin.to_csv(clin_csv, index=False)
    df_mask.to_csv(mask_csv, index=False)

    print(f"\n✅ Saved:")
    print(f"   - {seg_csv}")
    print(f"   - {clin_csv}")
    print(f"   - {mask_csv}")

    # --------------------------
    # 7) Subject-wise summary (inter-subject)
    # --------------------------
    # subject mean then global stats across subjects
    subj_seg = df_seg.groupby("Subject_ID").mean(numeric_only=True)
    subj_clin = df_clin.groupby("Subject_ID").mean(numeric_only=True)
    subj_mask = df_mask.groupby("Subject_ID").mean(numeric_only=True)

    # merge summaries
    df_subject = subj_seg.join(subj_clin, how="outer", rsuffix="_clin").join(subj_mask, how="outer", rsuffix="_mask")
    subject_csv = os.path.join(result_dir, "subject_summary.csv")
    df_subject.reset_index().to_csv(subject_csv, index=False)
    print(f"   - {subject_csv}")

    # --------------------------
    # 8) Global summary JSON (publication-ready)
    # --------------------------
    global_summary = {
        "exp_name": Config.EXP_NAME,
        "n_segments": int(len(df_seg)),
        "n_subjects": int(df_subject.shape[0]),
        "segment_level": {
            "PCC": _nan_summary(df_seg["PCC"].values),
            "MAE": _nan_summary(df_seg["MAE"].values),
            "RMSE": _nan_summary(df_seg["RMSE"].values),
        },
        "subject_level": {
            "PCC": _nan_summary(df_subject["PCC"].values),
            "MAE": _nan_summary(df_subject["MAE"].values),
            "RMSE": _nan_summary(df_subject["RMSE"].values),
            "HR_Error": _nan_summary(df_subject.get("HR_Error", pd.Series(dtype=float)).values),
            "RR_Error": _nan_summary(df_subject.get("RR_Error", pd.Series(dtype=float)).values),
            "QRS_Error": _nan_summary(df_subject.get("QRS_Error", pd.Series(dtype=float)).values),
            "QT_Error": _nan_summary(df_subject.get("QT_Error", pd.Series(dtype=float)).values),
            "Mask_F1": _nan_summary(df_subject.get("Mask_F1", pd.Series(dtype=float)).values),
        },
    }

    global_json = os.path.join(result_dir, "global_summary.json")
    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)
    print(f"   - {global_json}")

    # --------------------------
    # 9) Case export (best/worst/median/random)
    # --------------------------
    case_idx = _topk_case_indices(pcc_list, k=int(args.save_cases_k))
    print("\n🧪 Exporting representative cases:")
    print(f"   best   : {case_idx['best']}")
    print(f"   worst  : {case_idx['worst']}")
    print(f"   median : {case_idx['median']}")
    print(f"   random : {case_idx['random']}")

    # Map seg_id -> pooled dict
    # Here pool_for_cases index matches seg_id order, so direct indexing is fine.
    def _save_case(case_name: str, indices: List[int]):
        for j, idx in enumerate(indices):
            item = pool_for_cases[idx]
            out_path = os.path.join(
                cases_dir,
                f"{case_name}_k{j}_seg{item['seg_id']}_sid{item['Subject_ID']}_pcc{item['pcc']:.3f}.npz"
            )
            np.savez(
                out_path,
                radar=item["radar"],
                ecg_true=item["ecg_true"],
                ecg_pred=item["ecg_pred"],
                mask_true=item["mask_true"],
                mask_prob=item["mask_prob"],
                pcc=np.array([item["pcc"]], dtype=np.float32),
                seg_id=np.array([item["seg_id"]], dtype=np.int32),
                subject_id=np.array([item["Subject_ID"]], dtype=np.int32),
            )

    _save_case("best", case_idx["best"])
    _save_case("worst", case_idx["worst"])
    _save_case("median", case_idx["median"])
    _save_case("random", case_idx["random"])

    print(f"   - cases saved under: {cases_dir}")

    # Optional: full NPZ dump (huge)
    if args.export_full_npz and len(full_npz_buffers) > 0:
        full_dir = os.path.join(result_dir, "full_npz")
        _ensure_dir(full_dir)
        for item in full_npz_buffers:
            out_path = os.path.join(full_dir, f"seg{item['seg_id']}_sid{item['Subject_ID']}.npz")
            np.savez(out_path, **item)
        print(f"   - full npz saved under: {full_dir} (⚠️ may be very large)")

    # --------------------------
    # 10) Print quick table-like stats
    # --------------------------
    print("\n📊 Quick Stats (subject-wise mean ± std):")
    for col in ["PCC", "MAE", "RMSE", "HR_Error", "RR_Error", "QRS_Error", "QT_Error", "Mask_F1"]:
        if col in df_subject.columns:
            mu = np.nanmean(df_subject[col].values)
            sd = np.nanstd(df_subject[col].values)
            print(f"   {col:10s}: {mu:.4f} ± {sd:.4f}")

    print("\n✅ Test finished successfully.")
    print(f"   Results directory: {result_dir}")


if __name__ == "__main__":
    test()
