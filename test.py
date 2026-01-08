import os
import torch
import json
import h5py
import numpy as np
import pandas as pd
import argparse  # <--- [核心修复] 补上这一行
from scipy.signal import find_peaks
from torch.utils.data import DataLoader
from tqdm import tqdm
# ✅ [新增] 引入 defaultdict，自动处理所有指标 Key
from collections import defaultdict 

# --- 引入项目模块 ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.metrics import calculate_metrics, extract_clinical_features_nk
from utils.seeding import seed_everything

# ============================================================
# Helper: robust clinical feature extraction
# ============================================================
def _safe_feat(x_1d: np.ndarray, fs: int = 200) -> dict:
    try:
        feat = extract_clinical_features_nk(x_1d, fs=fs)
        return feat if isinstance(feat, dict) else {}
    except Exception:
        return {}


def _get(feat: dict, key: str) -> float:
    v = feat.get(key, np.nan)
    try:
        return float(v)
    except Exception:
        return np.nan


# ============================================================
# Helper: beat-level evaluation using mask peaks
# ============================================================
def _mask_to_peaks(mask_1d: np.ndarray, fs: int, prom_frac: float = 0.30) -> np.ndarray:
    """
    Convert a continuous mask (0~1) into peak indices.
    - prom_frac: prominence fraction relative to (p99 - p1) amplitude range
    """
    m = np.nan_to_num(mask_1d).astype(np.float64)

    amp = np.percentile(m, 99) - np.percentile(m, 1)
    prom = max(prom_frac * amp, 1e-6)

    # distance ~ 최소 RR 间격下界：0.35s @200Hz≈70 samples（与 verify_alignment_metrics.py 逻辑一致）
    min_dist = int(round(0.35 * fs))
    peaks, _ = find_peaks(m, prominence=prom, distance=min_dist)
    return peaks.astype(np.int32)


def _beat_metrics_from_peaks(
    pred_peaks: np.ndarray,
    true_peaks: np.ndarray,
    fs: int,
    tol_ms: float = 50.0
) -> dict:
    """
    Beat-level precision/recall/F1 and timing error (ms).
    Matching criterion: |pred - true| <= tol
    """
    tol = int(round(tol_ms * fs / 1000.0))

    if true_peaks.size == 0 and pred_peaks.size == 0:
        return {
            "Beat_Precision": 1.0,
            "Beat_Recall": 1.0,
            "Beat_F1": 1.0,
            "Beat_MAE_ms": 0.0,
            "Beat_MedianAE_ms": 0.0,
            "Beat_N_true": 0,
            "Beat_N_pred": 0,
            "Beat_N_match": 0,
        }

    if true_peaks.size == 0:
        return {
            "Beat_Precision": 0.0,
            "Beat_Recall": np.nan,
            "Beat_F1": np.nan,
            "Beat_MAE_ms": np.nan,
            "Beat_MedianAE_ms": np.nan,
            "Beat_N_true": 0,
            "Beat_N_pred": int(pred_peaks.size),
            "Beat_N_match": 0,
        }

    if pred_peaks.size == 0:
        return {
            "Beat_Precision": np.nan,
            "Beat_Recall": 0.0,
            "Beat_F1": np.nan,
            "Beat_MAE_ms": np.nan,
            "Beat_MedianAE_ms": np.nan,
            "Beat_N_true": int(true_peaks.size),
            "Beat_N_pred": 0,
            "Beat_N_match": 0,
        }

    # match pred -> true (precision) and true -> pred (recall)
    matched_pred = 0
    dt_errors = []

    # precision: for each pred peak, find nearest true
    for p in pred_peaks:
        j = int(np.argmin(np.abs(true_peaks - p)))
        dt = int(p - true_peaks[j])
        if abs(dt) <= tol:
            matched_pred += 1
            dt_errors.append(dt)

    matched_true = 0
    # recall: for each true peak, whether any pred within tol
    for t in true_peaks:
        if np.any(np.abs(pred_peaks - t) <= tol):
            matched_true += 1

    precision = matched_pred / float(pred_peaks.size) if pred_peaks.size > 0 else np.nan
    recall = matched_true / float(true_peaks.size) if true_peaks.size > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan

    if len(dt_errors) > 0:
        dt_errors = np.array(dt_errors, dtype=np.float64)
        mae_ms = float(np.mean(np.abs(dt_errors)) / fs * 1000.0)
        med_ms = float(np.median(np.abs(dt_errors)) / fs * 1000.0)
    else:
        mae_ms = np.nan
        med_ms = np.nan

    return {
        "Beat_Precision": float(precision) if np.isfinite(precision) else np.nan,
        "Beat_Recall": float(recall) if np.isfinite(recall) else np.nan,
        "Beat_F1": float(f1) if np.isfinite(f1) else np.nan,
        "Beat_MAE_ms": mae_ms,
        "Beat_MedianAE_ms": med_ms,
        "Beat_N_true": int(true_peaks.size),
        "Beat_N_pred": int(pred_peaks.size),
        "Beat_N_match": int(matched_pred),
    }


# ============================================================
# Helper: stats dump
# ============================================================
def _summary_stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    return {
        "n": int(x.notna().sum()),
        "mean": float(x.mean(skipna=True)) if x.notna().any() else np.nan,
        "std": float(x.std(skipna=True)) if x.notna().any() else np.nan,
        "median": float(x.median(skipna=True)) if x.notna().any() else np.nan,
        "q25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
        "q75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
        "nan_rate": float(x.isna().mean()),
    }


# ============================================================
# Main
# ============================================================
def test():
    parser = argparse.ArgumentParser(description="Test BeatAware R-M2Net (paper-ready export)")
    parser.add_argument("--alpha", type=float, default=0.5, help="STFT loss weight used in training")
    parser.add_argument("--beta", type=float, default=1.0, help="Anchor loss weight used in training")
    parser.add_argument("--gamma", type=float, default=0.1, help="Smooth loss weight used in training")
    parser.add_argument("--exp_tag", type=str, default="Default", help="Tag used for this experiment")

    # case sampling policy (paper-ready)
    parser.add_argument("--cases_per_subject", type=int, default=3, help="1=best only, 2=best+worst, 3=best+median+worst")
    parser.add_argument("--beat_tol_ms", type=float, default=50.0, help="Beat matching tolerance (ms)")
    parser.add_argument("--mask_prom_frac", type=float, default=0.30, help="Prominence fraction for mask peak detection")
    args = parser.parse_args()

    # sync experiment name with train.py
    new_exp_name = f"Exp_a{args.alpha}_b{args.beta}_g{args.gamma}_{args.exp_tag}"
    Config.update_paths(new_exp_name)

    seed_everything(Config.SEED)
    device = Config.DEVICE

    ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    fs = int(getattr(Config, "FS", 200))

    print(f"🚀 Starting Test for Experiment: {Config.EXP_NAME}")
    print(f"   - Test H5: {Config.TEST_H5}")
    print(f"   - FS: {fs} Hz")
    print(f"   - Results: {Config.RESULT_DIR}")

    # QC for test
    bad_path = getattr(Config, "TEST_BAD_INDICES_PATH", None)
    if bad_path is not None and (not os.path.exists(bad_path)):
        print(f"[QC] TEST_BAD_INDICES_PATH not found, disable: {bad_path}")
        bad_path = None
    print(f"[QC] TEST_BAD_INDICES_PATH = {bad_path}")

    test_set = RadarDataset(Config.TEST_H5, bad_indices_path=bad_path)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # model
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[ERROR] Checkpoint not found!\n"
            f"  Expected: {ckpt_path}\n"
            f"  Please check EXP_NAME and CLI args consistency."
        )

    model = BeatAwareRM2Net(in_channels=Config.IN_CHANNELS, base_channels=Config.BASE_CHANNELS).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ============================================================
    # Pass-1: full inference -> export tables + per-sample indices
    # ============================================================
    clinical_records = []
    beat_records = []
    sample_index_records = []  # (sample_idx in this filtered dataset order)

    # For subject-wise case selection: store (sid -> list of (pcc, sample_i))
    # sample_i refers to "iteration index" in test_loader (0..len(test_set)-1)
    per_subject = {}

    print("🚀 Running full inference and collecting paper-ready metrics ...")

    with torch.no_grad():
        for sample_i, (radar, ecg, mask, subject_id) in enumerate(tqdm(test_loader, desc="Testing")):
            radar = radar.to(device)
            ecg = ecg.to(device)
            mask = mask.to(device)

            pred_ecg, pred_mask_logits = model(radar)

            # numpy for feature extraction / saving
            p_np = pred_ecg.detach().cpu().numpy().squeeze()
            t_np = ecg.detach().cpu().numpy().squeeze()
            pm_np = torch.sigmoid(pred_mask_logits).detach().cpu().numpy().squeeze()  # prob
            tm_np = mask.detach().cpu().numpy().squeeze()

            # waveform metrics
            wave_m = calculate_metrics(pred_ecg, ecg)
            pcc_val = float(wave_m.get("Pearson", np.nan))
            mae_val = float(wave_m.get("MAE", np.nan))
            rmse_val = float(wave_m.get("RMSE", np.nan))

            # clinical features
            p_feat = _safe_feat(p_np, fs=fs)
            t_feat = _safe_feat(t_np, fs=fs)

            sid = int(subject_id.item())

            record = {
                "Sample_Index": int(sample_i),
                "Subject_ID": sid,

                "PCC": pcc_val,
                "MAE": mae_val,
                "RMSE": rmse_val,

                "HR_True": _get(t_feat, "HR"),
                "HR_Pred": _get(p_feat, "HR"),
                "RR_True": _get(t_feat, "RR"),
                "RR_Pred": _get(p_feat, "RR"),
                "QRS_True": _get(t_feat, "QRS"),
                "QRS_Pred": _get(p_feat, "QRS"),
                "QT_True": _get(t_feat, "QT"),
                "QT_Pred": _get(p_feat, "QT"),
            }
            record["HR_Error"] = abs(record["HR_Pred"] - record["HR_True"]) if np.isfinite(record["HR_Pred"]) and np.isfinite(record["HR_True"]) else np.nan
            record["RR_Error"] = abs(record["RR_Pred"] - record["RR_True"]) if np.isfinite(record["RR_Pred"]) and np.isfinite(record["RR_True"]) else np.nan
            record["QRS_Error"] = abs(record["QRS_Pred"] - record["QRS_True"]) if np.isfinite(record["QRS_Pred"]) and np.isfinite(record["QRS_True"]) else np.nan
            record["QT_Error"] = abs(record["QT_Pred"] - record["QT_True"]) if np.isfinite(record["QT_Pred"]) and np.isfinite(record["QT_True"]) else np.nan
            clinical_records.append(record)

            # beat-level metrics from masks
            true_peaks = _mask_to_peaks(tm_np, fs=fs, prom_frac=args.mask_prom_frac)
            pred_peaks = _mask_to_peaks(pm_np, fs=fs, prom_frac=args.mask_prom_frac)
            bm = _beat_metrics_from_peaks(pred_peaks, true_peaks, fs=fs, tol_ms=args.beat_tol_ms)
            bm.update({"Sample_Index": int(sample_i), "Subject_ID": sid})
            beat_records.append(bm)

            # per-sample index registry
            sample_index_records.append({
                "Sample_Index": int(sample_i),
                "Subject_ID": sid,
                "PCC": pcc_val,
                "MAE": mae_val,
                "RMSE": rmse_val,
            })

            # case selection bookkeeping
            per_subject.setdefault(sid, []).append((pcc_val, int(sample_i)))

    # ============================================================
    # Export-1: main csv tables
    # ============================================================
    df_full = pd.DataFrame(clinical_records)
    csv_path = os.path.join(Config.RESULT_DIR, "test_comprehensive.csv")
    df_full.to_csv(csv_path, index=False)

    df_beat = pd.DataFrame(beat_records)
    beat_csv_path = os.path.join(Config.RESULT_DIR, "beat_metrics.csv")
    df_beat.to_csv(beat_csv_path, index=False)

    df_idx = pd.DataFrame(sample_index_records)
    idx_csv_path = os.path.join(Config.RESULT_DIR, "sample_index_pcc.csv")
    df_idx.to_csv(idx_csv_path, index=False)

    # subject-wise summary
    numeric_cols = [
        "PCC", "MAE", "RMSE",
        "HR_Error", "RR_Error", "QRS_Error", "QT_Error",
        "HR_True", "HR_Pred", "RR_True", "RR_Pred", "QRS_True", "QRS_Pred", "QT_True", "QT_Pred",
    ]
    subject_grp = df_full.groupby("Subject_ID", as_index=False)

    # mean/std + sample counts
    subj_mean = subject_grp[numeric_cols].mean(numeric_only=True)
    subj_std = subject_grp[numeric_cols].std(numeric_only=True).add_suffix("_std")
    subj_n = subject_grp.size().rename(columns={"size": "N_samples"})

    df_subject = subj_mean.merge(subj_std, left_on="Subject_ID", right_on="Subject_ID").merge(subj_n, on="Subject_ID")

    subject_csv_path = os.path.join(Config.RESULT_DIR, "subject_summary.csv")
    df_subject.to_csv(subject_csv_path, index=False)

    # global summary json: (sample-level) & (subject-level)
    summary = {
        "experiment": {
            "EXP_NAME": Config.EXP_NAME,
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "exp_tag": str(args.exp_tag),
            "fs": int(fs),
            "test_h5": str(Config.TEST_H5),
            "test_bad_indices_path": str(bad_path) if bad_path is not None else None,
            "num_test_samples_after_qc": int(len(df_full)),
        },
        "sample_level": {},
        "subject_level": {},
        "beat_level": {},
    }

    # sample-level stats
    for col in numeric_cols:
        if col in df_full.columns:
            summary["sample_level"][col] = _summary_stats(df_full[col])

    # subject-level stats: take subject-wise mean first then stats across subjects
    subj_means_only = df_full.groupby("Subject_ID")[numeric_cols].mean(numeric_only=True)
    for col in numeric_cols:
        if col in subj_means_only.columns:
            summary["subject_level"][col] = _summary_stats(subj_means_only[col])

    # beat-level stats
    beat_cols = ["Beat_Precision", "Beat_Recall", "Beat_F1", "Beat_MAE_ms", "Beat_MedianAE_ms"]
    for col in beat_cols:
        if col in df_beat.columns:
            summary["beat_level"][col] = _summary_stats(df_beat[col])

    summary_path = os.path.join(Config.RESULT_DIR, "global_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # console print (inter-subject Table-style)
    subj_mean_table = subj_means_only
    print("\n📊 Inter-subject (Subject-wise mean -> mean ± std across subjects)")
    for col in ["PCC", "MAE", "RMSE", "HR_Error", "QT_Error"]:
        if col in subj_mean_table.columns:
            mu = float(subj_mean_table[col].mean())
            sd = float(subj_mean_table[col].std())
            print(f"   {col}: {mu:.4f} ± {sd:.4f}")

    # ============================================================
    # Export-2: casebank (per-subject best/median/worst)
    # ============================================================
    # select indices per subject
    cases_per_subject = int(args.cases_per_subject)
    selected = []  # list of (sid, tag, sample_i)

    for sid, pairs in per_subject.items():
        # pairs: [(pcc, sample_i), ...]
        pairs_sorted = sorted(pairs, key=lambda x: (np.nan_to_num(x[0], nan=-1e9)))
        if len(pairs_sorted) == 0:
            continue
        worst = pairs_sorted[0][1]
        best = pairs_sorted[-1][1]

        if cases_per_subject <= 1:
            selected.append((sid, "best", best))
        elif cases_per_subject == 2:
            selected.append((sid, "best", best))
            selected.append((sid, "worst", worst))
        else:
            mid = pairs_sorted[len(pairs_sorted) // 2][1]
            selected.append((sid, "best", best))
            selected.append((sid, "median", mid))
            selected.append((sid, "worst", worst))

    # To avoid huge file: cap maximum cases
    # (e.g. 30 subjects * 3 = 90 cases -> very manageable)
    max_cases = 120
    if len(selected) > max_cases:
        selected = selected[:max_cases]

    # build a quick index->data access by re-iterating loader
    # We do a second pass but only store selected samples (efficient memory usage).
    selected_set = set((sid, tag, idx) for sid, tag, idx in selected)
    selected_by_idx = {idx: [] for _, _, idx in selected}  # idx -> [(sid, tag), ...]

    for sid, tag, idx in selected:
        selected_by_idx[idx].append((sid, tag))

    case_payload = {}
    print(f"\n🧪 Building casebank.npz for {len(selected)} cases ...")

    with torch.no_grad():
        for sample_i, (radar, ecg, mask, subject_id) in enumerate(tqdm(test_loader, desc="Casebank")):
            if sample_i not in selected_by_idx:
                continue

            radar = radar.to(device)
            ecg = ecg.to(device)
            mask = mask.to(device)

            pred_ecg, pred_mask_logits = model(radar)

            r_np = radar.detach().cpu().numpy().squeeze()
            t_np = ecg.detach().cpu().numpy().squeeze()
            p_np = pred_ecg.detach().cpu().numpy().squeeze()
            tm_np = mask.detach().cpu().numpy().squeeze()
            pm_np = torch.sigmoid(pred_mask_logits).detach().cpu().numpy().squeeze()

            sid = int(subject_id.item())

            # attach to all tags that reference this sample_i
            for (sid2, tag) in selected_by_idx[sample_i]:
                # safety: sid check (should match)
                key_prefix = f"sid{sid2}_{tag}_idx{sample_i}"
                case_payload[f"{key_prefix}_radar"] = r_np
                case_payload[f"{key_prefix}_ecg_true"] = t_np
                case_payload[f"{key_prefix}_ecg_pred"] = p_np
                case_payload[f"{key_prefix}_mask_true"] = tm_np
                case_payload[f"{key_prefix}_mask_pred"] = pm_np

    casebank_path = os.path.join(Config.RESULT_DIR, "casebank.npz")
    np.savez(casebank_path, **case_payload)

    print("\n✅ Test export finished (paper-ready).")
    print(f"   - test_comprehensive.csv  : {csv_path}")
    print(f"   - subject_summary.csv     : {subject_csv_path}")
    print(f"   - global_summary.json     : {summary_path}")
    print(f"   - beat_metrics.csv        : {beat_csv_path}")
    print(f"   - sample_index_pcc.csv    : {idx_csv_path}")
    print(f"   - casebank.npz            : {casebank_path}")


if __name__ == "__main__":
    test()
