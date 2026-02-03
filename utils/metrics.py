# utils/metrics.py
import numpy as np
import neurokit2 as nk


# -----------------------------
# Basic helpers
# -----------------------------
def _to_2d_numpy(x):
    """Convert torch tensor or numpy to numpy array with shape [B, L]."""
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    # Accept [L], [B,L], [B,1,L], [B,C,L] (take first channel by default)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 2:
        pass
    elif x.ndim == 3:
        x = x[:, 0, :]
    else:
        raise ValueError(f"Unsupported input ndim={x.ndim}, shape={x.shape}")

    return x.astype(np.float64, copy=False)


def _nan_safe(x):
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _pearsonr_np(a, b, eps=1e-8):
    """Pearson correlation for 1D numpy arrays, robust to near-constant signals."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa = np.std(a)
    sb = np.std(b)
    if sa < eps or sb < eps:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _median_iqr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": np.nan, "p25": np.nan, "p75": np.nan, "iqr": np.nan, "n": 0}
    p25 = float(np.percentile(x, 25))
    p75 = float(np.percentile(x, 75))
    return {
        "median": float(np.median(x)),
        "p25": p25,
        "p75": p75,
        "iqr": float(p75 - p25),
        "n": int(x.size),
    }


# -----------------------------
# Segment-level metrics (paper protocol)
# -----------------------------
def calculate_metrics(pred, target, eps=1e-6):
    """
    Waveform-level similarity metrics.
    Returns batch-averaged MAE / RMSE / Pearson / AbsPearson.
    (Kept for backward compatibility / internal monitoring)
    """
    pred = _to_2d_numpy(pred)
    target = _to_2d_numpy(target)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred{pred.shape} vs target{target.shape}")

    batch_mae, batch_rmse, batch_pcc = [], [], []

    for p, t in zip(pred, target):
        batch_mae.append(np.mean(np.abs(p - t)))
        batch_rmse.append(np.sqrt(np.mean((p - t) ** 2)))
        batch_pcc.append(_pearsonr_np(p, t, eps=eps))

    pcc_mean = float(np.mean(batch_pcc))
    return {
        "MAE": float(np.mean(batch_mae)),
        "RMSE": float(np.mean(batch_rmse)),
        "Pearson": pcc_mean,
        "AbsPearson": float(np.mean(np.abs(batch_pcc))),
    }


def compute_segment_metrics(pred_1d, gt_1d, eps=1e-8):
    """
    Paper-aligned segment-level metrics:
      - PCC: Pearson correlation coefficient (segment/window-level)
      - MRE: Mean Relative Error (segment/window-level), robust formulation

    Returns dict with: pcc, mre, mae, rmse (mae/rmse optional but useful for debug)
    """
    p = _nan_safe(pred_1d)
    g = _nan_safe(gt_1d)

    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: pred{p.shape} vs gt{g.shape}")

    # PCC
    pcc = _pearsonr_np(p, g, eps=eps)

    # MAE/RMSE (debug-friendly)
    diff = p - g
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    # MRE (robust): mean( |p-g| / (|g| + eps) )
    # This is stable for near-zero gt values (common after normalization).
    mre = float(np.mean(np.abs(diff) / (np.abs(g) + eps)))

    return {"pcc": float(pcc), "mre": mre, "mae": mae, "rmse": rmse}


# -----------------------------
# Beat-level metrics (RR/QRS/QT) via NeuroKit2
# -----------------------------
def _nk_rpeaks(x, fs):
    """Return rpeaks indices, raise on failure."""
    # nk.ecg_peaks returns (signals, info)
    _, info = nk.ecg_peaks(x, sampling_rate=fs)
    r = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    return r


def _nk_delineate(x, rpeaks_info, fs, method="peak"):
    """
    Return delineation waves dict from nk.ecg_delineate.
    rpeaks_info should be the "info" dict returned by nk.ecg_peaks (or similar structure).
    """
    # nk.ecg_delineate expects rpeaks dict-like (the same "info" returned by ecg_peaks)
    _, waves = nk.ecg_delineate(x, rpeaks_info, sampling_rate=fs, method=method)
    return waves


def _as_int_array(x):
    """Convert list/array to int array with NaNs handled."""
    if x is None:
        return np.array([], dtype=int)
    a = np.asarray(x)
    # neurokit may return float with nan
    a = a[np.isfinite(a)]
    return a.astype(int, copy=False)


def _match_by_nearest(gt_idx, pr_idx, max_dist):
    """
    Nearest-neighbor matching by sample index.
    Returns list of tuples: (gt_i, pr_i)
    """
    gt_idx = np.asarray(gt_idx, dtype=int)
    pr_idx = np.asarray(pr_idx, dtype=int)
    if gt_idx.size == 0 or pr_idx.size == 0:
        return []

    pairs = []
    # For each gt, find nearest pred
    for g in gt_idx:
        j = int(np.argmin(np.abs(pr_idx - g)))
        p = int(pr_idx[j])
        if abs(p - g) <= max_dist:
            pairs.append((int(g), int(p)))
    return pairs


def compute_beat_metrics_nk(
    pred_1d,
    gt_1d,
    fs=200,
    method="peak",
    max_rpeak_match_ms=150,
    drop_on_delineation_fail=True,
):
    """
    Compute beat-level errors for RR/QRS/QT using NeuroKit2 delineation.

    Returns:
      beat_rows: List[dict] each row corresponds to one matched beat (or RR interval),
                 fields include rr_gt_ms/rr_pred_ms/rr_err_ms, qrs_*, qt_*, valid_flag, drop_reason, etc.
      meta: dict with counts (total beats, valid beats, drop rate, reasons)

    Notes (protocol-aligned):
      - RR/QRS/QT evaluation unit: all detected heartbeats (beat-level)
      - Allow discarding undetected/failed beats, but MUST report counts/drop_rate
    """
    p = _nan_safe(pred_1d)
    g = _nan_safe(gt_1d)

    max_dist = int(round((max_rpeak_match_ms / 1000.0) * fs))

    beat_rows = []
    drop_reasons = {}

    def _count_reason(reason):
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    # --- GT peaks ---
    try:
        _, gt_info = nk.ecg_peaks(g, sampling_rate=fs)
        r_gt = _as_int_array(gt_info.get("ECG_R_Peaks", []))
    except Exception:
        # Cannot evaluate beat metrics without GT R-peaks
        return [], {
            "n_beats_total": 0,
            "n_beats_valid": 0,
            "drop_rate": 1.0,
            "drop_reasons": {"gt_rpeaks_fail": 1},
        }

    # --- Pred peaks ---
    try:
        _, pr_info = nk.ecg_peaks(p, sampling_rate=fs)
        r_pr = _as_int_array(pr_info.get("ECG_R_Peaks", []))
    except Exception:
        r_pr = np.array([], dtype=int)

    # Match beats by nearest R-peak
    pairs = _match_by_nearest(r_gt, r_pr, max_dist=max_dist)

    # If too few matches, still return with statistics (test.py can aggregate)
    if len(pairs) < 2:
        # still can report drop
        return [], {
            "n_beats_total": int(max(r_gt.size - 1, 0)),
            "n_beats_valid": 0,
            "drop_rate": 1.0,
            "drop_reasons": {"rpeak_match_insufficient": 1},
        }

    # --- Delineation (optional but needed for QRS/QT) ---
    waves_gt = None
    waves_pr = None

    # neurokit delineate needs the info dict
    try:
        waves_gt = _nk_delineate(g, gt_info, fs, method=method)
    except Exception:
        waves_gt = None

    try:
        waves_pr = _nk_delineate(p, pr_info, fs, method=method) if r_pr.size > 0 else None
    except Exception:
        waves_pr = None

    # Helper to get per-beat fiducials near a given R
    def _nearest_fid(fid_arr, r0, max_ms=200):
        """Return nearest fid index to r0 within max_ms, else None."""
        if fid_arr is None:
            return None
        fid_arr = _as_int_array(fid_arr)
        if fid_arr.size == 0:
            return None
        j = int(np.argmin(np.abs(fid_arr - r0)))
        cand = int(fid_arr[j])
        if abs(cand - r0) <= int(round((max_ms / 1000.0) * fs)):
            return cand
        return None

    # Pre-fetch fiducial arrays
    q_on_gt = waves_gt.get("ECG_Q_Onsets", None) if isinstance(waves_gt, dict) else None
    s_off_gt = waves_gt.get("ECG_S_Offsets", None) if isinstance(waves_gt, dict) else None
    t_off_gt = waves_gt.get("ECG_T_Offsets", None) if isinstance(waves_gt, dict) else None

    q_on_pr = waves_pr.get("ECG_Q_Onsets", None) if isinstance(waves_pr, dict) else None
    s_off_pr = waves_pr.get("ECG_S_Offsets", None) if isinstance(waves_pr, dict) else None
    t_off_pr = waves_pr.get("ECG_T_Offsets", None) if isinstance(waves_pr, dict) else None

    # RR intervals: use consecutive matched beats in time order
    pairs = sorted(pairs, key=lambda x: x[0])  # sort by gt R
    n_total = len(pairs) - 1

    for k in range(n_total):
        r0_gt, r0_pr = pairs[k]
        r1_gt, r1_pr = pairs[k + 1]

        row = {
            "beat_index": k,
            "r_gt": int(r0_gt),
            "r_pred": int(r0_pr),
            "valid_flag": 1,
            "drop_reason": "",
        }

        # RR
        rr_gt_ms = (r1_gt - r0_gt) * (1000.0 / fs)
        rr_pr_ms = (r1_pr - r0_pr) * (1000.0 / fs)
        row["rr_gt_ms"] = float(rr_gt_ms)
        row["rr_pred_ms"] = float(rr_pr_ms)
        row["rr_err_ms"] = float(abs(rr_pr_ms - rr_gt_ms))

        # QRS/QT (may fail)
        # Define QRS duration = (S_offset - Q_onset), QT = (T_offset - Q_onset)
        # Use nearest fiducials around current beat r0
        q0_gt = _nearest_fid(q_on_gt, r0_gt)
        s0_gt = _nearest_fid(s_off_gt, r0_gt)
        t0_gt = _nearest_fid(t_off_gt, r0_gt)

        q0_pr = _nearest_fid(q_on_pr, r0_pr) if waves_pr is not None else None
        s0_pr = _nearest_fid(s_off_pr, r0_pr) if waves_pr is not None else None
        t0_pr = _nearest_fid(t_off_pr, r0_pr) if waves_pr is not None else None

        # Compute GT durations if available
        if q0_gt is not None and s0_gt is not None and s0_gt > q0_gt:
            qrs_gt_ms = (s0_gt - q0_gt) * (1000.0 / fs)
        else:
            qrs_gt_ms = np.nan

        if q0_gt is not None and t0_gt is not None and t0_gt > q0_gt:
            qt_gt_ms = (t0_gt - q0_gt) * (1000.0 / fs)
        else:
            qt_gt_ms = np.nan

        # Compute Pred durations if available
        if q0_pr is not None and s0_pr is not None and s0_pr > q0_pr:
            qrs_pr_ms = (s0_pr - q0_pr) * (1000.0 / fs)
        else:
            qrs_pr_ms = np.nan

        if q0_pr is not None and t0_pr is not None and t0_pr > q0_pr:
            qt_pr_ms = (t0_pr - q0_pr) * (1000.0 / fs)
        else:
            qt_pr_ms = np.nan

        row["qrs_gt_ms"] = float(qrs_gt_ms) if np.isfinite(qrs_gt_ms) else np.nan
        row["qrs_pred_ms"] = float(qrs_pr_ms) if np.isfinite(qrs_pr_ms) else np.nan
        row["qt_gt_ms"] = float(qt_gt_ms) if np.isfinite(qt_gt_ms) else np.nan
        row["qt_pred_ms"] = float(qt_pr_ms) if np.isfinite(qt_pr_ms) else np.nan

        # Errors (absolute)
        row["qrs_err_ms"] = float(abs(qrs_pr_ms - qrs_gt_ms)) if (np.isfinite(qrs_pr_ms) and np.isfinite(qrs_gt_ms)) else np.nan
        row["qt_err_ms"] = float(abs(qt_pr_ms - qt_gt_ms)) if (np.isfinite(qt_pr_ms) and np.isfinite(qt_gt_ms)) else np.nan

        # If delineation fails severely, optionally drop beat
        if drop_on_delineation_fail:
            # Keep RR even if QRS/QT missing, but you can enforce stricter policy if needed.
            # Here we only drop if RR is invalid (should not happen) or no match.
            pass

        beat_rows.append(row)

    # Summarize validity (RR always present for matched intervals)
    n_valid = len(beat_rows)
    drop_rate = 1.0 - (n_valid / max(n_total, 1))

    meta = {
        "n_beats_total": int(n_total),
        "n_beats_valid": int(n_valid),
        "drop_rate": float(drop_rate),
        "drop_reasons": drop_reasons,
        "rpeak_match_ms": float(max_rpeak_match_ms),
        "delineate_method": str(method),
    }
    return beat_rows, meta


# -----------------------------
# Global summary (median-first protocol)
# -----------------------------
def summarize_metrics(segment_rows, beat_rows):
    """
    Build median-first summary for Table III / global_summary.json.

    segment_rows: list of dict, each with keys at least ['pcc','mre']
    beat_rows: list of dict, each with keys at least ['rr_err_ms', 'qrs_err_ms', 'qt_err_ms']

    Returns dict with segment_level, beat_level, counts.
    """
    pcc = [r.get("pcc", np.nan) for r in segment_rows]
    mre = [r.get("mre", np.nan) for r in segment_rows]

    rr_err = [r.get("rr_err_ms", np.nan) for r in beat_rows]
    qrs_err = [r.get("qrs_err_ms", np.nan) for r in beat_rows]
    qt_err = [r.get("qt_err_ms", np.nan) for r in beat_rows]

    out = {
        "segment_level": {
            "PCC": _median_iqr(pcc),
            "MRE": _median_iqr(mre),
        },
        "beat_level": {
            "RR_err_ms": _median_iqr(rr_err),
            "QRS_err_ms": _median_iqr(qrs_err),
            "QT_err_ms": _median_iqr(qt_err),
        },
        "counts": {
            "n_segments": int(len(segment_rows)),
            "n_beats": int(len(beat_rows)),
        },
    }
    return out
