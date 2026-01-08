# utils/metrics.py
import numpy as np
import neurokit2 as nk

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
        # [B, 1, L] or [B, C, L]
        x = x[:, 0, :]
    else:
        raise ValueError(f"Unsupported input ndim={x.ndim}, shape={x.shape}")

    return x.astype(np.float64, copy=False)

def calculate_metrics(pred, target, eps=1e-6):
    """
    Waveform-level similarity metrics.
    Returns batch-averaged MAE / RMSE / Pearson / AbsPearson.
    """
    pred = _to_2d_numpy(pred)
    target = _to_2d_numpy(target)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred{pred.shape} vs target{target.shape}")

    batch_mae, batch_rmse, batch_pcc = [], [], []

    for p, t in zip(pred, target):
        # MAE / RMSE
        batch_mae.append(np.mean(np.abs(p - t)))
        batch_rmse.append(np.sqrt(np.mean((p - t) ** 2)))

        # Pearson (robust)
        sp = np.std(p)
        st = np.std(t)
        if sp < eps or st < eps:
            batch_pcc.append(0.0)
        else:
            batch_pcc.append(float(np.corrcoef(p, t)[0, 1]))

    pcc_mean = float(np.mean(batch_pcc))
    return {
        "MAE": float(np.mean(batch_mae)),
        "RMSE": float(np.mean(batch_rmse)),
        "Pearson": pcc_mean,
        "AbsPearson": float(np.mean(np.abs(batch_pcc))),
    }

def extract_clinical_features_nk(signal, fs=200):
    """
    Extract clinical features using NeuroKit2.
    Output: HR (bpm), RR (ms), QRS (ms), QT (ms).
    """
    x = np.asarray(signal, dtype=np.float64).copy()
    x = np.nan_to_num(x)

    out = {"HR": np.nan, "RR": np.nan, "QRS": np.nan, "QT": np.nan}

    try:
        _, rpeaks = nk.ecg_peaks(x, sampling_rate=fs)
        r_idx = np.asarray(rpeaks.get("ECG_R_Peaks", []), dtype=int)

        if r_idx.size < 2:
            return out

        rr_ms = np.diff(r_idx) * (1000.0 / fs)
        rr_mean = float(np.nanmean(rr_ms))
        out["RR"] = rr_mean
        out["HR"] = float(60000.0 / rr_mean) if rr_mean > 1e-6 else np.nan

    except Exception:
        # HR/RR失败就直接返回（因为后面 delineate 也没有意义）
        return out

    # QRS/QT：允许失败，不影响 HR/RR
    try:
        _, waves = nk.ecg_delineate(x, rpeaks, sampling_rate=fs, method="peak")

        def _safe_interval(a, b):
            if (a not in waves) or (b not in waves):
                return np.nan
            va = np.asarray(waves[a], dtype=np.float64)
            vb = np.asarray(waves[b], dtype=np.float64)
            dt = vb - va
            dt_ms = dt * (1000.0 / fs)
            v = float(np.nanmean(dt_ms))
            return v

        out["QRS"] = _safe_interval("ECG_Q_Onsets", "ECG_S_Offsets")
        out["QT"]  = _safe_interval("ECG_Q_Onsets", "ECG_T_Offsets")

    except Exception:
        pass

    return out