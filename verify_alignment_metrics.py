import os
import h5py
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================== 配置区 ==================
H5_PATH = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5"
FS = 200

# 你生成该 H5 时的滑窗参数（用于 effective unique windows 折算）
WINDOW_SECONDS = 8.0
STRIDE_SECONDS = 1.0   # ← 改成你真实用的（例如 0.5 / 1.0 / 1.6）

# 1) 残余 lag 检查阈值：±5 点（25 ms）
LAG_THRESHOLD = 5

# 2) 心跳频段（与 build_dataset 逻辑一致）
HEART_BAND = (0.8, 3.0)

# 3) mask-ECG 命中容忍：±50ms（经验上 40~80ms 合理）
HIT_TOL_MS = 50
HIT_TOL = int(round(HIT_TOL_MS * FS / 1000))

# 4) ECG 峰值检测参数（用于命中率/误差计算）
ECG_MIN_PEAK_DIST_SEC = 0.25
ECG_MIN_PEAK_DIST = int(round(ECG_MIN_PEAK_DIST_SEC * FS))
ECG_PROM_FRAC = 0.15

# 5) 输出目录
OUT_DIR = "dataset_verification_reports"
# ============================================


def _butter_bandpass(band, fs, order=4):
    nyq = 0.5 * fs
    low = band[0] / nyq
    high = band[1] / nyq
    return signal.butter(order, [low, high], btype="band")


def _safe_filtfilt(b, a, x):
    x = np.nan_to_num(x)
    if len(x) < 3 * max(len(a), len(b)):
        return x
    try:
        return signal.filtfilt(b, a, x)
    except Exception:
        return x


def _extract_anchor_centers_from_mask(mask_1d):
    m = (mask_1d > 0.5).astype(np.int32)
    if m.sum() == 0:
        return np.array([], dtype=np.int32)

    diff = np.diff(np.pad(m, (1, 1), constant_values=0))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    centers = ((starts + ends) / 2.0).round().astype(np.int32)
    centers = centers[(centers >= 0) & (centers < len(mask_1d))]
    return centers


def _detect_ecg_peaks(ecg_1d):
    e = np.nan_to_num(ecg_1d)
    prom = max(1e-6, ECG_PROM_FRAC * (np.max(e) - np.min(e)))
    peaks, _ = signal.find_peaks(
        e,
        distance=ECG_MIN_PEAK_DIST,
        prominence=prom
    )
    return peaks.astype(np.int32)


def verify_dataset(path):
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    with h5py.File(path, 'r') as f:
        radar_data = f['radar']          # [N,1,L]
        ecg_data = f['ecg']              # [N,1,L]
        mask_data = f['mask']            # [N,1,L]
        sid_data = f['subject_id'] if 'subject_id' in f else None

        N = radar_data.shape[0]
        L = radar_data.shape[-1]

        print(f"🚀 开始验证: {path}")
        print(f"   Samples N = {N}, Length L = {L}, FS = {FS} Hz")
        print(f"   WINDOW_SECONDS={WINDOW_SECONDS}, STRIDE_SECONDS={STRIDE_SECONDS}  (for effective-count)")

        # ============ L0 ============
        print("\n[L0] 基础合法性检查")
        assert radar_data.shape == ecg_data.shape == mask_data.shape, \
            f"Shape mismatch: radar{radar_data.shape}, ecg{ecg_data.shape}, mask{mask_data.shape}"
        assert radar_data.shape[1] == 1, "Expected channel dim = 1"
        assert L > 0 and N > 0, "Empty dataset"

        sample_idx = np.random.choice(N, size=min(200, N), replace=False)
        nan_inf_flag = False
        for i in sample_idx:
            r = radar_data[i, 0]
            e = ecg_data[i, 0]
            m = mask_data[i, 0]
            if (not np.isfinite(r).all()) or (not np.isfinite(e).all()) or (not np.isfinite(m).all()):
                nan_inf_flag = True
                break
        print(f"   NaN/Inf 抽样检查: {'❌发现异常' if nan_inf_flag else '✅未发现异常'}")

        # ============ 主计算 ============
        print("\n[L1/L2] 计算 residual lag / PCC / mask-ECG consistency / subject stats ...")
        b_h, a_h = _butter_bandpass(HEART_BAND, FS, order=4)

        residual_lags = np.zeros(N, dtype=np.int32)
        pcc_raw = np.zeros(N, dtype=np.float32)
        pcc_heart = np.zeros(N, dtype=np.float32)

        total_anchors = 0
        matched_anchors = 0
        total_ecg_peaks = 0
        matched_ecg_peaks = 0

        # 额外：记录匹配到的 Δt（ECG peak - anchor center）
        dt_errors = []

        failed_lag = []

        for i in tqdm(range(N), desc="Scanning"):
            r = np.asarray(radar_data[i, 0], dtype=np.float64)
            e = np.asarray(ecg_data[i, 0], dtype=np.float64)
            m = np.asarray(mask_data[i, 0], dtype=np.float64)

            r = np.nan_to_num(r)
            e = np.nan_to_num(e)
            m = np.nan_to_num(m)

            # ----- residual lag (heart-band xcorr) -----
            r_f = _safe_filtfilt(b_h, a_h, r)
            e_f = _safe_filtfilt(b_h, a_h, e)

            corr = signal.correlate(e_f - e_f.mean(), r_f - r_f.mean(), mode='full')
            lag_axis = signal.correlation_lags(len(e_f), len(r_f), mode='full')
            best_lag = int(lag_axis[np.argmax(np.abs(corr))])
            residual_lags[i] = best_lag
            if abs(best_lag) > LAG_THRESHOLD:
                failed_lag.append((i, best_lag))

            # ----- PCC (raw) -----
            try:
                pcc_raw[i] = float(pearsonr(r, e)[0])
            except Exception:
                pcc_raw[i] = np.nan

            # ----- PCC (heart-band) : 更能反映“对齐后同相” -----
            try:
                pcc_heart[i] = float(pearsonr(r_f, e_f)[0])
            except Exception:
                pcc_heart[i] = np.nan

            # ----- mask-ECG hit & dt error -----
            anchor_centers = _extract_anchor_centers_from_mask(m)
            ecg_peaks = _detect_ecg_peaks(e)

            total_anchors += len(anchor_centers)
            total_ecg_peaks += len(ecg_peaks)

            if len(anchor_centers) == 0 or len(ecg_peaks) == 0:
                continue

            # anchor -> ECG peak (precision)
            for c in anchor_centers:
                diffs = ecg_peaks - c
                j = np.argmin(np.abs(diffs))
                if abs(diffs[j]) <= HIT_TOL:
                    matched_anchors += 1
                    dt_errors.append(int(diffs[j]))  # 记录误差（samples）

            # ECG peak -> anchor (recall)
            for p in ecg_peaks:
                if np.any(np.abs(anchor_centers - p) <= HIT_TOL):
                    matched_ecg_peaks += 1

        # ============ 报告 ============
        print("\n" + "=" * 72)
        print("📊 Dataset Verification Report (Final)")
        print("=" * 72)

        # [1] lag
        ok_lag = N - len(failed_lag)
        print("\n[1] Residual Lag Check (heart-band xcorr)")
        print(f"   ✅ 合格样本数: {ok_lag} / {N}")
        print(f"   📈 合格率: {ok_lag / N * 100:.2f}%  (threshold=±{LAG_THRESHOLD} samples = ±{LAG_THRESHOLD/FS*1000:.1f} ms)")
        print(f"   🧾 mean lag: {residual_lags.mean():.2f} samples")
        print(f"   🧾 std  lag: {residual_lags.std():.2f} samples")
        if failed_lag:
            print(f"   ❌ 前 10 个超标样本(index, lag): {failed_lag[:10]}")

        plt.figure()
        plt.hist(residual_lags, bins=41)
        plt.xlabel("Residual lag (samples)")
        plt.ylabel("Count")
        plt.title("Residual Lag Histogram (heart-band xcorr)")
        lag_fig = os.path.join(OUT_DIR, "residual_lag_hist.png")
        plt.savefig(lag_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   🖼️ residual lag hist: {lag_fig}")

        # [2] PCC raw
        print("\n[2] PCC Distribution (raw segment)")
        pcc_raw_c = pcc_raw[np.isfinite(pcc_raw)]
        print(f"   mean |PCC_raw|: {np.mean(np.abs(pcc_raw_c)):.3f}")
        print(f"   median |PCC_raw|: {np.median(np.abs(pcc_raw_c)):.3f}")
        print(f"   5% / 95% quantile: {np.quantile(np.abs(pcc_raw_c), 0.05):.3f} / {np.quantile(np.abs(pcc_raw_c), 0.95):.3f}")

        plt.figure()
        plt.hist(np.abs(pcc_raw_c), bins=40)
        plt.xlabel("|PCC_raw|")
        plt.ylabel("Count")
        plt.title("PCC Histogram (abs, raw segments)")
        pcc_fig = os.path.join(OUT_DIR, "pcc_hist_raw.png")
        plt.savefig(pcc_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   🖼️ PCC_raw hist: {pcc_fig}")

        # [3] PCC heart-band
        print("\n[3] PCC Distribution (heart-band segment)")
        pcc_h_c = pcc_heart[np.isfinite(pcc_heart)]
        print(f"   mean |PCC_heart|: {np.mean(np.abs(pcc_h_c)):.3f}")
        print(f"   median |PCC_heart|: {np.median(np.abs(pcc_h_c)):.3f}")
        print(f"   5% / 95% quantile: {np.quantile(np.abs(pcc_h_c), 0.05):.3f} / {np.quantile(np.abs(pcc_h_c), 0.95):.3f}")

        plt.figure()
        plt.hist(np.abs(pcc_h_c), bins=40)
        plt.xlabel("|PCC_heart|")
        plt.ylabel("Count")
        plt.title("PCC Histogram (abs, heart-band)")
        pcc_h_fig = os.path.join(OUT_DIR, "pcc_hist_heart.png")
        plt.savefig(pcc_h_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   🖼️ PCC_heart hist: {pcc_h_fig}")

        # [4] mask hit + dt error
        print("\n[4] mask–ECG Consistency (beat-level)")
        print(f"   HIT tolerance: ±{HIT_TOL} samples (±{HIT_TOL/FS*1000:.1f} ms)")
        print(f"   Total anchors (from mask): {total_anchors}")
        print(f"   Total ECG peaks (detected): {total_ecg_peaks}")

        precision = (matched_anchors / total_anchors) if total_anchors > 0 else np.nan
        recall = (matched_ecg_peaks / total_ecg_peaks) if total_ecg_peaks > 0 else np.nan
        f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan

        print(f"   ✅ Anchor Precision (mask -> ECG peak): {precision:.3f}")
        print(f"   ✅ Peak Recall     (ECG peak -> mask): {recall:.3f}")
        print(f"   ✅ F1 score: {f1:.3f}")

        if len(dt_errors) > 0:
            dt_errors = np.array(dt_errors, dtype=np.int32)
            print(f"   Δt (peak - anchor) mean: {dt_errors.mean():.2f} samples ({dt_errors.mean()/FS*1000:.1f} ms)")
            print(f"   Δt (peak - anchor) std : {dt_errors.std():.2f} samples ({dt_errors.std()/FS*1000:.1f} ms)")
            print(f"   Δt 5%/95%: {np.quantile(dt_errors,0.05):.1f} / {np.quantile(dt_errors,0.95):.1f} samples")

            plt.figure()
            plt.hist(dt_errors, bins=61)
            plt.xlabel("Δt = ECG_peak - anchor_center (samples)")
            plt.ylabel("Count")
            plt.title("Beat-level Temporal Error Histogram (Δt)")
            dt_fig = os.path.join(OUT_DIR, "mask_peak_dt_hist.png")
            plt.savefig(dt_fig, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"   🖼️ Δt hist: {dt_fig}")
        else:
            print("   ⚠️ 未记录到 Δt（可能峰检测过严或 mask 太稀疏）")

        # [5] subject-wise + effective unique windows
        print("\n[5] Subject-wise Segment Statistics (+ Effective unique windows)")
        if sid_data is None:
            print("   ⚠️ H5 中未发现 subject_id，跳过按 subject 统计。")
        else:
            sids = np.asarray(sid_data[:], dtype=np.int32)
            unique_sids, counts = np.unique(sids, return_counts=True)
            order = np.argsort(-counts)

            print(f"   Subjects: {len(unique_sids)}")
            print(f"   Segments per subject: mean={counts.mean():.1f}, median={np.median(counts):.1f}, min={counts.min()}, max={counts.max()}")
            print(f"   Top-5 subjects (sid:count): {[ (int(unique_sids[i]), int(counts[i])) for i in order[:5] ]}")
            print(f"   Bottom-5 subjects (sid:count): {[ (int(unique_sids[i]), int(counts[i])) for i in order[-5:] ]}")

            # effective unique windows (overlap-adjusted)
            overlap_factor = float(STRIDE_SECONDS) / float(WINDOW_SECONDS)
            eff_counts = counts.astype(np.float64) * overlap_factor
            print(f"   Effective factor (stride/window): {STRIDE_SECONDS:.3f}/{WINDOW_SECONDS:.3f} = {overlap_factor:.4f}")
            print(f"   Effective segments per subject: mean={eff_counts.mean():.1f}, median={np.median(eff_counts):.1f}, "
                  f"min={eff_counts.min():.1f}, max={eff_counts.max():.1f}")
            print(f"   Total effective segments (sum): {eff_counts.sum():.1f}")

            # 保存为 txt（增加 eff_count）
            stat_path = os.path.join(OUT_DIR, "subject_segment_counts_effective.txt")
            with open(stat_path, "w", encoding="utf-8") as wf:
                wf.write("sid\tcount\teff_count\n")
                for sid, c, ec in sorted(zip(unique_sids, counts, eff_counts), key=lambda x: x[0]):
                    wf.write(f"{int(sid)}\t{int(c)}\t{ec:.2f}\n")
            print(f"   📝 已保存 subject-wise 统计(含eff): {stat_path}")

    print("\n✅ 验证完成。建议把 OUT_DIR 下的图与统计文件作为实验日志长期保留。")


if __name__ == "__main__":
    verify_dataset(H5_PATH)
