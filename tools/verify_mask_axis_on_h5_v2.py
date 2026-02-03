import h5py
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

# ========== 你需要改的两项 ==========
H5_PATH = "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent/train.h5"
FS = 200  # 改成 Config.FS_TARGET
# ===================================

# 判定阈值
TAU_MS = 10
TAU = int(FS * TAU_MS / 1000)

MAX_MATCH_MS = 150
MAX_MATCH = int(FS * MAX_MATCH_MS / 1000)

# radar 心跳频段（与你 align_signals_robust / micro-align 一致）
F_LOW, F_HIGH = 0.8, 3.0

def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)

def nearest_match(a, b, max_dist):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return []
    pairs = []
    for x in a:
        j = np.argmin(np.abs(b - x))
        if abs(b[j] - x) <= max_dist:
            pairs.append((x, b[j]))
    return pairs

def summarize_delta(delta):
    delta = np.asarray(delta)
    return {
        "n": int(len(delta)),
        "median_samples": float(np.median(delta)),
        "iqr_samples": float(np.percentile(delta, 75) - np.percentile(delta, 25)),
        "median_ms": float(np.median(delta) * 1000 / FS),
        "iqr_ms": float((np.percentile(delta, 75) - np.percentile(delta, 25)) * 1000 / FS),
        "median_abs_ms": float(np.median(np.abs(delta)) * 1000 / FS),
    }

with h5py.File(H5_PATH, "r") as f:
    ecg = f["ecg"][:]     # (N,1,T)
    mask = f["mask"][:]   # (N,1,T)
    radar = f["radar"][:] # (N,1,T)
    sid = f["subject_id"][:] if "subject_id" in f else None

N = ecg.shape[0]
T = ecg.shape[-1]
print("Loaded:", H5_PATH)
print("N samples:", N, "T:", T, "FS:", FS)

np.random.seed(0)
idxs = np.random.choice(N, size=min(150, N), replace=False)

# -------- A) mask vs ECG --------
all_delta_A = []
hit_A = tot_A = kept_A = 0

# -------- B) mask vs radar_heart --------
all_delta_B = []
hit_B = tot_B = kept_B = 0

for idx in idxs:
    e = ecg[idx, 0]
    m = mask[idx, 0]
    r = radar[idx, 0]

    # ECG peaks（用粗峰检；若你希望更严谨，可替换为你 ecg_dsp 的 detector）
    p_ecg, _ = find_peaks(e, distance=int(0.25 * FS), prominence=0.05)
    p_msk, _ = find_peaks(m, distance=int(0.25 * FS), prominence=0.05)

    pairs_A = nearest_match(p_ecg, p_msk, max_dist=MAX_MATCH)
    if len(pairs_A) >= 3:
        kept_A += 1
        delta_A = np.array([b - a for a, b in pairs_A], dtype=np.int32)
        all_delta_A.append(delta_A)
        hit_A += np.sum(np.abs(delta_A) <= TAU)
        tot_A += len(delta_A)

    # radar_heart peaks
    try:
        r_heart = bandpass(r, FS, F_LOW, F_HIGH, order=4)
    except Exception:
        continue

    # 为了减少“波形振铃/符号”影响，取绝对值包络的峰更稳健（不想用 abs 也可以）
    r_feat = np.abs(r_heart)

    p_rad, _ = find_peaks(r_feat, distance=int(0.25 * FS), prominence=np.std(r_feat) * 0.2 + 1e-6)
    pairs_B = nearest_match(p_msk, p_rad, max_dist=MAX_MATCH)  # 注意：以 mask 为 anchor
    if len(pairs_B) >= 3:
        kept_B += 1
        delta_B = np.array([b - a for a, b in pairs_B], dtype=np.int32)  # Δ = peak(radar) - peak(mask)
        all_delta_B.append(delta_B)
        hit_B += np.sum(np.abs(delta_B) <= TAU)
        tot_B += len(delta_B)

print("\n===== A) MASK vs ECG =====")
print("Checked:", len(idxs), "Kept:", kept_A, "Matched beats:", tot_A)
if tot_A > 0:
    print(f"Hit-rate within {TAU_MS}ms:", hit_A / tot_A)
    all_delta_A = np.concatenate(all_delta_A)
    print("Delta stats:", summarize_delta(all_delta_A))
else:
    print("[WARN] No matched beats in A.")

print("\n===== B) MASK vs RADAR_HEART =====")
print("Checked:", len(idxs), "Kept:", kept_B, "Matched beats:", tot_B)
if tot_B > 0:
    print(f"Hit-rate within {TAU_MS}ms:", hit_B / tot_B)
    all_delta_B = np.concatenate(all_delta_B)
    stats_B = summarize_delta(all_delta_B)
    print("Delta stats:", stats_B)

    # 判定建议
    if stats_B["median_abs_ms"] <= 20 and stats_B["iqr_ms"] <= 40:
        print("[OK] radar–mask beat alignment looks acceptable (<=~20ms median abs).")
    else:
        print("[FLAG] radar–mask beat alignment may be inconsistent; consider syncing ecg/mask with local_lag or revising micro-alignment.")
else:
    print("[WARN] No matched beats in B (possible prominence/FS issue).")

# --- 可视化：各画一张 Δ 分布 ---
if tot_A > 0:
    plt.figure()
    plt.hist(all_delta_A, bins=60)
    plt.title("A) Delta = peak(mask) - peak(ECG)  (samples)")
    plt.xlabel("Δ (samples)")
    plt.ylabel("count")
    plt.show()

if tot_B > 0:
    plt.figure()
    plt.hist(all_delta_B, bins=60)
    plt.title("B) Delta = peak(radar_heart) - peak(mask)  (samples)")
    plt.xlabel("Δ (samples)")
    plt.ylabel("count")
    plt.show()

# --- 叠加图：随机挑一个样本直观看 ---
pick = idxs[0]
plt.figure(figsize=(12, 3))
plt.plot(ecg[pick, 0], label="ECG(seg)")
plt.plot(mask[pick, 0], label="Mask(seg)")
plt.legend()
plt.title(f"Overlay A: idx={pick}, sid={sid[pick] if sid is not None else 'NA'}")
plt.tight_layout()
plt.show()

# radar_heart overlay（只做相对可视化）
try:
    r_heart = bandpass(radar[pick, 0], FS, F_LOW, F_HIGH, order=4)
    plt.figure(figsize=(12, 3))
    plt.plot(np.abs(r_heart), label="|Radar_heart|(seg)")
    plt.plot(mask[pick, 0], label="Mask(seg)")
    plt.legend()
    plt.title(f"Overlay B: idx={pick}, sid={sid[pick] if sid is not None else 'NA'}")
    plt.tight_layout()
    plt.show()
except Exception:
    pass
