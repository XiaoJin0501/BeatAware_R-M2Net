import os
import numpy as np
import pandas as pd

# ================== 配置区 ==================
CSV_PATH = "dataset_verification_reports/failed_lag_samples.csv"
OUT_DIR  = "data_preprocessing/qc_indices"
OUT_NAME = "bad_indices.npy"

# 极端 lag 阈值（与 verify_alignment_metrics.py 保持一致）
EXTREME_LAG_SAMPLES = 150
# ============================================


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cannot find CSV: {CSV_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # 基本检查
    required_cols = {"index", "lag_samples"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # 筛选 extreme outliers
    bad_df = df[np.abs(df["lag_samples"]) >= EXTREME_LAG_SAMPLES]

    bad_indices = np.sort(bad_df["index"].astype(np.int64).values)
    out_path = os.path.join(OUT_DIR, OUT_NAME)

    np.save(out_path, bad_indices)

    print("=" * 60)
    print("✅ bad_indices.npy generated successfully")
    print(f"  CSV source        : {CSV_PATH}")
    print(f"  Extreme threshold : |lag| >= {EXTREME_LAG_SAMPLES} samples")
    print(f"  Total bad samples : {len(bad_indices)}")
    print(f"  Saved to          : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
