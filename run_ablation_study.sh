#!/bin/bash
# ==============================================================================
# BeatAware_R-M2Net 消融实验 + GPU 自动排队 + 安全关机脚本
# ==============================================================================

set -e  # 任意命令失败即退出（防止 silent error）

# ---------------- 基本配置 ----------------
PROJECT_ROOT="$HOME/Projects/BeatAware_R-M2Net"
cd "$PROJECT_ROOT"

PYTHON_BIN="python"
GPU_WAITER="gpu_waiter.py"

# GPU 空闲阈值（4090 / 4090D 24GB）
GPU_FREE_GB=14.5

# 是否在全部实验完成后自动关机（true / false）
AUTO_POWEROFF=true
POWEROFF_DELAY=60   # 秒

# 完成标志文件（防止误关机）
FINISH_FLAG="ALL_ABLATION_DONE.flag"

# ---------------- 消融实验组 ----------------
experiments=(
    "0.0 0.0 0.5 Baseline_L1_Only"
    "0.0 1.0 0.5 L1_plus_Anchor"
    "0.5 1.0 0.5 Full_Proposed"
    "1.0 1.0 0.5 High_Alpha_Morphology"
)

echo "============================================================"
echo "🚀 BeatAware_R-M2Net Ablation Experiments Started"
echo "📂 Project: $PROJECT_ROOT"
echo "============================================================"

for exp in "${experiments[@]}"; do
    read -r ALPHA BETA GAMMA TAG <<< "$exp"

    echo
    echo "------------------------------------------------------------"
    echo "🧪 Experiment: $TAG"
    echo "   Alpha=$ALPHA | Beta=$BETA | Gamma=$GAMMA"
    echo "------------------------------------------------------------"

    TRAIN_CMD="$PYTHON_BIN train.py --alpha $ALPHA --beta $BETA --gamma $GAMMA --exp_tag $TAG"
    TEST_CMD="$PYTHON_BIN test.py  --alpha $ALPHA --beta $BETA --gamma $GAMMA --exp_tag $TAG"

    echo "⏳ Waiting for GPU (free >= ${GPU_FREE_GB} GB)..."
    $PYTHON_BIN $GPU_WAITER "$TRAIN_CMD"

    echo "📈 Training finished. Start testing..."
    $TEST_CMD

    echo "✅ Experiment [$TAG] finished successfully."
done

# ---------------- 所有实验完成 ----------------
touch "$FINISH_FLAG"

echo
echo "🎉 All ablation experiments completed!"
echo "📄 Finish flag created: $FINISH_FLAG"

# ---------------- 自动关机（可选） ----------------
if [ "$AUTO_POWEROFF" = true ] && [ -f "$FINISH_FLAG" ]; then
    echo "🧯 Auto poweroff enabled."
    echo "⏳ System will power off in ${POWEROFF_DELAY}s (Ctrl+C to cancel)..."
    sleep "$POWEROFF_DELAY"
    sudo poweroff || shutdown -h now
fi
