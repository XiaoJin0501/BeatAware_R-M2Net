#!/bin/bash

# ==============================================================================
# BeatAware_R-M2Net 消融实验 + GPU 自动排队脚本
# ==============================================================================

set -e

# 定义消融实验组
experiments=(
    "0.0 0.0 0.5 Baseline_L1_Only"
    "0.0 1.0 0.5 L1_plus_Anchor"
    "0.5 1.0 0.5 Full_Proposed_Balanced"
    "1.0 1.0 0.5 High_Alpha_Morphology"   # 探索形态极限
    
)

for exp in "${experiments[@]}"; do
    read -r ALPHA BETA GAMMA TAG <<< "$exp"
    
    echo "----------------------------------------------------------------"
    echo "准备实验: $TAG (Alpha=$ALPHA, Beta=$BETA, Gamma=$GAMMA)"
    
    # 构造具体的训练命令
    # 注意：这里我们加入了 --gamma 参数
    TRAIN_CMD="python train.py --alpha $ALPHA --beta $BETA --gamma $GAMMA --exp_tag $TAG"

    # --- [联动关键点] ---
    # 使用 gpu_waiter.py 来运行训练命令
    # 它会一直等到显存空闲到 14.5GB 以上才会真正启动训练
    python gpu_waiter.py $TRAIN_CMD

    # 训练完成后运行测试 (测试显存需求小，通常不需要再排队，直接运行即可)
    TEST_CMD="python test.py --alpha $ALPHA --beta $BETA --gamma $GAMMA --exp_tag $TAG"
    echo "📈 训练完成，开始测试: $TAG"
    $TEST_CMD

    echo "✅ 实验 $TAG 全部流程结束。"
done

echo "🎉 所有消融实验任务已按照 GPU 显存情况自动排队完成！"