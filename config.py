import os
import torch
import time

class Config:
    # =========================================================================
    # 1. 实验基本信息 (Experiment Metadata)
    # =========================================================================
    PROJECT_NAME = "RQ1_BeatAware_RM2Net"
    
    # 实验名称 (每次跑新实验改这里，比如 "Exp_B_Ablation_NoTFiLM")
    EXP_NAME = "Exp_A_SubjectIndependent_Baseline" 
    
    # 随机种子 (保证复现性)
    SEED = 42
    
    # 自动检测设备 (优先 CUDA > MPS (Mac) > CPU)
    # DEVICE = "cpu"
    DEVICE = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    # =========================================================================
    # 2. 路径配置 (Path Configuration) - 自动推导，无需硬编码
    # =========================================================================
    # 获取当前文件 (config.py) 所在目录，即项目根目录
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 数据集路径 (请确保 build_dataset.py 生成的文件在这里)
    # 注意：这里假设你之前生成的 h5 文件在 data_preprocessing/processed_to_h5 下
    DATA_DIR = os.path.join(ROOT_DIR, "data_preprocessing", "processed_to_h5", "experiment_A_SubjectIndependent")
    TRAIN_H5 = os.path.join(DATA_DIR, "train.h5")
    TEST_H5 = os.path.join(DATA_DIR, "test.h5")
    
    # 输出目录结构
    OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", EXP_NAME)
    
    # 路径初始化函数
    @classmethod
    def update_paths(cls, new_exp_name):
        cls.EXP_NAME = new_exp_name
        cls.OUTPUT_DIR = os.path.join(cls.ROOT_DIR, "experiments", cls.EXP_NAME)
        cls.CKPT_DIR = os.path.join(cls.OUTPUT_DIR, "checkpoints")  # 存模型权重
        cls.LOG_DIR = os.path.join(cls.OUTPUT_DIR, "logs")          # 存日志文件
        cls.RESULT_DIR = os.path.join(cls.OUTPUT_DIR, "results")    # 存测试图表和 .mat 文件
    
    # =========================================================================
    # 3. 数据与模型参数 (Data & Model Params)
    # =========================================================================
    INPUT_LEN = 1600         # 8s @ 200Hz
    IN_CHANNELS = 1          # Radar Displacement
    BASE_CHANNELS = 32       # 卷积基础通道数
    
    # =========================================================================
    # 4. 训练超参数 (Training Hyperparameters)
    # =========================================================================
    
    BATCH_SIZE = 32 # 显存不够可调小 (e.g., 16)
    
    # BATCH_SIZE = 1
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2      # AdamW 的权重衰减
    
    # 数据加载线程数
    # Mac (MPS) 上多线程有时会报错，设为 0 安全；Linux 上设为 4 或 8 加速
    NUM_WORKERS = 4 if DEVICE == "cuda" else 0 
    
    # =========================================================================
    # 5. Loss 参数 (Loss Function Params)
    # =========================================================================
    ALPHA = 0.5              # STFT Loss 的权重 (L_total = L1 + alpha * L_STFT + beta * L_anchor)
    BETA = 1.0    # Anchor Loss 权重，用于 R 峰定位
    GAMMA = 0.1  # Smooth Loss (TV Loss) 权重，用于波形平滑
    # 多分辨率 STFT 的参数配置 200Hz ECG 的参数
    # 逻辑：
    # 小窗口: 捕捉瞬时变化 (QRS波), 约 40-60ms
    # 中窗口: 捕捉波形形态 (P/T波), 约 150-300ms
    # 大窗口: 捕捉整体节律 (RR间期), 约 600ms-1s
    FFT_SIZES = [64, 128, 256]  # FFT 点数 (2的幂次)
    HOP_SIZES = [32, 64, 128]   # 窗长 (分别对应 0.16s, 0.32s, 0.64s)
    WIN_LENGTHS = [8, 16, 32]   # 步长 (通常为窗长的 1/4 或 1/2)
    PATIENCE = 30 # 早停法耐心值 (如果验证集 Loss 连续 10 个 Epoch 不下降，则停止)

    @classmethod
    def makedirs(cls):
        """自动创建所有必要的输出目录"""
        os.makedirs(cls.CKPT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)
        print(f"📁 Created experiment directories at: {cls.OUTPUT_DIR}")

# 默认初始化一次路径
Config.update_paths(Config.EXP_NAME)