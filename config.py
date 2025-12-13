# config.py
import os
import torch

class Config:
    # --- 1. 实验管理 ---
    PROJECT_NAME = "BeatAware_RM2Net"
    EXP_NAME = "Exp_A_SubjectIndependent_Baseline" # 每次改这里区分实验
    SEED = 42                                      # 随机种子，保证可复现
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # --- 2. 路径配置 ---
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "data_preprocessing/processed_to_h5/experiment_A_SubjectIndependent")
    TRAIN_PATH = os.path.join(DATA_DIR, "train.h5")
    TEST_PATH = os.path.join(DATA_DIR, "test.h5")
    
    # 输出目录 (自动归档)
    OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", EXP_NAME)
    CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    RESULT_DIR = os.path.join(OUTPUT_DIR, "results") # 存放测试结果和图表
    
    # --- 3. 模型参数 (Model Hyperparams) ---
    INPUT_LEN = 1600
    IN_CHANNELS = 1
    BASE_CHANNELS = 32
    
    # --- 4. 训练参数 (Training) ---
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4 if DEVICE == "cuda" else 0 # Mac 上多线程可能报错
    
    # --- 5. Loss 参数 ---
    ALPHA = 1.0  # STFT Loss 权重
    FFT_SIZES = [1024, 2048, 512]
    HOP_SIZES = [120, 240, 50]
    WIN_LENGTHS = [600, 1200, 240]
    
    @classmethod
    def makedirs(cls):
        os.makedirs(cls.CKPT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)

# 自动创建目录
Config.makedirs()