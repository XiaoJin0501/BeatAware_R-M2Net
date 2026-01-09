import os
import torch
import time

class Config:
    # =========================================================================
    # 1. 实验基本信息 (Experiment Metadata)
    # =========================================================================
    PROJECT_NAME = "RQ1_BeatAware_RM2Net"
    EXP_NAME = "Exp_A_SubjectIndependent_Baseline"
    SEED = 42

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # =========================================================================
    # 2. 路径配置 (Path Configuration)
    # =========================================================================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(
        ROOT_DIR, "data_preprocessing", "processed_to_h5", "experiment_A_SubjectIndependent"
    )
    TRAIN_H5 = os.path.join(DATA_DIR, "train.h5")
    TEST_H5  = os.path.join(DATA_DIR, "test.h5")

    # =========================
    # QC / Alignment Filtering
    # =========================
    QC_DIR = os.path.join(ROOT_DIR, "data_preprocessing", "qc_indices")
    TRAIN_BAD_INDICES_PATH = os.path.join(QC_DIR, "train_bad_indices.npy")
    TEST_BAD_INDICES_PATH  = os.path.join(QC_DIR, "test_bad_indices.npy")

    # =========================
    # Subject-wise Validation Split  ✅新增
    # =========================
    # 注意：VAL_SUBJECTS 必须来自 train.h5 中的 subjects（不能用 test subjects）
    VAL_SUBJECTS = [21, 23]

    # 输出目录（给一个默认值，随后会在 update_paths 里被覆盖）
    OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", EXP_NAME)

    @classmethod
    def update_paths(cls, new_exp_name):
        cls.EXP_NAME = new_exp_name
        cls.OUTPUT_DIR = os.path.join(cls.ROOT_DIR, "experiments", cls.EXP_NAME)
        cls.CKPT_DIR   = os.path.join(cls.OUTPUT_DIR, "checkpoints")
        cls.LOG_DIR    = os.path.join(cls.OUTPUT_DIR, "logs")
        cls.RESULT_DIR = os.path.join(cls.OUTPUT_DIR, "results")

    # =========================================================================
    # 3. 数据与模型参数 (Data & Model Params)
    # =========================================================================
    INPUT_LEN = 1600
    IN_CHANNELS = 1
    BASE_CHANNELS = 32

    # =========================================================================
    # 4. 训练超参数 (Training Hyperparameters)
    # =========================================================================
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4 if DEVICE == "cuda" else 0

    # =========================================================================
    # 5. Loss 参数 (Loss Function Params)
    # =========================================================================
    ALPHA = 0.5
    BETA  = 1.0
    GAMMA = 0.5

    FFT_SIZES   = [128, 256, 512]
    WIN_LENGTHS = [128, 256, 512]
    HOP_SIZES   = [32,  64,  128]

    PATIENCE = 30

    # ========= Sampling / STFT band =========
    FS = 200
    STFT_FMIN = 0.5
    STFT_FMAX = 40.0
    STFT_USE_BAND = True

    # ========= Anchor (mask) =========
    ANCHOR_FROM_LOGITS = True
    ANCHOR_POS_WEIGHT = 20.0

    @classmethod
    def makedirs(cls):
        os.makedirs(cls.CKPT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)
        print(f"📁 Created experiment directories at: {cls.OUTPUT_DIR}")

# 默认初始化一次路径
Config.update_paths(Config.EXP_NAME)
