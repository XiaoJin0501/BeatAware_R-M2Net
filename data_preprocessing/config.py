import os

class Config:
    # ------------------ 路径配置 ------------------
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录路径
    PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
    
    # 原始数据路径，需根据实际存放位置修改
    RAW_DATA_DIR = r'C:\Xiao\Datasets\CR_Radar'
    
    # 输出路径依然保持在项目文件夹内，方便管理
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data_preprocessing', 'processed_to_h5')
    
    # ------------------ 信号物理参数 ------------------
    FS_RADAR_RAW = 2000
    FS_ECG_RAW = 2000
    FS_TARGET = 200
    
    RADAR_BANDPASS = [0.1, 16] 
    ECG_BANDPASS = [1.0, 16]   
    
    # ------------------ 数据集动态划分策略 ------------------
    MIN_VALID_SEGMENTS_PER_SUBJECT = 10 
    TEST_RATIO_A = 0.2 
    TEST_RATIO_B = 0.2
    
    # ------------------ 切片与Anchor ------------------
    WINDOW_SECONDS = 8.0         
    STRIDE_SECONDS = 1.6         
    ANCHOR_SIGMA = 5             
    
    # ------------------ 质量控制 ------------------
    SQI_HR_MIN = 40              
    SQI_HR_MAX = 140