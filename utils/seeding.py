import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    固定所有可能的随机种子，确保实验可复现。
    """
    # 1. Python 随机库
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果有多张卡
    
    # 4. Cudnn (确保卷积算法选择确定)
    # 注意：这会轻微降低训练速度，但为了复现是值得的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🔒 Random seed set to {seed}")