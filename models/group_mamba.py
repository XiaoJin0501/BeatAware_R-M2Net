import torch
import torch.nn as nn
from .ssm import VSSSBlock1D

class GroupMambaBlock(nn.Module):
    def __init__(self, d_model, num_groups=4, d_state=16):
        super().__init__()
        self.num_groups = num_groups
        self.group_dim = d_model // num_groups
        
        self.norm = nn.LayerNorm(d_model)
        
        # 创建 4 个独立的 VSSS Block
        self.blocks = nn.ModuleList([
            VSSSBlock1D(self.group_dim, d_state=d_state) 
            for _ in range(num_groups)
        ])
        
        # Channel Affine Modulation (CAM)
        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Global Avg Pool
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model, 1),
            nn.Sigmoid()
        )
        
        self.proj = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        residual = x
        
        # 1. Layer Norm (需要转置处理 Channel)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        # 2. CAM 计算权重
        w_cam = self.cam(x)
        
        # 3. 分组处理 (Group Processing)
        chunks = torch.chunk(x, self.num_groups, dim=1)
        outs = []
        for i, block in enumerate(self.blocks):
            outs.append(block(chunks[i]))
            
        # 4. 合并 & 调制
        x = torch.cat(outs, dim=1)
        x = x * w_cam
        
        # 5. 最终投影
        x = self.proj(x)
        
        return x + residual