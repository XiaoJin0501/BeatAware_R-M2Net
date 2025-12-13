import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .scan import selective_scan_1d  # 引用上一步写的 scan.py

class VSSSBlock1D(nn.Module):
    """
    1D Visual Single-Selective Scanning (VSSS) Block
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        # 1. 线性投影 (Input Projection)
        self.in_proj = nn.Conv1d(d_model, self.d_inner * 2, kernel_size=1)

        # 2. 深度卷积 (Depthwise Conv) - 提取局部特征
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2
        )
        self.act = nn.SiLU()

        # 3. SSM 参数投影 (生成 dt, B, C)
        self.x_proj = nn.Conv1d(self.d_inner, self.dt_rank + self.d_state * 2, kernel_size=1, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 A (S4D) 和 D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 4. 输出投影
        self.out_proj = nn.Conv1d(self.d_inner, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Channel, Length]
        residual = x

        # 1. Proj & Split
        x_and_z = self.in_proj(x)
        x, z = x_and_z.chunk(2, dim=1) # x: Signal, z: Gate

        # 2. Conv1D
        x = self.conv1d(x)
        x = self.act(x)

        # 3. 生成 SSM 参数
        x_dbl = self.x_proj(x) # [B, Rank+2*State, L]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)

        dt = self.dt_proj(dt.transpose(1, 2)).transpose(1, 2)
        A = -torch.exp(self.A_log)

        # 4. Selective Scan (自动调用 CPU 或 CUDA)
        y = selective_scan_1d(
            x, dt, A, B, C, D=self.D, z=None, 
            delta_bias=self.dt_proj.bias, 
            delta_softplus=True
        )

        # 5. Gating & Output
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)

        return out + residual