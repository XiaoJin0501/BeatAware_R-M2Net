import torch
import torch.nn as nn

class TFiLMGenerator(nn.Module):
    """
    根据 Anchor 特征生成 Gamma 和 Beta
    """
    def __init__(self, input_dim, target_channels):
        super().__init__()
        self.fc_gamma = nn.Linear(input_dim, target_channels)
        self.fc_beta = nn.Linear(input_dim, target_channels)
        
        # 初始化为 Identity
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x):
        return self.fc_gamma(x), self.fc_beta(x)

class ConformerFusionBlock(nn.Module):
    """
    Conformer Fusion: FFN -> MHSA -> Conv -> FFN
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 31, padding=15, groups=dim), # Depthwise
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1) # Pointwise
        )
        
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, C, L] -> 转置为 [B, L, C] 供 Transformer 使用
        x_t = x.transpose(1, 2)
        
        # MHSA
        res = x_t
        x_t = self.norm1(x_t)
        x_t, _ = self.mhsa(x_t, x_t, x_t)
        x_t = res + x_t
        
        # Conv (需要 [B, C, L])
        x_c = x_t.transpose(1, 2)
        x_c = x_c + self.conv(self.norm2(x_t).transpose(1, 2))
        
        # FFN
        x_t = x_c.transpose(1, 2)
        x_t = x_t + self.ffn(self.norm3(x_t))
        
        return x_t.transpose(1, 2)