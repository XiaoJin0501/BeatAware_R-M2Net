import torch
import torch.nn as nn
import torch.nn.functional as F

# 引用同目录下的模块
from .layers import TFiLMGenerator, ConformerFusionBlock
from .group_mamba import GroupMambaBlock

class BeatAwareRM2Net(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        # 1. anchor branch 用于生成 TFiLM 条件和预测 R 波 Mask, anchor_enc (anchor_encoder)
        # 输入是 Radar (1通道), 输出也是 Mask (1通道)
        # 注意：这里不能过早 Pooling 掉时间维度，要保持分辨率
        self.anchor_enc = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
        )
        
        # ✅ 头1: 预测概率 Mask (用于计算 Anchor Loss)
        # 输出 [B, 1, L]，经过 Sigmoid 归一化到 [0, 1]
        self.anchor_head = nn.Conv1d(32, 1, 1)
        
        # ✅ 头2: 生成 TFiLM 参数 (用于驱动主干)
        # 这里才进行 Pooling，变回 [B, 32] 向量
        self.tfilm_adapter = nn.Sequential(
            nn.AdaptiveMaxPool1d(1), 
            nn.Flatten()
        )
        self.tfilm_gen = TFiLMGenerator(32, base_channels * 4)
        self.base_channels = base_channels
        
        # 2. Encoder
        self.enc_convs = nn.ModuleList()
        self.enc_bns = nn.ModuleList()
        # Stride=4 下采样
        for k in [3, 5, 7, 9]:
            self.enc_convs.append(nn.Conv1d(in_channels, base_channels, k, padding=k//2, stride=4))
            self.enc_bns.append(nn.BatchNorm1d(base_channels))

        # 3. Bottleneck (GroupMamba)
        self.bottleneck_dim = 4 * base_channels
        self.mamba_layers = nn.Sequential(
            GroupMambaBlock(self.bottleneck_dim, num_groups=4),
            GroupMambaBlock(self.bottleneck_dim, num_groups=4)
        )

        # 4. Fusion
        self.fusion = ConformerFusionBlock(self.bottleneck_dim)

        # 5. Decoder
        self.up1 = nn.ConvTranspose1d(self.bottleneck_dim, self.bottleneck_dim//2, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(self.bottleneck_dim//2, base_channels, 4, stride=2, padding=1)
        self.final = nn.Conv1d(base_channels, 1, 1)
        
    def forward(self, x, mask=None): # mask 仅用于训练时的 Loss 计算，推理时不需要
        # The input for the Anchor Branch must be Radar(x) [B, 1, L].
        anchor_feat = self.anchor_enc(x)  # [B, 32, L]
        # 1. 输出 Mask 预测 (用于 Loss)(使用 Sigmoid 归一化到 0-1)
        anchor_pred_mask = self.anchor_head(anchor_feat)   # logits, NO sigmoid
        
        # 生成 TFiLM 参数
        anchor_vec = self.tfilm_adapter(anchor_feat) # [B, 32]
        gamma, beta = self.tfilm_gen(anchor_vec) # [B, 4*C], [B, 4*C]
        
        # Reshape for multiplication: [B, 4*C] -> [B, 4, C, 1]
        gamma = gamma.view(x.size(0), 4, self.base_channels, 1)
        beta = beta.view(x.size(0), 4, self.base_channels, 1)
        
        # B. Encoder (应用 TFiLM)
        feats = []
        for i, (conv, bn) in enumerate(zip(self.enc_convs, self.enc_bns)):
            f = conv(x)
            # 1: 先过 BN 再做 TFiLM, 这样调制的偏移量不会被 BN 抹掉
            f = bn(f)
            f = f * (1.0 + gamma[:, i]) + beta[:, i] # TFiLM
            f = F.relu(f) # 2: 激活 ReLU
            feats.append(f)
        
        x_enc = torch.cat(feats, dim=1)

        # C. Mamba & Fusion
        x_mid = self.mamba_layers(x_enc)
        x_mid = self.fusion(x_mid)

        # D. Decoder
        x_up = F.relu(self.up1(x_mid))
        x_up = F.relu(self.up2(x_up))
        
        
        out = torch.sigmoid(self.final(x_up))
        # The activation function was changed to Sigmoid, which matches the data range [0, 1].
        # out = torch.sigmoid(self.final(x_up)) # 在可视化/保存时再 torch.sigmoid 或 clamp(0,1)
    
    # 返回: 重建ECG, 预测Mask
        return out, anchor_pred_mask

if __name__ == "__main__":
    model = BeatAwareRM2Net()
    x = torch.randn(2, 1, 1600)
    # 这里的 mask 参数在 forward 里其实没用到，只是为了保持接口一致性
    mask = torch.randn(2, 1, 1600)
    
    # 注意这里接收两个返回值
    y, y_anchor = model(x, mask)
    print(f"✅ BA_M2Net Ready!")
    print(f"   Main Output Shape: {y.shape}")      # 应为 [2, 1, 1600]
    print(f"   Anchor Output Shape: {y_anchor.shape}") # 应为 [2, 1]