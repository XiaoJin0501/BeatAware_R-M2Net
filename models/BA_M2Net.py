import torch
import torch.nn as nn
import torch.nn.functional as F

# 引用同目录下的模块
from .layers import TFiLMGenerator, ConformerFusionBlock
from .group_mamba import GroupMambaBlock

class BeatAwareRM2Net(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        # 1. Anchor Branch
        self.anchor_branch = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten()
        )
        self.base_channels = base_channels
        self.tfilm_gen = TFiLMGenerator(32, base_channels * 4)

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

    def forward(self, x, mask):
        # A. Condition
        anchor = self.anchor_branch(mask)
        gamma, beta = self.tfilm_gen(anchor)
        gamma = gamma.view(-1, 4, self.base_channels, 1)
        beta = beta.view(-1, 4, self.base_channels, 1)

        # B. Encoder
        feats = []
        for i, (conv, bn) in enumerate(zip(self.enc_convs, self.enc_bns)):
            f = conv(x)
            f = f * (1.0 + gamma[:, i]) + beta[:, i] # TFiLM
            feats.append(F.relu(bn(f)))
        
        x_enc = torch.cat(feats, dim=1)

        # C. Mamba & Fusion
        x_mid = self.mamba_layers(x_enc)
        x_mid = self.fusion(x_mid)

        # D. Decoder
        x_up = F.relu(self.up1(x_mid))
        x_up = F.relu(self.up2(x_up))
        return torch.tanh(self.final(x_up))

if __name__ == "__main__":
    model = BeatAwareRM2Net()
    x = torch.randn(2, 1, 1600)
    mask = torch.randn(2, 1, 1600)
    y = model(x, mask)
    print(f"✅ BA_M2Net Ready! Output Shape: {y.shape}")