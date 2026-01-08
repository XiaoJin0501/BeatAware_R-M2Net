# models/group_mamba.py
import torch
import torch.nn as nn
from .ssm import VSSSBlock1D


class GroupMambaBlock(nn.Module):
    """
    Group Mamba Block (1D)

    Input/Output:
        x: [B, C, T]  ->  y: [B, C, T]

    Key constraints:
        - d_model must be divisible by num_groups
        - Each VSSSBlock1D must preserve shape: [B, Cg, T] -> [B, Cg, T]
    """
    def __init__(self, d_model: int, num_groups: int = 4, d_state: int = 16, debug: bool = False):
        super().__init__()

        if d_model % num_groups != 0:
            raise ValueError(f"[GroupMambaBlock] d_model={d_model} must be divisible by num_groups={num_groups}")

        self.d_model = int(d_model)
        self.num_groups = int(num_groups)
        self.group_dim = self.d_model // self.num_groups
        self.debug = bool(debug)

        # LayerNorm over channel dimension (implemented by transposing to [B,T,C])
        self.norm = nn.LayerNorm(self.d_model)

        # Create independent VSSS blocks per group
        self.blocks = nn.ModuleList([
            VSSSBlock1D(self.group_dim, d_state=d_state)
            for _ in range(self.num_groups)
        ])

        # Channel Affine Modulation (CAM): produces [B, C, 1] weights
        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # [B,C,T] -> [B,C,1]
            nn.Conv1d(self.d_model, self.d_model // 4, 1),
            nn.ReLU(),
            nn.Conv1d(self.d_model // 4, self.d_model, 1),
            nn.Sigmoid()
        )

        # Final projection (shape-preserving)
        self.proj = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            y: [B, C, T]
        """
        if self.debug:
            assert x.dim() == 3, f"[GroupMambaBlock] Expect [B,C,T], got {tuple(x.shape)}"
            assert x.size(1) == self.d_model, f"[GroupMambaBlock] C mismatch: got {x.size(1)}, expect {self.d_model}"

        residual = x

        # 1) LayerNorm over channels
        y = x.transpose(1, 2)     # [B,T,C]
        y = self.norm(y)
        y = y.transpose(1, 2)     # [B,C,T]

        # 2) CAM weights
        w_cam = self.cam(y)       # [B,C,1]

        # 3) Strict group split (avoid uneven chunk)
        chunks = torch.split(y, self.group_dim, dim=1)  # tuple of [B,Cg,T], length=num_groups
        if self.debug:
            assert len(chunks) == self.num_groups, f"[GroupMambaBlock] split groups={len(chunks)} != {self.num_groups}"

        outs = []
        for i, block in enumerate(self.blocks):
            yi = block(chunks[i])
            if self.debug:
                assert yi.shape == chunks[i].shape, (
                    f"[GroupMambaBlock] VSSSBlock1D must preserve shape. "
                    f"Got {tuple(yi.shape)} vs {tuple(chunks[i].shape)}"
                )
            outs.append(yi)

        # 4) Merge + CAM modulation
        y = torch.cat(outs, dim=1)  # [B,C,T]
        y = y * w_cam               # broadcast along T

        # 5) Projection + residual
        y = self.proj(y)            # [B,C,T]
        return y + residual
