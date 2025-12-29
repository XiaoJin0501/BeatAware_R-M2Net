import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    """
    RQ1 核心 Loss: 多分辨率 STFT 损失 (Multi-Resolution Short-Time Fourier Transform Loss)
    用于在频域约束模型，确保 P波 和 T波 等微细结构的保真度。
    参考: Parallel WaveGAN / HiFi-GAN
    """
    def __init__(self, fft_sizes=[64, 128, 256], hop_sizes=[8, 16, 32], win_lengths=[32, 64, 128]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x, fft_size, hop_size, win_length):
        # x: [B, 1, L] -> [B, L]
        x = x.squeeze(1)
        # Hanning Window
        window = torch.hann_window(win_length, device=x.device)
        return torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)

    def forward(self, x_fake, x_real):
        """
        x_fake: 重建的 ECG [B, 1, L]
        x_real: 真实的 ECG [B, 1, L]
        """
        loss = 0.0
        for fs, hs, wl in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_fake_stft = self.stft(x_fake, fs, hs, wl)
            x_real_stft = self.stft(x_real, fs, hs, wl)
            
            # 计算幅度谱 (Magnitude Spectrogram)
            x_fake_mag = torch.abs(x_fake_stft)
            x_real_mag = torch.abs(x_real_stft)
            
            # 1. Spectral Convergence Loss (谱收敛损失)
            sc_loss = torch.norm(x_real_mag - x_fake_mag, p="fro") / (torch.norm(x_real_mag, p="fro") + 1e-6)
            
            # 2. Log Magnitude Loss (对数幅度损失 - 关注低能量细节)
            mag_loss = F.l1_loss(torch.log(x_real_mag + 1e-6), torch.log(x_fake_mag + 1e-6))
            
            loss += sc_loss + mag_loss
            
        return loss / len(self.fft_sizes)

class TotalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1): 
        """
        alpha: STFT Loss 权重 (建议 0.5)
        beta: Anchor Loss 权重 (建议 1.0)
        gamma: Smooth Loss (TV Loss) 权重 (建议 0.1)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss() # Time Domain
        self.mr_stft_loss = MultiResolutionSTFTLoss() # Frequency Domain
        # 因为 Mask 是 0/1 概率，用 BCE Loss 最合适
        self.anchor_criterion = nn.BCELoss()

    def forward(self, x_pred, x_target, anchor_pred=None, anchor_target=None):
        # 1. Time Domain Loss (MAE) 主任务 Loss 定位 R 峰
        loss_time = self.l1_loss(x_pred, x_target)
        # 2. (Multi-Resolution STFT Loss) 修饰波形细节
        loss_freq = self.mr_stft_loss(x_pred, x_target)
        
        # 3. Anchor Loss (带安全检查) 强化 R 峰检测
        if anchor_pred is not None and anchor_target is not None:
            loss_anchor = self.anchor_criterion(anchor_pred, anchor_target)
        else:
            # 如果没有提供标签，Loss 为 0 (不影响训练)
            loss_anchor = torch.tensor(0.0, device=x_pred.device)
        
        # 4. Smooth Loss (Total Variation Loss) - 消除锯齿，提升 Pearson
        # 计算相邻点之间的差异：x_pred 形状为 [B, 1, L]
        loss_smooth = torch.mean(torch.abs(x_pred[:, :, 1:] - x_pred[:, :, :-1]))
        
        # Total Loss 组合
        total = loss_time + self.alpha * loss_freq + self.beta * loss_anchor + self.gamma * loss_smooth
        
        # 注意：为了不破坏 train.py 的解包逻辑，我们将 loss_smooth 暂时合并输出或修改 train.py
        return total, loss_time, loss_freq, loss_anchor, loss_smooth
