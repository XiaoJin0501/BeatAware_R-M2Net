import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    """
    RQ1 核心 Loss: 多分辨率 STFT 损失 (Multi-Resolution Short-Time Fourier Transform Loss)
    用于在频域约束模型，确保 P波 和 T波 等微细结构的保真度。
    参考: Parallel WaveGAN / HiFi-GAN
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
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
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss() # Time Domain
        self.mr_stft_loss = MultiResolutionSTFTLoss() # Frequency Domain

    def forward(self, x_pred, x_target):
        # 1. Time Domain Loss (MAE)
        loss_time = self.l1_loss(x_pred, x_target)
        
        # 2. Frequency Domain Loss
        loss_freq = self.mr_stft_loss(x_pred, x_target)
        
        # Total Loss
        return loss_time + self.alpha * loss_freq, loss_time, loss_freq