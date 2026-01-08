# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT loss with optional ECG-band frequency selection.

    Args:
        fs: sampling rate (Hz), e.g., 200
        fft_sizes/hop_sizes/win_lengths: multi-resolution STFT configs
        fmin, fmax: frequency band to keep (Hz), e.g., 0.5–40
        use_band: whether to apply frequency band mask
        mag_weight: weight for log-magnitude loss term
        eps: numerical stability
    """
    def __init__(
        self,
        fs: float = 200.0,
        fft_sizes=(128, 256, 512),
        hop_sizes=(32, 64, 128),
        win_lengths=(128, 256, 512),
        fmin: float = 0.5,
        fmax: float = 40.0,
        use_band: bool = True,
        mag_weight: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "fft_sizes, hop_sizes, win_lengths must have same length."
        self.fs = float(fs)
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)

        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.use_band = bool(use_band)

        self.mag_weight = float(mag_weight)
        self.eps = float(eps)

    def _stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        """
        x: [B, 1, L] -> STFT complex [B, F, T]
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # [B, L]
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"Expected x shape [B,1,L] or [B,L], got {tuple(x.shape)}")

        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        X = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True
        )
        return X  # [B, F, T]

    def _freq_mask(self, fft_size: int, device: torch.device) -> torch.Tensor:
        """
        Return boolean mask for rfft frequency bins within [fmin, fmax].
        """
        freqs = torch.fft.rfftfreq(fft_size, d=1.0 / self.fs).to(device)  # [F]
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        return mask  # [F] bool

    def forward(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """
        x_fake/x_real: [B, 1, L]
        """
        total_loss = 0.0

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            Xf = self._stft(x_fake, fft_size, hop_size, win_length)
            Xr = self._stft(x_real, fft_size, hop_size, win_length)

            mag_f = torch.abs(Xf)  # [B, F, T]
            mag_r = torch.abs(Xr)  # [B, F, T]

            if self.use_band:
                m = self._freq_mask(fft_size, mag_f.device)  # [F]
                mag_f = mag_f[:, m, :]
                mag_r = mag_r[:, m, :]

            # ---- Spectral Convergence (per-sample normalized) ----
            # sc = |||S_r|-|S_f|||_F / (|||S_r|||_F + eps), computed per sample then averaged
            num = torch.norm(mag_r - mag_f, dim=(-2, -1))
            den = torch.norm(mag_r, dim=(-2, -1)) + self.eps
            sc_loss = torch.mean(num / den)

            # ---- Log-magnitude loss ----
            # Using log1p is more stable than log10
            mag_loss = F.l1_loss(torch.log1p(mag_r), torch.log1p(mag_f))

            total_loss = total_loss + (sc_loss + self.mag_weight * mag_loss)

        return total_loss / len(self.fft_sizes)


class TotalLoss(nn.Module):
    """
    Total loss for BeatAware R-M2Net:
        L = L_time + alpha * L_stft + beta * L_anchor + gamma * L_smooth

    Notes:
      - If anchor_from_logits=True, anchor_pred MUST be logits (no sigmoid in model).
      - If anchor_from_logits=False, anchor_pred MUST be probabilities in [0,1] (sigmoid already applied).
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.1,
        # STFT configs
        fs: float = 200.0,
        fft_sizes=(128, 256, 512),
        hop_sizes=(32, 64, 128),
        win_lengths=(128, 256, 512),
        stft_fmin: float = 0.5,
        stft_fmax: float = 40.0,
        stft_use_band: bool = True,
        stft_mag_weight: float = 0.1,
        # Anchor configs
        anchor_pos_weight: float = 20.0,
        anchor_from_logits: bool = False,
        # Stability
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

        # time-domain
        self.l1_loss = nn.L1Loss()

        # freq-domain
        self.mr_stft_loss = MultiResolutionSTFTLoss(
            fs=fs,
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            fmin=stft_fmin,
            fmax=stft_fmax,
            use_band=stft_use_band,
            mag_weight=stft_mag_weight,
            eps=eps,
        )

        # anchor loss
        self.anchor_from_logits = bool(anchor_from_logits)
        if self.anchor_from_logits:
            # BCEWithLogitsLoss expects logits; pos_weight handles severe imbalance
            # pos_weight must be a tensor, moved to correct device in forward()
            self.register_buffer("anchor_pos_weight", torch.tensor([float(anchor_pos_weight)]))
            self.anchor_criterion = nn.BCEWithLogitsLoss(pos_weight=self.anchor_pos_weight)
        else:
            # If anchor_pred is already sigmoid probability
            self.anchor_criterion = nn.BCELoss()

    def forward(
        self,
        x_pred: torch.Tensor,
        x_target: torch.Tensor,
        anchor_pred: torch.Tensor = None,
        anchor_target: torch.Tensor = None,
    ):
        # 1) time-domain reconstruction loss
        loss_time = self.l1_loss(x_pred, x_target)

        # 2) frequency-domain loss
        loss_freq = self.mr_stft_loss(x_pred, x_target)

        # 3) anchor loss (optional)
        if (anchor_pred is not None) and (anchor_target is not None):
            # device safety for pos_weight
            if isinstance(self.anchor_criterion, nn.BCEWithLogitsLoss):
                # ensure buffer on same device as anchor_pred
                self.anchor_criterion.pos_weight = self.anchor_pos_weight.to(anchor_pred.device)
            loss_anchor = self.anchor_criterion(anchor_pred, anchor_target)
        else:
            loss_anchor = torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)

        # 4) smoothness (TV) loss
        loss_smooth = torch.mean(torch.abs(x_pred[:, :, 1:] - x_pred[:, :, :-1]))

        total = loss_time + self.alpha * loss_freq + self.beta * loss_anchor + self.gamma * loss_smooth
        return total, loss_time, loss_freq, loss_anchor, loss_smooth
