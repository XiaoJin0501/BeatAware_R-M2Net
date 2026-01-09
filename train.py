import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# --- 引入搭建的基础设施 ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.losses import TotalLoss
from utils.logger import setup_logger
from utils.seeding import seed_everything


def _build_subject_split_subsets(train_set: RadarDataset, val_subjects):
    """
    在 train_set（已经做过 QC bad_indices 过滤）基础上，按 subject_id 做 train/val 切分。
    返回：train_subset, val_subset
    """
    if val_subjects is None or len(val_subjects) == 0:
        raise ValueError("VAL_SUBJECTS is empty. Please set Config.VAL_SUBJECTS, e.g., [21, 23].")

    # train_set.indices: 真实 H5 index（已过滤 bad）
    kept_real_indices = train_set.indices  # shape [N_kept]
    # 每个 kept 样本对应的 subject id
    kept_subjects = train_set.subject_ids[kept_real_indices]  # shape [N_kept]

    val_subjects = np.array(val_subjects, dtype=np.int32)
    is_val = np.isin(kept_subjects, val_subjects)

    val_pos = np.nonzero(is_val)[0].astype(np.int64)      # Subset 需要的是 dataset 内部位置索引
    train_pos = np.nonzero(~is_val)[0].astype(np.int64)

    if len(val_pos) == 0:
        # 说明这两个 subject 不在 train.h5 里（或被 QC 全过滤掉）
        unique_kept = np.unique(kept_subjects)
        raise RuntimeError(
            f"[VAL SPLIT ERROR] No samples found for VAL_SUBJECTS={val_subjects.tolist()} in train.h5 after QC.\n"
            f"  Unique subjects in kept train set: {unique_kept.tolist()}"
        )

    train_subset = Subset(train_set, train_pos.tolist())
    val_subset = Subset(train_set, val_pos.tolist())
    return train_subset, val_subset


def train():
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Train BeatAware R-M2Net")
    parser.add_argument('--alpha', type=float, default=Config.ALPHA, help='STFT loss weight')
    parser.add_argument('--beta', type=float, default=Config.BETA, help='Anchor loss weight')
    parser.add_argument('--gamma', type=float, default=Config.GAMMA, help='Smooth loss weight')
    parser.add_argument('--exp_tag', type=str, default="Default", help='Tag for this experiment')
    args = parser.parse_args()

    # 动态更新 Config 参数和路径
    new_name = f"Exp_a{args.alpha}_b{args.beta}_g{args.gamma}_{args.exp_tag}"
    Config.ALPHA = args.alpha
    Config.BETA = args.beta
    Config.GAMMA = args.gamma
    Config.update_paths(new_name)
    Config.makedirs()

    # 1. 固定随机种子
    seed_everything(Config.SEED)

    # 2. 初始化日志系统
    logger = setup_logger(Config.LOG_DIR, name="train")
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, 'tensorboard'))

    logger.info(f"🚀 Experiment Started: {Config.EXP_NAME}")
    logger.info(f"📊 Hyperparams: Alpha={Config.ALPHA}, Beta={Config.BETA}, Gamma={Config.GAMMA}")
    logger.info(f"📂 Data Directory: {Config.DATA_DIR}")
    logger.info(f"💻 Device: {Config.DEVICE}")

    # =========================
    # QC: bad indices (train/test)
    # =========================
    train_bad = getattr(Config, "TRAIN_BAD_INDICES_PATH", None)
    test_bad  = getattr(Config, "TEST_BAD_INDICES_PATH", None)

    if train_bad is not None and not os.path.exists(train_bad):
        logger.warning(f"[QC] TRAIN_BAD_INDICES_PATH not found, disable: {train_bad}")
        train_bad = None

    if test_bad is not None and not os.path.exists(test_bad):
        logger.warning(f"[QC] TEST_BAD_INDICES_PATH not found, disable: {test_bad}")
        test_bad = None

    logger.info(f"[QC] TRAIN_BAD_INDICES_PATH = {train_bad if train_bad else 'None (disabled)'}")
    logger.info(f"[QC] TEST_BAD_INDICES_PATH  = {test_bad if test_bad else 'None (disabled)'}")

    # =========================
    # 3. 数据准备
    # =========================
    # (1) 先加载 train.h5（带 QC），然后在其内部做 subject-wise val split
    full_train_set = RadarDataset(Config.TRAIN_H5, bad_indices_path=train_bad)

    val_subjects = getattr(Config, "VAL_SUBJECTS", None)
    logger.info(f"[VAL] VAL_SUBJECTS = {val_subjects}")

    train_subset, val_subset = _build_subject_split_subsets(full_train_set, val_subjects)

    # (2) test.h5 仍然可以加载，但严格只用于“最终评估”——训练过程不用于 early-stopping
    test_set = RadarDataset(Config.TEST_H5, bad_indices_path=test_bad)

    logger.info(f"[DATA] Train subset: {len(train_subset)} | Val subset: {len(val_subset)} | Test set: {len(test_set)}")

    # DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
        pin_memory=(Config.DEVICE == "cuda"),
        persistent_workers=(Config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
        pin_memory=(Config.DEVICE == "cuda"),
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    # （可选）test_loader：不参与早停，训练结束你可以手动/自动跑一次
    test_loader = DataLoader(
        test_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
        pin_memory=(Config.DEVICE == "cuda"),
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    # =========================
    # 4. 模型与 Loss 构建
    # =========================
    logger.info("⏳ Initializing model and optimizer...")
    model = BeatAwareRM2Net(
        in_channels=Config.IN_CHANNELS,
        base_channels=Config.BASE_CHANNELS
    ).to(Config.DEVICE)

    criterion = TotalLoss(
        alpha=Config.ALPHA,
        beta=Config.BETA,
        gamma=Config.GAMMA,
        fs=Config.FS,
        fft_sizes=Config.FFT_SIZES,
        hop_sizes=Config.HOP_SIZES,
        win_lengths=Config.WIN_LENGTHS,
        stft_fmin=Config.STFT_FMIN,
        stft_fmax=Config.STFT_FMAX,
        stft_use_band=Config.STFT_USE_BAND,
        anchor_pos_weight=Config.ANCHOR_POS_WEIGHT,
        anchor_from_logits=Config.ANCHOR_FROM_LOGITS,
    ).to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {n_params} trainable parameters.")

    # =========================
    # 5. Resume / Early stopping
    # =========================
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    last_ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_last.pth")
    best_ckpt_path = os.path.join(Config.CKPT_DIR, f"{Config.EXP_NAME}_best.pth")

    if os.path.exists(last_ckpt_path):
        logger.info(f"Found checkpoint at {last_ckpt_path}. Resuming training...")
        checkpoint = torch.load(last_ckpt_path, map_location=Config.DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

        logger.info(f"Resumed from Epoch {start_epoch}. Best Val Loss so far: {best_val_loss:.6f}")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

    # =========================
    # 6. 训练主循环
    # =========================
    for epoch in range(start_epoch, Config.EPOCHS):
        # -------- Train --------
        model.train()
        train_loss_avg = 0.0
        train_L1_avg, train_STFT_avg, train_Anchor_avg, train_Smooth_avg = 0.0, 0.0, 0.0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        for i, (radar, ecg, mask, subject_id) in enumerate(loop):
            global_step = epoch * len(train_loader) + i

            radar = radar.to(Config.DEVICE, non_blocking=True)
            ecg   = ecg.to(Config.DEVICE, non_blocking=True)
            mask  = mask.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_ecg, pred_mask = model(radar)
            loss, l_time, l_freq, l_anchor, l_smooth = criterion(pred_ecg, ecg, pred_mask, mask)

            loss.backward()
            optimizer.step()

            train_loss_avg += loss.item()
            train_L1_avg += l_time.item()
            train_STFT_avg += l_freq.item()
            train_Anchor_avg += l_anchor.item()
            train_Smooth_avg += l_smooth.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                L1=f"{l_time.item():.4f}",
                STFT=f"{l_freq.item():.4f}",
                Anchor=f"{l_anchor.item():.4f}",
                Smooth=f"{l_smooth.item():.4f}"
            )

            # TensorBoard：每 10 step 写一次即可（你原来逻辑保留）
            if i % 10 == 0:
                writer.add_scalar('Loss/Train_Total', loss.item(), global_step)
                writer.add_scalar('Loss/Train_L1', l_time.item(), global_step)
                writer.add_scalar('Loss/Train_STFT', l_freq.item(), global_step)
                writer.add_scalar('Loss/Train_Anchor', l_anchor.item(), global_step)
                writer.add_scalar('Loss/Train_Smooth', l_smooth.item(), global_step)

        # epoch 平均
        train_loss_avg /= max(len(train_loader), 1)
        train_L1_avg /= max(len(train_loader), 1)
        train_STFT_avg /= max(len(train_loader), 1)
        train_Anchor_avg /= max(len(train_loader), 1)
        train_Smooth_avg /= max(len(train_loader), 1)

        # -------- Val (来自 train.h5 的 subject-wise split) --------
        model.eval()
        val_loss_avg = 0.0
        val_L1_avg, val_STFT_avg, val_Anchor_avg, val_Smooth_avg = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for radar, ecg, mask, subject_id in val_loader:
                radar = radar.to(Config.DEVICE, non_blocking=True)
                ecg   = ecg.to(Config.DEVICE, non_blocking=True)
                mask  = mask.to(Config.DEVICE, non_blocking=True)

                pred_ecg, pred_mask = model(radar)
                loss, l_time, l_freq, l_anchor, l_smooth = criterion(pred_ecg, ecg, pred_mask, mask)

                val_loss_avg += loss.item()
                val_L1_avg += l_time.item()
                val_STFT_avg += l_freq.item()
                val_Anchor_avg += l_anchor.item()
                val_Smooth_avg += l_smooth.item()

        val_loss_avg /= max(len(val_loader), 1)
        val_L1_avg /= max(len(val_loader), 1)
        val_STFT_avg /= max(len(val_loader), 1)
        val_Anchor_avg /= max(len(val_loader), 1)
        val_Smooth_avg /= max(len(val_loader), 1)

        # TensorBoard：epoch 粒度
        writer.add_scalar('Loss/Val_Total', val_loss_avg, epoch)
        writer.add_scalar('Loss/Val_L1', val_L1_avg, epoch)
        writer.add_scalar('Loss/Val_STFT', val_STFT_avg, epoch)
        writer.add_scalar('Loss/Val_Anchor', val_Anchor_avg, epoch)
        writer.add_scalar('Loss/Val_Smooth', val_Smooth_avg, epoch)

        # -------- Logging --------
        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Train Total: {train_loss_avg:.4f} "
            f"(L1: {train_L1_avg:.4f}, STFT: {train_STFT_avg:.4f}, Anchor: {train_Anchor_avg:.4f}, Smooth: {train_Smooth_avg:.4f}) | "
            f"Val Total: {val_loss_avg:.4f} "
            f"(L1: {val_L1_avg:.4f}, STFT: {val_STFT_avg:.4f}, Anchor: {val_Anchor_avg:.4f}, Smooth: {val_Smooth_avg:.4f})"
        )

        # -------- Save last --------
        last_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss_avg,
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        }
        torch.save(last_checkpoint, last_ckpt_path)
        logger.info(f"Last checkpoint saved to {last_ckpt_path}")

        # -------- Save best / Early stopping --------
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0

            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss_avg,
                'best_val_loss': best_val_loss,
            }
            torch.save(best_checkpoint, best_ckpt_path)
            logger.info(f"Best model saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{Config.PATIENCE} epochs.")
            if epochs_no_improve >= Config.PATIENCE:
                logger.info(f"Early stopping triggered after {Config.PATIENCE} epochs with no improvement.")
                break

    writer.close()
    logger.info("Training Finished Successfully!")

    # （可选）训练结束后立即用 test_loader 跑一次 quick test loss（不写 best，不早停）
    # 你更推荐用 test.py 做全套指标输出，所以这里我默认不自动跑，避免重复。


if __name__ == "__main__":
    train()
