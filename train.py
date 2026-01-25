import os
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# --- project imports ---
from config import Config
from dataset import RadarDataset
from models.BA_M2Net import BeatAwareRM2Net
from utils.losses import TotalLoss
from utils.logger import setup_logger
from utils.seeding import seed_everything


# =========================
# Helper: subject-wise split
# =========================
def _build_subject_split_subsets(train_set: RadarDataset, val_subjects: List[int]) -> Tuple[Subset, Subset]:
    """
    在 train_set（已做 QC bad_indices 过滤）基础上，按 subject_id 做 train/val 切分。
    返回：train_subset, val_subset

    关键点：
    - train_set.indices 是“真实 H5 index”（已过滤 bad）
    - Subset 需要的是“dataset 内部位置 idx”（即 0..len(train_set)-1 对应 indices 的位置）
    """
    if val_subjects is None or len(val_subjects) == 0:
        raise ValueError("VAL_SUBJECTS is empty. Please set Config.VAL_SUBJECTS, e.g., [21, 23].")

    # train_set.indices: kept 的真实 H5 index（N_kept）
    kept_real_indices = train_set.indices
    kept_subjects = train_set.subject_ids[kept_real_indices]  # (N_kept,)

    val_subjects = np.array(val_subjects, dtype=np.int32)
    is_val = np.isin(kept_subjects, val_subjects)

    # 这些是“train_set 内部位置索引”
    val_pos = np.nonzero(is_val)[0].astype(np.int64)
    train_pos = np.nonzero(~is_val)[0].astype(np.int64)

    if len(val_pos) == 0:
        unique_kept = np.unique(kept_subjects)
        raise RuntimeError(
            f"[VAL SPLIT ERROR] No samples found for VAL_SUBJECTS={val_subjects.tolist()} in train.h5 after QC.\n"
            f"  Unique subjects in kept train set: {unique_kept.tolist()}"
        )

    train_subset = Subset(train_set, train_pos.tolist())
    val_subset = Subset(train_set, val_pos.tolist())
    return train_subset, val_subset


# =========================
# Meta info (saved to ckpt)
# =========================
@dataclass
class TrainMeta:
    # experiment
    exp_name: str
    exp_tag: str
    alpha: float
    beta: float
    gamma: float
    seed: int
    device: str
    timestamp: str

    # data
    train_h5: str
    test_h5: str
    train_bad_indices_path: Optional[str]
    test_bad_indices_path: Optional[str]
    val_subjects: List[int]
    n_train: int
    n_val: int
    n_test: int

    # model & training
    in_channels: int
    base_channels: int
    input_len: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int

    # loss/STFT/anchor config (optional but recommended)
    fs: int
    fft_sizes: List[int]
    hop_sizes: List[int]
    win_lengths: List[int]
    stft_fmin: float
    stft_fmax: float
    stft_use_band: bool
    anchor_from_logits: bool
    anchor_pos_weight: float


def train():
    # --------------------------
    # 0) CLI
    # --------------------------
    parser = argparse.ArgumentParser(description="Train BeatAware R-M2Net (subject-wise val split)")
    parser.add_argument('--alpha', type=float, default=Config.ALPHA, help='STFT loss weight')
    parser.add_argument('--beta', type=float, default=Config.BETA, help='Anchor loss weight')
    parser.add_argument('--gamma', type=float, default=Config.GAMMA, help='Smooth loss weight')
    parser.add_argument('--exp_tag', type=str, default="Default", help='Tag for this experiment')
    args = parser.parse_args()

    # --------------------------
    # 1) Update Config + dirs
    # --------------------------
    new_name = f"Exp_a{args.alpha}_b{args.beta}_g{args.gamma}_{args.exp_tag}"
    Config.ALPHA = float(args.alpha)
    Config.BETA = float(args.beta)
    Config.GAMMA = float(args.gamma)
    Config.update_paths(new_name)
    Config.makedirs()

    # seed + logger
    seed_everything(Config.SEED)
    logger = setup_logger(Config.LOG_DIR, name="train")
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, 'tensorboard'))

    logger.info(f"🚀 Experiment Started: {Config.EXP_NAME}")
    logger.info(f"📊 Hyperparams: Alpha={Config.ALPHA}, Beta={Config.BETA}, Gamma={Config.GAMMA}")
    logger.info(f"📂 Data Directory: {Config.DATA_DIR}")
    logger.info(f"💻 Device: {Config.DEVICE}")

    # --------------------------
    # 2) QC paths
    # --------------------------
    train_bad = getattr(Config, "TRAIN_BAD_INDICES_PATH", None)
    test_bad = getattr(Config, "TEST_BAD_INDICES_PATH", None)

    if train_bad is not None and not os.path.exists(train_bad):
        logger.warning(f"[QC] TRAIN_BAD_INDICES_PATH not found, disable: {train_bad}")
        train_bad = None
    if test_bad is not None and not os.path.exists(test_bad):
        logger.warning(f"[QC] TEST_BAD_INDICES_PATH not found, disable: {test_bad}")
        test_bad = None

    logger.info(f"[QC] TRAIN_BAD_INDICES_PATH = {train_bad if train_bad else 'None (disabled)'}")
    logger.info(f"[QC] TEST_BAD_INDICES_PATH  = {test_bad if test_bad else 'None (disabled)'}")

    # --------------------------
    # 3) Datasets + subject-wise val split
    # --------------------------
    full_train_set = RadarDataset(Config.TRAIN_H5, bad_indices_path=train_bad)

    val_subjects = getattr(Config, "VAL_SUBJECTS", None)
    if val_subjects is None:
        raise ValueError("Config.VAL_SUBJECTS not found. Please set e.g., VAL_SUBJECTS = [21, 23] in config.py")
    logger.info(f"[VAL] VAL_SUBJECTS = {val_subjects}")

    train_subset, val_subset = _build_subject_split_subsets(full_train_set, val_subjects)

    # test set: 只用于最终评估，不参与 early-stopping
    test_set = RadarDataset(Config.TEST_H5, bad_indices_path=test_bad)

    logger.info(f"[DATA] Train subset: {len(train_subset)} | Val subset: {len(val_subset)} | Test set: {len(test_set)}")

    # DataLoaders
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
    
    # --------------------------
    # 4) Build model + loss + optimizer
    # --------------------------
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

    # --------------------------
    # 5) (2.3) Build meta AFTER logger init (and save a copy to logs)
    # --------------------------
    meta = TrainMeta(
        exp_name=Config.EXP_NAME,
        exp_tag=str(args.exp_tag),
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        seed=int(Config.SEED),
        device=str(Config.DEVICE),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),

        train_h5=str(Config.TRAIN_H5),
        test_h5=str(Config.TEST_H5),
        train_bad_indices_path=train_bad,
        test_bad_indices_path=test_bad,
        val_subjects=[int(x) for x in val_subjects],
        n_train=int(len(train_subset)),
        n_val=int(len(val_subset)),
        n_test=int(len(test_set)),

        in_channels=int(Config.IN_CHANNELS),
        base_channels=int(Config.BASE_CHANNELS),
        input_len=int(Config.INPUT_LEN),
        batch_size=int(Config.BATCH_SIZE),
        epochs=int(Config.EPOCHS),
        lr=float(Config.LEARNING_RATE),
        weight_decay=float(Config.WEIGHT_DECAY),
        patience=int(Config.PATIENCE),

        fs=int(Config.FS),
        fft_sizes=list(Config.FFT_SIZES),
        hop_sizes=list(Config.HOP_SIZES),
        win_lengths=list(Config.WIN_LENGTHS),
        stft_fmin=float(Config.STFT_FMIN),
        stft_fmax=float(Config.STFT_FMAX),
        stft_use_band=bool(Config.STFT_USE_BAND),
        anchor_from_logits=bool(Config.ANCHOR_FROM_LOGITS),
        anchor_pos_weight=float(Config.ANCHOR_POS_WEIGHT),
    )

    # save meta.json (human readable) — optional but strongly recommended
    meta_path = os.path.join(Config.LOG_DIR, "train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)
    logger.info(f"[META] saved to: {meta_path}")
    logger.info("=" * 80)
    logger.info("[META] Snapshot")
    logger.info(f"[META] EXP_NAME          : {meta.exp_name}")
    logger.info(f"[META] EXP_TAG           : {meta.exp_tag}")
    logger.info(f"[META] ALPHA/BETA/GAMMA  : {meta.alpha}/{meta.beta}/{meta.gamma}")
    logger.info(f"[META] VAL_SUBJECTS      : {meta.val_subjects}")
    logger.info(f"[META] N_train/N_val/N_test : {meta.n_train}/{meta.n_val}/{meta.n_test}")
    logger.info(f"[META] TRAIN_H5          : {meta.train_h5}")
    logger.info(f"[META] TEST_H5           : {meta.test_h5}")
    logger.info(f"[META] Train bad idx     : {meta.train_bad_indices_path}")
    logger.info(f"[META] Test  bad idx     : {meta.test_bad_indices_path}")
    logger.info("=" * 80)


    # --------------------------
    # 6) Resume / Early stopping
    # --------------------------
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
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        tmp = checkpoint.get("best_val_loss", None)
        best_val_loss = float(tmp) if tmp is not None else float("inf")
        epochs_no_improve = int(checkpoint.get('epochs_no_improve', 0))

        logger.info(f"Resumed from Epoch {start_epoch}. Best Val Loss so far: {best_val_loss:.6f}")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

    # --------------------------
    # 7) Training loop
    # --------------------------
    for epoch in range(start_epoch, Config.EPOCHS):
        # -------- Train --------
        model.train()
        train_loss_avg = 0.0
        train_L1_avg, train_STFT_avg, train_Anchor_avg, train_Smooth_avg = 0.0, 0.0, 0.0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        for i, (radar, ecg, mask, subject_id) in enumerate(loop):
            global_step = epoch * max(len(train_loader), 1) + i

            radar = radar.to(Config.DEVICE, non_blocking=True)
            ecg = ecg.to(Config.DEVICE, non_blocking=True)
            mask = mask.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_ecg, pred_mask = model(radar)
            loss, l_time, l_freq, l_anchor, l_smooth = criterion(pred_ecg, ecg, pred_mask, mask)

            loss.backward()
            optimizer.step()

            train_loss_avg += float(loss.item())
            train_L1_avg += float(l_time.item())
            train_STFT_avg += float(l_freq.item())
            train_Anchor_avg += float(l_anchor.item())
            train_Smooth_avg += float(l_smooth.item())

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                L1=f"{l_time.item():.4f}",
                STFT=f"{l_freq.item():.4f}",
                Anchor=f"{l_anchor.item():.4f}",
                Smooth=f"{l_smooth.item():.4f}"
            )

            if i % 10 == 0:
                writer.add_scalar('Loss/Train_Total', loss.item(), global_step)
                writer.add_scalar('Loss/Train_L1', l_time.item(), global_step)
                writer.add_scalar('Loss/Train_STFT', l_freq.item(), global_step)
                writer.add_scalar('Loss/Train_Anchor', l_anchor.item(), global_step)
                writer.add_scalar('Loss/Train_Smooth', l_smooth.item(), global_step)

        denom_tr = max(len(train_loader), 1)
        train_loss_avg /= denom_tr
        train_L1_avg /= denom_tr
        train_STFT_avg /= denom_tr
        train_Anchor_avg /= denom_tr
        train_Smooth_avg /= denom_tr

        # -------- Val --------
        model.eval()
        val_loss_avg = 0.0
        val_L1_avg, val_STFT_avg, val_Anchor_avg, val_Smooth_avg = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for radar, ecg, mask, subject_id in val_loader:
                radar = radar.to(Config.DEVICE, non_blocking=True)
                ecg = ecg.to(Config.DEVICE, non_blocking=True)
                mask = mask.to(Config.DEVICE, non_blocking=True)

                pred_ecg, pred_mask = model(radar)
                loss, l_time, l_freq, l_anchor, l_smooth = criterion(pred_ecg, ecg, pred_mask, mask)

                val_loss_avg += float(loss.item())
                val_L1_avg += float(l_time.item())
                val_STFT_avg += float(l_freq.item())
                val_Anchor_avg += float(l_anchor.item())
                val_Smooth_avg += float(l_smooth.item())

        denom_v = max(len(val_loader), 1)
        val_loss_avg /= denom_v
        val_L1_avg /= denom_v
        val_STFT_avg /= denom_v
        val_Anchor_avg /= denom_v
        val_Smooth_avg /= denom_v

        # TensorBoard epoch-level
        writer.add_scalar('Loss/Val_Total', val_loss_avg, epoch)
        writer.add_scalar('Loss/Val_L1', val_L1_avg, epoch)
        writer.add_scalar('Loss/Val_STFT', val_STFT_avg, epoch)
        writer.add_scalar('Loss/Val_Anchor', val_Anchor_avg, epoch)
        writer.add_scalar('Loss/Val_Smooth', val_Smooth_avg, epoch)

        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Train Total: {train_loss_avg:.4f} "
            f"(L1: {train_L1_avg:.4f}, STFT: {train_STFT_avg:.4f}, Anchor: {train_Anchor_avg:.4f}, Smooth: {train_Smooth_avg:.4f}) | "
            f"Val Total: {val_loss_avg:.4f} "
            f"(L1: {val_L1_avg:.4f}, STFT: {val_STFT_avg:.4f}, Anchor: {val_Anchor_avg:.4f}, Smooth: {val_Smooth_avg:.4f})"
        )

        # --------------------------
        # 8) Save checkpoints (last + best) WITH meta  ✅(2.3)
        # --------------------------
        last_checkpoint = {
            "meta": asdict(meta),  # ✅ save meta inside checkpoint
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss_avg),
            "best_val_loss": float(best_val_loss),
            "epochs_no_improve": int(epochs_no_improve),
        }
        torch.save(last_checkpoint, last_ckpt_path)
        logger.info(f"Last checkpoint saved to {last_ckpt_path}")

        if val_loss_avg < best_val_loss:
            best_val_loss = float(val_loss_avg)
            epochs_no_improve = 0

            best_checkpoint = {
                "meta": asdict(meta),  # ✅ save meta inside checkpoint
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_loss_avg),
                "best_val_loss": float(best_val_loss),
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

    # 说明：
    # - test_loader 不参与 early-stopping；
    # - 训练结束后建议直接跑 test.py 输出全套论文级结果（CSV/JSON/cases）。


if __name__ == "__main__":
    train()
