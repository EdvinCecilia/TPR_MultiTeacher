"""
TPR training script: task-performance-based routing for multi-teacher knowledge distillation.
Uses two full teachers (with decoder) for distillation.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from dataset_pancreas import get_pancreas_dataloaders
from eval_metrics import compute_dice
from model.student import StudentModel, load_student_from_checkpoint
from model.teachers import load_teacher_full_models
from model.tpr_routing import TPR
from loss.tpr_loss import TPRLoss


class EMAModel:
    """Shadow copy for Exponential Moving Average of student weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = None

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self):
        if self.backup is None:
            return
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None


def format_time(seconds):
    """Format seconds as human-readable time."""
    if seconds < 0:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class WarmupCosineSchedule:
    """Warmup cosine learning rate scheduler"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, current_step):
        if current_step < self.warmup_steps:
            lr = self.base_lr * current_step / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_one_epoch(
    student,
    teacher_1,
    teacher_2,
    tpr,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    scaler,
    print_freq=10,
    use_logits_distillation=True,
):
    """Train one epoch."""
    student.train()
    tpr.train()
    teacher_1.eval()
    teacher_2.eval()

    total_loss = 0.0
    total_seg = 0.0
    total_align = 0.0
    total_logits = 0.0
    total_balance = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch} [Train]",
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Teacher forward (frozen): encoder feats + decoder logits
        with torch.no_grad():
            teacher1_feats, teacher1_logits = teacher_1(images, return_logits=True)
            teacher2_feats, teacher2_logits = teacher_2(images, return_logits=True)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            if isinstance(student, DDP):
                student_logits, _ = student(images, return_features=True)
                student_model = student.model_ref if hasattr(student, 'model_ref') else student.module
            else:
                student_logits, _ = student(images, return_features=True)
                student_model = student

            student_feats_dict = student_model.get_projected_features()

            # TPR: task-performance-based routing and feature mixing
            mixed_teacher_feats, routing_weights = tpr(
                student_logits=student_logits,
                teacher1_logits=teacher1_logits,
                teacher2_logits=teacher2_logits,
                labels=labels,
                teacher1_feats=teacher1_feats,
                teacher2_feats=teacher2_feats,
            )

            losses = criterion(
                student_logits=student_logits,
                student_feats=student_feats_dict,
                teacher1_feats=teacher1_feats,
                teacher2_feats=teacher2_feats,
                routing_weights=routing_weights,
                labels=labels,
                teacher1_logits=teacher1_logits if use_logits_distillation else None,
                teacher2_logits=teacher2_logits if use_logits_distillation else None,
                mixed_teacher_feats=mixed_teacher_feats,
            )

            loss = losses['total']

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN loss at batch {batch_idx}, skipping...")
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        student_params = list(student.module.parameters()) if isinstance(student, DDP) else list(student.parameters())
        tpr_params = list(tpr.module.parameters()) if isinstance(tpr, DDP) else list(tpr.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student_params + tpr_params,
            max_norm=0.5
        )

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            optimizer.zero_grad()
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_seg += losses['seg'].item()
        total_align += losses['align'].item()
        total_logits += losses['logits'].item()
        total_balance += losses['balance'].item()
        num_batches += 1

        if batch_idx % print_freq == 0:
            first_stage = list(routing_weights.keys())[0]
            routing = routing_weights[first_stage]
            weights = routing.get('weights')
            hard_region_mask = routing.get('hard_region_mask')
            avg_pi_1 = weights[:, :, 0].mean().item() if weights is not None else 0.5
            hard_ratio = hard_region_mask.float().mean().item() if hard_region_mask is not None else 0.0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "seg": f"{losses['seg'].item():.4f}",
                "align": f"{losses['align'].item():.4f}",
                "logits": f"{losses['logits'].item():.4f}",
                "pi1": f"{avg_pi_1:.2f}",
                "hard": f"{hard_ratio:.2f}",
            })

    pbar.close()
    avg_loss = total_loss / max(num_batches, 1)
    avg_seg = total_seg / max(num_batches, 1)
    avg_align = total_align / max(num_batches, 1)
    avg_logits = total_logits / max(num_batches, 1)
    avg_balance = total_balance / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "seg_loss": avg_seg,
        "align_loss": avg_align,
        "logits_loss": avg_logits,
        "balance_loss": avg_balance,
    }


def validate(
    student,
    val_loader,
    criterion,
    device,
    epoch,
    num_classes=14,
    roi_size=(96, 96, 96),
):
    """Validation."""
    student.eval()

    def predictor(x):
        with torch.no_grad():
            logits, _ = student(x, return_features=True)
        return logits

    dice_scores = {i: [] for i in range(num_classes)}
    total_loss = 0.0
    num_batches = 0

    from monai.losses import DiceCELoss
    val_criterion = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch} [Val]",
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=predictor,
                overlap=0.75,
            )

            if labels.dim() == 4:
                labels_for_loss = labels.unsqueeze(1).long()
            else:
                labels_for_loss = labels.long()

            loss = val_criterion(logits, labels_for_loss)
            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.squeeze(1).cpu().numpy() if labels.dim() == 5 else labels.cpu().numpy()

            for b in range(preds.shape[0]):
                dice_dict = compute_dice(preds[b], labels_np[b], num_classes)
                for cls_id, dice_val in dice_dict.items():
                    dice_scores[cls_id].append(dice_val)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{np.mean([sum(scores)/len(scores) for scores in dice_scores.values() if len(scores) > 0]):.4f}",
            })

    pbar.close()
    avg_dice = {}
    for cls_id, scores in dice_scores.items():
        if len(scores) > 0:
            avg_dice[cls_id] = np.mean(scores)
        else:
            avg_dice[cls_id] = 0.0
    
    organ_dice_values = [avg_dice[i] for i in range(1, num_classes)]
    mean_dice = np.mean(organ_dice_values) if len(organ_dice_values) > 0 else 0.0

    avg_loss = total_loss / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "mean_dice": mean_dice,
        "dice_per_class": avg_dice,
    }


def main():
    parser = argparse.ArgumentParser(description="TPR training (task-performance-based routing for multi-teacher)")

    parser.add_argument("--data-dir", type=str, default="/path/to/your/dataset", help="Dataset root directory")
    parser.add_argument("--teacher-1-path", type=str, default="/path/to/teacher1/best_model.pth", help="Teacher 1 checkpoint (e.g. SuPreM)")
    parser.add_argument("--teacher-2-path", type=str, default="/path/to/teacher2/best_model.pth", help="Teacher 2 checkpoint (e.g. VoCo)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--roi-size", type=int, nargs=3, default=[96, 96, 96], help="ROI size (D H W)")

    parser.add_argument("--student-feature-size", type=int, default=48, help="Student feature dim (match teacher)")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes (incl. background)")
    parser.add_argument("--hard-region-threshold", type=float, default=0.5, help="Hard region threshold for TPR routing")
    parser.add_argument("--routing-temperature", type=float, default=0.7, help="Routing softmax temperature")

    parser.add_argument("--epochs", type=int, default=5000, help="Total epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=20, help="Warmup epochs")
    parser.add_argument("--num-samples", type=int, default=4, help="Samples per image")
    parser.add_argument("--pos", type=int, default=9, help="Positive samples per crop")
    parser.add_argument("--neg", type=int, default=1, help="Negative samples per crop")
    parser.add_argument("--flip-prob", type=float, default=0.2, help="Flip augmentation prob")
    parser.add_argument("--rotate-prob", type=float, default=0.2, help="Rotate90 augmentation prob")

    parser.add_argument("--lambda-seg", type=float, default=1.0, help="Segmentation loss weight")
    parser.add_argument("--lambda-align", type=float, default=0.2, help="Feature alignment loss weight")
    parser.add_argument("--lambda-logits", type=float, default=0.2, help="Logits distillation weight")
    parser.add_argument("--lambda-balance", type=float, default=0.1, help="Load balance loss weight")
    parser.add_argument("--balance-temperature", type=float, default=0.4, help="Balance loss temperature")
    parser.add_argument("--use-logits-distillation", action="store_true", default=True, help="Use decoder logits distillation")
    parser.add_argument("--distill-stop-epoch", type=int, default=1200, help="Epoch after which to turn off distillation/align/balance")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA of student for validation")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay")

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (single-GPU)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_tpr", help="Checkpoint save directory")
    parser.add_argument("--val-interval", type=int, default=10, help="Validation interval (epochs)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (torchrun)")
    parser.add_argument("--world-size", type=int, default=1, help="World size")
    parser.add_argument("--rank", type=int, default=0, help="Rank")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        is_distributed = True
        is_main_process = (args.rank == 0)
    else:
        is_distributed = False
        is_main_process = True
        if torch.cuda.is_available():
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if visible_devices is not None:
                device = torch.device("cuda:0")
                print(f"CUDA_VISIBLE_DEVICES={visible_devices}, using logical GPU 0")
            else:
                device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cpu")
    
    if is_main_process:
        print(f"Device: {device}")
        if is_distributed:
            print(f"Distributed: world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}")

    set_determinism(seed=args.seed)
    if args.resume:
        checkpoint_path = Path(args.resume)
        save_dir = checkpoint_path.parent
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume}")
            print(f"Save dir: {save_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir_with_timestamp = f"{args.save_dir}_{timestamp}"
        save_dir = Path(save_dir_with_timestamp)
        if is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Save dir: {save_dir}")

    if is_main_process and not args.resume:
        config_file = save_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(vars(args), f, indent=2)

    history_file = save_dir / "training_history.json"
    if args.resume and history_file.exists():
        if is_main_process:
            print("Loading training history...")
        with open(history_file, 'r') as f:
            training_history = json.load(f)
    else:
        training_history = {
            "train_loss": [],
            "train_seg_loss": [],
            "train_align_loss": [],
            "train_logits_loss": [],
            "train_balance_loss": [],
            "val_loss": [],
            "val_mean_dice": [],
            "learning_rate": [],
            "epochs": [],
        }

    if is_main_process:
        print("Loading dataset...")
    train_loader, val_loader = get_pancreas_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        roi_size=tuple(args.roi_size),
        space_z=2.0,
        num_samples=args.num_samples,
        pos=args.pos,
        neg=args.neg,
        flip_prob=args.flip_prob,
        rotate_prob=args.rotate_prob,
        is_distributed=is_distributed,
        rank=args.rank,
        world_size=args.world_size,
    )
    if is_main_process:
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        if is_distributed:
            print(f"Batches per GPU: {len(train_loader)}")

    if is_main_process:
        print("Loading teacher models...")
    teacher_1, teacher_2 = load_teacher_full_models(
        args.teacher_1_path,
        args.teacher_2_path,
        device=str(device),
        num_classes=args.num_classes,
    )
    if is_main_process:
        print("  Teachers loaded.")

    if is_main_process:
        print("Creating student model (same architecture as teacher)...")
    student_model = StudentModel(
        in_channels=1,
        out_channels=args.num_classes,
        feature_size=args.student_feature_size,  # 48: same as teacher
        use_v2=True,
    ).to(device)
    if is_main_process:
        print("Initializing student from Teacher 2 (VoCo) weights.")
    load_student_from_checkpoint(student_model, args.teacher_2_path)
    
    if is_main_process:
        print("Verifying student init...")
        with torch.no_grad():
            test_input = torch.randn(1, 1, 96, 96, 96).to(device)
            student_model.eval()
            test_output, _ = student_model(test_input, return_features=True)
            print(f"  Student output shape: {test_output.shape}")
    if is_main_process:
        print("Creating TPR routing module...")
    tpr_model = TPR(
        num_classes=args.num_classes,
        hard_region_threshold=args.hard_region_threshold,
        epsilon=1e-6,
        temperature=args.routing_temperature,
    ).to(device)

    if is_distributed:
        student = DDP(student_model, device_ids=[args.local_rank], find_unused_parameters=False)
        tpr = DDP(tpr_model, device_ids=[args.local_rank], find_unused_parameters=False)
        student.model_ref = student_model
        tpr.model_ref = tpr_model
    else:
        student = student_model
        tpr = tpr_model

    if is_main_process:
        student_stats = student_model if is_distributed else student
        tpr_stats = tpr_model if is_distributed else tpr
        student_params = sum(p.numel() for p in student_stats.parameters())
        tpr_params = sum(p.numel() for p in tpr_stats.parameters())
        trainable_params = sum(p.numel() for p in student_stats.parameters() if p.requires_grad)
        trainable_params += sum(p.numel() for p in tpr_stats.parameters() if p.requires_grad)
        print(f"Student params: {student_params:,}, TPR params: {tpr_params:,}, Trainable: {trainable_params:,}")

    criterion = TPRLoss(
        lambda_seg=args.lambda_seg,
        lambda_align=args.lambda_align,
        lambda_logits=args.lambda_logits,
        lambda_balance=args.lambda_balance,
        balance_temperature=args.balance_temperature,
        num_classes=args.num_classes,
    )
    ema_model = EMAModel(student_model if not is_distributed else student.module,
                         decay=args.ema_decay) if args.use_ema else None

    if is_distributed:
        student_params = list(student.module.parameters())
        tpr_params = list(tpr.module.parameters())
    else:
        student_params = list(student.parameters())
        tpr_params = list(tpr.parameters())
    
    optimizer = AdamW(
        student_params + tpr_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    scaler = GradScaler(device='cuda')
    start_epoch = 1
    if args.resume:
        if is_main_process:
            print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        if is_distributed:
            student.module.load_state_dict(checkpoint['student_state_dict'])
            tpr.module.load_state_dict(checkpoint['tpr_state_dict'])
        else:
            student.load_state_dict(checkpoint['student_state_dict'])
            tpr.load_state_dict(checkpoint['tpr_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if ema_model is not None and 'ema_shadow' in checkpoint:
            ema_model.shadow = checkpoint['ema_shadow']
            if is_main_process:
                print("  EMA state restored.")
        
        start_epoch = checkpoint['epoch'] + 1
        best_dice_from_ckpt = checkpoint.get('best_dice', 0.0)
        if best_dice_from_ckpt == 0.0 and 'best_dice' in training_history:
            best_dice_from_ckpt = training_history.get('best_dice', 0.0)
        
        if is_main_process:
            print(f"  Checkpoint loaded. Resuming from epoch {start_epoch}")
            if best_dice_from_ckpt > 0:
                print(f"  Previous best Dice: {best_dice_from_ckpt:.4f}")
    else:
        best_dice_from_ckpt = 0.0

    if is_main_process:
        print("\nStarting training (TPR multi-teacher routing)...")
        if is_distributed:
            print(f"Distributed: {args.world_size} GPUs")
        if args.resume:
            print(f"Epoch {start_epoch} -> {args.epochs}")
    best_dice = best_dice_from_ckpt
    last_val_metrics = None
    global_step = (start_epoch - 1) * len(train_loader)

    epoch_times = []
    start_time_total = time.time()

    if not args.resume:
        if is_main_process:
            print("\nValidation at init (Epoch 0)...")
        val_metrics_init = validate(
            student=student,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=0,
            num_classes=args.num_classes,
            roi_size=tuple(args.roi_size),
        )
        if is_main_process:
            print(f"  Epoch 0 - Val Dice: {val_metrics_init['mean_dice']:.4f}")
            if val_metrics_init['mean_dice'] < 0.5:
                print("  Warning: Low init Dice; check teacher paths and model structure.")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        if is_distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if epoch >= args.distill_stop_epoch:
            criterion.lambda_align = 0.0
            criterion.lambda_logits = 0.0
            criterion.lambda_balance = 0.0
            if is_main_process and epoch == args.distill_stop_epoch:
                print(f"[Phase-2] Epoch {epoch}: distillation/align/balance off, supervision only.")
        else:
            criterion.lambda_align = args.lambda_align
            criterion.lambda_logits = args.lambda_logits
            criterion.lambda_balance = args.lambda_balance

        train_metrics = train_one_epoch(
            student=student,
            teacher_1=teacher_1,
            teacher_2=teacher_2,
            tpr=tpr,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_logits_distillation=args.use_logits_distillation,
        )

        if ema_model is not None:
            ema_model.update()

        global_step += len(train_loader)
        current_lr = scheduler.step(global_step)

        training_history["epochs"].append(epoch)
        training_history["train_loss"].append(float(train_metrics['loss']))
        training_history["train_seg_loss"].append(float(train_metrics['seg_loss']))
        training_history["train_align_loss"].append(float(train_metrics['align_loss']))
        training_history["train_logits_loss"].append(float(train_metrics['logits_loss']))
        training_history["train_balance_loss"].append(float(train_metrics['balance_loss']))
        training_history["learning_rate"].append(float(current_lr))

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        avg_epoch_time = np.mean(epoch_times[-10:]) if len(epoch_times) > 0 else epoch_time
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        if is_main_process:
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Seg: {train_metrics['seg_loss']:.4f}, "
                  f"Align: {train_metrics['align_loss']:.4f}, "
                  f"Logits: {train_metrics['logits_loss']:.4f}, "
                  f"Balance: {train_metrics['balance_loss']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Time: {format_time(epoch_time)} | Avg: {format_time(avg_epoch_time)} | ETA: {format_time(eta_seconds)}")

        if epoch % 50 == 0:
            torch.cuda.empty_cache()
            if is_main_process:
                print("  [GPU cache cleared]")

        if epoch == 1 or epoch % args.val_interval == 0 or epoch == args.epochs:
            if ema_model is not None:
                ema_model.apply_shadow()
            val_metrics = validate(
                student=student,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_classes=args.num_classes,
                roi_size=tuple(args.roi_size),
            )
            if ema_model is not None:
                ema_model.restore()
            last_val_metrics = val_metrics

            training_history["val_loss"].append(float(val_metrics['loss']))
            training_history["val_mean_dice"].append(float(val_metrics['mean_dice']))

            if is_main_process:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Mean Dice: {val_metrics['mean_dice']:.4f}")

            if is_main_process and val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                best_path = save_dir / "best_model.pth"
                student_state = student.module.state_dict() if is_distributed else student.state_dict()
                tpr_state = tpr.module.state_dict() if is_distributed else tpr.state_dict()
                save_dict = {
                    "epoch": epoch,
                    "student_state_dict": student_state,
                    "tpr_state_dict": tpr_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_dice": best_dice,
                    "val_metrics": val_metrics,
                }
                if ema_model is not None:
                    save_dict["ema_shadow"] = ema_model.shadow
                torch.save(save_dict, best_path)
                print(f"  Saved best model (Dice: {best_dice:.4f})")
        else:
            training_history["val_loss"].append(None)
            training_history["val_mean_dice"].append(None)

        if is_main_process:
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=2)

        if is_main_process:
            last_path = save_dir / "last_model.pth"
            student_state = student.module.state_dict() if is_distributed else student.state_dict()
            tpr_state = tpr.module.state_dict() if is_distributed else tpr.state_dict()
            save_dict = {
                "epoch": epoch,
                "student_state_dict": student_state,
                "tpr_state_dict": tpr_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": last_val_metrics,
            }
            if ema_model is not None:
                save_dict["ema_shadow"] = ema_model.shadow
            torch.save(save_dict, last_path)
    if is_main_process:
        total_time = time.time() - start_time_total
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0.0

        training_history["total_training_time_seconds"] = float(total_time)
        training_history["total_training_time_formatted"] = format_time(total_time)
        training_history["average_epoch_time_seconds"] = float(avg_epoch_time)
        training_history["average_epoch_time_formatted"] = format_time(avg_epoch_time)
        training_history["best_dice"] = float(best_dice)
        training_history["total_epochs"] = args.epochs

        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)

        print(f"\nTraining done. Best Dice: {best_dice:.4f}")
        print(f"  Total time: {format_time(total_time)}, Avg/epoch: {format_time(avg_epoch_time) if epoch_times else 'N/A'}")
        print(f"  History: {history_file}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
