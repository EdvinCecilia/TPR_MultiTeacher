"""
Losses for Task-Performance Routing (TPR).

The objective follows the paper:
    L = lambda_seg * L_seg
      + lambda_align * L_align
      + lambda_logits * L_logits
      + lambda_balance * L_balance
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


class TPRLoss(nn.Module):
    """Paper-aligned multi-component loss for TPR."""

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_align: float = 0.2,
        lambda_logits: float = 0.2,
        lambda_balance: float = 0.1,
        balance_temperature: float = 0.4,
        distillation_temperature: float = 1.0,
        num_classes: int = 14,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_align = lambda_align
        self.lambda_logits = lambda_logits
        self.lambda_balance = lambda_balance
        self.balance_temperature = balance_temperature
        self.distillation_temperature = distillation_temperature
        self.num_classes = num_classes
        self.seg_loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )

    @staticmethod
    def _region_average_5d(
        tensor: torch.Tensor,
        region_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Average [B, C, D, H, W] into [B, N_regions, C]."""
        pooled = F.adaptive_avg_pool3d(tensor, output_size=region_grid)
        return pooled.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_channel_first_5d(tensor: torch.Tensor) -> torch.Tensor:
        """Convert channel-last 5D features to channel-first when needed."""
        if tensor.dim() == 5 and tensor.shape[1] not in (48, 96, 192, 384, 768):
            return tensor.permute(0, 4, 1, 2, 3).contiguous()
        return tensor

    def _feature_alignment_loss(
        self,
        student_feats: dict[str, torch.Tensor],
        teacher_feats: list[dict[str, torch.Tensor]],
        routing_weights: dict,
    ) -> torch.Tensor:
        """Align region-averaged student features to each teacher using TPR weights."""
        losses = []
        device = next(iter(student_feats.values())).device

        for stage_name, routing in routing_weights.items():
            if stage_name == "logits" or stage_name not in student_feats:
                continue
            if any(stage_name not in feats for feats in teacher_feats):
                continue

            student_feat = self._to_channel_first_5d(student_feats[stage_name])
            if student_feat.dim() != 5:
                continue

            region_grid = routing["region_grid"]
            weights = routing["weights"].detach()
            student_regions = self._region_average_5d(student_feat, region_grid)

            stage_loss = 0.0
            for teacher_idx, feats in enumerate(teacher_feats):
                teacher_feat = self._to_channel_first_5d(feats[stage_name].detach())
                teacher_regions = self._region_average_5d(teacher_feat, region_grid)
                per_region = (student_regions - teacher_regions).pow(2).mean(dim=-1)
                stage_loss = stage_loss + (weights[:, :, teacher_idx] * per_region).mean()
            losses.append(stage_loss)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).sum()

    def _logits_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: list[torch.Tensor],
        routing_weights: dict,
    ) -> torch.Tensor:
        """Compute KL distillation on region-averaged logits."""
        routing = routing_weights.get("logits")
        if routing is None:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        region_grid = routing["region_grid"]
        weights = routing["weights"].detach()
        temperature = self.distillation_temperature

        student_regions = self._region_average_5d(student_logits, region_grid)
        student_log_probs = F.log_softmax(student_regions / temperature, dim=-1)

        loss = 0.0
        for teacher_idx, logits in enumerate(teacher_logits):
            teacher_regions = self._region_average_5d(logits.detach(), region_grid)
            teacher_probs = F.softmax(teacher_regions / temperature, dim=-1)
            kl_per_region = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)
            loss = loss + (weights[:, :, teacher_idx] * kl_per_region).mean()

        return (temperature**2) * loss

    def _balance_loss(self, routing_weights: dict) -> torch.Tensor:
        """Encourage non-collapsed average routing usage via entropy."""
        all_weights = []
        for routing in routing_weights.values():
            if "weights" in routing:
                all_weights.append(routing["weights"].reshape(-1, routing["weights"].shape[-1]))

        if not all_weights:
            return torch.tensor(0.0, requires_grad=True)

        weights = torch.cat(all_weights, dim=0)
        avg_weights = weights.mean(dim=0)
        entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum()
        return torch.exp(-entropy / self.balance_temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        student_feats: dict[str, torch.Tensor],
        teacher1_feats: dict[str, torch.Tensor],
        teacher2_feats: dict[str, torch.Tensor],
        routing_weights: dict,
        labels: torch.Tensor,
        teacher1_logits: torch.Tensor | None = None,
        teacher2_logits: torch.Tensor | None = None,
        mixed_teacher_feats: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        """Return total loss and individual loss terms."""
        if labels.dim() == 4:
            labels_for_loss = labels.unsqueeze(1).long()
        else:
            labels_for_loss = labels.long()

        seg_loss = self.seg_loss_fn(student_logits, labels_for_loss)
        align_loss = self._feature_alignment_loss(
            student_feats=student_feats,
            teacher_feats=[teacher1_feats, teacher2_feats],
            routing_weights=routing_weights,
        )

        if teacher1_logits is not None and teacher2_logits is not None:
            logits_loss = self._logits_distillation_loss(
                student_logits=student_logits,
                teacher_logits=[teacher1_logits, teacher2_logits],
                routing_weights=routing_weights,
            )
        else:
            logits_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        balance_loss = self._balance_loss(routing_weights).to(student_logits.device)
        total_loss = (
            self.lambda_seg * seg_loss
            + self.lambda_align * align_loss
            + self.lambda_logits * logits_loss
            + self.lambda_balance * balance_loss
        )

        return {
            "total": total_loss,
            "seg": seg_loss,
            "align": align_loss,
            "logits": logits_loss,
            "balance": balance_loss,
        }


__all__ = ["TPRLoss"]
