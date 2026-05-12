"""
Task-Performance Routing (TPR) core module.

Paper-aligned behavior:
- Region-wise routing is driven by prediction error.
- Hard regions use performance-based teacher selection.
- Easy regions use uniform teacher weights.
- Hierarchical routing follows stage-wise region grids.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskPerformanceRouter(nn.Module):
    """Compute region-wise teacher weights from student and teacher errors."""

    def __init__(
        self,
        num_classes: int = 14,
        hard_region_threshold: float = 0.5,
        temperature: float = 0.7,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hard_region_threshold = hard_region_threshold
        self.temperature = temperature
        self.epsilon = epsilon

    @staticmethod
    def _pool_to_regions(values: torch.Tensor, region_grid: tuple[int, int, int]) -> torch.Tensor:
        """Average [B, D, H, W] into [B, N_regions]."""
        pooled = F.adaptive_avg_pool3d(values.unsqueeze(1), output_size=region_grid)
        return pooled.flatten(1)

    def prediction_error(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        region_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute region-wise cross-entropy error from logits and ground truth."""
        if labels.dim() == 5:
            labels = labels.squeeze(1)
        labels = labels.long()
        pixel_error = F.cross_entropy(logits, labels, reduction="none")
        return self._pool_to_regions(pixel_error, region_grid)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: list[torch.Tensor],
        labels: torch.Tensor,
        region_grid: tuple[int, int, int],
    ) -> dict:
        """Return region-wise routing weights and diagnostics."""
        student_error = self.prediction_error(student_logits, labels, region_grid)
        teacher_errors = [
            self.prediction_error(logits, labels, region_grid)
            for logits in teacher_logits
        ]

        advantages = torch.stack(
            [student_error - teacher_error for teacher_error in teacher_errors],
            dim=-1,
        )
        hard_region_mask = student_error > self.hard_region_threshold

        hard_weights = F.softmax(advantages / max(self.temperature, self.epsilon), dim=-1)
        uniform_weights = torch.full_like(hard_weights, 1.0 / len(teacher_logits))
        weights = torch.where(hard_region_mask.unsqueeze(-1), hard_weights, uniform_weights)

        return {
            "weights": weights,
            "student_error": student_error,
            "teacher_errors": teacher_errors,
            "hard_region_mask": hard_region_mask,
            "region_grid": region_grid,
        }


class TPR(nn.Module):
    """Task-Performance Routing with hierarchical stage-wise region grids."""

    def __init__(
        self,
        num_classes: int = 14,
        stage_region_grids: dict[str, tuple[int, int, int]] | None = None,
        logits_region_grid: tuple[int, int, int] = (8, 8, 8),
        hard_region_threshold: float = 0.5,
        temperature: float = 0.7,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stage_region_grids = stage_region_grids or {
            "stage_0": (8, 8, 8),
            "stage_1": (4, 4, 4),
            "stage_2": (2, 2, 2),
            "stage_3": (1, 1, 1),
        }
        self.logits_region_grid = logits_region_grid
        self.router = TaskPerformanceRouter(
            num_classes=num_classes,
            hard_region_threshold=hard_region_threshold,
            temperature=temperature,
            epsilon=epsilon,
        )

    @staticmethod
    def _map_regions_to_spatial(
        region_weights: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        region_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Map [B, N, M] region weights to [B, M, D, H, W]."""
        batch, _, num_teachers = region_weights.shape
        d_regions, h_regions, w_regions = region_grid
        weights_grid = region_weights.transpose(1, 2).reshape(
            batch, num_teachers, d_regions, h_regions, w_regions
        )
        return F.interpolate(weights_grid, size=spatial_shape, mode="nearest")

    def _mix_stage_features(
        self,
        stage_name: str,
        teacher_feats: list[dict[str, torch.Tensor]],
        routing: dict,
    ) -> torch.Tensor | None:
        """Mix teacher features for one stage using spatial routing weights."""
        if any(stage_name not in feats for feats in teacher_feats):
            return None

        feat_0 = teacher_feats[0][stage_name]
        weights = routing["weights"]

        if feat_0.dim() != 5:
            mean_weights = weights.mean(dim=1)
            mixed = 0.0
            for teacher_idx, feats in enumerate(teacher_feats):
                view_shape = [mean_weights.shape[0]] + [1] * (feat_0.dim() - 1)
                mixed = mixed + mean_weights[:, teacher_idx].view(*view_shape) * feats[stage_name]
            return mixed

        spatial_shape = feat_0.shape[-3:]
        spatial_weights = self._map_regions_to_spatial(
            weights,
            spatial_shape=spatial_shape,
            region_grid=routing["region_grid"],
        )
        mixed = 0.0
        for teacher_idx, feats in enumerate(teacher_feats):
            mixed = mixed + spatial_weights[:, teacher_idx : teacher_idx + 1] * feats[stage_name]
        return mixed

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher1_logits: torch.Tensor,
        teacher2_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher1_feats: dict[str, torch.Tensor],
        teacher2_feats: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Compute stage-wise routing and mixed teacher features."""
        teacher_logits = [teacher1_logits, teacher2_logits]
        teacher_feats = [teacher1_feats, teacher2_feats]

        routing_weights: dict[str, dict] = {}
        mixed_teacher_feats: dict[str, torch.Tensor] = {}

        for stage_name, region_grid in self.stage_region_grids.items():
            routing = self.router(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                region_grid=region_grid,
            )
            routing_weights[stage_name] = routing
            mixed_feat = self._mix_stage_features(stage_name, teacher_feats, routing)
            if mixed_feat is not None:
                mixed_teacher_feats[stage_name] = mixed_feat

        routing_weights["logits"] = self.router(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            region_grid=self.logits_region_grid,
        )
        return mixed_teacher_feats, routing_weights


__all__ = ["TaskPerformanceRouter", "TPR"]
