"""Evaluation metrics for segmentation."""

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from scipy.ndimage import distance_transform_edt


def compute_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict[int, float]:
    """
    Compute Dice coefficient for each class.

    Args:
        pred: Prediction mask [H, W, Z] or [H, W, Z]
        gt: Ground truth mask [H, W, Z]
        num_classes: Number of classes

    Returns:
        Dictionary mapping class_id -> Dice score
    """
    dice_scores = {}
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).astype(np.float32)
        gt_mask = (gt == class_id).astype(np.float32)

        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = 2.0 * intersection / union

        dice_scores[class_id] = dice

    return dice_scores


def compute_hd95(pred: np.ndarray, gt: np.ndarray, num_classes: int, spacing: tuple = (1.0, 1.0, 1.0)) -> dict[int, float]:
    """
    Compute 95th percentile Hausdorff distance for each class.

    Args:
        pred: Prediction mask [H, W, Z]
        gt: Ground truth mask [H, W, Z]
        num_classes: Number of classes
        spacing: Voxel spacing (dx, dy, dz)

    Returns:
        Dictionary mapping class_id -> HD95 score
    """
    hd95_scores = {}

    for class_id in range(num_classes):
        pred_mask = (pred == class_id).astype(bool)
        gt_mask = (gt == class_id).astype(bool)

        if not np.any(gt_mask):
            # No ground truth for this class
            hd95_scores[class_id] = np.nan
            continue

        if not np.any(pred_mask):
            # No prediction for this class - distance is infinity
            hd95_scores[class_id] = np.inf
            continue

        # Compute distance transform
        # Distance from each point in pred to nearest point in gt
        # Ensure spacing matches the dimension of the mask
        mask_shape = gt_mask.shape
        if len(spacing) != len(mask_shape):
            # Use default spacing (1.0 for each dimension)
            spacing_actual = tuple(1.0 for _ in range(len(mask_shape)))
        else:
            spacing_actual = spacing
        
        dist_pred_to_gt = distance_transform_edt(~gt_mask, sampling=spacing_actual)
        dist_gt_to_pred = distance_transform_edt(~pred_mask, sampling=spacing_actual)

        # Get boundary points
        # A pixel is a boundary if it's in the mask and at least one neighbor is not in the mask
        boundary_conditions_pred = []
        boundary_conditions_gt = []
        
        # Generate boundary conditions for each dimension
        for axis in range(len(pred_mask.shape)):
            boundary_conditions_pred.append(np.roll(pred_mask, 1, axis=axis) != pred_mask)
            boundary_conditions_pred.append(np.roll(pred_mask, -1, axis=axis) != pred_mask)
            boundary_conditions_gt.append(np.roll(gt_mask, 1, axis=axis) != gt_mask)
            boundary_conditions_gt.append(np.roll(gt_mask, -1, axis=axis) != gt_mask)
        
        pred_boundary = pred_mask & np.logical_or.reduce(boundary_conditions_pred)
        gt_boundary = gt_mask & np.logical_or.reduce(boundary_conditions_gt)

        # Compute distances
        if np.any(pred_boundary):
            dists_pred = dist_pred_to_gt[pred_boundary]
        else:
            dists_pred = np.array([0.0])

        if np.any(gt_boundary):
            dists_gt = dist_gt_to_pred[gt_boundary]
        else:
            dists_gt = np.array([0.0])

        # Compute 95th percentile
        hd95 = np.percentile(np.concatenate([dists_pred, dists_gt]), 95)
        hd95_scores[class_id] = hd95

    return hd95_scores


def compute_boundary_f1(pred: np.ndarray, gt: np.ndarray, num_classes: int, tolerance: int = 2) -> dict[int, float]:
    """
    Compute Boundary F1 score for each class.

    Args:
        pred: Prediction mask [H, W, Z]
        gt: Ground truth mask [H, W, Z]
        num_classes: Number of classes
        tolerance: Tolerance in pixels for boundary matching

    Returns:
        Dictionary mapping class_id -> Boundary F1 score
    """
    f1_scores = {}

    for class_id in range(num_classes):
        pred_mask = (pred == class_id).astype(bool)
        gt_mask = (gt == class_id).astype(bool)

        if not np.any(gt_mask):
            f1_scores[class_id] = np.nan
            continue

        if not np.any(pred_mask):
            f1_scores[class_id] = 0.0
            continue

        # Extract boundaries using morphological operations
        from scipy.ndimage import binary_erosion, binary_dilation

        pred_boundary = pred_mask & ~binary_erosion(pred_mask)
        gt_boundary = gt_mask & ~binary_erosion(gt_mask)

        if not np.any(pred_boundary) or not np.any(gt_boundary):
            f1_scores[class_id] = 0.0
            continue

        # Dilate boundaries for tolerance
        pred_boundary_dilated = binary_dilation(pred_boundary, iterations=tolerance)
        gt_boundary_dilated = binary_dilation(gt_boundary, iterations=tolerance)

        # Compute precision and recall
        tp = np.sum(pred_boundary & gt_boundary_dilated)
        fp = np.sum(pred_boundary & ~gt_boundary_dilated)
        fn = np.sum(gt_boundary & ~pred_boundary_dilated)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[class_id] = f1

    return f1_scores


def evaluate_segmentation(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    spacing: tuple = (1.0, 1.0, 1.0),
    compute_hd: bool = True,
    compute_boundary_f1: bool = True,
) -> dict:
    """
    Comprehensive evaluation of segmentation results.

    Args:
        pred: Prediction mask [H, W, Z]
        gt: Ground truth mask [H, W, Z]
        num_classes: Number of classes
        spacing: Voxel spacing
        compute_hd: Whether to compute HD95
        compute_boundary_f1: Whether to compute Boundary F1

    Returns:
        Dictionary with metrics for each class
    """
    results = {}

    # Dice scores
    dice_scores = compute_dice(pred, gt, num_classes)
    results["dice"] = dice_scores

    # HD95 scores
    if compute_hd:
        hd95_scores = compute_hd95(pred, gt, num_classes, spacing=spacing)
        results["hd95"] = hd95_scores

    # Boundary F1 scores
    if compute_boundary_f1:
        # Use globals() to avoid name conflict with parameter
        f1_scores = globals()['compute_boundary_f1'](pred, gt, num_classes)
        results["boundary_f1"] = f1_scores

    # Summary statistics
    dice_values = [v for k, v in dice_scores.items() if k > 0]  # Exclude background
    results["mean_dice"] = np.nanmean(dice_values) if dice_values else np.nan

    if compute_hd:
        hd95_values = [v for k, v in hd95_scores.items() if k > 0 and not np.isinf(v) and not np.isnan(v)]
        results["mean_hd95"] = np.nanmean(hd95_values) if hd95_values else np.nan

    if compute_boundary_f1:
        f1_values = [v for k, v in f1_scores.items() if k > 0 and not np.isnan(v)]
        results["mean_boundary_f1"] = np.nanmean(f1_values) if f1_values else np.nan

    return results


def evaluate_batch(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> dict:
    """
    Evaluate a batch of predictions.

    Args:
        pred_logits: Prediction logits [B, C, H, W, Z]
        gt: Ground truth [B, 1, H, W, Z] or [B, H, W, Z]
        num_classes: Number of classes
        spacing: Voxel spacing

    Returns:
        Dictionary with aggregated metrics
    """
    # Convert to numpy
    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_mask = torch.argmax(pred_probs, dim=1).cpu().numpy()  # [B, H, W, Z]

    if gt.dim() == 5:
        gt_mask = gt.squeeze(1).cpu().numpy()  # [B, H, W, Z]
    else:
        gt_mask = gt.cpu().numpy()  # [B, H, W, Z]

    # Aggregate results across batch
    all_dice = []
    all_hd95 = []
    all_f1 = []

    for b in range(pred_mask.shape[0]):
        results = evaluate_segmentation(
            pred_mask[b],
            gt_mask[b],
            num_classes,
            spacing=spacing,
        )
        dice_values = [v for k, v in results["dice"].items() if k > 0]
        all_dice.extend(dice_values)

        if "hd95" in results:
            hd95_values = [v for k, v in results["hd95"].items() if k > 0 and not np.isinf(v) and not np.isnan(v)]
            all_hd95.extend(hd95_values)

        if "boundary_f1" in results:
            f1_values = [v for k, v in results["boundary_f1"].items() if k > 0 and not np.isnan(v)]
            all_f1.extend(f1_values)

    return {
        "mean_dice": np.nanmean(all_dice) if all_dice else np.nan,
        "mean_hd95": np.nanmean(all_hd95) if all_hd95 else np.nan,
        "mean_boundary_f1": np.nanmean(all_f1) if all_f1 else np.nan,
    }


if __name__ == "__main__":
    # Test metrics
    H, W, Z = 128, 128, 128
    num_classes = 25

    # Create dummy predictions and ground truth
    pred = np.random.randint(0, num_classes, size=(H, W, Z))
    gt = np.random.randint(0, num_classes, size=(H, W, Z))

    results = evaluate_segmentation(pred, gt, num_classes)
    print("Evaluation results:")
    print(f"  Mean Dice: {results['mean_dice']:.4f}")
    if "mean_hd95" in results:
        print(f"  Mean HD95: {results['mean_hd95']:.4f}")
    if "mean_boundary_f1" in results:
        print(f"  Mean Boundary F1: {results['mean_boundary_f1']:.4f}")

