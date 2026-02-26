"""
TRP loss: segmentation, feature alignment, logits distillation, and load-balance loss.
Supports adaptive region sizes per stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.losses import DiceCELoss


class TRPLoss(nn.Module):
    """
    TRP multi-component loss with adaptive region sizes per stage:
    segmentation, feature alignment, logits distillation, load balance.
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_align: float = 0.5,
        lambda_logits: float = 0.5,
        lambda_balance: float = 0.01,
        balance_temperature: float = 0.5,
        region_sizes: list = None,
        num_classes: int = 14,
    ):
        super().__init__()

        self.lambda_seg = lambda_seg
        self.lambda_align = lambda_align
        self.lambda_logits = lambda_logits
        self.lambda_balance = lambda_balance
        self.balance_temperature = balance_temperature
        
        # segmentation loss
        self.seg_loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )
        
        # default region sizes
        if region_sizes is None:
            region_sizes = [(8, 8, 8), (4, 4, 4), (2, 2, 2), (1, 1, 1)]
        
        self.region_sizes = region_sizes
        self.expected_dims = {48, 96, 192, 384}

    def _compute_region_alignment_loss(
        self,
        s_feat: torch.Tensor,
        t_feat: torch.Tensor,
        region_size: tuple,
    ) -> torch.Tensor:
        """Short docstring."""
        d_size, h_size, w_size = region_size
        
        # check format
        if s_feat.shape[-1] in self.expected_dims:


            B, D, H, W, C = s_feat.shape
        else:


            s_feat = s_feat.permute(0, 2, 3, 4, 1).contiguous()
            t_feat = t_feat.permute(0, 2, 3, 4, 1).contiguous()
            B, D, H, W, C = s_feat.shape
        
        # region extent
        d_step = max(1, D // d_size)
        h_step = max(1, H // h_size)
        w_step = max(1, W // w_size)
        
        s_regions = []
        t_regions = []
        
        for i in range(d_size):
            for j in range(h_size):
                for k in range(w_size):
                    d_start = i * d_step
                    d_end = min((i + 1) * d_step, D)
                    h_start = j * h_step
                    h_end = min((j + 1) * h_step, H)
                    w_start = k * w_step
                    w_end = min((k + 1) * w_step, W)
                    
                    if d_end > d_start and h_end > h_start and w_end > w_start:
                        s_region = s_feat[:, d_start:d_end, h_start:h_end, w_start:w_end, :].mean(dim=[1, 2, 3])  # [B, C]
                        t_region = t_feat[:, d_start:d_end, h_start:h_end, w_start:w_end, :].mean(dim=[1, 2, 3])  # [B, C]
                        s_regions.append(s_region)
                        t_regions.append(t_region)
        
        if s_regions:
            s_regions = torch.stack(s_regions, dim=1)  # [B, num_regions, C]
            t_regions = torch.stack(t_regions, dim=1)






            s_regions_norm = F.normalize(s_regions, p=2, dim=2)  # [B, num_regions, C]
            t_regions_norm = F.normalize(t_regions, p=2, dim=2)  # [B, num_regions, C]


            mse_loss = F.mse_loss(s_regions_norm, t_regions_norm)


            cosine_sim = (s_regions_norm * t_regions_norm).sum(dim=2)  # [B, num_regions]
            cosine_loss = (1 - cosine_sim).mean()




            loss = 0.3 * mse_loss + 0.7 * cosine_loss


            loss = torch.clamp(loss, max=10.0)
        else:


            s_global = s_feat.mean(dim=[1, 2, 3])  # [B, C]
            t_global = t_feat.mean(dim=[1, 2, 3])  # [B, C]


            s_global_norm = F.normalize(s_global, p=2, dim=1)
            t_global_norm = F.normalize(t_global, p=2, dim=1)
            mse_loss = F.mse_loss(s_global_norm, t_global_norm)
            
            cosine_sim = (s_global_norm * t_global_norm).sum(dim=1).mean()
            cosine_loss = 1 - cosine_sim


            loss = 0.3 * mse_loss + 0.7 * cosine_loss
            loss = torch.clamp(loss, max=10.0)
        
        return loss

    def _compute_alignment_loss(
        self,
        student_feats: dict,
        mixed_teacher_feats: dict,
        routing_weights: dict,  # {stage_name: {pi_1_spatial, pi_2_spatial, region_size, ...}}
    ) -> torch.Tensor:
        """Short docstring."""
        if not student_feats or not mixed_teacher_feats:
            device = next(iter(student_feats.values())).device if student_feats else torch.device('cuda')
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = 0.0
        num_stages = 0

        for stage_idx, (stage_name, s_feat) in enumerate(sorted(student_feats.items())):
            if stage_name not in mixed_teacher_feats:
                continue




            if s_feat.dim() == 5:
                # check format
                if s_feat.shape[-1] in self.expected_dims:


                    _, D, H, W, _ = s_feat.shape
                else:


                    _, _, D, H, W = s_feat.shape


                feat_size = min(D, H, W)
                if feat_size >= 32:
                    region_size = (8, 8, 8)
                elif feat_size >= 16:
                    region_size = (4, 4, 4)
                elif feat_size >= 8:
                    region_size = (2, 2, 2)
                elif feat_size >= 4:
                    region_size = (2, 2, 2)
                else:
                    region_size = (1, 1, 1)
            else:


                region_size = (1, 1, 1)
            
            t_feat = mixed_teacher_feats[stage_name]


            if s_feat.dim() == 3:  # [B, N, C]


                s_global = s_feat.mean(dim=1)  # [B, C]
                t_global = t_feat.mean(dim=1)  # [B, C]


                s_global_norm = F.normalize(s_global, p=2, dim=1)
                t_global_norm = F.normalize(t_global, p=2, dim=1)
                mse_loss = F.mse_loss(s_global_norm, t_global_norm)
                cosine_sim = (s_global_norm * t_global_norm).sum(dim=1).mean()
                cosine_loss = 1 - cosine_sim
                loss = 0.3 * mse_loss + 0.7 * cosine_loss
                loss = torch.clamp(loss, max=10.0)
            elif s_feat.dim() == 5:  # [B, D, H, W, C] or [B, C, D, H, W]


                loss = self._compute_region_alignment_loss(s_feat, t_feat, region_size)
            else:
                continue

            total_loss += loss
            num_stages += 1

        if num_stages == 0:
            device = next(iter(student_feats.values())).device
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss / num_stages

    def _compute_balance_loss(self, routing_weights: dict) -> torch.Tensor:
        """Short docstring."""


        stage_name = list(routing_weights.keys())[0] if routing_weights else None
        
        if stage_name is None or stage_name not in routing_weights:
            device = next(iter(routing_weights.values())).device if routing_weights else torch.device('cuda')
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        routing = routing_weights[stage_name]
        pi_1_spatial = routing.get('pi_1_spatial')
        pi_2_spatial = routing.get('pi_2_spatial')
        teacher1_error = routing.get('teacher1_error')
        teacher2_error = routing.get('teacher2_error')
        
        if pi_1_spatial is None or pi_2_spatial is None:
            device = pi_1_spatial.device if pi_1_spatial is not None else pi_2_spatial.device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        device = pi_1_spatial.device


        if teacher1_error is not None and teacher2_error is not None:
            t1_quality = (-teacher1_error.mean()).detach()
            t2_quality = (-teacher2_error.mean()).detach()
            quality_stack = torch.stack([t1_quality, t2_quality])
            target_weights = torch.softmax(quality_stack / max(self.balance_temperature, 1e-3), dim=0)
            target_pi_1 = target_weights[0]
        else:
            target_pi_1 = torch.tensor(0.5, device=device)


        avg_pi_1 = pi_1_spatial.mean()
        mse_loss = (avg_pi_1 - target_pi_1) ** 2


        pi_1_safe = pi_1_spatial + 1e-8  # [B, D, H, W]
        pi_2_safe = pi_2_spatial + 1e-8  # [B, D, H, W]
        entropy = -(pi_1_safe * torch.log(pi_1_safe) + 
                    pi_2_safe * torch.log(pi_2_safe))  # [B, D, H, W]
        entropy = entropy.mean()
        target_entropy = -(target_pi_1 * torch.log(target_pi_1 + 1e-8) + 
                           (1 - target_pi_1) * torch.log(1 - target_pi_1 + 1e-8))
        entropy_loss = (entropy - target_entropy) ** 2


        balance_loss = 0.5 * mse_loss + 0.5 * entropy_loss
        
        return balance_loss

    def _map_regions_to_spatial(
        self,
        region_weights: torch.Tensor,
        spatial_shape: tuple,
        region_size: tuple,
    ) -> torch.Tensor:
        """Short docstring."""
        B = region_weights.shape[0]
        D, H, W = spatial_shape
        d_size, h_size, w_size = region_size
        
        spatial_weights = torch.zeros(B, D, H, W, device=region_weights.device, dtype=region_weights.dtype)
        
        # spatial extent per region
        d_step = max(1, D // d_size)
        h_step = max(1, H // h_size)
        w_step = max(1, W // w_size)
        
        region_idx = 0
        for i in range(d_size):
            for j in range(h_size):
                for k in range(w_size):
                    if region_idx >= region_weights.shape[1]:
                        break
                    
                    d_start = i * d_step
                    d_end = min((i + 1) * d_step, D)
                    h_start = j * h_step
                    h_end = min((j + 1) * h_step, H)
                    w_start = k * w_step
                    w_end = min((k + 1) * w_step, W)
                    
                    if d_end > d_start and h_end > h_start and w_end > w_start:
                        spatial_weights[:, d_start:d_end, h_start:h_end, w_start:w_end] = \
                            region_weights[:, region_idx:region_idx+1].unsqueeze(-1).unsqueeze(-1)
                    
                    region_idx += 1
        
        return spatial_weights

    def _compute_logits_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher1_logits: torch.Tensor,
        teacher2_logits: torch.Tensor,
        routing_weights: dict,
    ) -> torch.Tensor:
        """Short docstring."""


        stage_name = list(routing_weights.keys())[0] if routing_weights else None
        
        if stage_name is None or stage_name not in routing_weights:


            mixed_teacher_logits = 0.5 * (teacher1_logits + teacher2_logits)
        else:
            routing = routing_weights[stage_name]
            pi_1_spatial = routing.get('pi_1_spatial')
            pi_2_spatial = routing.get('pi_2_spatial')
            
            if pi_1_spatial is None or pi_2_spatial is None:


                mixed_teacher_logits = 0.5 * (teacher1_logits + teacher2_logits)
            else:


                pi_1_expanded = pi_1_spatial.unsqueeze(1)  # [B, 1, D, H, W]
                pi_2_expanded = pi_2_spatial.unsqueeze(1)  # [B, 1, D, H, W]


                mixed_teacher_logits = pi_1_expanded * teacher1_logits + pi_2_expanded * teacher2_logits






        temperature = 4.0


        student_logits_clipped = torch.clamp(student_logits, min=-10.0, max=10.0)
        mixed_teacher_logits_clipped = torch.clamp(mixed_teacher_logits.detach(), min=-10.0, max=10.0)
        
        student_soft = F.log_softmax(student_logits_clipped / temperature, dim=1)  # [B, C, D, H, W]
        teacher_soft = F.softmax(mixed_teacher_logits_clipped / temperature, dim=1)  # [B, C, D, H, W]


        kl_div_per_pixel = F.kl_div(
            student_soft, teacher_soft, 
            reduction='none'  # [B, C, D, H, W]
        )


        kl_loss = kl_div_per_pixel.sum(dim=1).mean()


        kl_loss = torch.clamp(kl_loss, max=10.0)
        logits_loss = (temperature ** 2) * kl_loss


        logits_loss = torch.clamp(logits_loss, max=100.0)
        
        return logits_loss

    def forward(
        self,
        student_logits: torch.Tensor,
        student_feats: dict,
        mixed_teacher_feats: dict,
        routing_weights: dict,  # {stage_name: {pi_1, pi_2, ...}}
        labels: torch.Tensor = None,
        teacher1_logits: torch.Tensor = None,
        teacher2_logits: torch.Tensor = None,
    ) -> dict:
        """Short docstring."""
        device = next(iter(student_feats.values())).device


        if labels is not None:
            if labels.dim() == 4:
                labels = labels.unsqueeze(1).long()
            else:
                labels = labels.long()
            seg_loss = self.seg_loss_fn(student_logits, labels)
        else:
            seg_loss = torch.tensor(0.0, device=device, requires_grad=True)


        align_loss = self._compute_alignment_loss(student_feats, mixed_teacher_feats, routing_weights)


        if teacher1_logits is not None and teacher2_logits is not None:
            logits_loss = self._compute_logits_distillation_loss(
                student_logits, teacher1_logits, teacher2_logits, routing_weights
            )


            if torch.isnan(logits_loss) or torch.isinf(logits_loss) or logits_loss.item() > 1000.0:
                print(f"Warning: Abnormal logits_loss detected: {logits_loss.item()}, setting to 0")
                logits_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            logits_loss = torch.tensor(0.0, device=device, requires_grad=True)


        balance_loss = self._compute_balance_loss(routing_weights)


        total_loss = (
            self.lambda_seg * seg_loss +
            self.lambda_align * align_loss +
            self.lambda_logits * logits_loss +
            self.lambda_balance * balance_loss
        )


        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf detected! seg={seg_loss.item():.4f}, "
                  f"align={align_loss.item():.4f}, logits={logits_loss.item():.4f}, "
                  f"balance={balance_loss.item():.4f}")


            total_loss = self.lambda_seg * seg_loss

        return {
            'total': total_loss,
            'seg': seg_loss,
            'align': align_loss,
            'logits': logits_loss,
            'balance': balance_loss,
        }


MTRDLossAdaptive = TRPLoss  # alias
MTRDUnsupervisedLossAdaptive = TRPLoss


if __name__ == "__main__":


    print("Testing TRPLoss...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = TRPLoss(
        lambda_seg=1.0,
        lambda_align=0.5,
        lambda_logits=0.5,
        lambda_balance=0.01,
    )


    B, num_classes, D, H, W = 2, 14, 32, 32, 32
    
    student_logits = torch.randn(B, num_classes, D, H, W).to(device)
    labels = torch.randint(0, num_classes, (B, D, H, W)).to(device)
    teacher1_logits = torch.randn(B, num_classes, D, H, W).to(device)
    teacher2_logits = torch.randn(B, num_classes, D, H, W).to(device)
    
    student_feats = {
        'stage_0': torch.randn(2, 8, 8, 8, 48).to(device),
        'stage_1': torch.randn(2, 4, 4, 4, 96).to(device),
        'stage_2': torch.randn(2, 2, 2, 2, 192).to(device),
        'stage_3': torch.randn(2, 1, 1, 1, 384).to(device),
    }
    
    mixed_teacher_feats = {
        'stage_0': torch.randn(2, 8, 8, 8, 48).to(device),
        'stage_1': torch.randn(2, 4, 4, 4, 96).to(device),
        'stage_2': torch.randn(2, 2, 2, 2, 192).to(device),
        'stage_3': torch.randn(2, 1, 1, 1, 384).to(device),
    }
    
    routing_weights = {
        'stage_0': {
            'pi_1': torch.rand(2, 512).to(device),  # 8x8x8 = 512
            'pi_2': torch.rand(2, 512).to(device),
            'region_size': (8, 8, 8),
        },
        'stage_1': {
            'pi_1': torch.rand(2, 64).to(device),  # 4x4x4 = 64
            'pi_2': torch.rand(2, 64).to(device),
            'region_size': (4, 4, 4),
        },
        'stage_2': {
            'pi_1': torch.rand(2, 8).to(device),  # 2x2x2 = 8
            'pi_2': torch.rand(2, 8).to(device),
            'region_size': (2, 2, 2),
        },
        'stage_3': {
            'pi_1': torch.rand(2, 1).to(device),  # 1x1x1 = 1
            'pi_2': torch.rand(2, 1).to(device),
            'region_size': (1, 1, 1),
        },
    }
    
    losses = criterion(
        student_logits=student_logits,
        student_feats=student_feats,
        mixed_teacher_feats=mixed_teacher_feats,
        routing_weights=routing_weights,
        labels=labels,
        teacher1_logits=teacher1_logits,
        teacher2_logits=teacher2_logits,
    )
    
    print("Losses:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    print("Done.")
