"""
TRP (Task-performance-based Routing for multi-teacher) core module.
Task-performance-based complementary routing: student learns from each teacher's strength per region.

Key idea:
- In hard regions: learn from the teacher with better task performance.
- In easy regions: fuse complementary knowledge from both teachers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMorphologyEncoder(nn.Module):
    """Short docstring."""
    
    def __init__(
        self, 
        output_dim: int = 256, 
        num_stages: int = 4,
        feature_dims: list = None,
        use_spatial_attention: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_stages = num_stages
        self.per_stage_dim = output_dim // num_stages
        self.use_spatial_attention = use_spatial_attention
        
        if feature_dims is None:
            feature_dims = [48, 96, 192, 384]
        
        self.feature_dims = feature_dims
        
        # spatial attention
        self.spatial_attentions = nn.ModuleList()
        for dim in feature_dims:
            self.spatial_attentions.append(
                nn.Sequential(
                    nn.Linear(dim, dim // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim // 4, 1),
                )
            )
        
        # projection
        self.stage_projections = nn.ModuleList()
        for dim in feature_dims:
            self.stage_projections.append(
                nn.Sequential(
                    nn.Linear(dim, self.per_stage_dim),
                    nn.ReLU(inplace=True),
                )
            )
        
        total_dim = self.per_stage_dim * len(feature_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim),
        )
    
    def forward(self, features: dict) -> torch.Tensor:
        """Short docstring."""
        pooled_features = []
        
        for stage_idx, (name, feat) in enumerate(sorted(features.items())):
            if feat.dim() == 3:  # [B, N, C]
                if self.use_spatial_attention and stage_idx < len(self.spatial_attentions):
                    attention = self.spatial_attentions[stage_idx](feat)
                    attention = F.softmax(attention, dim=1)
                    pooled = (feat * attention).sum(dim=1)
                else:
                    pooled = feat.mean(dim=1)
                    
            elif feat.dim() == 5:
                is_channel_last = feat.shape[-1] in self.feature_dims
                
                if is_channel_last:
                    B, D, H, W, C = feat.shape
                    if self.use_spatial_attention and stage_idx < len(self.spatial_attentions):
                        feat_flat = feat.reshape(B, -1, C)
                        attention = self.spatial_attentions[stage_idx](feat_flat)
                        attention = F.softmax(attention, dim=1)
                        pooled = (feat_flat * attention).sum(dim=1)
                    else:
                        pooled = feat.mean(dim=(1, 2, 3))
                else:
                    B, C, D, H, W = feat.shape
                    if self.use_spatial_attention and stage_idx < len(self.spatial_attentions):
                        feat_permuted = feat.permute(0, 2, 3, 4, 1).contiguous()
                        feat_flat = feat_permuted.reshape(B, -1, C)
                        attention = self.spatial_attentions[stage_idx](feat_flat)
                        attention = F.softmax(attention, dim=1)
                        pooled = (feat_flat * attention).sum(dim=1)
                    else:
                        pooled = F.adaptive_avg_pool3d(feat, 1).flatten(1)
            else:
                continue
            
            pooled_features.append(pooled)
        
        if len(pooled_features) == 0:
            raise ValueError("No valid features found")
        
        # project and fuse
        projected = []
        for i, pooled in enumerate(pooled_features):
            if i < len(self.stage_projections):
                proj = self.stage_projections[i](pooled)
                projected.append(proj)
        
        concat = torch.cat(projected, dim=1)
        h_context = self.fusion(concat)
        
        return h_context


class TaskPerformanceRouting(nn.Module):
    """Short docstring."""
    
    def __init__(
        self,
        num_classes: int = 14,
        hard_region_threshold: float = 0.5,
        epsilon: float = 1e-6,
        temperature: float = 1.0,
        confidence_scale: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hard_region_threshold = hard_region_threshold
        self.epsilon = epsilon
        self.temperature = temperature
        self.confidence_scale = confidence_scale

    def _pool_region_values(
        self,
        values: torch.Tensor,
        region_size: tuple,
    ) -> torch.Tensor:
        """Short docstring."""
        B, D, H, W = values.shape
        d_size, h_size, w_size = region_size
        num_regions = d_size * h_size * w_size

        d_step = max(1, D // d_size)
        h_step = max(1, H // h_size)
        w_step = max(1, W // w_size)

        region_list = []
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
                        region_val = values[:, d_start:d_end, h_start:h_end, w_start:w_end].mean(dim=[1, 2, 3])  # [B]
                        region_list.append(region_val)

        if region_list:
            regions = torch.stack(region_list, dim=1)  # [B, num_regions]
        else:
            regions = values.mean(dim=[1, 2, 3]).unsqueeze(1).expand(-1, num_regions)

        return regions
    
    def compute_prediction_error(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        region_size: tuple,
    ) -> torch.Tensor:
        """Short docstring."""


        if labels.dim() == 5:
            labels = labels.squeeze(1)  # [B, 1, D, H, W] -> [B, D, H, W]
        labels = labels.long()
        
        B, num_classes, D, H, W = logits.shape
        d_size, h_size, w_size = region_size
        num_regions = d_size * h_size * w_size


        pixel_errors = F.cross_entropy(logits, labels, reduction='none')  # [B, D, H, W]
        return self._pool_region_values(pixel_errors, region_size)

    def compute_region_confidence(
        self,
        logits: torch.Tensor,
        region_size: tuple,
    ) -> torch.Tensor:
        """Short docstring."""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + self.epsilon)).sum(dim=1)  # [B, D, H, W]


        confidence = -entropy
        return self._pool_region_values(confidence, region_size)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher1_logits: torch.Tensor,
        teacher2_logits: torch.Tensor,
        labels: torch.Tensor,
        region_size: tuple,
    ) -> torch.Tensor:
        """Short docstring."""
        B = student_logits.shape[0]
        d_size, h_size, w_size = region_size
        num_regions = d_size * h_size * w_size


        student_error = self.compute_prediction_error(student_logits, labels, region_size)  # [B, num_regions]
        teacher1_error = self.compute_prediction_error(teacher1_logits, labels, region_size)  # [B, num_regions]
        teacher2_error = self.compute_prediction_error(teacher2_logits, labels, region_size)  # [B, num_regions]






        relative_advantage_t1 = student_error - teacher1_error
        relative_advantage_t2 = student_error - teacher2_error


        confidence_t1 = self.compute_region_confidence(teacher1_logits, region_size)  # [B, num_regions]
        confidence_t2 = self.compute_region_confidence(teacher2_logits, region_size)  # [B, num_regions]






        score_t1 = relative_advantage_t1 + self.confidence_scale * confidence_t1
        score_t2 = relative_advantage_t2 + self.confidence_scale * confidence_t2




        advantage_scores = torch.stack([score_t1, score_t2], dim=-1)  # [B, num_regions, 2]
        weights = F.softmax(advantage_scores / self.temperature, dim=-1)  # [B, num_regions, 2]
        
        pi_1 = weights[:, :, 0]  # [B, num_regions]
        pi_2 = weights[:, :, 1]  # [B, num_regions]


        hard_region_mask = (student_error > self.hard_region_threshold)  # [B, num_regions]
        
        return {
            'pi_1': pi_1,
            'pi_2': pi_2,
            'student_error': student_error,
            'teacher1_error': teacher1_error,
            'teacher2_error': teacher2_error,
            'hard_region_mask': hard_region_mask,
            'region_size': region_size,
            'num_regions': num_regions,
        }


class TRPRouting(nn.Module):
    """Short docstring."""
    
    def __init__(
        self,
        num_classes: int = 14,
        feature_dims: list = None,
        region_size: tuple = (8, 8, 8),  # region size in logits
        hard_region_threshold: float = 0.5,
        epsilon: float = 1e-6,
        temperature: float = 1.0,
        confidence_scale: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dims = feature_dims if feature_dims else [48, 96, 192, 384]
        self.region_size = region_size  # region size
        
        # task-performance routing (once)
        self.routing_module = TaskPerformanceRouting(
            num_classes=num_classes,
            hard_region_threshold=hard_region_threshold,
            epsilon=epsilon,
            temperature=temperature,
            confidence_scale=confidence_scale,
        )
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher1_logits: torch.Tensor,
        teacher2_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Short docstring."""
        B, num_classes, D, H, W = student_logits.shape
        
        # compute routing in logits space
        routing_result = self.routing_module(
            student_logits=student_logits,
            teacher1_logits=teacher1_logits,
            teacher2_logits=teacher2_logits,
            labels=labels,
            region_size=self.region_size,
        )


        pi_1 = routing_result['pi_1']  # [B, num_regions]
        pi_2 = routing_result['pi_2']  # [B, num_regions]
        
        pi_1_spatial = self._map_regions_to_spatial(pi_1, (D, H, W), self.region_size)  # [B, D, H, W]
        pi_2_spatial = self._map_regions_to_spatial(pi_2, (D, H, W), self.region_size)  # [B, D, H, W]
        
        return {
            'logits_spatial': {
                'pi_1_spatial': pi_1_spatial,
                'pi_2_spatial': pi_2_spatial,
                'student_error': routing_result['student_error'],
                'teacher1_error': routing_result['teacher1_error'],
                'teacher2_error': routing_result['teacher2_error'],
                'hard_region_mask': routing_result['hard_region_mask'],
                'region_size': self.region_size,
            }
        }
    
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


class TRPAdaptive(nn.Module):
    """Short docstring."""
    
    def __init__(
        self,
        num_classes: int = 14,
        feature_dims: list = None,
        region_size: tuple = (8, 8, 8),  # region size in logits
        hard_region_threshold: float = 0.5,
        epsilon: float = 1e-6,
        temperature: float = 1.0,
        confidence_scale: float = 0.3,
    ):
        super().__init__()
        
        if feature_dims is None:
            feature_dims = [48, 96, 192, 384]
        
        self.feature_dims = feature_dims
        self.num_classes = num_classes
        
        # routing in logits space
        self.teacher_routing = TRPRouting(
            num_classes=num_classes,
            feature_dims=feature_dims,
            region_size=region_size,  # region size
            hard_region_threshold=hard_region_threshold,
            epsilon=epsilon,
            temperature=temperature,
            confidence_scale=confidence_scale,
        )
    
    def map_regions_to_spatial(
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
    
    def mix_teacher_features(
        self,
        teacher1_feats: dict,
        teacher2_feats: dict,
        routing_weights: dict,  # {'logits_spatial': {pi_1_spatial, pi_2_spatial, ...}}
    ) -> dict:
        """Short docstring."""
        mixed = {}
        expected_dims = {48, 96, 192, 384}
        
        # get routing weights
        logits_routing = routing_weights.get('logits_spatial', {})
        pi_1_logits = logits_routing.get('pi_1_spatial')  # [B, D_logits, H_logits, W_logits]
        pi_2_logits = logits_routing.get('pi_2_spatial')  # [B, D_logits, H_logits, W_logits]
        
        if pi_1_logits is None or pi_2_logits is None:
            # Fallback: uniform weights
            for stage_name in teacher1_feats:
                if stage_name not in teacher2_feats:
                    continue
                mixed[stage_name] = 0.5 * teacher1_feats[stage_name] + 0.5 * teacher2_feats[stage_name]
            return mixed
        
        # adapt weights to feature size
        for stage_name in teacher1_feats:
            if stage_name not in teacher2_feats:
                continue
            
            feat1 = teacher1_feats[stage_name]
            feat2 = teacher2_feats[stage_name]
            
            B = feat1.shape[0]
            
            if feat1.dim() == 3:  # [B, N, C]
                # 3D/patch: use mean weight
                avg_pi_1 = pi_1_logits.mean(dim=[1, 2, 3], keepdim=True).unsqueeze(-1)  # [B, 1, 1]
                avg_pi_2 = pi_2_logits.mean(dim=[1, 2, 3], keepdim=True).unsqueeze(-1)
                mixed[stage_name] = avg_pi_1 * feat1 + avg_pi_2 * feat2
                
            elif feat1.dim() == 5:
                is_channel_last = feat1.shape[-1] in expected_dims
                
                if is_channel_last:


                    D, H, W, C = feat1.shape[1:5]
                    
                    # downsample weights to feature size
                    # pi_1_logits: [B, D_logits, H_logits, W_logits]


                    pi_1_logits_4d = pi_1_logits.unsqueeze(1)  # [B, 1, D_logits, H_logits, W_logits]
                    pi_2_logits_4d = pi_2_logits.unsqueeze(1)  # [B, 1, D_logits, H_logits, W_logits]
                    
                    # adaptive pool to feature size
                    pi_1_feat = F.adaptive_avg_pool3d(pi_1_logits_4d, (D, H, W)).squeeze(1)  # [B, D, H, W]
                    pi_2_feat = F.adaptive_avg_pool3d(pi_2_logits_4d, (D, H, W)).squeeze(1)  # [B, D, H, W]
                    
                    # expand to channel
                    pi_1_expanded = pi_1_feat.unsqueeze(-1)  # [B, D, H, W, 1]
                    pi_2_expanded = pi_2_feat.unsqueeze(-1)
                    
                    # mix features
                    mixed[stage_name] = pi_1_expanded * feat1 + pi_2_expanded * feat2
                    
                else:


                    C, D, H, W = feat1.shape[1:5]
                    
                    # downsample weights to feature size
                    pi_1_logits_4d = pi_1_logits.unsqueeze(1)  # [B, 1, D_logits, H_logits, W_logits]
                    pi_2_logits_4d = pi_2_logits.unsqueeze(1)  # [B, 1, D_logits, H_logits, W_logits]
                    
                    # adaptive pool to feature size
                    pi_1_feat = F.adaptive_avg_pool3d(pi_1_logits_4d, (D, H, W)).squeeze(1)  # [B, D, H, W]
                    pi_2_feat = F.adaptive_avg_pool3d(pi_2_logits_4d, (D, H, W)).squeeze(1)  # [B, D, H, W]
                    
                    # expand to channel
                    pi_1_expanded = pi_1_feat.unsqueeze(1)  # [B, 1, D, H, W]
                    pi_2_expanded = pi_2_feat.unsqueeze(1)
                    
                    # mix features
                    mixed[stage_name] = pi_1_expanded * feat1 + pi_2_expanded * feat2
            else:
                # Fallback: mean weight
                avg_pi_1 = pi_1_logits.mean(dim=[1, 2, 3], keepdim=True)
                avg_pi_2 = pi_2_logits.mean(dim=[1, 2, 3], keepdim=True)
                while avg_pi_1.dim() < feat1.dim():
                    avg_pi_1 = avg_pi_1.unsqueeze(-1)
                    avg_pi_2 = avg_pi_2.unsqueeze(-1)
                mixed[stage_name] = avg_pi_1 * feat1 + avg_pi_2 * feat2
        
        return mixed
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher1_logits: torch.Tensor,
        teacher2_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher1_feats: dict,
        teacher2_feats: dict,
    ) -> tuple:
        """Short docstring."""
        # 1. routing in logits
        routing_weights = self.teacher_routing(
            student_logits=student_logits,
            teacher1_logits=teacher1_logits,
            teacher2_logits=teacher2_logits,
            labels=labels,
        )
        
        # 2. mix teacher feats
        mixed_feats = self.mix_teacher_features(
            teacher1_feats, teacher2_feats, routing_weights
        )
        
        # 3. build stage routing for loss


        logits_routing = routing_weights.get('logits_spatial', {})
        pi_1_logits = logits_routing.get('pi_1_spatial')  # [B, D, H, W]
        pi_2_logits = logits_routing.get('pi_2_spatial')  # [B, D, H, W]
        region_size = logits_routing.get('region_size', (8, 8, 8))
        
        # build per-stage routing
        # pi_1/pi_2 in logits space
        stage_routing_weights = {}
        for stage_name in teacher1_feats:
            stage_routing_weights[stage_name] = {
                'pi_1_spatial': pi_1_logits,  # [B, D_logits, H_logits, W_logits]
                'pi_2_spatial': pi_2_logits,  # [B, D_logits, H_logits, W_logits]
                'region_size': region_size,
                'student_error': logits_routing.get('student_error'),
                'teacher1_error': logits_routing.get('teacher1_error'),
                'teacher2_error': logits_routing.get('teacher2_error'),
                'hard_region_mask': logits_routing.get('hard_region_mask'),
            }
        
        return mixed_feats, stage_routing_weights


if __name__ == "__main__":
    print("Testing TRPAdaptive...")
    model = TRPAdaptive(num_classes=3, feature_dims=[48, 96, 192, 384])
    B, D, H, W = 2, 6, 6, 6
    sl = torch.randn(B, 3, D, H, W)
    t1l = torch.randn(B, 3, D, H, W)
    t2l = torch.randn(B, 3, D, H, W)
    labels = torch.randint(0, 3, (B, D, H, W))
    t1f = {"stage_0": torch.randn(B, 6, 6, 6, 48), "stage_1": torch.randn(B, 3, 3, 3, 96)}
    t2f = {"stage_0": torch.randn(B, 6, 6, 6, 48), "stage_1": torch.randn(B, 3, 3, 3, 96)}
    mixed, rw = model(sl, t1l, t2l, labels, t1f, t2f)
    print("Mixed keys:", list(mixed.keys()), "Routing keys:", list(rw.keys()))
