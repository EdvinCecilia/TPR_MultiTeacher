"""Student model: same SwinUNETR architecture as teacher, outputs multi-stage encoder features for distillation."""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


# Same SwinUNETR config as teacher (TeacherFullModel): feature_size=48, depths=(2,2,2,2), etc.
STUDENT_TEACHER_FEATURE_SIZE = 48
STUDENT_TEACHER_DEPTHS = (2, 2, 2, 2)


class StudentModel(nn.Module):
    """Student: same network architecture as teacher (SwinUNETR). Outputs encoder features for distillation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 14,
        feature_size: int = STUDENT_TEACHER_FEATURE_SIZE,
        depths: tuple = STUDENT_TEACHER_DEPTHS,
        use_v2: bool = True,
    ):
        super().__init__()
        # Same architecture as teacher
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            patch_size=2,
            depths=depths,
            num_heads=(3, 6, 12, 24),
            window_size=7,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            spatial_dims=3,
            downsample="merging",
            use_v2=use_v2,
        )
        self.feature_dims = [
            feature_size,
            feature_size * 2,
            feature_size * 4,
            feature_size * 8,
        ]
        # Student and teacher share the same architecture -> no projection
        self.teacher_dims = [48, 96, 192, 384]
        self.need_projection = any(s != t for s, t in zip(self.feature_dims, self.teacher_dims))
        if self.need_projection:
            self.projections = nn.ModuleDict()
            for i, (s_dim, t_dim) in enumerate(zip(self.feature_dims, self.teacher_dims)):
                if s_dim != t_dim:
                    self.projections[f'proj_{i}'] = nn.Linear(s_dim, t_dim)
                else:
                    self.projections[f'proj_{i}'] = nn.Identity()
        else:
            self.projections = None
        self.encoder_features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture encoder features."""
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.encoder_features[name] = output[0]
                else:
                    self.encoder_features[name] = output
            return hook
        
        if hasattr(self.model, 'swinViT'):
            swin = self.model.swinViT
            for i, layer_name in enumerate(['layers1', 'layers2', 'layers3', 'layers4']):
                if hasattr(swin, layer_name):
                    layer = getattr(swin, layer_name)
                    if len(layer) > 0 and hasattr(layer[0], 'blocks') and len(layer[0].blocks) > 0:
                        layer[0].blocks[-1].register_forward_hook(get_hook(f'stage_{i}'))
    
    def get_projected_features(self) -> dict:
        """Return projected encoder features (aligned to teacher dims)."""
        if not self.need_projection:
            return self.encoder_features.copy()
        projected = {}
        for i, (name, feat) in enumerate(self.encoder_features.items()):
            proj_key = f'proj_{i}'
            if proj_key in self.projections:
                proj_layer = self.projections[proj_key]
                if hasattr(proj_layer, 'weight'):
                    feat = feat.to(proj_layer.weight.device)
                projected[name] = proj_layer(feat)
            else:
                projected[name] = feat
        return projected
    
    def forward(self, x: torch.Tensor, return_features: bool = True):
        """
        Forward pass
        
        Args:
            x: input [B,1,D,H,W], return_features: whether to return encoder features.
        Returns logits and optionally features dict."""
        self.encoder_features.clear()
        
        logits = self.model(x)
        
        if return_features:
            features = self.get_projected_features()
            return logits, features
        
        return logits


def count_parameters(model: nn.Module) -> tuple:
    """Count parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_student_from_checkpoint(student: StudentModel, checkpoint_path: str) -> None:
    """
    Initialize student from teacher checkpoint (VoCo/SuPreM compatible)."""
    print(f"Loading Student init from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "net" in checkpoint:
        state_dict = checkpoint["net"]
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = student.model.load_state_dict(state_dict, strict=False)
    print("  Student weights loaded.")
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)} (e.g. {missing_keys[:5]})")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)} (e.g. {unexpected_keys[:5]})")


if __name__ == "__main__":
    # Test 1: feature_size=48
    print("=" * 60)
    print("Test 1: feature_size=48")
    print("=" * 60)
    model1 = StudentModel(
        in_channels=1,
        out_channels=14,
        feature_size=48,
    )
    
    total1, trainable1 = count_parameters(model1)
    print(f"Total params: {total1:,}")
    print(f"Trainable params: {trainable1:,}")
    print(f"Need projection: {model1.need_projection}")
    
    x = torch.randn(1, 1, 128, 128, 128)
    logits1, features1 = model1(x)
    
    print(f"\nOutput logits: {logits1.shape}")
    print("Encoder features:")
    for k, v in features1.items():
        print(f"  {k}: {v.shape}")
    
    # Test 2: feature_size=24
    print("\n" + "=" * 60)
    print("Test 2: feature_size=24")
    print("=" * 60)
    model2 = StudentModel(
        in_channels=1,
        out_channels=14,
        feature_size=24,
    )
    
    total2, trainable2 = count_parameters(model2)
    print(f"Total params: {total2:,}")
    print(f"Trainable params: {trainable2:,}")
    print(f"Need projection: {model2.need_projection}")
    
    logits2, features2 = model2(x)
    
    print(f"\nOutput logits: {logits2.shape}")
    print("Encoder features (after projection):")
    for k, v in features2.items():
        print(f"  {k}: {v.shape}")

