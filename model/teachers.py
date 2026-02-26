"""Teacher model: same SwinUNETR architecture as student, frozen; outputs encoder features and decoder logits."""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class TeacherFullModel(nn.Module):
    """Teacher: same network as student (SwinUNETR), frozen. Outputs encoder features and decoder logits."""

    def __init__(
        self, 
        checkpoint_path: str, 
        name: str = "Teacher",
        use_v2: bool = None,
        num_classes: int = 14,
    ):
        super().__init__()
        self.name = name
        
        if use_v2 is None:
            if "suprem" in checkpoint_path.lower():
                use_v2 = False
                print(f"  [{name}] SuPreM detected, use_v2=False")
            elif "voco" in checkpoint_path.lower():
                use_v2 = True
                print(f"  [{name}] VoCo detected, use_v2=True")
            else:
                use_v2 = False
                print(f"  [{name}] Unknown model, default use_v2=False")
        else:
            print(f"  [{name}] use_v2={use_v2}")
        
        # Same architecture as student (SwinUNETR: feature_size=48, depths=(2,2,2,2))
        self.model = SwinUNETR(
            in_channels=1,
            out_channels=num_classes,
            feature_size=48,
            patch_size=2,
            depths=(2, 2, 2, 2),
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
        
        self.feature_dims = [48, 96, 192, 384, 768]
        self._load_checkpoint(checkpoint_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.features = {}
        self._register_hooks()
    
    def _load_checkpoint(self, checkpoint_path: str):
        print(f"Loading {self.name} from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ["state_dict", "net", "network_weights", "model_state_dict", "student_state_dict"]:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ["module.", "backbone.", "model."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_key = new_key.replace("swin_vit", "swinViT")
            
            if any(skip in new_key for skip in ["organ_embedding", "precls_conv", "GAP", "optimizer", "epoch", "scaler"]):
                continue
            new_state_dict[new_key] = value
        
        model_state = self.model.state_dict()
        matched = 0
        for key in model_state:
            if key in new_state_dict and model_state[key].shape == new_state_dict[key].shape:
                model_state[key] = new_state_dict[key]
                matched += 1
        
        self.model.load_state_dict(model_state, strict=False)
        print(f"  Loaded {matched}/{len(model_state)} layers")
    
    def _register_hooks(self):
        """Register hooks to capture encoder features per stage."""
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.features[name] = output[0]
                else:
                    self.features[name] = output
            return hook
        
        if hasattr(self.model, 'swinViT'):
            swin = self.model.swinViT
            for i, layer_name in enumerate(['layers1', 'layers2', 'layers3', 'layers4']):
                if hasattr(swin, layer_name):
                    layer = getattr(swin, layer_name)
                    if len(layer) > 0 and hasattr(layer[0], 'blocks') and len(layer[0].blocks) > 0:
                        layer[0].blocks[-1].register_forward_hook(get_hook(f'stage_{i}'))
    
    def forward(self, x: torch.Tensor, return_logits: bool = True):
        """
        Forward pass (encoder + decoder). Returns encoder features and optionally logits."""
        self.features.clear()
        
        with torch.no_grad():
            logits = self.model(x)
        
        encoder_features = self.features.copy()
        
        if return_logits:
            return encoder_features, logits
        else:
            return encoder_features


def load_teacher_full_models(
    teacher_1_path: str,
    teacher_2_path: str,
    device: str = "cuda",
    num_classes: int = 14,
):
    """
    Load two full teacher models (encoder + decoder).
    
    Args:
        teacher_1_path: Path to teacher 1 checkpoint
        teacher_2_path: Path to teacher 2 checkpoint
        device: Device string
        num_classes: Number of classes
    
    Returns:
        (teacher_1, teacher_2)
    """
    teacher_1 = TeacherFullModel(
        checkpoint_path=teacher_1_path,
        name="Teacher 1 (SuPreM)",
        num_classes=num_classes,
    ).to(device)
    
    teacher_2 = TeacherFullModel(
        checkpoint_path=teacher_2_path,
        name="Teacher 2 (VoCo)",
        num_classes=num_classes,
    ).to(device)
    
    return teacher_1, teacher_2


if __name__ == "__main__":
    teacher_1_path = "/path/to/teacher1/best_model.pth"
    teacher_2_path = "/path/to/teacher2/best_model.pth"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_1, teacher_2 = load_teacher_full_models(
        teacher_1_path, teacher_2_path, device=device, num_classes=14
    )
    
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    
    feats_1, logits_1 = teacher_1(x)
    feats_2, logits_2 = teacher_2(x)
    
    print("Teacher 1:")
    print("  Encoder features:")
    for k, v in feats_1.items():
        print(f"    {k}: {v.shape}")
    print(f"  Decoder logits: {logits_1.shape}")
    
    print("\nTeacher 2:")
    print("  Encoder features:")
    for k, v in feats_2.items():
        print(f"    {k}: {v.shape}")
    print(f"  Decoder logits: {logits_2.shape}")

