import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from dino_finetune.model.dino_lora import DINOEncoderLoRA
from dino_finetune.model.sam2_lora import SAM2EncoderLoRA
from dino_finetune.model.sam2.build_sam import build_sam2
import math
from functools import reduce
from operator import mul

from dino_finetune.model.segformer import SegFormerHead


class VisualPrompt(nn.Module):
    """Visual Prompt module that adds learnable prompts to ViT block outputs"""
    
    def __init__(self, embed_dim: int, prompt_length: int = 10):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        
        if prompt_length > 0:
            self.prompt_embeddings = nn.Parameter(
                torch.randn(1, prompt_length, embed_dim)
            )
            val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + embed_dim))  # noqa
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            self.prompt_embeddings = None

    def forward(self, x: torch.Tensor, prefix_tokens: int = 1) -> torch.Tensor:
        if self.prompt_length == 0 or self.prompt_embeddings is None:
            return x
        batch_size = x.shape[0]
        prompts = self.prompt_embeddings.expand(batch_size, -1, -1)
        prefix_tokens = max(0, min(prefix_tokens, x.shape[1]))
        x_with_prompts = torch.cat(
            (
                x[:, :prefix_tokens, :],     # CLS + storage/register tokens
                prompts,                     # Prompt tokens
                x[:, prefix_tokens:, :],     # Patch tokens
            ),
            dim=1,
        )
        return x_with_prompts

class SAM2PromptTokens(nn.Module):
    """Per-block learnable prompt tokens for SAM2 attention KV prefix."""

    def __init__(self, embed_dim: int, prompt_length: int, scale_init: float = 1.0):
        super().__init__()
        if prompt_length <= 0:
            raise ValueError(f"prompt_length must be > 0, got {prompt_length}")
        self.embed_dim = int(embed_dim)
        self.prompt_length = int(prompt_length)
        self.prompt = nn.Parameter(torch.randn(1, self.prompt_length, self.embed_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        prompt_tokens = self.prompt.expand(batch_size, -1, -1)
        prompt_tokens = prompt_tokens.to(device=device, dtype=dtype)
        return self.scale * prompt_tokens


class MoELayer(nn.Module):
    """Mixture of Experts layer for multimodal fusion"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        num_modalities: int,
        specific_mask_ratio: float = 0.3,
        expert_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.num_modalities = int(num_modalities)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.concat_dim = input_dim * num_modalities
        self.specific_mask_ratio = float(specific_mask_ratio)
        self.expert_dropout = float(expert_dropout)
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        # 1 general expert + (num_experts - 1) specific experts.
        self.num_specific_experts = self.num_experts - 1

        self.gate = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )

        self.shared_proj = nn.Sequential(
            nn.Linear(self.concat_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

        self.general_expert = self._build_residual_expert()
        self.specific_experts = nn.ModuleList(
            [self._build_residual_expert() for _ in range(self.num_specific_experts)]
        )

    def _build_residual_expert(self) -> nn.Sequential:
        layers = [
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if self.expert_dropout > 0.0:
            layers.append(nn.Dropout(self.expert_dropout))
        layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        return nn.Sequential(*layers)

    def _random_mask_expert_input(self, expert_input: torch.Tensor) -> torch.Tensor:
        if self.specific_mask_ratio <= 0.0 or not self.training:
            return expert_input
        keep_prob = 1.0 - self.specific_mask_ratio
        keep_prob = max(1e-6, keep_prob)
        # Token-level random masking on expert input.
        keep_mask = (torch.rand(expert_input.shape[:2], device=expert_input.device) < keep_prob).to(
            expert_input.dtype
        )
        keep_mask = keep_mask.unsqueeze(-1)
        return expert_input * keep_mask / keep_prob

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        # modality_features: list of [B, N, D]
        concat_features = torch.cat(modality_features, dim=-1)  # [B, N, D*num_modalities]
        gate_input = concat_features.mean(dim=1)  # [B, D*num_modalities]
        gate_weights = self.gate(gate_input)  # [B, num_experts]

        base_features = self.shared_proj(concat_features)  # [B, N, D]
        general_out = self.general_expert(base_features)
        # general_out = base_features + self.general_scale * general_delta

        specific_outputs = []
        for idx, expert in enumerate(self.specific_experts):
            expert_in = self._random_mask_expert_input(base_features)
            expert_delta = expert(expert_in)
            specific_outputs.append(expert_delta)

        expert_outputs = [general_out] + specific_outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, N, D]
        gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_experts, 1, 1]
        fused_output = base_features + (expert_outputs * gate_weights).sum(dim=1)
        return fused_output


class MultiModalMoEFusion(nn.Module):
    """Multi-modal fusion model with MoE and visual prompts (prompt_layers and fusion_layers decoupled)"""
    
    def __init__(
        self,
        modality_configs: Dict,
        num_experts: int = 4,
        expert_hidden_dim: int = 512,
        prompt_length: int = 10,
        fusion_layers: List[int] = [8, 11], 
        prompt_layers: List[int] = [], 
        n_classes: int = 150,
        img_dim: tuple = (490, 490),
        seg_embed_dim: int = 256,
        specific_mask_ratio: float = 0.3,
    ):
        super().__init__()

        self.modality_names = list(modality_configs.keys())
        self.num_modalities = len(self.modality_names)
        self.fusion_layers = [int(x) for x in fusion_layers]
        self.prompt_layers = prompt_layers if isinstance(prompt_layers, str) else list(prompt_layers)
        self.n_classes = n_classes
        self.img_dim = img_dim
        self.prompt_length = prompt_length
        self.specific_mask_ratio = float(specific_mask_ratio)
        self.modality_prefix_tokens: Dict[str, int] = {}
        self.modality_backbone_type: Dict[str, str] = {}
        self.prefix_tokens = 0

        self.modality_encoders = nn.ModuleDict()
        self.embed_dim = None

        for modality_name, config in modality_configs.items():
            is_rgb = modality_name.lower() == 'img'
            vfm_type = str(config.get("VFM_TYPE", "dinov2")).lower()
            self.modality_backbone_type[modality_name] = vfm_type

            if vfm_type in {"dinov2", "dinov3"}:
                repo_or_dir = config.get("VFM_REPO_DIR", f"/home/yajing/.cache/torch/hub/facebookresearch/{vfm_type}")
                encoder = torch.hub.load(
                    repo_or_dir=repo_or_dir,
                    model=config['BACKBONE'],
                    source="local",
                )
                if is_rgb:
                    encoder_lora = DINOEncoderLoRA(
                        encoder=encoder,
                        r=config['R'],
                        emb_dim=config['EMB_DIM'],
                        use_lora=False,
                        decoder_type="task",
                        img_dim=self.img_dim,
                        channel_num=3,
                    )
                else:
                    encoder_lora = DINOEncoderLoRA(
                        encoder=encoder,
                        r=config['R'],
                        emb_dim=config['EMB_DIM'],
                        use_lora=True,
                        decoder_type="task",
                        img_dim=self.img_dim,
                        channel_num=config['IN_CHANNELS'],
                    )
            else:
                if vfm_type != "sam2":
                    raise ValueError(
                        f"Unsupported VFM_TYPE `{config.get('VFM_TYPE')}` for modality `{modality_name}`. "
                        "Expected one of [dinov2, dinov3, sam2]."
                    )
                sam2_cfg = config.get("SAM2_CFG")
                sam2_ckpt = config.get("SAM2_CKPT")
                if not sam2_cfg or not sam2_ckpt:
                    raise ValueError(
                        f"Modality `{modality_name}` uses sam2 but SAM2_CFG/SAM2_CKPT is missing."
                    )
                sam2_model = build_sam2(
                    config_file=sam2_cfg,
                    ckpt_path=sam2_ckpt,
                    device="cpu",
                    mode="eval",
                )
                hidden_dim = getattr(sam2_model, "hidden_dim", None)
                if hidden_dim is None:
                    hidden_dim = getattr(getattr(getattr(sam2_model, "image_encoder", None), "neck", None), "d_model", None)
                emb_dim = int(hidden_dim if hidden_dim is not None else config.get("EMB_DIM", 256))
                sam2_feature_level = int(config.get("SAM2_FEATURE_LEVEL", -1))
                sam2_patch_size = int(config.get("PATCH_SIZE", getattr(sam2_model, "backbone_stride", 16)))

                encoder_lora = SAM2EncoderLoRA(
                    sam_model=sam2_model,
                    r=config['R'],
                    emb_dim=emb_dim,
                    use_lora=(not is_rgb),
                    decoder_type="task",
                    img_dim=self.img_dim,
                    channel_num=(3 if is_rgb else config['IN_CHANNELS']),
                    feature_level=sam2_feature_level,
                    patch_size=sam2_patch_size,
                )

            if not is_rgb:
                lora_weights = config.get('LORA_WEIGHTS')
                if lora_weights:
                    encoder_lora.load_parameters(lora_weights)
                    print(f"Loaded LoRA weights for {modality_name} from {lora_weights}")
                self._freeze_lora_parameters(encoder_lora)
                for name, param in encoder_lora.named_parameters():
                    if 'patch_embed.proj' in name:
                        param.requires_grad = False
                        print(f"Freezing: {name}")

            self.modality_encoders[modality_name] = encoder_lora
            cur_prefix = self._get_prefix_tokens_count(encoder_lora.encoder)
            self.modality_prefix_tokens[modality_name] = cur_prefix
            if self.embed_dim is None:
                self.prefix_tokens = cur_prefix
            elif cur_prefix != self.prefix_tokens:
                raise ValueError(
                    f"All modalities must have same number of prefix tokens, got {self.prefix_tokens} and {cur_prefix}."
                )
            if self.embed_dim is None:
                self.embed_dim = int(getattr(encoder_lora, "emb_dim", config.get('EMB_DIM', 256)))
            else:
                cur_embed_dim = int(getattr(encoder_lora, "emb_dim", config.get('EMB_DIM', self.embed_dim)))
                if cur_embed_dim != self.embed_dim:
                    raise ValueError(
                        f"All modalities must have same embed dim, got {self.embed_dim} and {cur_embed_dim}."
                    )

        unique_backbone_types = set(self.modality_backbone_type.values())
        if len(unique_backbone_types) != 1:
            raise NotImplementedError(
                f"Mixed backbone families are not supported yet: {self.modality_backbone_type}. "
                "Please use all DINO or all SAM2 modalities in one run."
            )
        self.backbone_type = unique_backbone_types.pop()

        ref_modality = self.modality_names[0]
        self.patch_size = int(getattr(self.modality_encoders[ref_modality], "patch_size", 16))
        self.visual_prompts = nn.ModuleDict()
        if self.backbone_type == "sam2":
            ref_trunk_blocks = getattr(self.modality_encoders[ref_modality].encoder, "trunk", None).blocks
            self.sam2_num_blocks = len(ref_trunk_blocks)
            for modality_name in self.modality_names:
                cur_blocks = getattr(self.modality_encoders[modality_name].encoder, "trunk", None).blocks
                if len(cur_blocks) != self.sam2_num_blocks:
                    raise ValueError(
                        f"All SAM2 modalities must have same number of trunk blocks, got "
                        f"{self.sam2_num_blocks} and {len(cur_blocks)}."
                    )

            if self.prompt_length <= 0:
                self.prompt_layers = []
            else:
                self.prompt_layers = self._normalize_sam2_prompt_layers(self.prompt_layers, self.sam2_num_blocks)
                for modality_name in self.modality_names:
                    trunk_blocks = self.modality_encoders[modality_name].encoder.trunk.blocks
                    for block_idx in self.prompt_layers:
                        block_dim = int(getattr(trunk_blocks[block_idx], "dim"))
                        key = self._sam2_prompt_key(modality_name, block_idx)
                        self.visual_prompts[key] = SAM2PromptTokens(
                            embed_dim=block_dim,
                            prompt_length=self.prompt_length,
                            scale_init=1.0,
                        )
        else:
            if isinstance(self.prompt_layers, str):
                if self.prompt_layers.strip().lower() == "all":
                    dino_num_blocks = len(self.modality_encoders[ref_modality].encoder.blocks)
                    self.prompt_layers = list(range(dino_num_blocks))
                else:
                    self.prompt_layers = [int(x) for x in self.prompt_layers.split(",") if x.strip() != ""]
            else:
                self.prompt_layers = [int(x) for x in self.prompt_layers]
            if self.prompt_length > 0:
                for modality_name in self.modality_names:
                    for layer_idx in self.prompt_layers:
                        key = self._dino_prompt_key(modality_name, layer_idx)
                        self.visual_prompts[key] = VisualPrompt(self.embed_dim, prompt_length)

        # MoE layers for each fusion layer (decoupled)
        self.moe_layers = nn.ModuleDict()
        for layer_idx in self.fusion_layers:
            self.moe_layers[f'layer_{layer_idx}'] = MoELayer(
                self.embed_dim,
                expert_hidden_dim,
                num_experts,
                self.num_modalities,
                specific_mask_ratio=self.specific_mask_ratio,
            )
        
        # SegFormer head that accepts len(fusion_layers) inputs
        dims = [self.embed_dim for _ in self.fusion_layers]
        self.seg_head = SegFormerHead(dims=dims, embed_dim=seg_embed_dim, num_classes=self.n_classes)
        # initialize_segformer_head(self.seg_head)

    def _freeze_lora_parameters(self, encoder_lora: nn.Module):
        if hasattr(encoder_lora, "use_lora") and encoder_lora.use_lora:
            for w_a in encoder_lora.w_a:
                for param in w_a.parameters():
                    param.requires_grad = False
            for w_b in encoder_lora.w_b:
                for param in w_b.parameters():
                    param.requires_grad = False
            print("Froze LoRA parameters.")

    def _get_prefix_tokens_count(self, encoder: nn.Module) -> int:
        n_prefix = 0
        if hasattr(encoder, "cls_token") and getattr(encoder, "cls_token") is not None:
            n_prefix += 1
        if hasattr(encoder, "n_storage_tokens"):
            n_prefix += int(getattr(encoder, "n_storage_tokens"))
        elif hasattr(encoder, "num_register_tokens"):
            n_prefix += int(getattr(encoder, "num_register_tokens"))
        elif hasattr(encoder, "storage_tokens") and getattr(encoder, "storage_tokens") is not None:
            n_prefix += int(getattr(encoder, "storage_tokens").shape[1])
        elif hasattr(encoder, "register_tokens") and getattr(encoder, "register_tokens") is not None:
            n_prefix += int(getattr(encoder, "register_tokens").shape[1])
        return n_prefix

    def _prepare_tokens_standard(self, encoder: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        # Keep compatibility with fixed 3-channel pretrained patch embedding.
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if hasattr(encoder, "prepare_tokens_with_masks"):
            tokens_hw = encoder.prepare_tokens_with_masks(x, masks=None)
            if isinstance(tokens_hw, tuple) and len(tokens_hw) == 2:
                return tokens_hw[0], tokens_hw[1]
            return tokens_hw, None

        if hasattr(encoder, "prepare_tokens"):
            tokens = encoder.prepare_tokens(x)
            return tokens, None

        x_embed = encoder.patch_embed(x)
        if x_embed.ndim == 4:
            b, h, w, _ = x_embed.shape
            x_tokens = x_embed.flatten(1, 2)
            hw = (h, w)
        elif x_embed.ndim == 3:
            b = x_embed.shape[0]
            x_tokens = x_embed
            hw = None
        else:
            raise RuntimeError(f"Unexpected patch_embed output ndim={x_embed.ndim}")

        token_parts = []
        if hasattr(encoder, "cls_token") and encoder.cls_token is not None:
            token_parts.append(encoder.cls_token.expand(b, -1, -1))
        if hasattr(encoder, "storage_tokens") and getattr(encoder, "storage_tokens") is not None:
            token_parts.append(encoder.storage_tokens.expand(b, -1, -1))
        elif hasattr(encoder, "register_tokens") and getattr(encoder, "register_tokens") is not None:
            token_parts.append(encoder.register_tokens.expand(b, -1, -1))
        token_parts.append(x_tokens)
        return torch.cat(token_parts, dim=1), hw

    def _get_block_rope(self, encoder: nn.Module, hw: Optional[Tuple[int, int]]):
        if hw is None:
            return None
        if hasattr(encoder, "rope_embed") and getattr(encoder, "rope_embed") is not None:
            h, w = hw
            return encoder.rope_embed(H=h, W=w)
        return None

    def _forward_block_compat(self, blk: nn.Module, x: torch.Tensor, rope):
        if rope is not None:
            try:
                return blk(x, rope)
            except TypeError:
                pass
        return blk(x)

    def _normalize_sam2_prompt_layers(self, prompt_layers_raw, num_blocks: int) -> List[int]:
        if isinstance(prompt_layers_raw, str):
            if prompt_layers_raw.strip().lower() == "all":
                return list(range(num_blocks))
            prompt_layers_raw = [x for x in prompt_layers_raw.split(",") if x.strip() != ""]

        if not isinstance(prompt_layers_raw, (list, tuple)):
            raise ValueError(
                f"Invalid SAM2 prompt layers `{prompt_layers_raw}`. Use list of block indices or `all`."
            )

        if len(prompt_layers_raw) == 0:
            return list(range(num_blocks))
        if len(prompt_layers_raw) == 1 and isinstance(prompt_layers_raw[0], str):
            if prompt_layers_raw[0].strip().lower() == "all":
                return list(range(num_blocks))

        parsed = []
        for v in prompt_layers_raw:
            idx = int(v)
            if idx < 0:
                idx = num_blocks + idx
            if idx < 0 or idx >= num_blocks:
                raise ValueError(
                    f"SAM2 prompt layer index {v} is out of range for {num_blocks} blocks."
                )
            parsed.append(idx)
        # Deduplicate while preserving order.
        return list(dict.fromkeys(parsed))

    def _sam2_prompt_key(self, modality_name: str, block_idx: int) -> str:
        return f"sam2_{modality_name}_block_{int(block_idx)}"

    def _dino_prompt_key(self, modality_name: str, layer_idx: int) -> str:
        return f"dino_{modality_name}_layer_{int(layer_idx)}"

    def _get_sam2_block_prompt_tokens(
        self, modality_name: str, block_idx: int, x_grid: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.prompt_length <= 0:
            return None
        if int(block_idx) not in self.prompt_layers:
            return None
        key = self._sam2_prompt_key(modality_name, block_idx)
        if key not in self.visual_prompts:
            return None

        if x_grid.ndim != 4:
            raise RuntimeError(f"SAM2 block prompt expects [B,H,W,C], got shape {tuple(x_grid.shape)}.")
        bsz, _, _, _ = x_grid.shape
        return self.visual_prompts[key](batch_size=bsz, dtype=x_grid.dtype, device=x_grid.device)
    
    def forward_with_vpt_and_moe(self, modality_name: str, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward pass for a single modality.
           - prompts inserted at self.prompt_layers
           - features for fusion collected at self.fusion_layers
        Returns:
            x: final token sequence after encoder.norm
            modality_features: dict mapping fusion_layer_idx -> cloned features (the pre-block snapshot, to match previous behavior)
        """
        if self.backbone_type == "sam2":
            raise RuntimeError("SAM2 path should call `_forward_sam2_moe` instead of token-block forwarding.")

        encoder = self.modality_encoders[modality_name].encoder
        
        x, hw = self._prepare_tokens_standard(encoder, x)
        
        modality_features = {}
        prefix_tokens = self.modality_prefix_tokens.get(modality_name, self.prefix_tokens)
        key = None
        for i, blk in enumerate(encoder.blocks):
            rope = self._get_block_rope(encoder, hw)
            # If this layer is a prompt layer, insert prompts before passing to block
            if self.prompt_length > 0 and i in self.prompt_layers:
                key = self._dino_prompt_key(modality_name, i)
                if key in self.visual_prompts:
                    x = self.visual_prompts[key](x, prefix_tokens=prefix_tokens)
            
            # Forward through transformer block
            x = self._forward_block_compat(blk, x, rope)
            
            # If we had inserted prompts at this layer, remove them after block
            if key is not None and key in self.visual_prompts:
                total_len = x.shape[1]
                if total_len >= prefix_tokens + self.prompt_length:
                    x = torch.cat([
                        x[:, :prefix_tokens, :],                           # prefix tokens
                        x[:, prefix_tokens + self.prompt_length:, :],      # patch tokens (skip prompts)
                    ], dim=1)
            if i in self.fusion_layers:
                modality_features[i] = x.clone()

        # final norm
        x = encoder.norm(x)
        return x, modality_features

    def _resolve_feature_index(self, num_levels: int, level_idx: int) -> int:
        idx = int(level_idx)
        if idx < 0:
            idx = num_levels + idx
        if idx < 0 or idx >= num_levels:
            raise ValueError(
                f"Fusion level {level_idx} is out of range for SAM2 backbone_fpn with {num_levels} levels."
            )
        return idx

    def _forward_sam2_modality_features(
        self, modality_name: str, x: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        wrapper = self.modality_encoders[modality_name]
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.dim() != 4:
            raise RuntimeError(
                f"Expected input tensor [B,C,H,W] for modality `{modality_name}`, got shape {tuple(x.shape)}."
            )
        if x.shape[1] != 3:
            raise RuntimeError(
                f"SAM2 path currently expects 3-channel inputs, got {x.shape[1]} channels for `{modality_name}`."
            )

        sam = wrapper.sam
        image_encoder = sam.image_encoder
        trunk = image_encoder.trunk

        # Manual trunk forward to pass per-block prompt tokens into attention KV.
        x_grid = trunk.patch_embed(x)  # [B, H, W, C]
        x_grid = x_grid + trunk._get_pos_embed(x_grid.shape[1:3])

        trunk_outputs = []
        for block_idx, blk in enumerate(trunk.blocks):
            prompt_tokens = self._get_sam2_block_prompt_tokens(modality_name, block_idx, x_grid)
            x_grid = blk(x_grid, prompt_tokens=prompt_tokens)
            if (block_idx == trunk.stage_ends[-1]) or (
                block_idx in trunk.stage_ends and trunk.return_interm_layers
            ):
                trunk_outputs.append(x_grid.permute(0, 3, 1, 2))

        feature_maps, _ = image_encoder.neck(trunk_outputs)
        if image_encoder.scalp > 0:
            feature_maps = feature_maps[: -image_encoder.scalp]

        if len(feature_maps) == 0:
            raise RuntimeError(f"SAM2 image encoder produced no FPN features for `{modality_name}`.")

        selected_features = {}
        for level_idx in self.fusion_layers:
            resolved_idx = self._resolve_feature_index(len(feature_maps), level_idx)
            feat_map = feature_maps[resolved_idx]
            if feat_map.shape[1] != self.embed_dim:
                raise RuntimeError(
                    f"SAM2 feature channels ({feat_map.shape[1]}) do not match embed_dim ({self.embed_dim}) "
                    f"for modality `{modality_name}` at level {level_idx}."
                )
            selected_features[level_idx] = feat_map
        return selected_features

    def _forward_sam2_moe(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        all_modality_features: Dict[str, Dict[int, torch.Tensor]] = {}
        for modality_name in self.modality_names:
            if modality_name not in inputs:
                raise KeyError(f"Missing modality `{modality_name}` in model inputs.")
            all_modality_features[modality_name] = self._forward_sam2_modality_features(
                modality_name, inputs[modality_name]
            )

        fused_feature_maps = []
        for layer_idx in self.fusion_layers:
            layer_features = []
            target_hw = None
            for modality_name in self.modality_names:
                feat_map = all_modality_features[modality_name][layer_idx]  # [B, C, H, W]
                if target_hw is None:
                    target_hw = feat_map.shape[-2:]
                elif feat_map.shape[-2:] != target_hw:
                    feat_map = F.interpolate(feat_map, size=target_hw, mode="bilinear", align_corners=False)
                layer_features.append(feat_map.flatten(2).transpose(1, 2).contiguous())  # [B, N, D]

            moe_key = f'layer_{layer_idx}'
            layer_fused = self.moe_layers[moe_key](layer_features)  # [B, N, D]

            bsz, n_tokens, dim = layer_fused.shape
            h, w = target_hw
            if n_tokens != h * w:
                raise RuntimeError(
                    f"SAM2 fused tokens length {n_tokens} does not match spatial size {h}x{w} at fusion level {layer_idx}."
                )
            fused_map = layer_fused.transpose(1, 2).contiguous().view(bsz, dim, h, w)
            fused_feature_maps.append(fused_map)

        fused_tuple = tuple(fused_feature_maps)
        logits = self.seg_head(fused_tuple)
        logits = F.interpolate(logits, size=self.img_dim, mode="bilinear", align_corners=False)
        return logits
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.backbone_type == "sam2":
            return self._forward_sam2_moe(inputs)

        batch_size = list(inputs.values())[0].shape[0]
        
        # Extract features from each modality at fusion layers
        all_modality_features = {}
        
        for modality_name in self.modality_names:
            if modality_name not in inputs:
                raise KeyError(f"Missing modality `{modality_name}` in model inputs.")
            x = inputs[modality_name]
            _, layer_features = self.forward_with_vpt_and_moe(modality_name, x)
            all_modality_features[modality_name] = layer_features
        
        # For each fusion layer, compute MoE fused tokens and convert to [B, D, H, W]
        fused_feature_maps = []  # list of tensors [B, D, H, W] in same order as self.fusion_layers
        for layer_idx in self.fusion_layers:
            layer_features = []
            for modality_name in self.modality_names:
                if layer_idx not in all_modality_features[modality_name]:
                    raise KeyError(f"Modality {modality_name} did not produce features for fusion layer {layer_idx}. Check encoder depth and fusion_layers/prompt_layers settings.")
                feat = all_modality_features[modality_name][layer_idx]  # [B, N, D]
                layer_features.append(feat)
            
            moe_key = f'layer_{layer_idx}'
            layer_fused = self.moe_layers[moe_key](layer_features)  # [B, N, D]

            # compute expected patch dims
            patch_h = self.img_dim[0] // self.patch_size
            patch_w = self.img_dim[1] // self.patch_size
            expected_patch_tokens = patch_h * patch_w

            # remove prefix tokens (cls + storage/register) if present
            prefix_tokens = self.prefix_tokens
            if layer_fused.shape[1] > expected_patch_tokens and prefix_tokens > 0:
                fused_tokens = layer_fused[:, prefix_tokens:, :]
            else:
                fused_tokens = layer_fused

            if fused_tokens.shape[1] != expected_patch_tokens:
                # warning but continue (user can handle)
                print(f"Warning: fusion layer {layer_idx} expected {expected_patch_tokens} patch tokens, got {fused_tokens.shape[1]}")

            # [B, N, D] -> [B, D, H, W]
            fused_tokens = fused_tokens.transpose(1, 2).contiguous()  # [B, D, N]
            try:
                fused_map = fused_tokens.view(batch_size, self.embed_dim, patch_h, patch_w)
            except Exception as e:
                # fallback: try to reshape with nearest possible square if N is square
                N = fused_tokens.shape[-1]
                side = int(N ** 0.5)
                if side * side == N:
                    fused_map = fused_tokens.view(batch_size, self.embed_dim, side, side)
                    print(f"Note: reshaped fusion layer {layer_idx} tokens to {side}x{side} spatial dims as fallback.")
                else:
                    raise RuntimeError(f"Cannot reshape fused tokens of length {N} into a square map for layer {layer_idx}") from e

            fused_feature_maps.append(fused_map)

        fused_tuple = tuple(fused_feature_maps)
        logits = self.seg_head(fused_tuple)  # [B, n_classes, H, W] with H,W = fused_feature_maps[0] spatial dims

        # upsample logits to desired img_dim
        logits = F.interpolate(
            logits,
            size=self.img_dim,
            mode="bilinear",
            align_corners=False,
        )

        return logits
