import math

import torch
import torch.nn as nn


class _LoRAQKVHiera(nn.Module):
    """LoRA wrapper for Hiera qkv projection.

    Hiera uses qkv as `Linear(dim, dim_out * 3)`, where dim_out can differ from dim.
    We therefore add LoRA updates to q and v slices with width `qkv.out_features // 3`.
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.qkv_dim = int(qkv.out_features // 3)

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        qkv[..., : self.qkv_dim] += new_q
        qkv[..., -self.qkv_dim :] += new_v
        return qkv


class SAM2EncoderLoRA(nn.Module):
    """MAE-style LoRA finetuning wrapper for SAM2 image encoder."""

    def __init__(
        self,
        sam_model,
        r: int = 3,
        emb_dim: int = 256,
        use_lora: bool = True,
        decoder_type: str = "mae",
        img_dim: tuple = (520, 520),
        channel_num: int = 3,
        feature_level: int = -1,
        patch_size: int = 16,
    ):
        super().__init__()
        assert r > 0
        if decoder_type not in {"mae", "task"}:
            raise ValueError(f"Unsupported decoder_type for SAM2EncoderLoRA: {decoder_type}")

        self.sam = sam_model
        self.encoder = sam_model.image_encoder
        self.use_lora = bool(use_lora)
        self.decoder_type = decoder_type
        self.img_dim = img_dim
        # SAM2 pretrained patch embedding expects 3-channel input.
        self.channel_num = 3 if channel_num == 1 else channel_num
        self.feature_level = int(feature_level)
        self.patch_size = int(patch_size)
        self.patch_dim = self.patch_size * self.patch_size * self.channel_num
        self._shape_checked = False

        hidden_dim = getattr(self.sam, "hidden_dim", None)
        if hidden_dim is None:
            hidden_dim = getattr(getattr(self.encoder, "neck", None), "d_model", None)
        self.emb_dim = int(hidden_dim if hidden_dim is not None else emb_dim)

        # Freeze full SAM2 by default; only LoRA and MAE decoder are trainable.
        for param in self.sam.parameters():
            param.requires_grad = False

        self.w_a = []
        self.w_b = []
        if self.use_lora:
            blocks = getattr(getattr(self.encoder, "trunk", None), "blocks", None)
            if blocks is None:
                raise RuntimeError("SAM2 image_encoder.trunk.blocks not found; cannot inject LoRA.")

            self.lora_layers = list(range(len(blocks)))
            for i, block in enumerate(blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim_in = int(w_qkv_linear.in_features)
                qkv_dim = int(w_qkv_linear.out_features // 3)

                w_a_q, w_b_q = self._create_lora_layer(dim_in, qkv_dim, r)
                w_a_v, w_b_v = self._create_lora_layer(dim_in, qkv_dim, r)

                self.w_a.extend([w_a_q, w_a_v])
                self.w_b.extend([w_b_q, w_b_v])

                block.attn.qkv = _LoRAQKVHiera(
                    w_qkv_linear,
                    w_a_q,
                    w_b_q,
                    w_a_v,
                    w_b_v,
                )
            self._reset_lora_parameters()

        if self.decoder_type == "mae":
            self.decoder = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.GELU(),
                nn.Linear(self.emb_dim, self.patch_dim),
            )

    def _create_lora_layer(self, dim_in: int, dim_out: int, r: int):
        w_a = nn.Linear(dim_in, r, bias=False)
        w_b = nn.Linear(r, dim_out, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def _select_feature_map(self, backbone_out: dict) -> torch.Tensor:
        feature_maps = backbone_out.get("backbone_fpn", None)
        if feature_maps is None or len(feature_maps) == 0:
            raise RuntimeError("SAM2 forward_image output missing `backbone_fpn` feature maps.")
        idx = self.feature_level if self.feature_level >= 0 else len(feature_maps) + self.feature_level
        if idx < 0 or idx >= len(feature_maps):
            raise ValueError(
                f"SAM2 feature_level={self.feature_level} is out of range for {len(feature_maps)} FPN levels."
            )
        return feature_maps[idx]

    def _validate_patch_alignment(self, x: torch.Tensor, feat: torch.Tensor) -> None:
        if self._shape_checked:
            return
        _, _, h, w = x.shape
        _, c, hf, wf = feat.shape

        if c != self.emb_dim:
            raise RuntimeError(
                f"SAM2 feature channel ({c}) != emb_dim ({self.emb_dim}). "
                "Please set EMB_DIM to SAM2 hidden_dim / neck.d_model."
            )
        if h % hf != 0 or w % wf != 0:
            raise RuntimeError(
                f"Input size {(h, w)} is not divisible by selected feature size {(hf, wf)}."
            )

        eff_patch_h = h // hf
        eff_patch_w = w // wf
        if eff_patch_h != eff_patch_w:
            raise RuntimeError(
                f"Non-square effective patch size from SAM2 feature map: ({eff_patch_h}, {eff_patch_w})."
            )
        if eff_patch_h != self.patch_size:
            raise RuntimeError(
                f"PATCH_SIZE mismatch: configured {self.patch_size}, but SAM2 feature level implies {eff_patch_h}. "
                "Adjust PATCH_SIZE or SAM2_FEATURE_LEVEL."
            )
        self._shape_checked = True

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        backbone_out = self.sam.forward_image(x)
        feat = self._select_feature_map(backbone_out)  # [B, C, Hf, Wf]
        self._validate_patch_alignment(x, feat)
        patch_tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        return patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.dim() != 4:
            raise RuntimeError(f"Expected input tensor [B,C,H,W], got shape {tuple(x.shape)}.")
        if x.shape[1] != 3:
            raise RuntimeError(
                "SAM2 pretrained image encoder expects 3-channel input. "
                "Please convert input to 3 channels before feeding."
            )

        patch_tokens = self._extract_patch_tokens(x)
        if self.decoder_type == "mae":
            return self.decoder(patch_tokens)
        return patch_tokens

    def save_parameters(self, filename: str) -> None:
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_b))}

        if self.decoder_type == "mae" and hasattr(self, "decoder"):
            decoder_weights = self.decoder.state_dict()
            torch.save({**w_a, **w_b, **decoder_weights}, filename)
        else:
            torch.save({**w_a, **w_b}, filename)

    def load_parameters(self, filename: str) -> None:
        state_dict = torch.load(filename, map_location="cpu")

        if self.use_lora:
            for i, w_a_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                if saved_key in state_dict:
                    w_a_linear.weight = nn.Parameter(state_dict[saved_key])
            for i, w_b_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                if saved_key in state_dict:
                    w_b_linear.weight = nn.Parameter(state_dict[saved_key])

        if self.decoder_type == "mae" and hasattr(self, "decoder"):
            decoder_head_dict = self.decoder.state_dict()
            decoder_state_dict = {}
            for k in decoder_head_dict.keys():
                if k in state_dict:
                    decoder_state_dict[k] = state_dict[k]
                elif f"decoder.{k}" in state_dict:
                    decoder_state_dict[k] = state_dict[f"decoder.{k}"]
            if len(decoder_state_dict) > 0:
                self.decoder.load_state_dict(decoder_state_dict, strict=False)
