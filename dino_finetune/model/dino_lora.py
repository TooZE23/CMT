import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRA
from .linear_decoder import LinearClassifier
from .fpn_decoder import FPNDecoder
from .patch_embed import PatchEmbed


class DINOEncoderLoRA(nn.Module):
    def __init__(
        self,
        encoder,
        r: int = 3,
        emb_dim: int = 1024,
        use_lora: bool = True,
        decoder_type: str = "mae", # ["mae","dino", "task"]
        img_dim: tuple = (520, 520),
        channel_num: int = 3,  
        cls_head_type: str = "cls" # ["cls","avgpool"]
    ):
        """The DINOv2 encoder-decoder model for finetuning to downstream tasks.

        Args:
            encoder (nn.Module): The ViT encoder model loaded with the DINOv2 model weights.
            r (int, optional): The rank parameter of the LoRA weights. Defaults to 3.
            emb_dim (int, optional): The embedding dimension of the encoder. Defaults to 1024.
            use_lora (bool, optional): Determines whether to use LoRA. Defaults to False.
            img_dim (tuple[int, int], optional): The input image dimension. Defaults to
                (520, 520).
        """
        super().__init__()
        assert img_dim[0] % encoder.patch_size == 0, "Wrong input shape for patches"
        assert r > 0

        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.use_lora = use_lora
        # DINO pretrained patch embedding expects 3-channel inputs.
        # When caller sets channel_num=1 (grayscale), we duplicate channels to RGB.
        self.channel_num = 3 if channel_num == 1 else channel_num

        patch_size = encoder.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.patch_dim = self.patch_size * self.patch_size * self.channel_num

        # Number of previous layers to use as input
        self.inter_layers = 4
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.cls_head_type = cls_head_type
        for param in self.encoder.parameters():
            param.requires_grad = False

        if decoder_type == "mae":
            self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.patch_dim),
        )
            
        if self.use_lora:
            self.lora_layers = list(range(len(self.encoder.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.encoder.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feature = self.encoder.forward_features(x)
        if self.decoder_type == "mae":
            patch_tokens = feature.get("x_norm_patchtokens", None)
            if patch_tokens is None:
                patch_tokens = next(v for k, v in feature.items() if isinstance(v, torch.Tensor) and v.dim() == 3)
            pred_patches = self.decoder(patch_tokens)  # (B, N, patch_dim)
            return pred_patches
        elif self.decoder_type == "dino":
            rep = None
            if "x_norm_clstoken" in feature and self.cls_head_type == "cls":
                rep = feature["x_norm_clstoken"]
            elif "pooled" in feature and self.cls_head_type == "cls":
                rep = feature["pooled"]
            else:
                pt = feature.get("x_norm_patchtokens", None)
                if pt is None:
                    pt = next(v for k,v in feature.items() if isinstance(v, torch.Tensor) and v.dim()==3)
                rep = pt.mean(dim=1)

            # logits = self.decoder(rep)  # (B, n_classes)
            return rep
        elif self.decoder_type == "task":
            return feature
        
    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights and decoder weights to a .pt file

        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}
        if self.decoder_type == "mae":
            decoder_weights = self.decoder.state_dict()
            torch.save({**w_a, **w_b, **decoder_weights}, filename)
        elif self.decoder_type in ["dino", "task"]:
            torch.save({**w_a, **w_b}, filename)
    
    def load_parameters(self, filename: str) -> None:
        """Load the LoRA and decoder weights from a file

        Args:
            filename (str): Filename of the weights
        """
        state_dict = torch.load(filename, map_location="cpu")
        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                if saved_key in state_dict:
                    saved_tensor = state_dict[saved_key]
                    w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                if saved_key in state_dict:
                    saved_tensor = state_dict[saved_key]
                    w_B_linear.weight = nn.Parameter(saved_tensor)

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
        