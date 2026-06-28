import math
import torch
import torch.nn as nn
from .lora import LoRA  

class SAMEncoderLoRA(nn.Module):
    def __init__(self, sam_model, r=3, use_lora=True, patch_size=16, in_chans=3):
        super().__init__()
        self.encoder = sam_model.model.visual  
        self.use_lora = use_lora

        for param in self.encoder.parameters():
            param.requires_grad = False

        if in_chans != 3:
            old_proj = self.encoder.patch_embed.proj
            self.encoder.patch_embed.proj = nn.Conv2d(
                in_chans,
                old_proj.out_channels,
                kernel_size=patch_size,
                stride=patch_size
            )

        if self.use_lora:
            self.w_a = []
            self.w_b = []

            for block in self.encoder.blocks:
                w_qkv = block.attn.qkv
                dim = w_qkv.in_features

                w_a_q, w_b_q = self._create_lora_layer(dim, r)
                w_a_v, w_b_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_q, w_a_v])
                self.w_b.extend([w_b_q, w_b_v])

                block.attn.qkv = LoRA(
                    w_qkv,
                    w_a_q,
                    w_b_q,
                    w_a_v,
                    w_b_v
                )

            self._reset_lora_parameters()

    def _create_lora_layer(self, dim, r):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self):
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x):
        features = self.encoder.forward_features(x)  # [B, N+1, D]
        patch_features = features[:, 1:, :]
        return patch_features

    def save_lora(self, path):
        state = {}
        for i, w_a in enumerate(self.w_a):
            state[f"w_a_{i}"] = w_a.weight.data
        for i, w_b in enumerate(self.w_b):
            state[f"w_b_{i}"] = w_b.weight.data
        torch.save(state, path)

    def load_lora(self, path):
        state = torch.load(path)
        for i, w_a in enumerate(self.w_a):
            w_a.weight = nn.Parameter(state[f"w_a_{i}"])
        for i, w_b in enumerate(self.w_b):
            w_b.weight = nn.Parameter(state[f"w_b_{i}"])
