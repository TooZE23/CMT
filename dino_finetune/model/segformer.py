import torch
from torch import nn, Tensor
from typing import Tuple, List
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] -> flatten spatial -> linear -> back to [B, embed_dim, H, W] after caller reshape
        x = x.flatten(2).transpose(1, 2)   # [B, H*W, C]
        x = self.proj(x)                  # [B, H*W, embed_dim]
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    """
    SegFormer head that supports variable number of input feature maps.
    dims: list of channel dims for each input feature (token embedding dim).
    embed_dim: the head internal embedding dimension (default 256).
    num_classes: segmentation classes.
    """
    def __init__(self, dims: List[int], embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        self.num_inputs = len(dims)
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim * self.num_inputs, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, ...]) -> Tensor:
        """
        features: tuple of tensors, each is [B, C, H_i, W_i]
        We map each with an MLP (linear_cX) to embed_dim, then upsample to the highest resolution (features[0]'s H,W)
        and concatenate.
        """
        assert len(features) == self.num_inputs, f"Expect {self.num_inputs} features, got {len(features)}"

        B, _, H, W = features[0].shape
        outs = []

        # first feature
        out0 = getattr(self, "linear_c1")(features[0])  # returns [B, H0*W0, embed_dim]
        out0 = out0.permute(0, 2, 1).reshape(B, -1, features[0].shape[-2], features[0].shape[-1])  # [B, embed_dim, H0, W0]
        outs.append(out0)

        # other features
        for i, feat in enumerate(features[1:]):
            linear = getattr(self, f"linear_c{i+2}")
            cf = linear(feat)  # [B, H_i*W_i, embed_dim]
            cf = cf.permute(0, 2, 1).reshape(B, -1, feat.shape[-2], feat.shape[-1])  # [B, embed_dim, Hi, Wi]
            cf = F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(cf)

        # concat in reversed order as original SegFormer did (optional)
        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg
