import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class MAEDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        img_dim=(224, 224),
        n_permutations=1000,
        mask_ratio=0.75,
        patch_size=16,
        channel_num=3,
        transform=None,
    ):
        self.root = root
        self.split = split
        self.img_dim = img_dim
        self.n_permutations = n_permutations  # kept for compatibility, unused in MAE mode
        self.mask_ratio = float(mask_ratio)
        self.patch_size = int(patch_size)
        self.channel_num = int(channel_num)
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio should be in [0, 1], got {self.mask_ratio}")

        self.h, self.w = self.img_dim
        if self.h % self.patch_size != 0 or self.w % self.patch_size != 0:
            raise ValueError(
                f"img_dim {self.img_dim} must be divisible by patch_size {self.patch_size}"
            )
        self.num_patches = (self.h // self.patch_size) * (self.w // self.patch_size)
        self.num_masked = int(round(self.mask_ratio * self.num_patches))
        self.num_masked = min(self.num_patches, max(0, self.num_masked))

        txt_file = os.path.join(self.root, f"{self.split}.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Split file not found: {txt_file}")
        self.img_paths = []
        with open(txt_file, "r") as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                if not path.lower().endswith(ALLOWED_EXTS):
                    continue
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                self.img_paths.append(path)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No valid image paths found in {txt_file}")

        self.transform = transform or transforms.Compose([
            transforms.Resize(img_dim),
            transforms.ToTensor()
        ])

    def _random_patch_mask(self, img: torch.Tensor):
        # img: (C, H, W), returns masked_img and patch-level mask (N,)
        mask = torch.zeros(self.num_patches, dtype=torch.float32)
        if self.num_masked == 0:
            return img.clone(), mask

        masked_indices = torch.randperm(self.num_patches)[: self.num_masked]
        mask[masked_indices] = 1.0

        patches = F.unfold(
            img.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # (1, C*p*p, N)
        patches[:, :, masked_indices] = 0.0
        masked_img = F.fold(
            patches,
            output_size=(self.h, self.w),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(0)
        return masked_img, mask

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        mode = "L" if self.channel_num == 1 else "RGB"
        img = Image.open(self.img_paths[idx]).convert(mode)
        img = self.transform(img)
        if self.channel_num == 1:
            img = img.repeat(3, 1, 1)
        masked_img, patch_mask = self._random_patch_mask(img)
        return masked_img, img, patch_mask

def patchify_images(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert image tensor (B, C, H, W) to patch tensor (B, N, patch_dim)."""
    b, c, h, w = imgs.shape
    p = patch_size
    assert h % p == 0 and w % p == 0, "Image size must be divisible by patch size"
    h_patches = h // p
    w_patches = w // p
    x = imgs.reshape(b, c, h_patches, p, w_patches, p)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, h_patches * w_patches, p * p * c)
    return x


def apply_patch_masks_to_images(
    imgs: torch.Tensor,
    patch_masks: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Apply patch-level bool masks to images.
    imgs: (B, C, H, W), patch_masks: (B, N), True means masked.
    """
    b, c, h, w = imgs.shape
    gh, gw = h // patch_size, w // patch_size
    assert patch_masks.shape == (b, gh * gw), "Mask shape does not match image patch grid"
    pixel_mask = patch_masks.view(b, gh, gw).unsqueeze(1).expand(b, c, gh, gw)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    masked = imgs.clone()
    masked[pixel_mask] = 0.0
    return masked


def build_random_patch_masks(batch_size: int, n_tokens: int, mask_ratio: float, device: str) -> torch.Tensor:
    n_mask = int(round(mask_ratio * n_tokens))
    n_mask = min(max(n_mask, 0), n_tokens)
    masks = torch.zeros((batch_size, n_tokens), dtype=torch.bool, device=device)
    if n_mask == 0:
        return masks
    for i in range(batch_size):
        idx = torch.randperm(n_tokens, device=device)[:n_mask]
        masks[i, idx] = True
    return masks


def masked_mae_recon_loss(
    pred_patches: torch.Tensor,
    target_patches: torch.Tensor,
    patch_masks: torch.Tensor,
):
    """
    MAE reconstruction loss on masked patches.
    Returns:
        batch_loss: scalar mean over images
        per_img_loss: (B,) each image masked-patch MSE
    """
    per_patch_mse = (pred_patches - target_patches).pow(2).mean(dim=-1)
    patch_masks = patch_masks.float()
    denom = patch_masks.sum(dim=-1).clamp(min=1.0)
    per_img_loss = (per_patch_mse * patch_masks).sum(dim=-1) / denom
    return per_img_loss.mean(), per_img_loss