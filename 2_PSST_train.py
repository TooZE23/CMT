# file: finetune_dino_student_teacher.py
import copy
import json
import logging
import argparse
import os
import yaml
from typing import List
from functools import partial
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from dino_finetune import (
    DINOEncoderLoRA,
    SAM2EncoderLoRA,
    DINOHead,
    BlockDiagonalMask,
    DataAugmentationDINO,
    MaskingGenerator,
    collate_data_and_cast,
    DINOLoss,
    iBOTPatchLoss,
    KoLeoLoss,
    build_sam2,
)
from dino_finetune.data import ADE20kDatasetImage
from dino_finetune.schedulers import PolyLR, WarmupPolyLR
from dino_finetune.utils import fix_seeds
from dino_finetune.data.jigsaw_dataset import (
    build_random_patch_masks,
    apply_patch_masks_to_images,
    patchify_images,
    masked_mae_recon_loss,
)


def _dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_rank() -> int:
    return torch.distributed.get_rank() if _dist_available_and_initialized() else 0


def _get_world_size() -> int:
    return torch.distributed.get_world_size() if _dist_available_and_initialized() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _none_if_empty(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _load_backbone(config: dict, device: torch.device) -> nn.Module:
    vfm_type = str(config["VFM_TYPE"]).lower()
    size_key = str(config["SIZE"]).lower()

    if vfm_type in {"dinov2", "dinov3"}:
        config["PATCH_SIZE"] = 16 if vfm_type == "dinov3" else 14
        backbones = {
            "small": f"{vfm_type}_vits{config['PATCH_SIZE']}{'_reg' if vfm_type == 'dinov2' else ''}",
            "base": f"{vfm_type}_vitb{config['PATCH_SIZE']}{'_reg' if vfm_type == 'dinov2' else ''}",
            "large": f"{vfm_type}_vitl{config['PATCH_SIZE']}{'_reg' if vfm_type == 'dinov2' else ''}",
            "giant": f"{vfm_type}_vitg{config['PATCH_SIZE']}{'_reg' if vfm_type == 'dinov2' else ''}",
        }
        if size_key not in backbones:
            raise ValueError(f"Unsupported SIZE for {vfm_type}: {config['SIZE']}")

        repo_or_dir = config.get("VFM_REPO_DIR", f"/home/yajing/.cache/torch/hub/facebookresearch/{vfm_type}")
        encoder = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=backbones[size_key],
            source="local",
        ).to(device)
        config["EMB_DIM"] = encoder.num_features
        return encoder

    if vfm_type == "sam2":
        sam2_cfg = _none_if_empty(config.get("SAM2_CFG"))
        sam2_ckpt = _none_if_empty(config.get("SAM2_CKPT"))
        if not sam2_cfg or not sam2_ckpt:
            raise ValueError("When VFM_TYPE='sam2', both SAM2_CFG and SAM2_CKPT must be set in config.")

        sam2_model = build_sam2(
            config_file=sam2_cfg,
            ckpt_path=sam2_ckpt,
            device=str(device),
            mode="eval",
        )
        hidden_dim = getattr(sam2_model, "hidden_dim", None)
        if hidden_dim is None:
            hidden_dim = getattr(getattr(getattr(sam2_model, "image_encoder", None), "neck", None), "d_model", None)
        config["EMB_DIM"] = int(hidden_dim if hidden_dim is not None else config.get("EMB_DIM", 256))
        config["SAM2_FEATURE_LEVEL"] = int(config.get("SAM2_FEATURE_LEVEL", -1))

        if config.get("PATCH_SIZE", None) is None:
            config["PATCH_SIZE"] = int(getattr(sam2_model, "backbone_stride", 16))
        else:
            config["PATCH_SIZE"] = int(config["PATCH_SIZE"])
        return sam2_model.to(device)

    raise ValueError(f"Unsupported VFM_TYPE: {config['VFM_TYPE']}. Expected one of [dinov2, dinov3, sam2].")


def _broadcast_long_list(values: List[int], device: torch.device) -> List[int]:
    if not _dist_available_and_initialized():
        return values
    if _is_main_process():
        value_tensor = torch.tensor(values, dtype=torch.long, device=device)
        length_tensor = torch.tensor([value_tensor.numel()], dtype=torch.long, device=device)
    else:
        value_tensor = None
        length_tensor = torch.zeros(1, dtype=torch.long, device=device)
    torch.distributed.broadcast(length_tensor, src=0)
    out_tensor = value_tensor
    if not _is_main_process():
        out_tensor = torch.empty(int(length_tensor.item()), dtype=torch.long, device=device)
    if int(length_tensor.item()) > 0:
        torch.distributed.broadcast(out_tensor, src=0)
    return out_tensor.cpu().tolist()


@torch.no_grad()
def update_ema(teacher_model: nn.Module, student_model: nn.Module, m: float):
    """
    EMA update for every parameter (in-place)
    """
    for tp, sp in zip(teacher_model.parameters(), student_model.parameters()):
        tp.data.mul_(m).add_(sp.data * (1.0 - m))


@torch.no_grad()
def update_ema_module(teacher_mod: nn.Module, student_mod: nn.Module, m: float):
    """EMA for separate small modules (e.g., projectors)"""
    for tp, sp in zip(teacher_mod.parameters(), student_mod.parameters()):
        tp.data.mul_(m).add_(sp.data * (1.0 - m))


def _ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def _extract_patch_tokens(feats: dict) -> torch.Tensor:
    if isinstance(feats, torch.Tensor):
        if feats.dim() != 3:
            raise RuntimeError(f"Expected patch token tensor [B,N,D], got shape {tuple(feats.shape)}.")
        return feats
    patch_tokens = feats.get("x_norm_patchtokens", None)
    if patch_tokens is not None:
        return patch_tokens
    # fallback for different backbone key names
    return next(v for _, v in feats.items() if isinstance(v, torch.Tensor) and v.dim() == 3)


def _extract_cls_tokens(feats: dict) -> torch.Tensor:
    if isinstance(feats, torch.Tensor):
        return feats.mean(dim=1)
    if "x_norm_clstoken" in feats:
        return feats["x_norm_clstoken"]
    if "pooled" in feats:
        return feats["pooled"]
    return _extract_patch_tokens(feats).mean(dim=1)

def _save_mae_filter_cache(cache_path: str, image_paths: List[str], keep_indices: List[int], config: dict) -> None:
    keep_paths = [str(image_paths[i]) for i in keep_indices]
    payload = {
        "version": 1,
        "exp_name": str(config.get("EXP_NAME", "")),
        "dataset_root": str(config.get("DATASET_ROOT", "")),
        "total_samples": int(len(image_paths)),
        "keep_count": int(len(keep_paths)),
        "keep_paths": keep_paths,
        "threshold": config.get("MAE_FILTER_THRESHOLD"),
        "top_ratio": config.get("MAE_FILTER_TOP_RATIO"),
    }
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logging.info("Saved MAE filter cache: %s (keep %d)", cache_path, len(keep_paths))


def _load_mae_filter_cache(cache_path: str, image_paths: List[str]) -> List[int]:
    if not os.path.isfile(cache_path):
        raise RuntimeError(
            f"MAE filter cache file not found: {cache_path}. "
            "Run once with DISABLE_MAE_FILTER=false to build it."
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    keep_paths = payload.get("keep_paths", [])
    if not isinstance(keep_paths, list) or len(keep_paths) == 0:
        raise RuntimeError(f"Invalid or empty keep_paths in MAE filter cache: {cache_path}")

    path_to_indices = {}
    for idx, p in enumerate(image_paths):
        key = str(p)
        if key not in path_to_indices:
            path_to_indices[key] = []
        path_to_indices[key].append(idx)

    keep_indices = []
    missing = 0
    for p in keep_paths:
        key = str(p)
        idx_list = path_to_indices.get(key, [])
        if len(idx_list) == 0:
            missing += 1
            continue
        keep_indices.append(idx_list.pop(0))

    if len(keep_indices) == 0:
        raise RuntimeError(
            f"Loaded MAE cache from {cache_path}, but none of cached paths match current dataset."
        )
    if missing > 0:
        logging.warning(
            "MAE cache had %d unmatched samples (matched %d).",
            missing,
            len(keep_indices),
        )
    keep_indices.sort()
    logging.info("Loaded MAE filter cache: %s (matched %d)", cache_path, len(keep_indices))
    return keep_indices


def build_stage1_style_mae_head(emb_dim: int, patch_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(emb_dim, emb_dim),
        nn.GELU(),
        nn.Linear(emb_dim, patch_dim),
    )

def load_stage1_mae_head_weights(mae_head: nn.Module, ckpt_path: str) -> None:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    head_state = mae_head.state_dict()
    load_state = {}
    for k in head_state.keys():
        if k in state_dict:
            load_state[k] = state_dict[k]
        elif f"decoder.{k}" in state_dict:
            load_state[k] = state_dict[f"decoder.{k}"]
    if len(load_state) == 0:
        raise RuntimeError(f"Cannot find MAE head weights in checkpoint: {ckpt_path}")
    mae_head.load_state_dict(load_state, strict=False)
    print(f"Loaded MAE head weights from {ckpt_path}")


@torch.no_grad()
def filter_indices_by_mae_error(
    image_paths: List[str],
    student: nn.Module,
    mae_head: nn.Module,
    config: dict,
    device: torch.device,
) -> List[int]:
    preprocess = transforms.Compose(
        [
            transforms.Resize(tuple(config['IMG_DIM'])),
            transforms.ToTensor(),
        ]
    )
    mode = "L" if config['CHANNEL_NUM'] == 1 else "RGB"
    n_tokens = (config['IMG_DIM'][0] // config['PATCH_SIZE']) * (config['IMG_DIM'][1] // config['PATCH_SIZE'])
    all_errors = []

    student.eval()
    mae_head.eval()

    for start in range(0, len(image_paths), config['MAE_FILTER_BATCH_SIZE']):
        batch_paths = image_paths[start : start + config['MAE_FILTER_BATCH_SIZE']]
        imgs = []
        for path in batch_paths:
            resolved_path = path if os.path.isabs(path) else os.path.join(config['DATASET_ROOT'], path)
            img = Image.open(resolved_path).convert(mode)
            imgs.append(preprocess(img))
        imgs = torch.stack(imgs, dim=0).to(device)
        imgs = _ensure_three_channels(imgs)
        patch_masks = build_random_patch_masks(
            batch_size=imgs.shape[0],
            n_tokens=n_tokens,
            mask_ratio=config['MAE_MASK_RATIO'],
            device=device,
        )
        masked_imgs = apply_patch_masks_to_images(imgs, patch_masks, config['PATCH_SIZE'])
        feats = student(masked_imgs)
        pred_patches = mae_head(_extract_patch_tokens(feats))
        target_patches = patchify_images(imgs, config['PATCH_SIZE'])
        _, per_img_loss = masked_mae_recon_loss(pred_patches, target_patches, patch_masks)
        all_errors.append(per_img_loss.detach().cpu())

    mae_errors = torch.cat(all_errors, dim=0)

    if config['MAE_FILTER_THRESHOLD'] is not None:
        keep_indices = (mae_errors > config['MAE_FILTER_THRESHOLD']).nonzero(as_tuple=False).flatten()
    else:
        keep_ratio = min(max(config['MAE_FILTER_TOP_RATIO'], 0.0), 1.0)
        keep_num = max(1, int(round(keep_ratio * mae_errors.numel())))
        keep_indices = torch.topk(mae_errors, k=keep_num, largest=True).indices

    keep_indices = keep_indices.sort().values.tolist()
    logging.info(
        "MAE filter: keep %d / %d samples (mean error %.6f, max %.6f)",
        len(keep_indices),
        len(image_paths),
        float(mae_errors.mean().item()),
        float(mae_errors.max().item()),
    )
    return keep_indices


# -----------------------
# Training loop (student-teacher)
# -----------------------
def finetune_dino_student_teacher(config: dict, encoder: nn.Module):
    use_ddp = bool(config.get("DDP", False)) and _dist_available_and_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    use_amp = bool(config.get("AMP", False)) and device.type == "cuda"
    inputs_dtype = torch.float32
    vfm_type = str(config["VFM_TYPE"]).lower()

    # Student backbone (DINO/SAM2 + LoRA)
    if vfm_type == "sam2":
        student = SAM2EncoderLoRA(
            sam_model=encoder,
            r=config['R'],
            emb_dim=config['EMB_DIM'],
            use_lora=True,
            decoder_type="task",
            img_dim=config['IMG_DIM'],
            channel_num=config['CHANNEL_NUM'],
            feature_level=config.get("SAM2_FEATURE_LEVEL", -1),
            patch_size=config['PATCH_SIZE'],
        )
    else:
        student = DINOEncoderLoRA(
            encoder=encoder,
            r=config['R'],
            emb_dim=config['EMB_DIM'],
            img_dim=config['IMG_DIM'],
            channel_num=config['CHANNEL_NUM'],
            use_lora=True,
            decoder_type="task",
        )

    # Load LoRA weights before moving model to target device, then move once.
    lora_weights = _none_if_empty(config.get('LORA_WEIGHTS'))
    if lora_weights:
        student.load_parameters(lora_weights)
        print(f"Loaded LoRA weights from {lora_weights}")
    student = student.to(device)

    # SSL loss modules
    dino_clstoken_loss = DINOLoss(out_dim=config['PROJ_DIM'], center_momentum=config['CENTER_M']).to(device)
    ibot_loss_fn = iBOTPatchLoss(patch_out_dim=config['PROJ_DIM'], center_momentum=config['CENTER_M']).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)

    # MAE head (same structure as Stage-1) and warm-start from Stage-1 checkpoint
    effective_channel_num = 3 if config['CHANNEL_NUM'] == 1 else config['CHANNEL_NUM']
    patch_dim = config['PATCH_SIZE'] * config['PATCH_SIZE'] * effective_channel_num
    student_mae_head = build_stage1_style_mae_head(config['EMB_DIM'], patch_dim).to(device)
    stage1_mae_weights = _none_if_empty(config.get('STAGE1_MAE_WEIGHTS')) or lora_weights
    if stage1_mae_weights is None:
        raise ValueError("Please provide STAGE1_MAE_WEIGHTS or LORA_WEIGHTS to warm-start MAE head.")
    load_stage1_mae_head_weights(student_mae_head, stage1_mae_weights)

    # Teacher is a deepcopy of student backbone (same init), frozen grads
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # DINO projectors
    student_dino_head = DINOHead(config['EMB_DIM'], out_dim=config['PROJ_DIM'], hidden_dim=2048, bottleneck_dim=256, nlayers=3).to(device)
    teacher_dino_head = DINOHead(config['EMB_DIM'], out_dim=config['PROJ_DIM'], hidden_dim=2048, bottleneck_dim=256, nlayers=3).to(device)
    teacher_dino_head.load_state_dict(student_dino_head.state_dict())
    for p in teacher_dino_head.parameters():
        p.requires_grad = False

    img_size = config['GLOBAL_CROPS_SIZE']
    patch_size = config['PATCH_SIZE']
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    data_transform = DataAugmentationDINO(
        config['GLOBAL_CROPS_SCALE'],
        config['LOCAL_CROPS_SCALE'],
        config['LOCAL_CROPS_NUMBER'],
        global_crops_size=config['GLOBAL_CROPS_SIZE'],
        local_crops_size=config['LOCAL_CROPS_SIZE'],
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=config['MASK_RATIO_MIN_MAX'],
        mask_probability=config['MASK_SAMPLE_PROBABILITY'],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # Stage-2 uses the same data root as Stage-1 and supports MAE-error filtering.
    train_dataset = ADE20kDatasetImage(
        root=config['DATASET_ROOT'],
        split="train",
        transform=data_transform,
    )
    val_dataset = ADE20kDatasetImage(
        root=config['DATASET_ROOT'],
        split="val",
        transform=data_transform,
    )

    keep_indices = []
    mae_cache_path = _none_if_empty(config.get("MAE_FILTER_CACHE_FILE"))
    if not config['DISABLE_MAE_FILTER']:
        if (not use_ddp) or _is_main_process():
            keep_indices = filter_indices_by_mae_error(
                image_paths=train_dataset.img_paths,
                student=student,
                mae_head=student_mae_head,
                config=config,
                device=device,
            )
            if len(keep_indices) == 0:
                raise RuntimeError("MAE-based filtering removed all samples. Please lower threshold or increase top ratio.")
            _save_mae_filter_cache(
                cache_path=mae_cache_path,
                image_paths=train_dataset.img_paths,
                keep_indices=keep_indices,
                config=config,
            )
        if use_ddp:
            keep_indices = _broadcast_long_list(keep_indices, device)
    else:
        if (not use_ddp) or _is_main_process():
            keep_indices = _load_mae_filter_cache(
                cache_path=mae_cache_path,
                image_paths=train_dataset.img_paths,
            )
        if use_ddp:
            keep_indices = _broadcast_long_list(keep_indices, device)

    if len(keep_indices) > 0:
        train_dataset = Subset(train_dataset, keep_indices)
        if _is_main_process():
            logging.info("Using filtered training set with %d samples", len(keep_indices))

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=_get_world_size(),
            rank=_get_rank(),
            shuffle=True,
            drop_last=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        persistent_workers=(config['NUM_WORKERS'] > 0),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        persistent_workers=(config['NUM_WORKERS'] > 0),
        collate_fn=collate_fn,
    )

    if use_ddp:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        student_dino_head = DDP(student_dino_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        student_mae_head = DDP(student_mae_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    student_core = _unwrap_module(student)
    student_dino_head_core = _unwrap_module(student_dino_head)
    student_mae_head_core = _unwrap_module(student_mae_head)

    # Optimizer for: student LoRA + DINO head + MAE head
    trainable_params = list(filter(lambda p: p.requires_grad, student_core.parameters()))
    trainable_params += list(student_dino_head_core.parameters())
    trainable_params += list(student_mae_head_core.parameters())

    opt_cfg = config['OPTIMIZER']
    sched_cfg = config['SCHEDULER']
    iters_per_epoch = max(1, len(train_loader))
    if opt_cfg['NAME'] == 'adamw':
        optimizer = optim.AdamW(trainable_params, lr=float(opt_cfg['LR']), weight_decay=opt_cfg['WEIGHT_DECAY'])
    else:
        optimizer = optim.SGD(trainable_params, lr=float(opt_cfg['LR']), momentum=0.9, weight_decay=opt_cfg['WEIGHT_DECAY'])
    
    if sched_cfg['NAME'] == 'warmuppolylr':
        scheduler = WarmupPolyLR(optimizer, 
                                sched_cfg['POWER'], int((config['EPOCHS'] + 1) * iters_per_epoch),
                                iters_per_epoch * sched_cfg['WARMUP'],
                                sched_cfg['WARMUP_RATIO'],)
    else:
        scheduler = PolyLR(optimizer, int((config['EPOCHS'] + 1) * iters_per_epoch))
        
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics = {"train_loss": [], "train_ibot_loss": [], "train_mae_loss": []}
    global_step = 0
    ema_m = config['EMA_M']
    writer = SummaryWriter(log_dir=f"runs/{config['EXP_NAME']}") if _is_main_process() else None

    os.makedirs("output", exist_ok=True)

    for epoch in range(config['EPOCHS']):
        student.train()
        teacher.eval()
        student_dino_head.train()
        teacher_dino_head.eval()
        student_mae_head.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_ibot = 0.0
        running_mae = 0.0
        for iter_idx, images in enumerate(train_loader, start=1):
            n_global_crops = 2
            assert n_global_crops == 2
            n_local_crops = config['LOCAL_CROPS_NUMBER']

            # Use same masked image for MAE and DINO/iBOT branches.
            global_crops_unmasked = images["collated_global_crops"].to(device, non_blocking=True)
            local_crops = images["collated_local_crops"].to(device, non_blocking=True)
            global_crops_unmasked = _ensure_three_channels(global_crops_unmasked)
            local_crops = _ensure_three_channels(local_crops)
            masks = images["collated_masks"].to(device, non_blocking=True).bool()
            mask_indices_list = images["mask_indices_list"].to(device, non_blocking=True)
            n_masked_patches = mask_indices_list.shape[0]
            upperbound = images["upperbound"]
            masks_weight = images["masks_weight"].to(device, non_blocking=True)
            global_crops = apply_patch_masks_to_images(global_crops_unmasked, masks, config['PATCH_SIZE'])

            n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
            n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
            ibot_loss_scale = 1.0 / n_global_crops

            optimizer.zero_grad(set_to_none=True)

            autocast_context = torch.cuda.amp.autocast(enabled=use_amp) if device.type == "cuda" else nullcontext()
            with autocast_context:
                # ---- Student forward features ----
                student_global_backbone_output_dict = student(global_crops)
                student_local_backbone_output_dict = student(local_crops)

                # ---- Teacher forward features (no grad) ----
                with torch.no_grad():
                    n_global_crops_teacher = n_global_crops
                    teacher_backbone_output_dict = teacher(global_crops)
                teacher_cls_tokens = _extract_cls_tokens(teacher_backbone_output_dict)
                teacher_cls_tokens =teacher_cls_tokens.chunk(n_global_crops_teacher)
                teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
                ibot_teacher_patch_tokens = _extract_patch_tokens(teacher_backbone_output_dict)
                _dim = ibot_teacher_patch_tokens.shape[-1]
                n_cls_tokens =teacher_cls_tokens.shape[0]

                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = teacher_dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]

                teacher_dino_softmaxed_centered_list = dino_clstoken_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=config['TEACHER_TEMP']
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                dino_clstoken_loss.update_center(teacher_cls_tokens_after_head)

                masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                masked_teacher_ibot_softmaxed_centered = ibot_loss_fn.softmax_center_teacher(
                    masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=config['TEACHER_TEMP']
                )
                masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                ibot_loss_fn.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            inputs_for_student_head_list = []
            student_local_cls_tokens = _extract_cls_tokens(student_local_backbone_output_dict)
            inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))
            student_global_cls_tokens = _extract_cls_tokens(student_global_backbone_output_dict)
            inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

            _dim = student_global_cls_tokens.shape[-1]
            ibot_student_patch_tokens = _extract_patch_tokens(student_global_backbone_output_dict)
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))

            _attn_bias, cat_inputs = BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
            outputs_list = _attn_bias.split(student_dino_head(cat_inputs))

            # 3a: local crops cls tokens
            student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

            # 3b: global crops cls tokens
            student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

            # 1) DINO cls-token
            dino_local_crops_loss = torch.tensor(0.0, device=device)
            if n_local_crops > 0:
                dino_local_crops_loss = dino_clstoken_loss(
                    student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                    teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
                ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            loss_scales = 2
            dino_global_crops_loss = (
                dino_clstoken_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dino_cls = dino_local_crops_loss + dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens
            loss_koleo = sum(koleo_loss_fn(p) for p in student_cls_tokens.chunk(2))

            # 2) MAE reconstruction branch on the same masked global crops
            mae_pred_patches = student_mae_head(ibot_student_patch_tokens)
            mae_target_patches = patchify_images(global_crops_unmasked, config['PATCH_SIZE'])
            loss_mae, per_img_mae = masked_mae_recon_loss(mae_pred_patches, mae_target_patches, masks)

            # 3) iBOT patch loss re-weighted by per-image reconstruction complexity
            ibot_img_weight = per_img_mae.detach()
            ibot_img_weight = ibot_img_weight / ibot_img_weight.mean().clamp(min=1e-6)
            ibot_img_weight = ibot_img_weight.clamp(min=0.1, max=10.0)
            ibot_masks_weight = masks_weight * ibot_img_weight.unsqueeze(-1).expand_as(masks)[masks].to(masks_weight.dtype)

            loss_ibot = (
                ibot_loss_fn.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=ibot_masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            ssl_loss = (
                config['W_DINO'] * loss_dino_cls
                + config['W_IBOT'] * loss_ibot
                + config['W_KOLEO'] * loss_koleo
                + config['W_MAE'] * loss_mae
            )
            total_loss = ssl_loss

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # EMA update teacher <- student (backbone + DINO projector)
            update_ema(teacher, student_core, ema_m)
            update_ema_module(teacher_dino_head, student_dino_head_core, ema_m)

            running_loss += float(total_loss.item())
            running_ibot += float(loss_ibot.item())
            running_mae += float(loss_mae.item())
            global_step += 1

            if writer is not None:
                writer.add_scalar("Loss/step_total", total_loss.item(), global_step)
                writer.add_scalar("Loss/step_dino", loss_dino_cls.item(), global_step)
                writer.add_scalar("Loss/step_ibot", loss_ibot.item(), global_step)
                writer.add_scalar("Loss/step_koleo", loss_koleo.item(), global_step)
                writer.add_scalar("Loss/step_mae", loss_mae.item(), global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            if _is_main_process() and (iter_idx % 20 == 0 or iter_idx == len(train_loader)):
                avg_total = running_loss / iter_idx
                logging.info(
                    f"[S2][Epoch {epoch+1}/{config['EPOCHS']}] "
                    f"Iter {iter_idx}/{len(train_loader)} | "
                    f"total {total_loss.item():.4f} | avg {avg_total:.4f} | "
                    f"dino {loss_dino_cls.item():.4f} | ibot {loss_ibot.item():.4f} | "
                    f"mae {loss_mae.item():.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
                )
        scheduler.step()

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_ibot = running_ibot / max(1, len(train_loader))
        epoch_mae = running_mae / max(1, len(train_loader))
        if use_ddp:
            epoch_metrics = torch.tensor([epoch_loss, epoch_ibot, epoch_mae], device=device)
            torch.distributed.all_reduce(epoch_metrics, op=torch.distributed.ReduceOp.SUM)
            epoch_metrics = epoch_metrics / _get_world_size()
            epoch_loss, epoch_ibot, epoch_mae = epoch_metrics.tolist()
        metrics["train_loss"].append(epoch_loss)
        metrics["train_ibot_loss"].append(epoch_ibot)
        metrics["train_mae_loss"].append(epoch_mae)
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
            writer.add_scalar("Loss/train_ibot_epoch", epoch_ibot, epoch)
            writer.add_scalar("Loss/train_mae_epoch", epoch_mae, epoch)
        if _is_main_process():
            logging.info(
                f"Epoch: {epoch+1} - train total: {epoch_loss:.4f} - "
                f"ibot: {epoch_ibot:.4f} - mae: {epoch_mae:.4f}"
            )

        if _is_main_process() and config['SAVE_EVERY'] > 0 and (epoch + 1) % config['SAVE_EVERY'] == 0:
            ckpt_path = f"output/{config['EXP_NAME']}_epoch{epoch+1}.pt"
            student_core.save_parameters(ckpt_path)
            logging.info(f"Saved LoRA weights: {ckpt_path}")
        if use_ddp:
            torch.distributed.barrier()

    if _is_main_process():
        if writer is not None:
            writer.close()
        student_core.save_parameters(f"output/{config['EXP_NAME']}.pt")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "student_dino_head": student_dino_head_core.state_dict(),
            "student_mae_head": student_mae_head_core.state_dict(),
        }, f"output/{config['EXP_NAME']}_extras.pt")
        with open(f"output/{config['EXP_NAME']}_metrics.json", "w") as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument("--debug", action="store_true", help="Debug visualize")
    parser.add_argument("--config", type=str, default="configs/PSST_train.yaml", help="Path to config YAML")
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    logging.basicConfig(level=logging.INFO)
    request_ddp = bool(config.get("DDP", False))
    has_ddp_env = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    if request_ddp and has_ddp_env and not _dist_available_and_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    elif request_ddp and not has_ddp_env:
        logging.warning("DDP=True but distributed env vars are missing. Fallback to single-process training.")
        config["DDP"] = False

    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if _dist_available_and_initialized() else 0
    main_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if main_device.type == "cuda":
        torch.cuda.set_device(local_rank)

    if config['LOCAL_CROPS_NUMBER'] <= 0:
        raise ValueError("local_crops_number must be > 0 because collate_data_and_cast expects local crops.")

    fix_seeds(seed=3407 + local_rank)
    encoder = _load_backbone(config, main_device)

    if config['IMG_DIM'][0] % config['PATCH_SIZE'] != 0 or config['IMG_DIM'][1] % config['PATCH_SIZE'] != 0:
        logging.info(f"The image size ({config['IMG_DIM']}) should be divisible by patch size {config['PATCH_SIZE']}.")
        config['IMG_DIM'] = (config['IMG_DIM'][0] - config['IMG_DIM'][0] % config['PATCH_SIZE'],
                          config['IMG_DIM'][1] - config['IMG_DIM'][1] % config['PATCH_SIZE'])
        logging.info(f"The image size is lowered to ({config['IMG_DIM']}).")

    finetune_dino_student_teacher(config, encoder)
    if _dist_available_and_initialized():
        torch.distributed.destroy_process_group()
