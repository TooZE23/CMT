import os
import json, logging, argparse, torch, yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from dino_finetune import MAEDataset
from dino_finetune.data.jigsaw_dataset import masked_mae_recon_loss, patchify_images
from dino_finetune import DINOEncoderLoRA, SAM2EncoderLoRA
from dino_finetune.schedulers import PolyLR, WarmupPolyLR
from dino_finetune.utils import fix_seeds
from dino_finetune import build_sam2, get_scheduler, get_optimizer

def _dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_rank() -> int:
    return torch.distributed.get_rank() if _dist_available_and_initialized() else 0


def _get_world_size() -> int:
    return torch.distributed.get_world_size() if _dist_available_and_initialized() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


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
        sam2_cfg = config.get("SAM2_CFG", None)
        sam2_ckpt = config.get("SAM2_CKPT", None)
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

def get_dataloader_mae(
    root: str,
    img_dim=(224,224),
    batch_size=64,
    mask_ratio=0.75,
    patch_size=16,
    channel_num=3,
    num_workers=24,
    use_ddp=False,
):
    
    train_set = MAEDataset(
        root=root,
        split='train',
        img_dim=img_dim,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        channel_num=channel_num,
    )
    val_set = MAEDataset(
        root=root,
        split='val',
        img_dim=img_dim,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        channel_num=channel_num,
    )
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=_get_world_size(),
            rank=_get_rank(),
            shuffle=True,
            drop_last=False,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader, train_sampler

@torch.no_grad()
def validate_epoch(
    dino_lora: nn.Module,
    val_loader: DataLoader,
    metrics: dict,
    device: torch.device,
) -> None:
    dino_lora.eval()
    val_loss = 0.0
    for masked_images, target_images, patch_mask in val_loader:
        masked_images = masked_images.float().to(device, non_blocking=True)
        target_images = target_images.float().to(device, non_blocking=True)
        patch_mask = patch_mask.float().to(device, non_blocking=True)

        pred_patches = dino_lora(masked_images)
        target_patches = patchify_images(target_images, int(dino_lora.patch_size))
        loss, _ = masked_mae_recon_loss(pred_patches, target_patches, patch_mask)
        val_loss += loss.item()
    metrics["val_loss"].append(val_loss / max(1,len(val_loader)))

def finetune_dino(config: dict, encoder: nn.Module, device: torch.device):
    vfm_type = str(config["VFM_TYPE"]).lower()
    use_ddp = bool(config.get("DDP", False)) and _dist_available_and_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0

    if vfm_type == "sam2":
        dino_lora = SAM2EncoderLoRA(
            sam_model=encoder,
            r=config['R'],
            emb_dim=config['EMB_DIM'],
            use_lora=True,
            decoder_type="mae",
            img_dim=config['IMG_DIM'],
            channel_num=config['CHANNEL_NUM'],
            feature_level=config.get("SAM2_FEATURE_LEVEL", -1),
            patch_size=config['PATCH_SIZE'],
        )
    else:
        dino_lora = DINOEncoderLoRA(
            encoder=encoder,
            r=config['R'],
            emb_dim=config['EMB_DIM'],
            use_lora=True,
            decoder_type="mae",
            img_dim=config['IMG_DIM'],
            channel_num=config['CHANNEL_NUM'],
            cls_head_type=config['CLS_HEAD_TYPE'],
        )

    # Load LoRA weights before moving model to target device, then move once.
    if config['LORA_WEIGHTS']:
        dino_lora.load_parameters(config['LORA_WEIGHTS'])
    dino_lora = dino_lora.to(device)

    if use_ddp:
        ddp_kwargs = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        dino_lora = DDP(dino_lora, **ddp_kwargs)
    model_without_ddp = dino_lora.module if hasattr(dino_lora, "module") else dino_lora

    train_loader, val_loader, train_sampler = get_dataloader_mae(
        root=config['DATASET_ROOT'],
        img_dim=config['IMG_DIM'],
        batch_size=config['BATCH_SIZE'],
        mask_ratio=config['MASK_RATIO'],
        patch_size=config['PATCH_SIZE'],
        channel_num=config['CHANNEL_NUM'],
        num_workers=config.get('NUM_WORKERS', 24),
        use_ddp=use_ddp,
    )

    os.makedirs(f"output/{config['EXP_NAME']}", exist_ok=True)
    opt_cfg = config['OPTIMIZER']
    sched_cfg = config['SCHEDULER']
    iters_per_epoch = max(1, len(train_loader))
    if opt_cfg['NAME'] == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dino_lora.parameters()), lr=float(opt_cfg['LR']), weight_decay=opt_cfg['WEIGHT_DECAY'])
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, dino_lora.parameters()), lr=float(opt_cfg['LR']), momentum=0.9, weight_decay=opt_cfg['WEIGHT_DECAY'])
    
    if sched_cfg['NAME'] == 'warmuppolylr':
        scheduler = WarmupPolyLR(optimizer, 
                                sched_cfg['POWER'], int((config['EPOCHS'] + 1) * iters_per_epoch),
                                iters_per_epoch * sched_cfg['WARMUP'],
                                sched_cfg['WARMUP_RATIO'],)
    else:
        scheduler = PolyLR(optimizer, int((config['EPOCHS'] + 1) * iters_per_epoch))
    metrics = {"train_loss": [], "val_loss": []}
    global_step = 0
    writer = SummaryWriter(log_dir=f"runs/{config['EXP_NAME']}") if _is_main_process() else None

    trainable_encoder = sum(p.numel() for p in model_without_ddp.encoder.parameters() if p.requires_grad)
    trainable_decoder = sum(p.numel() for p in model_without_ddp.decoder.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model_without_ddp.parameters() if not p.requires_grad)
    total = trainable_encoder  + trainable_decoder+ non_trainable
    
    if _is_main_process():
        print(f'Trainable Encoder Parameters: {trainable_encoder/1e6:,}M')
        print(f'Trainable Decoder Parameters: {trainable_decoder/1e6:,}M')
        print(f'Non-trainable Parameters: {non_trainable/1e6:,}M')
        print(f'Total Parameters: {total/1e6:,}M')

    for epoch in range(config['EPOCHS']):
        dino_lora.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for iter_idx, (masked_images, target_images, patch_mask) in enumerate(train_loader, start=1):
            masked_images = masked_images.float().to(device, non_blocking=True)
            target_images = target_images.float().to(device, non_blocking=True)
            patch_mask = patch_mask.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_patches = dino_lora(masked_images)
            target_patches = patchify_images(target_images, int(model_without_ddp.patch_size))
            loss, _ = masked_mae_recon_loss(pred_patches, target_patches, patch_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            global_step += 1
            # print(f"Epoch [{epoch+1}/{config.epochs}], Step [{global_step}], Loss: {loss.item():.4f}")
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
            if _is_main_process() and (iter_idx % 20 == 0 or iter_idx == len(train_loader)):
                avg_loss = running_loss / iter_idx
                logging.info(
                    f"[S1][Epoch {epoch+1}/{config['EPOCHS']}] "
                    f"Iter {iter_idx}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | avg {avg_loss:.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )

        epoch_loss = running_loss / max(1,len(train_loader))
        if use_ddp:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            torch.distributed.all_reduce(epoch_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            epoch_loss = (epoch_loss_tensor / _get_world_size()).item()
        metrics["train_loss"].append(epoch_loss)
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)

        if (epoch + 1) % 1 == 0:
            if _is_main_process():
                validate_epoch(model_without_ddp, val_loader, metrics, device)
                if writer is not None:
                    writer.add_scalar("Loss/val", metrics["val_loss"][-1], epoch)
                logging.info(
                    f"Epoch {epoch+1}: train loss {epoch_loss:.4f} | "
                    f"val loss {metrics['val_loss'][-1]:.4f}"
                )
                model_without_ddp.save_parameters(f"output/{config['EXP_NAME']}/s1_lora_epoch{epoch}.pt")
            
            if use_ddp:
                torch.distributed.barrier()
        
        scheduler.step()  # StepLR调度器
    if writer is not None:
        writer.close()
    
    if _is_main_process():
        with open(f"output/{config['EXP_NAME']}_metrics.json", "w") as f:
            json.dump(metrics, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")

    parser.add_argument("--debug", action="store_true", help="Debug by visualizing some of the outputs to file for a sanity check",)
    parser.add_argument("--config", type=str, default="configs/PSST_warm.yaml", help="Path to config YAML")

    args = parser.parse_args()
    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    request_ddp = bool(config.get("DDP", False))
    has_ddp_env = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    if request_ddp and has_ddp_env and not _dist_available_and_initialized():
        backend = "nccl" if (torch.cuda.is_available() and torch.distributed.is_nccl_available()) else "gloo"
        torch.distributed.init_process_group(backend=backend)
    elif request_ddp and not has_ddp_env:
        logging.warning("DDP=True but distributed env vars are missing. Fallback to single-process training.")
        config["DDP"] = False

    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if _dist_available_and_initialized() else 0
    main_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if main_device.type == "cuda":
        torch.cuda.set_device(local_rank)

    fix_seeds(seed=3407 + local_rank)
    logging.basicConfig(level=logging.INFO)
    # Model configuration
    encoder = _load_backbone(config, main_device)
    if config['IMG_DIM'][0] % config['PATCH_SIZE'] != 0 or config['IMG_DIM'][1] % config['PATCH_SIZE'] != 0:
        logging.info(f"The image size ({config['IMG_DIM']}) should be divisible "
            f"by the patch size {config['PATCH_SIZE']}.")
        # subtract the difference from image size and set a new size.
        config['IMG_DIM'] = (config['IMG_DIM'][0] - config['IMG_DIM'][0] % config['PATCH_SIZE'],
                          config['IMG_DIM'][1] - config['IMG_DIM'][1] % config['PATCH_SIZE'])
        logging.info(f"The image size is lowered to ({config['IMG_DIM']}).")
    finetune_dino(config, encoder, main_device)
    if _dist_available_and_initialized():
        torch.distributed.destroy_process_group()
