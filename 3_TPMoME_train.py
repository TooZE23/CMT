import os
import json
import logging
import argparse
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from dino_finetune.model.utils import get_logger, cal_flops, fix_seeds, print_iou
from dino_finetune.datasets import *

from dino_finetune import (
    MultiModalMoEFusion,
    get_loss,
    get_optimizer,
    get_scheduler,
    get_train_augmentation,
    get_val_augmentation,
    scores,
    Metrics,
)


def _dist_available_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_rank():
    return torch.distributed.get_rank() if _dist_available_and_initialized() else 0


def _get_world_size():
    return torch.distributed.get_world_size() if _dist_available_and_initialized() else 1


def _is_main_process():
    return _get_rank() == 0

def _save_checkpoint(model_without_ddp, metrics, epoch, save_path, extra: dict = None):
    state_dict_to_save = {
        "visual_prompts": model_without_ddp.visual_prompts.state_dict(),
        "moe_layers": model_without_ddp.moe_layers.state_dict(),
        "final_decoder": model_without_ddp.seg_head.state_dict(),
    }
    ckpt = {
        'model_state_dict': state_dict_to_save,
        'metrics': metrics,
        'epoch': epoch,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, save_path)

@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    core_model = model.module if hasattr(model, "module") else model
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    for inputs_list, masks in dataloader:
        if isinstance(inputs_list, (list, tuple)):
            assert len(inputs_list) == len(core_model.modality_names), (
                f"modal count mismatch: got {len(inputs_list)}, expected {len(core_model.modality_names)}"
            )
            inputs_dict = {
                k: v.float().to(device, non_blocking=True)
                for k, v in zip(core_model.modality_names, inputs_list)
            }
        else:
            inputs_dict = {k: v.float().to(device, non_blocking=True) for k, v in inputs_list.items()}
        masks = masks.long().to(device, non_blocking=True)
        preds = model(inputs_dict).softmax(dim=1)
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        metrics.update(preds, masks)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou

def evaluate_only(config: dict, model_path: str = None):
    """Load a trained checkpoint and run evaluation on the validation set only."""
    modality_configs = config['MODALITIES']
    train_cfg = config['TRAIN']
    eval_cfg = config['EVAL']
    dataset_cfg = config['DATASET']
    test_cfg = config.get('TEST', {})

    if model_path is None:
        model_path = test_cfg.get('MODEL_PATH', None)
    if model_path is None:
        raise ValueError("eval-only requires a checkpoint path via TEST.MODEL_PATH (or --model-path).")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_layers_cfg = train_cfg.get("SAM2_PROMPT_LAYERS", train_cfg.get("PROMPT_LAYERS", []))
    model = MultiModalMoEFusion(
        modality_configs=modality_configs,
        num_experts=train_cfg['NUM_EXPERTS'],
        expert_hidden_dim=train_cfg['EXPERT_HIDDEN_DIM'],
        prompt_length=train_cfg['PROMPT_LENGTH'],
        fusion_layers=train_cfg['FUSION_LAYERS'],
        prompt_layers=prompt_layers_cfg,
        n_classes=train_cfg['N_CLASSES'],
        img_dim=train_cfg['IMAGE_SIZE'],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    submodule_map = {
        'visual_prompts': model.visual_prompts,
        'moe_layers': model.moe_layers,
        'final_decoder': model.seg_head,
    }
    missing = []
    for key, module in submodule_map.items():
        if key in state_dict:
            module.load_state_dict(state_dict[key])
        else:
            missing.append(key)
    if missing:
        logging.warning(f"Checkpoint is missing sub-modules: {missing}")

    epoch_info = f" (epoch {checkpoint['epoch']})" if isinstance(checkpoint, dict) and 'epoch' in checkpoint else ""
    logging.info(f"Loaded checkpoint from {model_path}{epoch_info}")

    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    class_names = valset.CLASSES
    val_loader = DataLoader(
        valset,
        batch_size=eval_cfg['BATCH_SIZE'],
        num_workers=train_cfg['NUM_WORKERS'],
        pin_memory=False,
        shuffle=False,
    )

    acc, macc, f1, mf1, ious, miou = evaluate(model, val_loader, device)
    logging.info(print_iou(0, ious, miou, acc, macc, class_names))
    logging.info(f"[EVAL-ONLY] mIoU: {miou:.4f} | mAcc: {macc:.4f} | mF1: {mf1:.4f}")
    return acc, macc, f1, mf1, ious, miou

def train_vpt_multimodal_moe(config: dict):
    modality_configs = config['MODALITIES']
    train_cfg = config['TRAIN']
    eval_cfg = config['EVAL']
    dataset_cfg = config['DATASET']
    sched_cfg = config['SCHEDULER']
    opt_cfg = config['OPTIMIZER']

    use_ddp_cfg = bool(train_cfg['DDP'])
    if use_ddp_cfg and not _dist_available_and_initialized():
        logging.warning("DDP=True but torch.distributed is not initialized. Falling back to single-process mode.")
    use_ddp = use_ddp_cfg and _dist_available_and_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    prompt_layers_cfg = train_cfg.get("SAM2_PROMPT_LAYERS", train_cfg.get("PROMPT_LAYERS", []))
    model = MultiModalMoEFusion(
            modality_configs=modality_configs,
            num_experts=train_cfg['NUM_EXPERTS'],
            expert_hidden_dim=train_cfg['EXPERT_HIDDEN_DIM'],
            prompt_length=train_cfg['PROMPT_LENGTH'],
            fusion_layers=train_cfg['FUSION_LAYERS'],
            prompt_layers=prompt_layers_cfg,
            n_classes=train_cfg['N_CLASSES'],
            img_dim=train_cfg['IMAGE_SIZE']
        ).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    model_without_ddp = model.module if hasattr(model, "module") else model

    backbone_type = getattr(model_without_ddp, "backbone_type", "unknown")
    if _is_main_process():
        logging.info(f"MultiModalMoEFusion backbone_type={backbone_type}")
        if backbone_type == "sam2":
            logging.info(
                f"SAM2 fusion levels={train_cfg['FUSION_LAYERS']}."
            )
            logging.info(
                f"SAM2 prompt config: PROMPT_LENGTH={train_cfg['PROMPT_LENGTH']}, "
                f"SAM2_PROMPT_LAYERS={prompt_layers_cfg}."
            )

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    class_names = trainset.CLASSES
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            trainset,
            num_replicas=_get_world_size(),
            rank=_get_rank(),
            shuffle=True,
            drop_last=True,
        )
    train_loader = DataLoader(
        trainset,
        batch_size=train_cfg['BATCH_SIZE'],
        num_workers=train_cfg['NUM_WORKERS'],
        drop_last=True,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        valset,
        batch_size=eval_cfg['BATCH_SIZE'],
        num_workers=train_cfg['NUM_WORKERS'],
        pin_memory=False,
        shuffle=False,
    )

    criterion = get_loss(config['LOSS']['NAME'], ignore_label=dataset_cfg['IGNORE_LABEL']).to(device)
    iters_per_epoch = max(1, len(train_loader))
    base_lr = float(opt_cfg['LR'])
    vp_lr = float(train_cfg.get('VP_LR', base_lr))
    moe_lr =float(train_cfg.get('MOE_LR', base_lr))
    dec_lr = float(train_cfg.get('DEC_LR', base_lr))

    optimizer_vp = None
    vp_trainable_params = sum(
        p.numel() for p in model_without_ddp.visual_prompts.parameters() if p.requires_grad
    )
    if vp_trainable_params > 0:
        optimizer_vp = get_optimizer(
            model_without_ddp.visual_prompts,
            opt_cfg['NAME'],
            vp_lr,
            weight_decay=opt_cfg['WEIGHT_DECAY'],
        )
    optimizer_moe = get_optimizer(
        model_without_ddp.moe_layers,
        opt_cfg['NAME'],
        moe_lr,
        weight_decay=opt_cfg['WEIGHT_DECAY'],
    )
    optimizer_dec = get_optimizer(
        model_without_ddp.seg_head,
        opt_cfg['NAME'],
        dec_lr,
        weight_decay=opt_cfg['WEIGHT_DECAY'],
    )

    scheduler_vp = None
    if optimizer_vp is not None:
        scheduler_vp = get_scheduler(
            sched_cfg['NAME'],
            optimizer_vp,
            int((train_cfg['EPOCHS'] + 1) * iters_per_epoch),
            sched_cfg['POWER'],
            iters_per_epoch * sched_cfg['WARMUP'],
            sched_cfg['WARMUP_RATIO'],
        )
    scheduler_moe = get_scheduler(
        sched_cfg['NAME'],
        optimizer_moe,
        int((train_cfg['EPOCHS'] + 1) * iters_per_epoch),
        sched_cfg['POWER'],
        iters_per_epoch * sched_cfg['WARMUP'],
        sched_cfg['WARMUP_RATIO'],
    )
    scheduler_dec = get_scheduler(
        sched_cfg['NAME'],
        optimizer_dec,
        int((train_cfg['EPOCHS'] + 1) * iters_per_epoch),
        sched_cfg['POWER'],
        iters_per_epoch * sched_cfg['WARMUP'],
        sched_cfg['WARMUP_RATIO'],
    )

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
    }

    mome_params = sum(p.numel() for p in model_without_ddp.moe_layers.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model_without_ddp.seg_head.parameters() if p.requires_grad)
    scaler = torch.amp.GradScaler(enabled=True)
    writer = None
    if _is_main_process():
        writer = SummaryWriter(log_dir=f"runs/{train_cfg['EXP_NAME']}")
        logger.info('================== training config =====================')
        logger.info(config)
    
    global_step = 0
    best_iou = 0.0  

    trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model_without_ddp.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    if _is_main_process():
        logging.info(f"vp parameters: {vp_trainable_params/ 1e6:.2f}M")
        logging.info(f"mome parameters: {mome_params/ 1e6:.2f}M")
        logging.info(f"decoder parameters: {decoder_params/ 1e6:.2f}M")
        logging.info(f"trainable parameters: {trainable/1e6:,}M")
        logging.info(f"non-trainable parameters: {non_trainable/1e6:,}M")
        logging.info(f"total parameters: {total/1e6:,}M")
        logger.info('================== model complexity =====================')
        cal_flops(model_without_ddp, list(model_without_ddp.modality_names), logger, device="cpu") 
        logger.info('================== model structure =====================')
        logger.info(model_without_ddp)


    def zero_all_optimizers():
        if optimizer_vp is not None:
            optimizer_vp.zero_grad(set_to_none=True)
        optimizer_moe.zero_grad(set_to_none=True)
        optimizer_dec.zero_grad(set_to_none=True)

    def step_all_optimizers():
            if optimizer_vp is not None:
                scaler.step(optimizer_vp)
            scaler.step(optimizer_moe)
            scaler.step(optimizer_dec)

    def step_all_schedulers():
            if scheduler_vp is not None:
                scheduler_vp.step()
            scheduler_moe.step()
            scheduler_dec.step()

    for epoch in range(train_cfg['EPOCHS']):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        val_iou = None

        for batch_idx, (inputs_list, masks) in enumerate(train_loader):
            if isinstance(inputs_list, (list, tuple)):
                assert len(inputs_list) == len(model_without_ddp.modality_names), (
                    f"modal count mismatch: got {len(inputs_list)}, expected {len(model_without_ddp.modality_names)}"
                )
                inputs_dict = {
                    k: v.float().to(device, non_blocking=True)
                    for k, v in zip(model_without_ddp.modality_names, inputs_list)
                }
            else: 
                inputs_dict = {k: v.float().to(device, non_blocking=True) for k, v in inputs_list.items()}
            masks = masks.long().to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda',enabled=train_cfg['AMP']):
                logits = model(inputs_dict)
                loss = criterion(logits, masks)

            zero_all_optimizers()
            scaler.scale(loss).backward()
           
            step_all_optimizers()
            scaler.update()

            step_all_schedulers()

            running_loss += loss.item()
            global_step += 1
            
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                if optimizer_vp is not None:
                    writer.add_scalar("LR/visual_prompts", optimizer_vp.param_groups[0]['lr'], global_step)
                writer.add_scalar("LR/moe_layers", optimizer_moe.param_groups[0]['lr'], global_step)
                writer.add_scalar("LR/final_decoder", optimizer_dec.param_groups[0]['lr'], global_step)
            if _is_main_process() and ((batch_idx + 1) % 20 == 0 or batch_idx + 1 == len(train_loader)):
                avg_loss = running_loss / (batch_idx + 1)
                msg = (
                    f"[S3][Epoch {epoch+1}/{train_cfg['EPOCHS']}] "
                    f"Iter {batch_idx+1}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | avg {avg_loss:.4f} | "
                    f"lr_moe {optimizer_moe.param_groups[0]['lr']:.2e} | "
                    f"lr_dec {optimizer_dec.param_groups[0]['lr']:.2e}"
                )
                if optimizer_vp is not None:
                    msg += f" | lr_vp {optimizer_vp.param_groups[0]['lr']:.2e}"
                logger.info(msg)
        # Epoch metrics
        epoch_loss = running_loss / len(train_loader)
        if use_ddp:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            torch.distributed.all_reduce(epoch_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            epoch_loss = (epoch_loss_tensor / _get_world_size()).item()
        metrics["train_loss"].append(epoch_loss)
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
        
        if ((epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == train_cfg['EPOCHS']:
            if _is_main_process():
                acc, macc, _, _, ious, miou = evaluate(model_without_ddp, val_loader, device)
                val_iou = float(miou)
                metrics["val_iou"].append(val_iou)
                if writer is not None:
                    writer.add_scalar("IoU/val_epoch", val_iou, epoch)
                if val_iou > best_iou:
                    prev_best = best_iou
                    best_iou = val_iou
                    best_path = config['SAVE_DIR']+f"/{train_cfg['EXP_NAME']}_best.pt"
                    # Save best model
                    _save_checkpoint(
                        model_without_ddp, metrics, epoch, best_path,
                        extra={'best_iou': best_iou},
                    )
                    logging.info(
                        f"[Best] *** New best model saved *** epoch={epoch+1} | "
                        f"val_iou={val_iou:.4f} (prev_best={prev_best:.4f}, "
                        f"improvement={val_iou - prev_best:+.4f}) | ckpt={best_path}"
                    )


                logging.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                summary_msg = (
                    f"Epoch: {epoch+1} - train loss: {epoch_loss:.4f} "
                    f"Best val IoU: {best_iou:.4f} "
                    f"- val IoU: {miou:.4f} "
                )
                logging.info(summary_msg)
            if use_ddp:
                torch.distributed.barrier()
    if _is_main_process():
        writer.close()
        # Save final model
        final_path = config['SAVE_DIR']+ f"/{train_cfg['EXP_NAME']}_final.pt"
        _save_checkpoint(model_without_ddp, metrics, train_cfg['EPOCHS'] - 1, final_path)

        with open(f"output/{train_cfg['EXP_NAME']}_metrics.json", "w") as f:
            json.dump(metrics, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--config", type=str, default="configs/mcubes_rgbad_sam2.yaml", help="Path to config YAML")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation using the checkpoint at TEST.MODEL_PATH")
    parser.add_argument("--model-path", type=str, default=None, help="Override TEST.MODEL_PATH for eval-only mode")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.eval_only:
        fix_seeds(seed=3407)
        log_file = f'output/{config["TRAIN"]["EXP_NAME"]}_eval.log'
        logger = get_logger(log_file)
        logging.basicConfig(level=logging.INFO)
        evaluate_only(config, model_path=args.model_path)
    else:
        request_ddp = bool(config.get('TRAIN', {}).get('DDP', False))
        has_ddp_env = ("RANK" in os.environ and "WORLD_SIZE" in os.environ) or "SLURM_PROCID" in os.environ
        if request_ddp and has_ddp_env and not _dist_available_and_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
        elif request_ddp and not has_ddp_env:
            logging.warning("DDP=True but distributed env vars are missing. Fallback to single-process training.")

        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if _dist_available_and_initialized() else 0
        fix_seeds(seed=3407 + local_rank)

        log_file = f'output/{config["TRAIN"]["EXP_NAME"]}_train.log' if _is_main_process() else None
        logger = get_logger(log_file)
        logging.basicConfig(level=logging.INFO)

        train_vpt_multimodal_moe(config)

        if _dist_available_and_initialized():
            torch.distributed.destroy_process_group()