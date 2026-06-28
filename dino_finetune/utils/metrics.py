from typing import Optional, Tuple

import torch
from torch import Tensor
import numpy as np


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes ** 2).view(
            self.num_classes, self.num_classes
        )

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        tp = self.hist.diag()
        denom = self.hist.sum(0) + self.hist.sum(1) - tp
        valid = denom > 0

        ious = torch.zeros_like(tp)
        ious[valid] = tp[valid] / denom[valid]

        # Standard segmentation protocol: average only over valid classes.
        miou = ious[valid].mean().item() if valid.any() else 0.0
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        tp = self.hist.diag()
        denom = self.hist.sum(0) + self.hist.sum(1)
        valid = denom > 0

        f1 = torch.zeros_like(tp)
        f1[valid] = 2 * tp[valid] / denom[valid]

        mf1 = f1[valid].mean().item() if valid.any() else 0.0
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        tp = self.hist.diag()
        denom = self.hist.sum(1)
        valid = denom > 0

        acc = torch.zeros_like(tp)
        acc[valid] = tp[valid] / denom[valid]

        macc = acc[valid].mean().item() if valid.any() else 0.0
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)


def compute_iou_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int | None] = None,
    eps: float = 1e-6,
) -> float:
    """Compute the Intersection over Union metric for the predictions and labels.

    Args:
        y_hat (torch.Tensor): The prediction of dimensions (B, C, H, W), C being
            equal to the number of classes.
        y (torch.Tensor): The label for the prediction of dimensions (B, H, W)
        ignore_index (int | None, optional): ignore label to omit predictions in
            given region.
        eps (float, optional): To smooth the division and prevent division
        by zero. Defaults to 1e-6.

    Returns:
        float: The mean IoU
    """

    y_hat = torch.argmax(y_hat, dim=1)
    y_hat = y_hat.int()
    y = y.int()

    if ignore_index is not None:
        mask = y != ignore_index
        y_hat = y_hat * mask
        y = y * mask

    intersection = (y_hat & y).float().sum((1, 2))
    union = (y_hat | y).float().sum((1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


import torch


def multiclass_iou(y_hat, y, num_classes, ignore_index=None, eps=1e-6):
    """
    计算多类别语义分割的 IoU

    参数：
        y_hat: [N, C, H, W] 模型输出 logits 或概率
        y: [N, H, W] 真实标签
        num_classes: 类别数
        ignore_index: 忽略的类别索引
        eps: 防止除零
    返回：
        mean_iou: 平均 IoU
        per_class_iou: 每个类别的 IoU
    """
    # 预测类别索引
    y_hat = torch.argmax(y_hat, dim=1)  # [N,H,W]

    if ignore_index is not None:
        mask = y != ignore_index
        y_hat = y_hat * mask
        y = y * mask

    per_class_iou = []

    for cls in range(num_classes):
        # 对每个类别生成二值掩码
        pred_cls = (y_hat == cls).float()
        target_cls = (y == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))  # 每张图的交集
        union = (pred_cls + target_cls - pred_cls * target_cls).sum(dim=(1, 2))  # 并集

        iou_cls = (intersection + eps) / (union + eps)
        per_class_iou.append(iou_cls)

    # 转成 [num_classes, N]
    per_class_iou = torch.stack(per_class_iou, dim=0)

    mean_iou = per_class_iou.mean()  # 所有类别和批量平均
    return mean_iou, per_class_iou


import numpy as np


def _fast_hist(label_true, label_pred, num_classes):
    # 如果是 torch.Tensor，先转为 numpy
    if isinstance(label_true, torch.Tensor):
        label_true = label_true.cpu().numpy()
    if isinstance(label_pred, torch.Tensor):
        label_pred = label_pred.cpu().numpy()

    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def scores(label_trues, label_preds, num_classes=9):
    # print("lable_trues:",label_trues)
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }
