import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)

class FocalLossWithClassWeight(nn.Module):
    """
    结合Focal Loss和类别权重的损失函数
    """
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25, ignore_index=255):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        pred: (B, C, H, W)
        target: (B, H, W)
        """
        # 计算交叉熵（不reduction）
        ce_loss = F.cross_entropy(
            pred, target, 
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                   reduction='none', 
                                   ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
# class Dice(nn.Module):
#     def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
#         """
#         delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
#         """
#         super().__init__()
#         self.delta = delta
#         self.aux_weights = aux_weights

#     def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
#         # preds in shape [B, C, H, W] and labels in shape [B, H, W]
#         num_classes = preds.shape[1]
#         labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
#         tp = torch.sum(labels*preds, dim=(2, 3))
#         fn = torch.sum(labels*(1-preds), dim=(2, 3))
#         fp = torch.sum((1-labels)*preds, dim=(2, 3))

#         dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
#         dice_score = torch.sum(1 - dice_score, dim=-1)

#         dice_score = dice_score / num_classes
# #         return dice_score.mean()

#     def forward(self, preds, targets: Tensor) -> Tensor:
#         if isinstance(preds, tuple):
#             return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
#         return self._forward(preds, targets)

class Dice(nn.Module):
    def __init__(self, num_classes, ignore_index=255, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (raw logits)
        targets: [B, H, W] (long) with values in [0, C-1] or ignore_index
        """
        # softmax -> probs
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        B, C, H, W = probs.shape
        assert C == self.num_classes, f"num_classes mismatch: {C} vs {self.num_classes}"

        # create valid mask where target != ignore_index
        valid_mask = (targets != self.ignore_index).unsqueeze(1)  # [B,1,H,W] bool

        # create one-hot safely: we will only fill positions where valid_mask is True
        # initialize target_onehot zeros
        device = logits.device
        target_onehot = torch.zeros((B, C, H, W), dtype=probs.dtype, device=device)

        # get indices for valid positions (on CPU or GPU - both fine)
        if valid_mask.any():
            # to avoid one_hot with invalid indices, mask and then scatter_
            targets_clamped = targets.clone()
            targets_clamped[~(targets_clamped != self.ignore_index)] = 0  # set ignore positions to 0 temporarily
            # scatter to onehot
            target_onehot.scatter_(1, targets_clamped.unsqueeze(1).long(), 1.0)
            # zero out ignore positions explicitly
            target_onehot = target_onehot * valid_mask.type_as(target_onehot)

        # compute per-class dice
        dims = (0, 2, 3)  # sum over batch and spatial
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        # dice_score shape: [C]
        # average over classes that actually appear? We will average over all classes
        loss = 1.0 - dice_score.mean()
        return loss
    

__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)