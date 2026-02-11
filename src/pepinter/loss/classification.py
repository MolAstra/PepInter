import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss module for classification tasks with class imbalance.

    This implementation supports three explicit task types:
    - "binary": Binary classification using sigmoid + BCE focal loss.
    - "multiclass": Single-label multi-class classification using softmax + CE focal loss.
    - "multilabel": Multi-label classification using sigmoid + BCE focal loss.

    Args:
        num_classes (int, optional):
            Number of classes. Required for "multiclass" and "multilabel".
        task (str):
            One of {"binary", "multiclass", "multilabel"}.
        alpha (float, optional):
            Balancing factor for positive vs. negative samples. Default is 0.25.
        gamma (float, optional):
            Focusing parameter that down-weights easy examples. Default is 2.
        reduction (str, optional):
            One of {"none", "mean", "sum"}. Specifies the reduction applied to the output.
            Default is "mean".
        ignore_index (int, optional):
            Specifies a target value that is ignored and not included in the loss.
            Default is -1.

    Shape:
        - logits: (N, C) or (N,) for binary classification
        - target:
            * (N,) for "binary" or "multiclass"
            * (N, C) for "multilabel" (multi-hot format)

    Returns:
        torch.Tensor:
            The computed focal loss. Shape depends on ``reduction``:
            - "none" → same shape as input loss
            - "mean" or "sum" → scalar tensor
    """

    def __init__(
        self,
        num_classes=None,
        task="multiclass",
        alpha=0.25,
        gamma=2,
        reduction="mean",
        ignore_index=-1,
    ):
        super().__init__()
        self.task = task
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        if task == "multiclass" and (num_classes is None or num_classes < 2):
            raise ValueError("num_classes must be >=2 for multiclass")
        if task == "binary":
            self.sigmoid = True
        elif task == "multilabel":
            self.sigmoid = True
        else:
            self.sigmoid = False

    def forward(self, logits, target):
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            if not valid.any():
                return torch.tensor(0.0, device=logits.device)
            logits = logits[valid]
            target = target[valid]

        if self.task == "binary":
            logits = logits.view(-1)
            target = target.float().view(-1)
            prob = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            pt = prob * target + (1 - prob) * (1 - target)
            loss = ce * ((1 - pt) ** self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                loss = alpha_t * loss

        elif self.task == "multilabel":
            prob = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none"
            )
            pt = prob * target + (1 - prob) * (1 - target)
            loss = ce * ((1 - pt) ** self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                loss = alpha_t * loss

        else:  # multiclass
            ce = F.cross_entropy(logits, target, reduction="none")
            prob = torch.softmax(logits, dim=1)
            pt = prob[torch.arange(len(target)), target]
            loss = ce * ((1 - pt) ** self.gamma)
            if self.alpha is not None:
                loss = loss * self.alpha

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
