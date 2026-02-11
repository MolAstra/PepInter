import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, task_types: dict):
        """
        task_types: dict, e.g. {"absorption": "regression", "emission": "regression",
                                "quantum_yield": "bce", "log_e": "regression"}
        """
        super().__init__()
        self.task_types = task_types
        self.task_names = list(task_types.keys())
        self.num_tasks = len(self.task_names)

        # 每个任务一个可学习参数 log_sigma
        self.log_sigma = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(self, logits, labels):
        device = logits.device
        total_loss = 0.0
        losses = {}

        for i, task in enumerate(self.task_names):
            task_type = self.task_types[task]
            logit = logits[:, i]
            label = labels[:, i]

            mask = ~torch.isnan(label)
            if mask.sum() == 0:
                continue  # 当前 batch 没标签，跳过

            logit = logit[mask]
            label = label[mask]

            if task_type == "regression":
                task_loss = F.mse_loss(logit, label, reduction="mean")
            elif task_type == "bce":
                task_loss = F.binary_cross_entropy_with_logits(
                    logit, label, reduction="mean"
                )
            elif task_type == "classification":
                task_loss = F.cross_entropy(logit, label.float(), reduction="mean")
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Uncertainty weighting
            sigma = torch.exp(self.log_sigma[i])
            weighted_loss = (1.0 / (2 * sigma**2)) * task_loss + self.log_sigma[i]

            losses[task] = task_loss.item()
            total_loss += weighted_loss

        return total_loss, losses
