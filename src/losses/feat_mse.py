import torch
from torch import nn

class FeatLoss(nn.Module):
    def __init__(self, mse_weight, ce_weight, vq_weight) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        self.vq_weight = vq_weight
    
    def forward(self, feat_in, logits_in, feat_target, label_target, vq_loss, split=None):
        mse_loss = nn.functional.mse_loss(feat_in, feat_target)
        ce_loss = nn.functional.cross_entropy(logits_in, label_target)
        total_loss = self.mse_weight * mse_loss + self.ce_weight * ce_loss + self.vq_weight * vq_loss
        if not split:
            split = "train" if self.training else "val"
        return total_loss, {
            f"{split}/mse_loss": mse_loss,
            f"{split}/ce_loss": ce_loss,
            f"{split}/vq_loss": vq_loss
        }


class FeatLoss_sq_vq(nn.Module):
    def __init__(self, mse_weight, ce_weight, sq_weight, vq_weight) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        self.sq_weight = sq_weight
        self.vq_weight = vq_weight
    
    def forward(self, feat_in, logits_in, feat_target, label_target, vq_loss, sq_loss, split=None):
        mse_loss = nn.functional.mse_loss(feat_in, feat_target)
        ce_loss = nn.functional.cross_entropy(logits_in, label_target)
        total_loss = self.mse_weight * mse_loss + self.ce_weight * ce_loss
        total_loss += self.vq_weight * vq_loss + self.sq_weight * sq_loss
        if not split:
            split = "train" if self.training else "val"
        return total_loss, {
            f"{split}/mse_loss": mse_loss,
            f"{split}/ce_loss": ce_loss,
            f"{split}/sq_loss": sq_loss,
            f"{split}/vq_loss": vq_loss,
            f"{split}/sq_lambda": self.sq_weight,
        }