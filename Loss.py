import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class TemperatureScaledLoss(nn.Module):
    def __init__(self, init_temperature=0.25, margin=0.15, margin_weight=0.5):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.margin = margin
        self.margin_weight = margin_weight

    def get_temperature(self):
        return torch.exp(self.log_temperature).clamp(0.07, 0.5)

    def compute_margin_loss(self, z_anchor, z_positive, z_negatives):
        """Push positives above hardest negative by a cosine margin (768d space)."""
        pos_sim = torch.sum(z_anchor * z_positive, dim=1)
        if z_negatives is None:
            return torch.zeros((), device=z_anchor.device)

        neg_sim = torch.einsum('bd,bkd->bk', z_anchor, z_negatives).max(dim=1).values
        return F.relu(neg_sim - pos_sim + self.margin).mean()

    def forward(self, z_anchor, z_positive, z_negatives):
        B = z_anchor.size(0)
        temp = self.get_temperature()

        pos_sim = torch.sum(z_anchor * z_positive, dim=1, keepdim=True) / temp

        if z_negatives is not None:
            neg_sim = torch.einsum('bd,bkd->bk', z_anchor, z_negatives) / temp
            logits = torch.cat([pos_sim, neg_sim], dim=1)
        else:
            logits = pos_sim

        labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
        infonce_loss = F.cross_entropy(logits, labels)
        margin_loss = self.compute_margin_loss(z_anchor, z_positive, z_negatives)
        loss = infonce_loss + self.margin_weight * margin_loss

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            top3 = (torch.topk(logits, min(3, logits.size(1)), dim=1)
                    .indices == 0).any(dim=1).float().mean()

        return loss, {
            'accuracy': acc.item(),
            'top3_acc': top3.item(),
            'temperature': temp.item(),
            'infonce_loss': infonce_loss.item(),
            'margin_loss': margin_loss.item(),
        }
