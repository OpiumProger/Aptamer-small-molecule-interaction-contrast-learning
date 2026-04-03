import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')




class TemperatureScaledLoss(nn.Module):
    def __init__(self, init_temperature=0.25):
        super().__init__()


        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature))
        )

    def forward(self, z_anchor, z_positive, z_negatives):

        temperature = torch.exp(self.log_temperature)

        B = z_anchor.size(0)
        K = z_negatives.size(1)

        # Positive similarity
        pos_sim = torch.sum(z_anchor * z_positive, dim=1) / temperature

        # Negative similarity
        z_anchor_exp = z_anchor.unsqueeze(1)
        neg_sim = torch.sum(z_anchor_exp * z_negatives, dim=2) / temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)

        loss = F.cross_entropy(logits, labels)

        # ===== метрики =====
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == 0).float().mean().item()

        top3 = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
        top3_acc = (top3 == 0).any(dim=1).float().mean().item()

        return loss, {
            'accuracy': accuracy,
            'top3_acc': top3_acc,
            'temperature': temperature.item()
        }