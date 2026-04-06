import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')




class TemperatureScaledLoss(nn.Module):
    def __init__(self, init_temperature=0.25):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(init_temperature))), dtype=torch.float32)
        )

    def get_temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, z_anchor, z_positive, z_negatives):
        temperature = torch.exp(self.log_temperature.clamp(-10, 10))
        B = z_anchor.size(0)
        temp = self.get_temperature()

        # POS
        pos_sim = torch.sum(z_anchor * z_positive, dim=1, keepdim=True) / temp

        # TRUE NEGATIVES ONLY
        if z_negatives is not None:
            neg_sim = torch.einsum('bd,bkd->bk', z_anchor, z_negatives) / temp
            logits = torch.cat([pos_sim, neg_sim], dim=1)
        else:
            logits = pos_sim

        labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)

        loss = F.cross_entropy(logits, labels)

        # metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

            top3 = (torch.topk(logits, min(3, logits.size(1)), dim=1)
                    .indices == 0).any(dim=1).float().mean()

        return loss, {
            'accuracy': acc.item(),
            'top3_acc': top3.item(),
            'temperature': temp.item()
        }
