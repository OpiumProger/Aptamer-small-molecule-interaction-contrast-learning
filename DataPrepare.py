import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')



class FinalContrastiveDataset(Dataset):
    def __init__(self, apt_pos, smi_pos, apt_neg, smi_neg, negative_ratio=3):

        self.apt_pos = torch.FloatTensor(apt_pos)
        self.smi_pos = torch.FloatTensor(smi_pos)

        self.apt_neg = torch.FloatTensor(apt_neg)
        self.smi_neg = torch.FloatTensor(smi_neg)

        self.pos_count = len(self.apt_pos)
        self.neg_count = len(self.smi_neg)

        self.negative_ratio = min(negative_ratio, self.neg_count)

        print(f"Dataset created:")
        print(f"  Positives: {self.pos_count}")
        print(f"  Negatives: {self.neg_count}")

    def __len__(self):
        # каждый индекс = один anchor (aptamer)
        return self.pos_count

    def __getitem__(self, idx):

        # Anchor (aptamer)
        anchor_apt = self.apt_pos[idx]

        # Positive (правильная молекула)
        positive_smi = self.smi_pos[idx]

        # Negatives (случайные или hard)
        neg_indices = torch.randint(0, self.neg_count, (self.negative_ratio,))
        negative_smis = self.smi_neg[neg_indices]

        # Лёгкая аугментация
        if torch.rand(1).item() > 0.5:
            anchor_apt = anchor_apt + torch.randn_like(anchor_apt) * 0.01

        if torch.rand(1).item() > 0.5:
            positive_smi = positive_smi + torch.randn_like(positive_smi) * 0.01

        return {
            'anchor_apt': anchor_apt,
            'positive_smi': positive_smi,
            'negative_smis': negative_smis
        }
