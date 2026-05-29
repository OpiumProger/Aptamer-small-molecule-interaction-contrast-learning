import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')


class FinalContrastiveDataset(Dataset):
    def __init__(self, apt_pos, smi_pos, apt_neg, smi_neg, negative_ratio=3):

        self.smi_to_pos = defaultdict(list)
        self.smi_to_neg = defaultdict(list)

        for apt, smi in zip(apt_pos, smi_pos):
            self.smi_to_pos[tuple(smi)].append(apt)

        for apt, smi in zip(apt_neg, smi_neg):
            self.smi_to_neg[tuple(smi)].append(apt)

        self.smis = list(self.smi_to_pos.keys())

        self.negative_ratio = negative_ratio

        print(f"Dataset created:")
        print(f"  Unique molecules: {len(self.smis)}")
        print(f"  Total positive pairs: {sum(len(v) for v in self.smi_to_pos.values())}")
        print(f"  Total negative pairs: {sum(len(v) for v in self.smi_to_neg.values())}")

    def __len__(self):
        return len(self.smis)

    def __getitem__(self, idx):

        # ВЫБИРАЕМ Молекулу
        smi = self.smis[idx]
        anchor_smi = torch.FloatTensor(smi)

        # POSITIVE (СЛУЧАЙНЫЙ ИЗ МНОЖЕСТВА)
        pos_list = self.smi_to_pos[smi]
        positive_apts = torch.FloatTensor(
            pos_list[random.randint(0, len(pos_list) - 1)]
        )

        # NEGATIVES (ТОЛЬКО ДЛЯ ЭТОй же молекулы)
        neg_list = self.smi_to_neg.get(smi, [])

        if len(neg_list) > 0:
            indices = torch.randint(0, len(neg_list), (self.negative_ratio,))
            negative_apts = torch.stack([
                torch.FloatTensor(neg_list[i]) for i in indices
            ])
        else:
            # fallback если нет явных негативов
            negative_apts = torch.zeros(
                self.negative_ratio,
                positive_apts.shape[0]
            )

        # АУГМЕНТАЦИЯ
        if torch.rand(1).item() > 0.5:
            anchor_smi = anchor_smi + torch.randn_like(anchor_smi) * 0.01

        if torch.rand(1).item() > 0.5:
            positive_apts = positive_apts + torch.randn_like(positive_apts) * 0.01

        return {
            'anchor_smi': anchor_smi,
            'positive_apts': positive_apts,
            'negative_apts': negative_apts
        }
