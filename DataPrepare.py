import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')


class FinalContrastiveDataset(Dataset):
    def __init__(self, apt_pos, smi_pos, apt_neg, smi_neg, negative_ratio=3):

        self.apt_to_pos = defaultdict(list)
        self.apt_to_neg = defaultdict(list)

        for apt, smi in zip(apt_pos, smi_pos):
            self.apt_to_pos[tuple(apt)].append(smi)

        for apt, smi in zip(apt_neg, smi_neg):
            self.apt_to_neg[tuple(apt)].append(smi)

        self.apts = list(self.apt_to_pos.keys())

        self.negative_ratio = negative_ratio

        print(f"Dataset created:")
        print(f"  Unique aptamers: {len(self.apts)}")
        print(f"  Total positive pairs: {sum(len(v) for v in self.apt_to_pos.values())}")
        print(f"  Total negative pairs: {sum(len(v) for v in self.apt_to_neg.values())}")

    def __len__(self):
        return len(self.apts)

    def __getitem__(self, idx):

        # ===== ВЫБИРАЕМ АПТАМЕР =====
        apt = self.apts[idx]
        anchor_apt = torch.FloatTensor(apt)

        # ===== POSITIVE (СЛУЧАЙНЫЙ ИЗ МНОЖЕСТВА) =====
        pos_list = self.apt_to_pos[apt]
        positive_smi = torch.FloatTensor(
            pos_list[random.randint(0, len(pos_list) - 1)]
        )

        # ===== NEGATIVES (ТОЛЬКО ДЛЯ ЭТОГО ЖЕ АПТАМЕРА) =====
        neg_list = self.apt_to_neg.get(apt, [])

        if len(neg_list) > 0:
            indices = torch.randint(0, len(neg_list), (self.negative_ratio,))
            negative_smis = torch.stack([
                torch.FloatTensor(neg_list[i]) for i in indices
            ])
        else:
            # fallback если нет явных негативов
            negative_smis = torch.zeros(
                self.negative_ratio,
                positive_smi.shape[0]
            )

        # # ===== АУГМЕНТАЦИЯ (опционально) =====
        # if torch.rand(1).item() > 0.5:
        #     anchor_apt = anchor_apt + torch.randn_like(anchor_apt) * 0.01
        #
        # if torch.rand(1).item() > 0.5:
        #     positive_smi = positive_smi + torch.randn_like(positive_smi) * 0.01

        return {
            'anchor_apt': anchor_apt,
            'positive_smi': positive_smi,
            'negative_smis': negative_smis
        }
