import random
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

DEFAULT_NEGATIVE_RATIO = 16


def _negative_mix_counts(negative_ratio):
    """25% same-molecule, 50% semi-hard global, 25% random cross-molecule."""
    n_same = max(0, int(round(0.25 * negative_ratio)))
    n_semi = max(0, int(round(0.50 * negative_ratio)))
    n_random = max(0, negative_ratio - n_same - n_semi)
    if negative_ratio > 0 and n_same + n_semi + n_random == 0:
        n_random = negative_ratio
    return n_same, n_semi, n_random


def _smiles_column(df):
    for col in df.columns:
        if col.lower() in ('canonical_smiles', 'smiles'):
            return col
    raise ValueError("Need canonical_smiles or smiles column for molecule split")


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        norm = np.linalg.norm(values)
        return values / np.clip(norm, 1e-8, None)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def molecule_disjoint_split(df, seed=42, train_frac=0.7, val_frac=0.15):
    """
    Split positives and negatives by molecule (canonical_smiles).
    All pairs for a molecule land in the same fold — no train/val leakage.
    """
    smiles_col = _smiles_column(df)

    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]

    unique_smiles = pos_df[smiles_col].astype(str).unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_smiles))

    n = len(unique_smiles)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train_smiles = set(unique_smiles[perm[:train_end]])
    val_smiles = set(unique_smiles[perm[train_end:val_end]])
    test_smiles = set(unique_smiles[perm[val_end:]])

    pos_smiles = pos_df[smiles_col].astype(str).values
    neg_smiles = neg_df[smiles_col].astype(str).values

    train_pos_idx = np.where(np.isin(pos_smiles, list(train_smiles)))[0]
    val_pos_idx = np.where(np.isin(pos_smiles, list(val_smiles)))[0]
    test_pos_idx = np.where(np.isin(pos_smiles, list(test_smiles)))[0]

    train_neg_idx = np.where(np.isin(neg_smiles, list(train_smiles)))[0]
    val_neg_idx = np.where(np.isin(neg_smiles, list(val_smiles)))[0]
    test_neg_idx = np.where(np.isin(neg_smiles, list(test_smiles)))[0]

    stats = {
        'train_molecules': len(train_smiles),
        'val_molecules': len(val_smiles),
        'test_molecules': len(test_smiles),
        'train_pos': len(train_pos_idx),
        'val_pos': len(val_pos_idx),
        'test_pos': len(test_pos_idx),
        'train_neg': len(train_neg_idx),
        'val_neg': len(val_neg_idx),
        'test_neg': len(test_neg_idx),
    }
    return (
        train_pos_idx, val_pos_idx, test_pos_idx,
        train_neg_idx, val_neg_idx, test_neg_idx,
        stats,
    )


def _mol_group_key(smi_vec, mol_key=None):
    if mol_key is not None:
        return str(mol_key)
    return tuple(np.asarray(smi_vec, dtype=np.float32).round(6).tolist())


class FinalContrastiveDataset(Dataset):
    def __init__(
        self,
        apt_pos,
        smi_pos,
        apt_neg,
        smi_neg,
        mol_keys_pos=None,
        mol_keys_neg=None,
        negative_ratio=DEFAULT_NEGATIVE_RATIO,
        semi_hard_low_pct=0.10,
        semi_hard_high_pct=0.40,
    ):
        self.smi_to_pos = defaultdict(list)
        self.smi_to_neg = defaultdict(list)
        self.key_to_smi = {}

        for i, (apt, smi) in enumerate(zip(apt_pos, smi_pos)):
            mk = _mol_group_key(smi, mol_keys_pos[i] if mol_keys_pos is not None else None)
            self.smi_to_pos[mk].append(apt)
            self.key_to_smi.setdefault(mk, np.asarray(smi, dtype=np.float32))

        for i, (apt, smi) in enumerate(zip(apt_neg, smi_neg)):
            mk = _mol_group_key(smi, mol_keys_neg[i] if mol_keys_neg is not None else None)
            self.smi_to_neg[mk].append(apt)
            self.key_to_smi.setdefault(mk, np.asarray(smi, dtype=np.float32))

        self.smis = list(self.smi_to_pos.keys())
        self.negative_ratio = negative_ratio
        self.n_same_mol, self.n_semi_hard, self.n_random = _negative_mix_counts(negative_ratio)
        self.semi_hard_low_pct = semi_hard_low_pct
        self.semi_hard_high_pct = semi_hard_high_pct

        self.global_neg_pool = [np.asarray(a, dtype=np.float32) for a in apt_neg]
        if self.global_neg_pool:
            self.global_neg_norm = _normalize_rows(np.stack(self.global_neg_pool))
        else:
            self.global_neg_norm = np.zeros((0, apt_pos.shape[1]), dtype=np.float32)

        same_mol_with_neg = sum(1 for s in self.smis if self.smi_to_neg.get(s))
        print("Dataset created:")
        print(f"  Unique molecules: {len(self.smis)}")
        print(f"  Total positive pairs: {sum(len(v) for v in self.smi_to_pos.values())}")
        print(f"  Total negative pairs (split): {len(self.global_neg_pool)}")
        print(f"  Molecules with same-mol negatives: {same_mol_with_neg} ({100 * same_mol_with_neg / max(len(self.smis), 1):.1f}%)")
        print(
            f"  Negative mix per sample: {self.n_same_mol} same-mol + "
            f"{self.n_semi_hard} semi-hard + {self.n_random} random "
            f"(total={negative_ratio})"
        )

    def __len__(self):
        return len(self.smis)

    def _choose_indices(self, pool_size, n, exclude, replace=False):
        if pool_size <= 0 or n <= 0:
            return []
        available = [i for i in range(pool_size) if i not in exclude]
        if not available:
            available = list(range(pool_size))
        if len(available) >= n and not replace:
            return np.random.choice(available, size=n, replace=False).tolist()
        return np.random.choice(available, size=n, replace=True).tolist()

    def _sample_semi_hard_from_pool(self, positive_vec, n, pool=None, pool_norm=None, exclude=None):
        if pool is None:
            pool = self.global_neg_pool
        if pool_norm is None:
            pool_norm = self.global_neg_norm
        if not pool or n <= 0:
            return []

        exclude = set(exclude or [])
        pos_norm = _normalize_rows(positive_vec)
        sims = pool_norm @ pos_norm
        ranked = np.argsort(-sims)

        pool_size = len(pool)
        lo = int(self.semi_hard_low_pct * pool_size)
        hi = max(lo + 1, int(self.semi_hard_high_pct * pool_size))
        band = [i for i in ranked[lo:hi] if i not in exclude]

        if len(band) < n:
            extra = [i for i in ranked[:lo] if i not in exclude]
            band = band + extra

        if not band:
            band = [i for i in ranked if i not in exclude]
        if not band:
            band = list(range(pool_size))

        chosen = np.random.choice(band, size=n, replace=len(band) < n).tolist()
        return [pool[i] for i in chosen]

    def _sample_random_from_pool(self, n, pool=None, exclude=None):
        if pool is None:
            pool = self.global_neg_pool
        if not pool or n <= 0:
            return []
        exclude = set(exclude or [])
        chosen = self._choose_indices(len(pool), n, exclude, replace=True)
        return [pool[i] for i in chosen]

    def _sample_same_mol_negatives(self, neg_list, positive_vec, n):
        if n <= 0 or not neg_list:
            return []

        neg_arr = np.stack([np.asarray(a, dtype=np.float32) for a in neg_list])
        neg_norm = _normalize_rows(neg_arr)
        pos_norm = _normalize_rows(positive_vec)
        sims = neg_norm @ pos_norm
        ranked = np.argsort(-sims)

        lo = max(0, int(0.05 * len(neg_list)))
        hi = max(lo + 1, int(0.50 * len(neg_list)))
        band = ranked[lo:hi].tolist()
        if len(band) < n:
            band = ranked[: max(n, hi)].tolist()

        chosen = np.random.choice(band, size=n, replace=len(band) < n).tolist()
        return [neg_list[i] for i in chosen]

    def _sample_negatives_mixed(self, mol_key, positive_apts_np):
        neg_list = self.smi_to_neg.get(mol_key, [])
        n_same = self.n_same_mol if neg_list else 0
        n_semi = self.n_semi_hard
        n_random = self.n_random + max(0, self.n_same_mol - n_same)

        negatives = []
        used_pool_idx = set()

        if n_same > 0:
            negatives.extend(self._sample_same_mol_negatives(neg_list, positive_apts_np, n_same))

        if n_semi > 0:
            semi = self._sample_semi_hard_from_pool(
                positive_apts_np, n_semi, exclude=used_pool_idx
            )
            negatives.extend(semi)

        if n_random > 0:
            negatives.extend(self._sample_random_from_pool(n_random, exclude=used_pool_idx))

        if len(negatives) < self.negative_ratio:
            extra = self._sample_semi_hard_from_pool(
                positive_apts_np,
                self.negative_ratio - len(negatives),
            )
            negatives.extend(extra)

        return negatives[: self.negative_ratio]

    def get_negatives_for_molecule(self, mol_key, max_local=50, max_global=50):
        """Negatives for generation / clustering: local + semi-hard/random global."""
        local = [np.asarray(a, dtype=np.float32) for a in self.smi_to_neg.get(mol_key, [])]
        local_arr = local[:max_local]

        n_needed = max(0, max_global - len(local_arr))
        global_arr = []
        if n_needed > 0 and self.global_neg_pool:
            pos_list = self.smi_to_pos.get(mol_key, [])
            ref = _normalize_rows(np.asarray(pos_list[0], dtype=np.float32)) if pos_list else None
            if ref is not None:
                n_semi = max(1, int(round(0.6 * n_needed)))
                n_rand = n_needed - n_semi
                global_arr.extend(self._sample_semi_hard_from_pool(ref, n_semi))
                global_arr.extend(self._sample_random_from_pool(n_rand))
            else:
                global_arr.extend(self._sample_random_from_pool(n_needed))

        combined = local_arr + global_arr
        if not combined and self.global_neg_pool:
            combined = self._sample_random_from_pool(min(max_global, len(self.global_neg_pool)))
        return combined

    def __getitem__(self, idx):
        smi = self.smis[idx]
        anchor_smi = torch.FloatTensor(self.key_to_smi[smi])

        pos_list = self.smi_to_pos[smi]
        positive_apts_np = np.asarray(pos_list[random.randint(0, len(pos_list) - 1)], dtype=np.float32)
        positive_apts = torch.FloatTensor(positive_apts_np)

        negatives = self._sample_negatives_mixed(smi, positive_apts_np)
        negative_apts = torch.stack([
            torch.FloatTensor(np.asarray(n, dtype=np.float32)) for n in negatives
        ])

        if torch.rand(1).item() > 0.5:
            anchor_smi = anchor_smi + torch.randn_like(anchor_smi) * 0.01

        if torch.rand(1).item() > 0.5:
            positive_apts = positive_apts + torch.randn_like(positive_apts) * 0.01

        return {
            'anchor_smi': anchor_smi,
            'positive_apts': positive_apts,
            'negative_apts': negative_apts,
        }
