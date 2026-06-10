import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import Levenshtein
from sklearn.model_selection import train_test_split

# Словарь: нуклеотиды + служебные токены
CHAR_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
IDX_TO_CHAR = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
PAD_IDX = 4
SOS_IDX = 5
EOS_IDX = 6
VOCAB_SIZE = 7
VALID_BASES = set(CHAR_TO_IDX)


def encode_sequences(sequences, max_len, min_len=20):
    """
    Кодирует строки в тензоры: [нуклеотиды..., EOS, PAD, ...].
    PAD не участвует в loss; EOS останавливает генерацию.
    """
    encoded = []
    lengths = []
    for seq in sequences:
        seq = str(seq).upper().replace('U', 'T')
        indices = []
        for char in seq:
            if len(indices) >= max_len - 1:
                break
            if char in VALID_BASES:
                indices.append(CHAR_TO_IDX[char])
        lengths.append(len(indices))
        indices.append(EOS_IDX)
        while len(indices) < max_len:
            indices.append(PAD_IDX)
        encoded.append(indices)
    return torch.LongTensor(encoded), torch.LongTensor(lengths)


def decode_indices(indices, min_len=20):
    """Декодирует индексы в строку аптамера."""
    chars = []
    for idx in indices:
        idx = int(idx)
        if idx == EOS_IDX:
            break
        if idx == PAD_IDX or idx in (SOS_IDX,):
            continue
        if idx in IDX_TO_CHAR:
            chars.append(IDX_TO_CHAR[idx])
    seq = ''.join(chars)
    if len(seq) < min_len:
        return seq
    return seq


def top_k_filter(logits, top_k):
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


class ConditionalGRUDecoder(nn.Module):
    """
    Условный GRU декодер: генерирует аптамер по эмбеддингу молекулы и латентной точке аптамера.
    """

    def __init__(self, mol_dim=768, latent_dim=768, hidden_dim=512, vocab_size=VOCAB_SIZE,
                 max_len=50, min_len=20, num_layers=2, dropout=0.3, embed_dim=64,
                 context_dim=128):
        super().__init__()
        self.mol_dim = mol_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.min_len = min_len
        self.num_layers = num_layers
        self.context_dim = context_dim

        combined_dim = mol_dim + latent_dim

        self.fc_hidden = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * num_layers),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.context_proj = nn.Sequential(
            nn.Linear(combined_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.gru = nn.GRU(
            input_size=embed_dim + context_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )

    def _init_hidden(self, mol_embedding, z):
        combined = torch.cat([mol_embedding, z], dim=1)
        return self.fc_hidden(combined).view(self.num_layers, mol_embedding.size(0), self.hidden_dim)

    def _make_context(self, mol_embedding, z):
        combined = torch.cat([mol_embedding, z], dim=1)
        return self.context_proj(combined)

    def _step(self, input_idx, hidden, context):
        embedded = self.char_embed(input_idx)
        if embedded.dim() == 2:
            embedded = embedded.unsqueeze(1)
        step_context = context.unsqueeze(1).expand(-1, embedded.size(1), -1)
        embedded = torch.cat([embedded, step_context], dim=-1)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden

    def forward(self, mol_embedding, z, target_seq=None, temperature=0.8,
                teacher_forcing_ratio=1.0, top_k=None):
        batch_size = mol_embedding.size(0)
        hidden = self._init_hidden(mol_embedding, z)
        context = self._make_context(mol_embedding, z)

        if target_seq is None:
            return self._generate(
                batch_size, hidden, context, mol_embedding.device,
                temperature=temperature, top_k=top_k
            )

        # Teacher forcing: вход [SOS, t0, t1, ...], цель [t0, t1, ..., EOS, ...]
        sos = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=mol_embedding.device)
        decoder_input = torch.cat([sos, target_seq[:, :-1]], dim=1)

        if teacher_forcing_ratio < 1.0 and self.training:
            logits_list = []
            input_idx = decoder_input[:, 0]
            for step in range(self.max_len):
                logits, hidden = self._step(input_idx, hidden, context)
                logits_list.append(logits.unsqueeze(1))
                use_teacher = torch.rand(batch_size, device=mol_embedding.device) < teacher_forcing_ratio
                teacher_idx = target_seq[:, step]
                pred_idx = logits.argmax(dim=-1)
                input_idx = torch.where(use_teacher, teacher_idx, pred_idx)
            return torch.cat(logits_list, dim=1)

        embedded = self.char_embed(decoder_input)
        step_context = context.unsqueeze(1).expand(-1, embedded.size(1), -1)
        embedded = torch.cat([embedded, step_context], dim=-1)
        output, _ = self.gru(embedded, hidden)
        return self.fc_out(output)

    def _mask_generation_logits(self, logits, step):
        logits = logits.clone()
        logits[:, PAD_IDX] = float('-inf')
        logits[:, SOS_IDX] = float('-inf')
        if step < self.min_len - 1:
            logits[:, EOS_IDX] = float('-inf')
        return logits

    def _generate(self, batch_size, hidden, context, device, temperature=0.8, top_k=None):
        input_idx = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=device)
        generated = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(self.max_len):
            logits, hidden = self._step(input_idx, hidden, context)
            if finished.all():
                break

            logits = logits / max(temperature, 1e-5)
            logits = self._mask_generation_logits(logits, step)
            logits = top_k_filter(logits, top_k)

            probs = F.softmax(logits, dim=-1).clone()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            probs[finished] = 0.0
            probs[finished, PAD_IDX] = 1.0

            next_idx = torch.multinomial(probs, 1).squeeze(1)
            next_idx = torch.where(finished, torch.full_like(next_idx, PAD_IDX), next_idx)

            generated.append(next_idx)
            finished = finished | (next_idx == EOS_IDX)
            input_idx = next_idx

        if not generated:
            return torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        return torch.stack(generated, dim=1)


def build_aligned_training_pairs(df, model, device, negative_only=True):
    """
    Строит выровненные тройки (mol_emb, apt_emb, sequence) из CSV.
    Использует те же строки, что и контрастивное обучение (label 0 = негативные пары).
    """
    seq_col = next((c for c in df.columns if 'sequence' in c.lower()), None)
    if seq_col is None:
        raise ValueError("Колонка sequence не найдена в CSV")

    seq_emb_cols = [c for c in df.columns if c.startswith('seq_emb_')]
    smi_emb_cols = [c for c in df.columns if c.startswith('smi_emb_')]
    if not seq_emb_cols or not smi_emb_cols:
        raise ValueError("Не найдены колонки seq_emb_* или smi_emb_*")

    if negative_only:
        subset = df[df['label'] == 0]
    else:
        subset = df

    mol_list, apt_list, seq_list = [], [], []
    model.eval()

    with torch.no_grad():
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Building aligned pairs"):
            smi_t = torch.FloatTensor(row[smi_emb_cols].values.astype(np.float32)).unsqueeze(0).to(device)
            apt_t = torch.FloatTensor(row[seq_emb_cols].values.astype(np.float32)).unsqueeze(0).to(device)

            mol_list.append(model.encode_molecule(smi_t).cpu().numpy()[0])
            apt_list.append(model.encode_aptamer(apt_t).cpu().numpy()[0])
            seq_list.append(str(row[seq_col]).upper().replace('U', 'T'))

    return np.array(mol_list), np.array(apt_list), seq_list


def train_conditional_decoder(decoder, mol_embeddings, apt_embeddings, sequences,
                              epochs=100, batch_size=32, lr=1e-3, device='cpu',
                              scheduled_sampling_max=0.4):
    """
    Обучает условный декодер: молекула + латентная точка → аптамер.
    sequences должны быть выровнены с mol_embeddings и apt_embeddings (одинаковая длина).
    """
    decoder = decoder.to(device)

    targets, seq_lengths = encode_sequences(sequences, decoder.max_len, decoder.min_len)

    min_len_data = min(len(mol_embeddings), len(apt_embeddings), len(targets))
    mol_embeddings = mol_embeddings[:min_len_data]
    apt_embeddings = apt_embeddings[:min_len_data]
    targets = targets[:min_len_data]
    seq_lengths = seq_lengths[:min_len_data]

    train_mol, val_mol, train_apt, val_apt, train_seq, val_seq, train_lens, val_lens = train_test_split(
        mol_embeddings, apt_embeddings, targets, seq_lengths,
        test_size=0.15, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_mol), torch.FloatTensor(train_apt), train_seq, train_lens),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_mol), torch.FloatTensor(val_apt), val_seq, val_lens),
        batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    print(f"\nTraining CONDITIONAL decoder on {len(train_mol)} samples...")
    print(f"  Input: molecule ({decoder.mol_dim}) + latent ({decoder.latent_dim})")
    print(f"  Seq length: {decoder.min_len}–{decoder.max_len}, vocab={decoder.vocab_size}")
    print(f"  Device: {device}")

    best_val_loss = float('inf')
    patience_counter = 0
    save_path = 'best_conditional_decoder.pth'

    for epoch in range(epochs):
        decoder.train()
        tf_ratio = 1.0 - scheduled_sampling_max * (epoch / max(epochs - 1, 1))
        train_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for mol_emb, apt_emb, target, _ in pbar:
            mol_emb = mol_emb.to(device)
            apt_emb = apt_emb.to(device)
            target = target.to(device)

            logits = decoder(
                mol_emb, apt_emb, target_seq=target,
                teacher_forcing_ratio=tf_ratio
            )
            loss = criterion(
                logits.reshape(-1, decoder.vocab_size),
                target.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'tf': f'{tf_ratio:.2f}'})

        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for mol_emb, apt_emb, target, _ in val_loader:
                mol_emb = mol_emb.to(device)
                apt_emb = apt_emb.to(device)
                target = target.to(device)
                logits = decoder(mol_emb, apt_emb, target_seq=target, teacher_forcing_ratio=1.0)
                loss = criterion(
                    logits.reshape(-1, decoder.vocab_size),
                    target.reshape(-1)
                )
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, tf={tf_ratio:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if os.path.exists(save_path):
        decoder.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nBest model loaded, val loss: {best_val_loss:.4f}")
    return decoder


def generate_aptamer_for_molecule(decoder, mol_embedding, latent_point, device='cpu',
                                  temperature=0.8, top_k=3, num_samples=1):
    """
    Генерирует аптамер(ы) для молекулы.

    Returns:
        str или list[str], если num_samples > 1
    """
    decoder.eval()

    if len(mol_embedding.shape) == 1:
        mol_tensor = torch.FloatTensor(mol_embedding).unsqueeze(0).to(device)
    else:
        mol_tensor = torch.FloatTensor(mol_embedding).to(device)

    if len(latent_point.shape) == 1:
        latent_tensor = torch.FloatTensor(latent_point).unsqueeze(0).to(device)
    else:
        latent_tensor = torch.FloatTensor(latent_point).to(device)

    sequences = []
    with torch.no_grad():
        for _ in range(num_samples):
            seq_idx = decoder(
                mol_tensor, latent_tensor, target_seq=None,
                temperature=temperature, top_k=top_k
            )
            if seq_idx.numel() == 0:
                sequences.append('')
                continue
            seq = decode_indices(seq_idx[0].cpu().tolist(), min_len=decoder.min_len)
            sequences.append(seq if seq else 'A' * decoder.min_len)

    return sequences[0] if num_samples == 1 else sequences


def latent_cosine_scores(mol_embedding, latent_points):
    """Возвращает cosine similarity молекулы с кандидатными негативными латентами."""
    mol = np.asarray(mol_embedding, dtype=np.float32).reshape(1, -1)
    points = np.asarray(latent_points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    mol = mol / np.clip(np.linalg.norm(mol, axis=1, keepdims=True), 1e-8, None)
    points = points / np.clip(np.linalg.norm(points, axis=1, keepdims=True), 1e-8, None)
    return (points @ mol.T).reshape(-1)


def _deduplicate_latent_points(points: np.ndarray, ndigits: int = 4) -> np.ndarray:
    """Drop exact/near-duplicate latent rows before diversity selection."""
    if len(points) == 0:
        return points
    unique = []
    seen = set()
    for point in points:
        key = tuple(np.round(np.asarray(point, dtype=np.float32), ndigits).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(np.asarray(point, dtype=np.float32))
    if not unique:
        return points[:1]
    return np.vstack(unique)


def _select_diverse_latents(mol_embedding, candidate_points, candidate_scores, n_points, diversity_threshold):
    order = np.argsort(candidate_scores)
    selected = []
    selected_scores = []
    selected_norm = []

    for idx in order:
        point = candidate_points[idx]
        score = float(candidate_scores[idx])
        point_norm = point / max(np.linalg.norm(point), 1e-8)
        if selected_norm:
            diversity = np.max(np.dot(np.vstack(selected_norm), point_norm))
            if diversity > diversity_threshold:
                continue

        selected.append(point)
        selected_scores.append(score)
        selected_norm.append(point_norm)
        if len(selected) >= n_points:
            break

    return selected, selected_scores


def select_negative_latent_points(mol_embedding, local_negative_points=None, global_negative_points=None,
                                  n_points=20, max_similarity=0.15, diversity_threshold=0.98,
                                  max_local_points=8):
    """
    Выбирает negative latent seeds: немного локальных + разнообразные глобальные точки.
    """
    mol_dim = np.asarray(mol_embedding).shape[-1]
    selected = []
    selected_scores = []

    if local_negative_points is not None and len(local_negative_points) > 0:
        local = _deduplicate_latent_points(np.asarray(local_negative_points, dtype=np.float32))
        local_scores = latent_cosine_scores(mol_embedding, local)
        local_keep = min(max_local_points, n_points)
        local_selected, local_selected_scores = _select_diverse_latents(
            mol_embedding, local, local_scores, local_keep, diversity_threshold
        )
        selected.extend(local_selected)
        selected_scores.extend(local_selected_scores)

    remaining = max(0, n_points - len(selected))
    if remaining > 0 and global_negative_points is not None and len(global_negative_points) > 0:
        global_pool = _deduplicate_latent_points(np.asarray(global_negative_points, dtype=np.float32))
        global_scores = latent_cosine_scores(mol_embedding, global_pool)
        global_order = np.argsort(global_scores)

        global_selected = []
        global_selected_scores = []
        global_norms = []
        for idx in global_order:
            if global_scores[idx] > max_similarity and len(global_order) > remaining:
                continue
            point = global_pool[idx]
            score = float(global_scores[idx])
            point_norm = point / max(np.linalg.norm(point), 1e-8)
            if global_norms:
                diversity = np.max(np.dot(np.vstack(global_norms), point_norm))
                if diversity > diversity_threshold:
                    continue
            if selected:
                overlap = np.max(np.dot(np.vstack([p / max(np.linalg.norm(p), 1e-8) for p in selected]), point_norm))
                if overlap > diversity_threshold:
                    continue

            global_selected.append(point)
            global_selected_scores.append(score)
            global_norms.append(point_norm)
            if len(global_selected) >= remaining:
                break

        selected.extend(global_selected)
        selected_scores.extend(global_selected_scores)

    if not selected:
        return np.empty((0, mol_dim), dtype=np.float32), np.array([])

    return np.asarray(selected, dtype=np.float32), np.asarray(selected_scores, dtype=np.float32)


def passes_basic_aptamer_filters(seq, min_len=20, max_len=50, max_homopolymer=6, gc_range=(0.2, 0.8)):
    """Быстрые sanity-фильтры для сгенерированной DNA-последовательности."""
    if not seq or len(seq) < min_len or len(seq) > max_len:
        return False
    if any(ch not in VALID_BASES for ch in seq):
        return False

    run = 1
    for prev, cur in zip(seq, seq[1:]):
        run = run + 1 if prev == cur else 1
        if run > max_homopolymer:
            return False

    gc = (seq.count('G') + seq.count('C')) / max(len(seq), 1)
    return gc_range[0] <= gc <= gc_range[1]


def _resolve_smiles_column(df):
    for col in df.columns:
        if col.lower() in {"canonical_smiles", "smiles"}:
            return col
    return "canonical_smiles"


def compute_molecule_baseline_sims(contrastive_model, df, smiles, device, smiles_col=None):
    """
    Per-molecule contrastive cosine sims using stored seq_emb/smi_emb from CSV.
    Returns empty dict when the molecule has no positive/negative rows.
    """
    seq_cols = [c for c in df.columns if c.startswith("seq_emb_")]
    smi_cols = [c for c in df.columns if c.startswith("smi_emb_")]
    if not seq_cols or not smi_cols:
        return {}

    smiles_col = smiles_col or _resolve_smiles_column(df)
    subset = df[df[smiles_col].astype(str) == str(smiles)]
    if subset.empty:
        return {}

    pos_rows = subset[subset["label"] == 1]
    neg_rows = subset[subset["label"] == 0]
    if pos_rows.empty or neg_rows.empty:
        return {}

    smi_embedding = subset.iloc[0][smi_cols].to_numpy(dtype=np.float32)

    with torch.no_grad():
        smi_t = torch.tensor(smi_embedding, dtype=torch.float32, device=device).unsqueeze(0)
        smi_z = contrastive_model.encode_molecule(smi_t)

        pos_t = torch.tensor(pos_rows[seq_cols].values.astype(np.float32), device=device)
        neg_t = torch.tensor(neg_rows[seq_cols].values.astype(np.float32), device=device)
        pos_sims = F.cosine_similarity(
            smi_z.expand(len(pos_rows), -1),
            contrastive_model.encode_aptamer(pos_t),
            dim=-1,
        )
        neg_sims = F.cosine_similarity(
            smi_z.expand(len(neg_rows), -1),
            contrastive_model.encode_aptamer(neg_t),
            dim=-1,
        )

    positive_mean = float(pos_sims.mean().cpu())
    negative_mean = float(neg_sims.mean().cpu())
    return {
        "positive_mean": positive_mean,
        "positive_min": float(pos_sims.min().cpu()),
        "negative_mean": negative_mean,
        "negative_min": float(neg_sims.min().cpu()),
        "negative_max": float(neg_sims.max().cpu()),
        "contrastive_separation": positive_mean - negative_mean,
    }


def enrich_generation_targets_with_baselines(
    generation_targets,
    df,
    contrastive_model,
    device,
    smiles_col=None,
):
    """Attach per-molecule contrastive baselines; return enriched targets and skip log."""
    smiles_col = smiles_col or _resolve_smiles_column(df)
    enriched = []
    skipped = []

    for target in generation_targets:
        smiles = target.get("smiles", "")
        if not smiles or str(smiles).startswith("Unknown_SMILES_"):
            skipped.append((smiles, None, "unknown_smiles"))
            continue

        baseline = compute_molecule_baseline_sims(
            contrastive_model, df, smiles, device, smiles_col=smiles_col
        )
        if not baseline:
            skipped.append((smiles, None, "missing_baseline"))
            continue

        item = dict(target)
        item.update(baseline)
        enriched.append(item)

    return enriched, skipped


def summarize_gating_thresholds(enriched_targets, thresholds=(0.02, 0.03, 0.05)):
    """Print how many molecules pass each min_contrastive_separation threshold."""
    if not enriched_targets:
        print("  No molecules with valid baselines for gating summary.")
        return

    print("\nGating threshold preview (contrastive_separation = positive_mean - negative_mean):")
    for threshold in thresholds:
        passing = [
            t for t in enriched_targets
            if t.get("contrastive_separation", float("-inf")) >= threshold
        ]
        if passing:
            sep_mean = float(np.mean([t["contrastive_separation"] for t in passing]))
            pos_mean = float(np.mean([t["positive_mean"] for t in passing]))
            neg_mean = float(np.mean([t["negative_mean"] for t in passing]))
            print(
                f"  >= {threshold:.2f}: {len(passing)}/{len(enriched_targets)} molecules | "
                f"sep_mean={sep_mean:.4f}, pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}"
            )
        else:
            print(f"  >= {threshold:.2f}: 0/{len(enriched_targets)} molecules")


def filter_generation_targets_by_separation(
    generation_targets,
    df,
    contrastive_model,
    device,
    min_separation=0.05,
    max_targets=None,
    smiles_col=None,
):
    """
    Keep molecules with contrastive_separation >= min_separation, then take top-K
    by separation (best-separated targets first).
    """
    enriched, skipped = enrich_generation_targets_with_baselines(
        generation_targets, df, contrastive_model, device, smiles_col=smiles_col
    )
    summarize_gating_thresholds(enriched)

    eligible = [
        t for t in enriched
        if t.get("contrastive_separation", float("-inf")) >= min_separation
    ]
    eligible.sort(key=lambda t: t.get("contrastive_separation", float("-inf")), reverse=True)

    if max_targets is not None and max_targets > 0:
        kept = eligible[:max_targets]
    else:
        kept = eligible

    kept_smiles = {t.get("smiles") for t in kept}
    low_sep_skipped = [
        (t.get("smiles", ""), t.get("contrastive_separation"), "below_top_k_or_min_sep")
        for t in enriched
        if t.get("smiles") not in kept_smiles
    ]

    if kept:
        print(
            f"\nTop generation targets by contrastive_separation "
            f"(showing up to {len(kept)}):"
        )
        for i, target in enumerate(kept[:10]):
            print(
                f"  #{i} sep={target['contrastive_separation']:.4f} "
                f"pos={target['positive_mean']:.4f} neg={target['negative_mean']:.4f} "
                f"smiles={str(target['smiles'])[:70]}"
            )
        if len(kept) > 10:
            print(f"  ... and {len(kept) - 10} more")

    return kept, skipped + low_sep_skipped


def score_and_filter_candidates_by_sequence_sim(
    candidates,
    raw_smi_embedding,
    contrastive_model,
    device,
    aptamer_encode_fn,
    baseline_positive_mean=None,
):
    """
    Encode generated sequences with GENA-LM, score in contrastive space, and keep
    only candidates with sequence_sim below the molecule positive baseline.

    raw_smi_embedding must be the stored ChemBERTa vector (smi_emb_*), not the
    already contrastive-encoded molecule embedding.
    """
    if not candidates:
        return candidates

    sequences = [item["sequence"] for item in candidates]
    seq_embeddings = aptamer_encode_fn(sequences).astype(np.float32)

    mol_tensor = torch.tensor(np.asarray(raw_smi_embedding, dtype=np.float32), device=device).unsqueeze(0)
    seq_tensor = torch.tensor(seq_embeddings, dtype=torch.float32, device=device)

    contrastive_model.eval()
    with torch.no_grad():
        mol_z = contrastive_model.encode_molecule(mol_tensor)
        apt_z = contrastive_model.encode_aptamer(seq_tensor)
        sims = F.cosine_similarity(mol_z.expand(len(sequences), -1), apt_z, dim=-1).cpu().numpy()

    filtered = []
    for candidate, sim in zip(candidates, sims):
        candidate = dict(candidate)
        candidate["sequence_sim"] = float(sim)
        if baseline_positive_mean is None or sim < baseline_positive_mean:
            filtered.append(candidate)

    filtered.sort(key=lambda item: item.get("sequence_sim", float("inf")))
    return filtered


def augment_latent_points(latent_points, latent_scores, jitter_copies=0, jitter_std=0.05, seed=42):
    """Add small Gaussian perturbations around selected negative latents."""
    if jitter_copies <= 0 or len(latent_points) == 0:
        return latent_points, latent_scores

    rng = np.random.default_rng(seed)
    augmented_points = []
    augmented_scores = []
    for point, score in zip(latent_points, latent_scores):
        augmented_points.append(point)
        augmented_scores.append(score)
        point_norm = point / max(np.linalg.norm(point), 1e-8)
        for _ in range(jitter_copies):
            noise = rng.normal(0.0, jitter_std, size=point.shape).astype(np.float32)
            jittered = point + noise
            jittered = jittered / max(np.linalg.norm(jittered), 1e-8)
            augmented_points.append(jittered.astype(np.float32))
            augmented_scores.append(score)

    return np.asarray(augmented_points, dtype=np.float32), np.asarray(augmented_scores, dtype=np.float32)


def generate_ranked_aptamers_for_molecule(decoder, mol_embedding, local_negative_points=None,
                                          global_negative_points=None, device='cpu',
                                          n_latent_points=20, samples_per_latent=2,
                                          temperature=0.9, top_k=4,
                                          max_similarity=0.15,
                                          diversity_threshold=0.98,
                                          latent_jitter_copies=0, latent_jitter_std=0.05,
                                          min_seq_len=None, max_seq_len=None,
                                          contrastive_model=None,
                                          aptamer_encode_fn=None,
                                          raw_smi_embedding=None,
                                          baseline_positive_mean=None,
                                          use_sequence_sim_filter=True):
    """
    Генерирует кандидаты для целевой молекулы и возвращает их, отсортированными от
    наиболее вероятно не взаимодействующих к менее надёжным по latent cosine score.
    Optionally applies post-decode contrastive sequence_sim filtering.
    """
    latent_points, latent_scores = select_negative_latent_points(
        mol_embedding=mol_embedding,
        local_negative_points=local_negative_points,
        global_negative_points=global_negative_points,
        n_points=n_latent_points,
        max_similarity=max_similarity,
        diversity_threshold=diversity_threshold,
    )
    latent_points, latent_scores = augment_latent_points(
        latent_points,
        latent_scores,
        jitter_copies=latent_jitter_copies,
        jitter_std=latent_jitter_std,
    )

    min_len = decoder.min_len if min_seq_len is None else min_seq_len
    max_len = decoder.max_len if max_seq_len is None else max_seq_len

    candidates = []
    seen = set()
    for latent_point, _ in zip(latent_points, latent_scores):
        point_score = float(
            latent_cosine_scores(
                mol_embedding,
                np.asarray(latent_point, dtype=np.float32).reshape(1, -1),
            )[0]
        )
        generated = generate_aptamer_for_molecule(
            decoder=decoder,
            mol_embedding=mol_embedding,
            latent_point=latent_point,
            device=device,
            temperature=temperature,
            top_k=top_k,
            num_samples=samples_per_latent,
        )
        if isinstance(generated, str):
            generated = [generated]

        for seq in generated:
            if seq in seen:
                continue
            seen.add(seq)
            if not passes_basic_aptamer_filters(seq, min_len, max_len):
                continue
            candidates.append({
                'sequence': seq,
                'latent_similarity': point_score,
                'length': len(seq),
                'gc': float((seq.count('G') + seq.count('C')) / max(len(seq), 1)),
            })

    if (
        use_sequence_sim_filter
        and contrastive_model is not None
        and aptamer_encode_fn is not None
        and raw_smi_embedding is not None
        and baseline_positive_mean is not None
    ):
        candidates = score_and_filter_candidates_by_sequence_sim(
            candidates=candidates,
            raw_smi_embedding=raw_smi_embedding,
            contrastive_model=contrastive_model,
            device=device,
            aptamer_encode_fn=aptamer_encode_fn,
            baseline_positive_mean=baseline_positive_mean,
        )
    else:
        candidates.sort(key=lambda item: item['latent_similarity'])

    return candidates


def evaluate_conditional_decoder(decoder, mol_embeddings, apt_embeddings, sequences,
                                 device='cpu', temperature=0.8, top_k=3, n_examples=5):
    """Оценка декодера на тестовой выборке."""
    generated = []
    for mol, apt in zip(mol_embeddings, apt_embeddings):
        seq = generate_aptamer_for_molecule(
            decoder, mol, apt, device=device, temperature=temperature, top_k=top_k
        )
        generated.append(seq)

    exact = 0
    identities = []
    lev_dists = []

    for orig, gen in zip(sequences, generated):
        orig = str(orig).upper().replace('U', 'T')
        if orig == gen:
            exact += 1
        max_l = max(len(orig), len(gen), 1)
        lev_dists.append(Levenshtein.distance(orig, gen) / max_l)
        matches = sum(1 for a, b in zip(orig, gen) if a == b)
        identities.append(matches / max_l)

    n = len(sequences)
    print("=" * 60)
    print("CONDITIONAL GRU DECODER EVALUATION")
    print("=" * 60)
    print(f"Exact match: {exact}/{n} ({100 * exact / n:.1f}%)")
    print(f"Mean identity: {np.mean(identities):.3f}")
    print(f"Mean norm. Levenshtein: {np.mean(lev_dists):.3f}")

    print("\nExamples:")
    for i in range(min(n_examples, n)):
        print(f"  True:  {sequences[i][:60]}")
        print(f"  Pred:  {generated[i][:60]}")
        print(f"  Identity: {identities[i] * 100:.1f}%\n")

    return {
        'exact_match_rate': exact / n,
        'mean_identity': float(np.mean(identities)),
        'mean_levenshtein': float(np.mean(lev_dists)),
        'generated': generated,
    }
