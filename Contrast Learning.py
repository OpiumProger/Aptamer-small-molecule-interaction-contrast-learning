import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')

from FinalTrainer import FinalTrainer
from DataPrepare import FinalContrastiveDataset, molecule_disjoint_split, DEFAULT_NEGATIVE_RATIO
from Model import MicroContrastiveModel
from load_data_and_visual_data import load_data, analyze_results, visualize_embeddings_correct, visualize_embeddings_2d_simple

DATA_FILE = "aptabench_with_embeddings_v2.csv"
MODEL_CHECKPOINT = "final_micro_model.pth"
USE_PRETRAINED = False  # False = заново обучить contrastive + GRU; True = загрузить готовые веса


def main(train_contrastive: bool = True, model_checkpoint: str = MODEL_CHECKPOINT):
    print("=" * 70)
    print("CONTRASTIVE LEARNING WITH VISUALIZATION")
    print("=" * 70)

    # Data load (used again in generation block for SMILES lookup)
    data_file = DATA_FILE
    try:
        apt_pos, smi_pos, apt_neg, smi_neg, seq_dim, smi_dim, mol_keys_pos, mol_keys_neg = load_data(data_file)
    except Exception as e:
        print(f" Error loading data: {e}")
        return

    import pandas as pd

    np.random.seed(42)
    df = pd.read_csv(data_file, low_memory=False)

    (
        train_pos_idx, val_pos_idx, test_pos_idx,
        train_neg_idx, val_neg_idx, test_neg_idx,
        split_stats,
    ) = molecule_disjoint_split(df, seed=42)

    n_pos = len(smi_pos)
    n_neg = len(smi_neg)

    print("\nMolecule-disjoint split (by canonical_smiles):")
    print(f"  Molecules: train={split_stats['train_molecules']}, val={split_stats['val_molecules']}, test={split_stats['test_molecules']}")
    print(f"  Total positive pairs: {n_pos}")
    print(f"    Train pos: {split_stats['train_pos']}")
    print(f"    Val pos: {split_stats['val_pos']}")
    print(f"    Test pos: {split_stats['test_pos']}")
    print(f"  Total negative pairs: {n_neg}")
    print(f"    Train neg: {split_stats['train_neg']}")
    print(f"    Val neg: {split_stats['val_neg']}")
    print(f"    Test neg: {split_stats['test_neg']}")
    print(f"  Negatives per sample: {DEFAULT_NEGATIVE_RATIO} (4 same-mol + 8 semi-hard + 4 random)")

    train_dataset = FinalContrastiveDataset(
        apt_pos[train_pos_idx],
        smi_pos[train_pos_idx],
        apt_neg[train_neg_idx],
        smi_neg[train_neg_idx],
        mol_keys_pos[train_pos_idx],
        mol_keys_neg[train_neg_idx],
    )

    val_dataset = FinalContrastiveDataset(
        apt_pos[val_pos_idx],
        smi_pos[val_pos_idx],
        apt_neg[val_neg_idx],
        smi_neg[val_neg_idx],
        mol_keys_pos[val_pos_idx],
        mol_keys_neg[val_neg_idx],
    )

    test_dataset = FinalContrastiveDataset(
        apt_pos[test_pos_idx],
        smi_pos[test_pos_idx],
        apt_neg[test_neg_idx],
        smi_neg[test_neg_idx],
        mol_keys_pos[test_pos_idx],
        mol_keys_neg[test_neg_idx],
    )

    # DataLoaders
    def collate_fn(batch):
        return {
            'anchor_smi': torch.stack([b['anchor_smi'] for b in batch]),
            'positive_apts': torch.stack([b['positive_apts'] for b in batch]),
            'negative_apts': torch.stack([b['negative_apts'] for b in batch])
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"\n Final dataset sizes:")
    print(f"    Train: {len(train_dataset)} positive pairs, {len(train_neg_idx)} negative pairs")
    print(f"    Validation: {len(val_dataset)} positive pairs, {len(val_neg_idx)} negative pairs")
    print(f"    Test: {len(test_dataset)} positive pairs, {len(test_neg_idx)} negative pairs")

    print("Creating mini model...")
    model = MicroContrastiveModel(
        input_dim_apt=768,  # размерность эмбеддингов аптамеров
        input_dim_mol=768,  # размерность эмбеддингов молекул
        latent_dim=768,
        projection_dim=768
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")

    trainer = FinalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    history = {}
    if (not train_contrastive) and os.path.exists(model_checkpoint):
        print(f"\nLoading pretrained contrastive model from '{model_checkpoint}' (skip training)")
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("\n Starting contrastive training...")
        history = trainer.train(n_epochs=30, save_path=model_checkpoint)
        plot_final_results(history)
        try:
            checkpoint = torch.load(model_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(" Загружена лучшая модель после обучения")
        except Exception:
            print(" Используется текущая модель")

    print("\n Analyzing on test set...")
    test_results = analyze_results(model, test_loader, device)

    print(f"\nFINAL TEST RESULTS:")
    print(f"    Accuracy: {test_results['accuracy']:.3f}")
    print(f"    Separation: {test_results['separation']:.3f}")
    print(f"    Overlap (neg >= pos): {test_results['overlap_pct']:.1%}")
    print(f"    ROC-AUC (global): {test_results['roc_auc']:.3f}")
    print(f"    PR-AUC (global): {test_results['pr_auc']:.3f}")
    print(f"    ROC-AUC (pair-level mean): {test_results['pair_roc_auc']:.3f}")
    print(f"    Positive mean similarity: {test_results['pos_mean']:.3f}")
    print(f"    Negative mean similarity: {test_results['neg_mean']:.3f}")

    if history:
        plot_similarity_distributions(test_results)

    visualize_embeddings_correct(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path='embedding_visualization_correct.png'
    )

    from wasserstein_utils import analyze_model_with_loader

    ot_analysis = analyze_model_with_loader(
        model=model,
        data_loader=test_loader,  # используем test_loader с правильными парами!
        device=device
    )

    #save embeddings
    save_embeddings(model, apt_pos, smi_pos, apt_neg, smi_neg, device)



    return model, history, test_results, test_loader, device


def plot_final_results(history):

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(history['val_acc'], label='Val', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Accuracy (Final gap: {history["train_val_gap"][-1]:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if 'train_separation' in history:
        axes[0, 2].plot(history['train_separation'], label='Train', linewidth=2, marker='o', markersize=4)
        axes[0, 2].plot(history['val_separation'], label='Val', linewidth=2, marker='s', markersize=4)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Separation')
        axes[0, 2].set_title('Latent Separation (pos - neg)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].plot(history['train_top3'], label='Train', linewidth=2, marker='o', markersize=4)
        axes[0, 2].plot(history['val_top3'], label='Val', linewidth=2, marker='s', markersize=4)
        axes[0, 2].set_title('Top-3 Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    if 'val_overlap_pct' in history:
        axes[1, 0].plot(history['train_overlap_pct'], label='Train', linewidth=2, marker='o', markersize=4)
        axes[1, 0].plot(history['val_overlap_pct'], label='Val', linewidth=2, marker='s', markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Overlap fraction')
        axes[1, 0].set_title('Neg >= Pos overlap')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].plot(history['temperature'], label='Temperature', linewidth=2, color='red', marker='o', markersize=4)
        axes[1, 0].set_title('Temperature during training')
        axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['train_val_gap'], label='Gap', linewidth=2, color='orange', marker='o', markersize=4)
    axes[1, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Critical (0.3)')
    axes[1, 1].axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Warning (0.2)')
    axes[1, 1].axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (0.1)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train-Val Accuracy Gap')
    axes[1, 1].set_title('Overfitting Monitor')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Summary
    axes[1, 2].axis('off')
    best_val_acc = max(history['val_acc'])
    best_train_acc = max(history['train_acc'])
    final_gap = history['train_val_gap'][-1]
    best_val_sep = max(history.get('val_separation', [0]))
    best_val_score = max(history.get('val_score', [0]))
    final_overlap = history.get('val_overlap_pct', [0])[-1] if history.get('val_overlap_pct') else 0

    summary_text = (
        f"Final Results:\n\n"
        f"Best Val score: {best_val_score:.3f}\n"
        f"Best Val Sep: {best_val_sep:.3f}\n"
        f"Best Val Acc: {best_val_acc:.3f}\n"
        f"Best Train Acc: {best_train_acc:.3f}\n"
        f"Final Gap: {final_gap:.3f}\n"
        f"Val Overlap: {final_overlap:.1%}\n\n"
    )

    axes[1, 2].text(0.5, 0.5, summary_text,
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig('final_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_similarity_distributions(results):
    """Визуализация распределений сходств"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#Hist

    axes[0].hist(results['all_pos_sims'], bins=30, alpha=0.6,
                 label=f'Positive (μ={results["pos_mean"]:.3f}, σ={results["pos_std"]:.3f})',
                 color='green', density=True)
    axes[0].hist(results['all_neg_sims'], bins=30, alpha=0.6,
                 label=f'Negative (μ={results["neg_mean"]:.3f}, σ={results["neg_std"]:.3f})',
                 color='red', density=True)
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Similarity Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data_to_plot = [results['all_pos_sims'], results['all_neg_sims']]
    axes[1].boxplot(data_to_plot, labels=['Positive', 'Negative'])
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Similarity Statistics')
    axes[1].grid(True, alpha=0.3)

    axes[1].scatter([1, 2], [results['pos_mean'], results['neg_mean']],
                    color='red', zorder=3, label='Mean')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('final_similarity_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_embeddings(model, apt_pos, smi_pos, apt_neg, smi_neg, device):

    print("\nSaving adapted embeddings...")

    model.eval()

    def encode_batch(data, encoder_fn, batch_size=32):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = torch.FloatTensor(data[i:i + batch_size]).to(device)
                emb = encoder_fn(batch).cpu().numpy()
                embeddings.append(emb)

        if embeddings:
            return np.vstack(embeddings)
        return np.array([])

    # Take embeddings
    apt_pos_adapted = encode_batch(apt_pos, model.encode_aptamer)
    smi_pos_adapted = encode_batch(smi_pos, model.encode_molecule)

    if len(apt_neg) > 0:
        apt_neg_adapted = encode_batch(apt_neg, model.encode_aptamer)
        smi_neg_adapted = encode_batch(smi_neg, model.encode_molecule)
    else:
        apt_neg_adapted = np.array([])
        smi_neg_adapted = np.array([])

    # Save
    np.save('apt_pos_final.npy', apt_pos_adapted)
    np.save('smi_pos_final.npy', smi_pos_adapted)

    if len(apt_neg_adapted) > 0:
        np.save('apt_neg_final.npy', apt_neg_adapted)
        np.save('smi_neg_final.npy', smi_neg_adapted)

    print(f"    Embeddings saved:")
    print(f"   • apt_pos_final.npy: {apt_pos_adapted.shape}")
    print(f"   • smi_pos_final.npy: {smi_pos_adapted.shape}")
    if len(apt_neg_adapted) > 0:
        print(f"   apt_neg_final.npy: {apt_neg_adapted.shape}")
        print(f"   smi_neg_final.npy: {smi_neg_adapted.shape}")


if __name__ == '__main__':
    import os
    import pandas as pd

    from decoder import full_pipeline, get_negative_cluster_embeddings

    use_pretrained = USE_PRETRAINED
    model, history, test_results, test_loader, device = main(
        train_contrastive=not use_pretrained,
        model_checkpoint=MODEL_CHECKPOINT,
    )

    # ===== ПРОВЕРКА: ЕСЛИ КЛАСТЕРИЗАЦИЯ УЖЕ ВЫПОЛНЕНА =====
    cluster_files_exist = all([
        os.path.exists('cluster_embeddings_768d.npy'),
        os.path.exists('cluster_labels_768d.npy'),
        os.path.exists('cluster_types.npy')
    ])

    if cluster_files_exist and use_pretrained:
        print("\nЗагрузка сохранённых кластеров...")
        embeddings_768d = np.load('cluster_embeddings_768d.npy')
        cluster_labels = np.load('cluster_labels_768d.npy')
        types = np.load('cluster_types.npy')
        print(f"   Загружено {len(embeddings_768d)} эмбеддингов")

        # Получаем негативный кластер
        negative_embeddings, neg_cluster_id, neg_count = get_negative_cluster_embeddings(
            embeddings_768d, cluster_labels, types
        )
        print(f"\nНегативный кластер {neg_cluster_id}: {neg_count} эмбеддингов")
    else:
        print("\nВыполнение кластеризации...")
        # Запускаем пайплайн
        cluster_labels, embeddings_768d, types = full_pipeline(
            model=model,
            test_loader=test_loader,
            device=device,
            n_clusters=5,
        )

        # Получаем негативный кластер
        negative_embeddings, neg_cluster_id, neg_count = get_negative_cluster_embeddings(
            embeddings_768d, cluster_labels, types
        )

        print(f"\nНегативный кластер {neg_cluster_id}: {neg_count} эмбеддингов")
        print(f"   Shape: {negative_embeddings.shape}")

    # ===== ПРОВЕРКА: ЕСЛИ УСЛОВНЫЙ ДЕКОДЕР УЖЕ ОБУЧЕН =====
    conditional_decoder_path = 'best_conditional_decoder.pth'

    # ===== СОБИРАЕМ ЭМБЕДДИНГИ МОЛЕКУЛ И ПАРЫ С SMILES =====
    print("\nCollecting molecule embeddings and negative aptamer latents from test_dataset...")
    model.eval()

    # Загружаем тот же датасет, на котором обучалась contrastive-модель (smi_emb_* должны совпадать)
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Находим колонку с SMILES
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower() or 'canonical_smiles' in col.lower():
            smiles_col = col
            break

    if smiles_col is None:
        print("  Warning: SMILES column not found!")
        smiles_col = 'canonical_smiles'

    smi_emb_cols = [col for col in df.columns if col.startswith('smi_emb_')]
    if not smi_emb_cols:
        raise ValueError("Не найдены колонки smi_emb_* в CSV")

    def embedding_key(values, ndigits=6):
        return tuple(np.asarray(values, dtype=np.float32).round(ndigits).tolist())

    smi_to_smiles = {
        embedding_key(row[smi_emb_cols].values): str(row[smiles_col])
        for _, row in df.iterrows()
    }

    test_dataset = test_loader.dataset
    generation_targets = []
    global_negative_latents = []

    with torch.no_grad():
        for mol_idx, mol_key in enumerate(test_dataset.smis):
            raw_mol = test_dataset.key_to_smi[mol_key]
            raw_negatives = test_dataset.get_negatives_for_molecule(mol_key)
            if len(raw_negatives) == 0:
                continue

            mol_tensor = torch.FloatTensor(raw_mol).unsqueeze(0).to(device)
            mol_emb = model.encode_molecule(mol_tensor).cpu().numpy()[0]

            neg_tensor = torch.FloatTensor(np.asarray(raw_negatives, dtype=np.float32)).to(device)
            neg_latents = model.encode_aptamer(neg_tensor).cpu().numpy()

            if isinstance(mol_key, str):
                mol_smiles = mol_key
            else:
                mol_smiles = smi_to_smiles.get(
                    embedding_key(raw_mol), f"Unknown_SMILES_{mol_idx}"
                )
            generation_targets.append({
                'mol_embedding': mol_emb,
                'raw_smi_embedding': raw_mol,
                'local_negative_latents': neg_latents,
                'smiles': mol_smiles,
            })
            global_negative_latents.append(neg_latents)

    if global_negative_latents:
        global_negative_latents = np.vstack(global_negative_latents)
    else:
        global_negative_latents = np.empty((0, 768), dtype=np.float32)

    print(f"  Целевых молекул с негативами: {len(generation_targets)}")
    print(f"  Негативных latent-точек в общем пуле: {len(global_negative_latents)}")

    # ===== ПОДГОТОВКА ДАННЫХ ДЛЯ GRU (768d mol + 768d apt latent → sequence) =====
    from Embeddings import aptamer_encode
    from GRU import (
        ConditionalGRUDecoder,
        build_aligned_training_pairs,
        filter_generation_targets_by_separation,
        generate_ranked_aptamers_for_molecule,
        select_diverse_output_candidates,
        select_top_candidates_for_tools,
        train_conditional_decoder,
    )

    print("\nBuilding aligned (molecule_emb, aptamer_emb, sequence) data for GRU...")
    mol_embeddings_array, apt_embeddings_for_decoder, sequences = build_aligned_training_pairs(
        df=df,
        model=model,
        device=device,
        negative_only=True,
    )

    print(f"\nData for GRU decoder:")
    print(f"   Aligned negative samples: {len(mol_embeddings_array)}")
    print(f"   Molecule embeddings: {mol_embeddings_array.shape}")
    print(f"   Aptamer latents (768d): {apt_embeddings_for_decoder.shape}")

    # Общий пул negative latent: test negatives + чистый negative-кластер (768d)
    if len(global_negative_latents) > 0 and len(negative_embeddings) > 0:
        global_negative_pool = np.vstack([global_negative_latents, negative_embeddings])
    elif len(negative_embeddings) > 0:
        global_negative_pool = negative_embeddings
    else:
        global_negative_pool = global_negative_latents

    print(
        f"   Global negative pool for generation: {len(global_negative_pool)} "
        f"(test={len(global_negative_latents)}, cluster={len(negative_embeddings)})"
    )

    # ===== ОБУЧЕНИЕ ИЛИ ЗАГРУЗКА GRU =====
    conditional_decoder = ConditionalGRUDecoder(
        mol_dim=768,
        latent_dim=768,
        hidden_dim=512,
        max_len=50,
        min_len=20,
        num_layers=2,
        dropout=0.3,
    )

    if use_pretrained and os.path.exists(conditional_decoder_path):
        print(f"\nЗагрузка предобученного GRU из '{conditional_decoder_path}'")
        try:
            conditional_decoder.load_state_dict(
                torch.load(conditional_decoder_path, map_location=device)
            )
            conditional_decoder = conditional_decoder.to(device)
            conditional_decoder.eval()
            print("   GRU decoder загружен успешно!")
        except RuntimeError as e:
            print(f"   Старые веса GRU несовместимы: {e}")
            print("   Будет выполнено переобучение GRU")
            use_pretrained = False

    if not use_pretrained or not os.path.exists(conditional_decoder_path):
        print("\n" + "=" * 70)
        print("TRAINING GRU DECODER (768d Molecule + 768d Negative Latent → Aptamer)")
        print("=" * 70)

        conditional_decoder = train_conditional_decoder(
            decoder=conditional_decoder,
            mol_embeddings=mol_embeddings_array,
            apt_embeddings=apt_embeddings_for_decoder,
            sequences=sequences,
            epochs=50,
            batch_size=32,
            lr=1e-3,
            device=device,
            scheduled_sampling_max=0.4,
        )

        torch.save(conditional_decoder.state_dict(), conditional_decoder_path)
        print(f"GRU decoder сохранён в '{conditional_decoder_path}'")

    # ГЕНЕРАЦИЯ NON-INTERACTING АПТАМЕРОВ ЧЕРЕЗ GRU + NEGATIVE LATENTS
    # Режим по умолчанию: молекулы из test split датасета (top-K по contrastive_separation).
    # Для одной мишени задайте target_smiles = подстрока SMILES из CSV.
    GEN_CONFIG = {
        "target_smiles": None,
        "target_smiles_fallbacks": [],
        "min_contrastive_separation": 0.05,
        "max_generation_targets": 10,
        "sequence_sim_filter": True,
        "max_latent_sim_for_decode": -0.10,
        "allow_relaxed_fallback": False,
        "n_latent_points": 128,
        "samples_per_latent": 8,
        "latent_jitter_copies": 2,
        "latent_jitter_std": 0.10,
        "diversity_threshold": 0.88,
        "temperature": 1.0,
        "top_k": 8,
        "max_latent_similarity": 0.15,
        "n_keep": 50,
        "output_top_k_for_tools": 3,
        "max_motif_penalty_for_tools": 0.05,
        "min_seq_len": 18,
        "max_seq_len": 50,
        "motif_penalty_weight": 0.15,
        "max_kmer_repeat": 2,
        "motif_kmer_sizes": (5, 6),
        "max_homopolymer_motif": 5,
        "reject_high_motif_repeat": True,
        "max_same_prefix": 2,
        "motif_prefix_len": 8,
        "resume_from_molecule_index": 0,
    }

    n_targets_before_gate = len(generation_targets)
    generation_targets, skipped_targets = filter_generation_targets_by_separation(
        generation_targets=generation_targets,
        df=df,
        contrastive_model=model,
        device=device,
        min_separation=GEN_CONFIG["min_contrastive_separation"],
        max_targets=GEN_CONFIG.get("max_generation_targets"),
        smiles_col=smiles_col,
        target_smiles=GEN_CONFIG.get("target_smiles"),
        target_smiles_fallbacks=GEN_CONFIG.get("target_smiles_fallbacks"),
        global_negative_latents=global_negative_latents,
    )
    gate_label = (
        f"target='{GEN_CONFIG['target_smiles'][:40]}...'"
        if GEN_CONFIG.get("target_smiles")
        else (
            f"min separation={GEN_CONFIG['min_contrastive_separation']}, "
            f"max_targets={GEN_CONFIG.get('max_generation_targets')}"
        )
    )
    print(
        f"\nMolecule gating ({gate_label}): "
        f"kept {len(generation_targets)}/{n_targets_before_gate}"
    )
    if skipped_targets:
        print(f"  Skipped molecules: {len(skipped_targets)}")
        for smiles, separation, reason in skipped_targets[:5]:
            sep_text = "n/a" if separation is None else f"{separation:.4f}"
            print(f"    [{reason}] separation={sep_text} smiles={str(smiles)[:70]}")
        if len(skipped_targets) > 5:
            print(f"    ... and {len(skipped_targets) - 5} more")

    print("\n" + "=" * 70)
    print("GENERATING NON-INTERACTING APTAMERS WITH GRU + NEGATIVE LATENTS")
    print("=" * 70)
    if GEN_CONFIG.get("target_smiles"):
        print(f"Mode: single target SMILES")
    else:
        print(
            f"Mode: dataset molecules from test split "
            f"(top {GEN_CONFIG.get('max_generation_targets') or 'all'} by contrastive_separation)"
        )
    print(f"GEN_CONFIG: {GEN_CONFIG}")

    max_latent_similarity = GEN_CONFIG["max_latent_similarity"]
    n_keep = GEN_CONFIG["n_keep"]
    pair_rows = []
    top_tool_rows = []
    resume_from = int(GEN_CONFIG.get("resume_from_molecule_index") or 0)
    top_k_tools = int(GEN_CONFIG.get("output_top_k_for_tools", 3))

    print("Preloading GENA-LM encoder (cached for all molecules)...")
    aptamer_encode(["ACGTACGTACGTACGTACGTACGT"], batch_size=1, device=str(device))

    file_mode = "a" if resume_from > 0 else "w"
    with open('generated_pairs_molecule_aptamer.txt', file_mode, encoding='utf-8') as f:
        if resume_from == 0:
            f.write("=" * 80 + "\n")
            f.write("GRU GENERATED NON-INTERACTING APTAMERS (negative 768d latent seeds)\n")
            f.write("=" * 80 + "\n\n")
        else:
            print(f"Resuming generation from molecule #{resume_from} (append mode)")

        generated_count = 0
        generation_stats = []

        for mol_counter, target in enumerate(generation_targets):
            if mol_counter < resume_from:
                continue
            mol_emb = target['mol_embedding']
            mol_smiles = target['smiles']
            local_neg = target['local_negative_latents']

            f.write(f"\n{'=' * 70}\n")
            f.write(f"MOLECULE #{mol_counter}\n")
            f.write(f"SMILES: {mol_smiles}\n")
            f.write(f"{'=' * 70}\n")

            print(f"\nМолекула #{mol_counter}: {mol_smiles[:80]}...")

            candidates = generate_ranked_aptamers_for_molecule(
                decoder=conditional_decoder,
                mol_embedding=mol_emb,
                local_negative_points=local_neg,
                global_negative_points=global_negative_pool,
                device=device,
                n_latent_points=GEN_CONFIG["n_latent_points"],
                samples_per_latent=GEN_CONFIG["samples_per_latent"],
                temperature=GEN_CONFIG["temperature"],
                top_k=GEN_CONFIG["top_k"],
                max_similarity=max_latent_similarity,
                diversity_threshold=GEN_CONFIG["diversity_threshold"],
                latent_jitter_copies=GEN_CONFIG["latent_jitter_copies"],
                latent_jitter_std=GEN_CONFIG["latent_jitter_std"],
                min_seq_len=GEN_CONFIG["min_seq_len"],
                max_seq_len=GEN_CONFIG["max_seq_len"],
                contrastive_model=model,
                aptamer_encode_fn=aptamer_encode,
                raw_smi_embedding=target.get("raw_smi_embedding"),
                baseline_positive_mean=target.get("positive_mean"),
                use_sequence_sim_filter=GEN_CONFIG["sequence_sim_filter"],
                max_latent_sim_for_decode=GEN_CONFIG.get("max_latent_sim_for_decode", 0.0),
                motif_penalty_weight=GEN_CONFIG.get("motif_penalty_weight", 0.15),
                max_kmer_repeat=GEN_CONFIG.get("max_kmer_repeat", 2),
                motif_kmer_sizes=tuple(GEN_CONFIG.get("motif_kmer_sizes", (5, 6))),
                max_homopolymer_motif=GEN_CONFIG.get("max_homopolymer_motif", 5),
                reject_high_motif_repeat=GEN_CONFIG.get("reject_high_motif_repeat", True),
                motif_prefix_len=GEN_CONFIG.get("motif_prefix_len", 8),
            )

            max_decode_sim = GEN_CONFIG.get("max_latent_sim_for_decode", 0.0)
            strict_candidates = [
                c for c in candidates
                if c['latent_similarity'] <= max_latent_similarity
                and c.get('latent_sim', c.get('sequence_sim', 1.0)) <= max_decode_sim
            ]
            output_candidates = select_diverse_output_candidates(
                strict_candidates,
                n_keep=n_keep,
                max_same_prefix=GEN_CONFIG.get("max_same_prefix", 2),
                prefix_len=GEN_CONFIG.get("motif_prefix_len", 8),
            )
            used_relaxed = False
            if not output_candidates and GEN_CONFIG.get("allow_relaxed_fallback", False):
                used_relaxed = True
                output_candidates = select_diverse_output_candidates(
                    candidates,
                    n_keep=n_keep,
                    max_same_prefix=GEN_CONFIG.get("max_same_prefix", 2),
                    prefix_len=GEN_CONFIG.get("motif_prefix_len", 8),
                )

            f.write(
                f"\n  n_latent_points={GEN_CONFIG['n_latent_points']}, "
                f"samples_per_latent={GEN_CONFIG['samples_per_latent']}, "
                f"jitter_copies={GEN_CONFIG['latent_jitter_copies']}, "
                f"total_candidates={len(candidates)}, "
                f"strict_pass={len(strict_candidates)}, "
                f"used_relaxed_fallback={used_relaxed}\n"
            )
            if target.get("positive_mean") is not None:
                f.write(
                    f"  baseline_positive_mean={target['positive_mean']:.4f}, "
                    f"baseline_negative_mean={target.get('negative_mean', float('nan')):.4f}, "
                    f"contrastive_separation={target.get('contrastive_separation', float('nan')):.4f}\n"
                )
            if candidates:
                sims = [c['latent_similarity'] for c in candidates]
                f.write(
                    f"  latent_similarity: min={min(sims):.4f}, "
                    f"mean={np.mean(sims):.4f}, max={max(sims):.4f}\n"
                )
                if any("latent_sim" in c or "sequence_sim" in c for c in candidates):
                    seq_sims = [
                        c.get("latent_sim", c.get("sequence_sim"))
                        for c in candidates
                        if "latent_sim" in c or "sequence_sim" in c
                    ]
                    f.write(
                        f"  latent_sim(decoded): min={min(seq_sims):.4f}, "
                        f"mean={np.mean(seq_sims):.4f}, max={max(seq_sims):.4f}, "
                        f"pass_vs_positive_mean={sum(1 for s in seq_sims if s < target.get('positive_mean', 1.0))}/{len(seq_sims)}, "
                        f"pass_latent_sim_filter={sum(1 for s in seq_sims if s <= max_decode_sim)}/{len(seq_sims)}\n"
                    )
                    generation_stats.append({
                        "molecule_index": mol_counter,
                        "smiles": mol_smiles,
                        "contrastive_separation": target.get("contrastive_separation"),
                        "sequence_sim_min": float(min(seq_sims)),
                        "sequence_sim_mean": float(np.mean(seq_sims)),
                        "sequence_sim_max": float(max(seq_sims)),
                        "n_candidates": len(candidates),
                        "n_saved": len(output_candidates),
                    })

            if not output_candidates:
                f.write("\n  No candidates after GRU generation.\n")
                print("  Нет кандидатов после GRU-генерации")
                continue

            tool_candidates = select_top_candidates_for_tools(
                strict_candidates if strict_candidates else candidates,
                top_k=top_k_tools,
                max_motif_penalty=GEN_CONFIG.get("max_motif_penalty_for_tools", 0.05),
            )
            f.write(f"\n  TOP {top_k_tools} FOR TOOLS (RSAPred/Boltz):\n")
            for tool_rank, tool_cand in enumerate(tool_candidates, start=1):
                tool_decoded = tool_cand.get("latent_sim", tool_cand.get("sequence_sim"))
                decoded_val = float(tool_decoded) if tool_decoded is not None else float("nan")
                f.write(
                    f"    #{tool_rank} DNA={tool_cand['sequence']} | "
                    f"RNA={tool_cand['sequence'].replace('T', 'U')} | "
                    f"decoded_sim={decoded_val:.4f} | "
                    f"motif_penalty={tool_cand.get('motif_penalty', 0.0):.3f}\n"
                )
                top_tool_rows.append({
                    "molecule_index": mol_counter,
                    "tool_rank": tool_rank,
                    "canonical_smiles": mol_smiles,
                    "contrastive_separation": target.get("contrastive_separation"),
                    "dna_sequence": tool_cand["sequence"],
                    "rna_sequence": tool_cand["sequence"].replace("T", "U"),
                    "decoded_sim": decoded_val,
                    "motif_penalty": tool_cand.get("motif_penalty", 0.0),
                    "composite_score": tool_cand.get("composite_score"),
                    "seed_sim": tool_cand.get("latent_similarity"),
                })

            for i, candidate in enumerate(output_candidates):
                new_aptamer = candidate['sequence']
                relaxed = used_relaxed or candidate['latent_similarity'] > max_latent_similarity
                seed_sim = candidate['latent_similarity']
                decoded_sim = candidate.get(
                    'latent_sim',
                    candidate.get('sequence_sim'),
                )
                f.write(f"\n  Aptamer #{i + 1}:\n")
                f.write(f"  {new_aptamer}\n")
                decoded_text = (
                    f", decoded_sim={decoded_sim:.4f}"
                    if decoded_sim is not None
                    else ""
                )
                f.write(
                    f"  seed_sim={seed_sim:.4f}{decoded_text}, "
                    f"length={candidate['length']}, GC={candidate['gc']:.2f}, "
                    f"motif_penalty={candidate.get('motif_penalty', 0.0):.3f}, "
                    f"relaxed={relaxed}\n"
                )

                decoded_display = f"{decoded_sim:.4f}" if decoded_sim is not None else "n/a"
                motif_pen = candidate.get("motif_penalty")
                motif_text = f", motif_pen={motif_pen:.3f}" if motif_pen is not None else ""
                print(
                    f"  Aptamer {i + 1}: {new_aptamer[:60]}... "
                    f"(seed_sim={seed_sim:.4f}, decoded_sim={decoded_display}{motif_text}, "
                    f"relaxed={relaxed})"
                )
                pair_rows.append({
                    "molecule_index": mol_counter,
                    "canonical_smiles": mol_smiles,
                    "contrastive_separation": target.get("contrastive_separation"),
                    "positive_mean": target.get("positive_mean"),
                    "negative_mean": target.get("negative_mean"),
                    "aptamer_rank": i + 1,
                    "dna_sequence": new_aptamer,
                    "rna_sequence": new_aptamer.replace("T", "U"),
                    "seed_sim": seed_sim,
                    "decoded_sim": decoded_sim,
                    "motif_penalty": candidate.get("motif_penalty", 0.0),
                    "length": candidate["length"],
                    "gc_content": candidate["gc"],
                    "relaxed": relaxed,
                })
                generated_count += 1

    print(f"\n" + "=" * 70)
    print(f"Сгенерировано {generated_count} аптамеров для {len(generation_targets)} молекул")
    print(f"Результаты сохранены в 'generated_pairs_molecule_aptamer.txt'")
    if pair_rows:
        pairs_df = pd.DataFrame(pair_rows)
        summary_path = "generation_summary.csv"
        if resume_from > 0 and os.path.exists(summary_path):
            old_df = pd.read_csv(summary_path)
            pairs_df = pd.concat([old_df, pairs_df], ignore_index=True)
        pairs_df.to_csv(summary_path, index=False)
        print(f"Таблица пар сохранена в '{summary_path}' ({len(pairs_df)} строк)")
    if generation_stats:
        print("\nGeneration decoded_sim summary by molecule:")
        for row in generation_stats:
            print(
                f"  #{row['molecule_index']} sep={row['contrastive_separation']:.4f} "
                f"decoded_sim min/mean/max="
                f"{row['sequence_sim_min']:.4f}/"
                f"{row['sequence_sim_mean']:.4f}/"
                f"{row['sequence_sim_max']:.4f} "
                f"saved={row['n_saved']}/{row['n_candidates']}"
            )
        print(
            f"  Overall mean decoded_sim: "
            f"{np.mean([r['sequence_sim_mean'] for r in generation_stats]):.4f}"
        )

    if top_tool_rows:
        tools_df = pd.DataFrame(top_tool_rows)
        tools_path = "top_candidates_for_tools.csv"
        tools_df.to_csv(tools_path, index=False, encoding="utf-8-sig")
        print(f"\nТоп-{top_k_tools} для тулов сохранён в '{tools_path}' ({len(tools_df)} строк)")

        print("\n" + "=" * 70)
        print(f"TOP {top_k_tools} CANDIDATES PER MOLECULE (RSAPred / Boltz)")
        print("=" * 70)
        for mol_idx in sorted(tools_df["molecule_index"].unique()):
            mol_block = tools_df[tools_df["molecule_index"] == mol_idx]
            smiles = str(mol_block.iloc[0]["canonical_smiles"])
            sep = mol_block.iloc[0].get("contrastive_separation")
            sep_text = f"{float(sep):.4f}" if pd.notna(sep) else "n/a"
            print(f"\nMOLECULE #{int(mol_idx)} | sep={sep_text}")
            print(f"  SMILES: {smiles[:100]}{'...' if len(smiles) > 100 else ''}")
            for _, row in mol_block.sort_values("tool_rank").iterrows():
                print(
                    f"  #{int(row['tool_rank'])} decoded={row['decoded_sim']:.4f} "
                    f"motif={row['motif_penalty']:.3f}"
                )
                print(f"      DNA: {row['dna_sequence']}")
                print(f"      RNA: {row['rna_sequence']}")
