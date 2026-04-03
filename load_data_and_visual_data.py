import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap


def load_data(file_path):

    print(f"\nLoading data from {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)

    seq_emb_cols = [col for col in df.columns if col.startswith('seq_emb_')]
    smi_emb_cols = [col for col in df.columns if col.startswith('smi_emb_')]

    if not seq_emb_cols or not smi_emb_cols:
        raise ValueError("No embedding columns found")

    print(f"  Sequence embeddings: {len(seq_emb_cols)} columns")
    print(f"  SMILES embeddings: {len(smi_emb_cols)} columns")

    pos_mask = df['label'] == 1
    neg_mask = df['label'] == 0

    apt_pos = df.loc[pos_mask, seq_emb_cols].values.astype(np.float32)
    smi_pos = df.loc[pos_mask, smi_emb_cols].values.astype(np.float32)
    apt_neg = df.loc[neg_mask, seq_emb_cols].values.astype(np.float32)
    smi_neg = df.loc[neg_mask, smi_emb_cols].values.astype(np.float32)

    print(f"\nData statistics:")
    print(f"  • Positive pairs: {len(apt_pos)}")
    print(f"  • Negative pairs: {len(apt_neg)}")
    print(f"  • Positive/Negative ratio: {len(apt_pos) / len(apt_neg):.2f}" if len(
        apt_neg) > 0 else "  • No negative pairs")

    return apt_pos, smi_pos, apt_neg, smi_neg, len(seq_emb_cols), len(smi_emb_cols)



def analyze_results(model, data_loader, device):
    """Анализ результатов модели"""
    model.eval()

    all_pos_sims = []
    all_neg_sims = []
    accuracies = []

    with torch.no_grad():
        for batch in data_loader:
            anchor_apt = batch['anchor_apt'].to(device)
            positive_smi = batch['positive_smi'].to(device)
            negative_smis = batch['negative_smis'].to(device)

            outputs = model(anchor_apt, positive_smi, negative_smis)

            # Positive similarities
            pos_sim = F.cosine_similarity(outputs['z_anchor'], outputs['z_positive'], dim=1)
            all_pos_sims.extend(pos_sim.cpu().numpy())

            # Negative similarities
            batch_size, n_neg, _ = outputs['z_negatives'].shape
            for i in range(batch_size):
                neg_sims = F.cosine_similarity(
                    outputs['z_anchor'][i:i + 1].expand(n_neg, -1),
                    outputs['z_negatives'][i]
                )
                all_neg_sims.extend(neg_sims.cpu().numpy())

                # Accuracy для этого примера
                all_sims = torch.cat([pos_sim[i:i + 1], neg_sims])
                if torch.argmax(all_sims) == 0:
                    accuracies.append(1)
                else:
                    accuracies.append(0)

    accuracy = np.mean(accuracies) if accuracies else 0
    separation = np.mean(all_pos_sims) - np.mean(all_neg_sims) if all_pos_sims and all_neg_sims else 0

    return {
        'accuracy': accuracy,
        'separation': separation,
        'pos_mean': np.mean(all_pos_sims) if all_pos_sims else 0,
        'neg_mean': np.mean(all_neg_sims) if all_neg_sims else 0,
        'pos_std': np.std(all_pos_sims) if all_pos_sims else 0,
        'neg_std': np.std(all_neg_sims) if all_neg_sims else 0,
        'all_pos_sims': all_pos_sims,
        'all_neg_sims': all_neg_sims
    }


# ==================== 7. ВИЗУАЛИЗАЦИЯ ЭМБЕДДИНГОВ (T-SNE, UMAP, PCA) ====================

def visualize_embeddings_correct(model, test_loader, device, save_path='embedding_visualization_correct.png'):
    """
    ПРАВИЛЬНАЯ визуализация - использует ТЕ ЖЕ hard negatives, что и в тесте
    """
    print("\n" + "=" * 60)
    print("ПРАВИЛЬНАЯ ВИЗУАЛИЗАЦИЯ (с использованием test_loader)")
    print("=" * 60)

    model.eval()
    all_embeddings = []
    all_labels = []
    all_types = []  # 'anchor', 'positive', 'negative'

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            anchor_apt = batch['anchor_apt'].to(device)
            positive_smi = batch['positive_smi'].to(device)
            negative_smis = batch['negative_smis'].to(device)

            # Получаем эмбеддинги
            z_anchor = model.encode_aptamer(anchor_apt)  # (B, dim)
            z_positive = model.encode_molecule(positive_smi)  # (B, dim)

            batch_size = z_anchor.size(0)
            n_neg = negative_smis.size(1)

            # Для каждого элемента в батче
            for i in range(batch_size):
                # Anchor (аптамер)
                all_embeddings.append(z_anchor[i].cpu().numpy())
                all_labels.append(1)  # positive class
                all_types.append('anchor')

                # Positive (малая молекула из той же пары)
                all_embeddings.append(z_positive[i].cpu().numpy())
                all_labels.append(1)  # positive class
                all_types.append('positive')

                # Hard negatives (сложные негативные молекулы)
                for j in range(n_neg):
                    z_neg = model.encode_molecule(negative_smis[i, j].unsqueeze(0))
                    all_embeddings.append(z_neg[0].cpu().numpy())
                    all_labels.append(0)  # negative class
                    all_types.append('hard_negative')

            # Ограничиваем для визуализации (если данных слишком много)
            if batch_idx > 50 and len(all_embeddings) > 5000:
                print(f"   Ограничиваем выборку для визуализации: {len(all_embeddings)} точек")
                break

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    print(f"\nСтатистика выборки для визуализации:")
    print(f"  • Всего точек: {len(all_embeddings)}")
    print(f"  • Positive (anchor + positive): {(all_labels == 1).sum()}")
    print(f"  • Negative (hard negatives): {(all_labels == 0).sum()}")

    # Вычисляем метрики (должны совпадать с тестовыми!)
    pos_embeddings = all_embeddings[all_labels == 1]
    neg_embeddings = all_embeddings[all_labels == 0]

    # Считаем сходства для anchor-positive пар
    pos_sims = []
    for i in range(0, len(pos_embeddings), 2):
        if i + 1 < len(pos_embeddings):
            sim = np.dot(pos_embeddings[i], pos_embeddings[i + 1])
            pos_sims.append(sim)

    # Считаем сходства для anchor-hard_negative пар
    neg_sims = []
    for i in range(0, len(neg_embeddings), 1):
        # Каждый hard negative сравниваем с соответствующим anchor
        anchor_idx = (i // 1) * 2
        if anchor_idx < len(pos_embeddings):
            sim = np.dot(pos_embeddings[anchor_idx], neg_embeddings[i])
            neg_sims.append(sim)

    print(f"\n📊 МЕТРИКИ НА ВИЗУАЛИЗАЦИИ (должны совпадать с тестом):")
    print(f"  • Positive similarity: {np.mean(pos_sims):.4f} (σ={np.std(pos_sims):.4f})")
    print(f"  • Negative similarity: {np.mean(neg_sims):.4f} (σ={np.std(neg_sims):.4f})")
    print(f"  • Separation: {np.mean(pos_sims) - np.mean(neg_sims):.4f}")

    # Нормализация для визуализации
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(all_embeddings)

    # PCA
    n_components = min(50, embeddings_scaled.shape[1], embeddings_scaled.shape[0])
    print(f"\nПрименение PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    print(f"  • Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")

    # Создаем визуализации
    fig = plt.figure(figsize=(20, 12))

    # 1. t-SNE
    print("\nПрименение t-SNE...")
    ax1 = fig.add_subplot(2, 3, 1)
    try:
        perplexity_val = min(30, len(embeddings_pca) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity_val,
                    random_state=42, max_iter=1000, verbose=0)
        embeddings_tsne = tsne.fit_transform(embeddings_pca)

        # Рисуем
        pos_mask = all_labels == 1
        neg_mask = all_labels == 0

        ax1.scatter(embeddings_tsne[pos_mask, 0], embeddings_tsne[pos_mask, 1],
                    c='green', alpha=0.6, s=20, label=f'Positive ({pos_mask.sum()})')
        ax1.scatter(embeddings_tsne[neg_mask, 0], embeddings_tsne[neg_mask, 1],
                    c='red', alpha=0.6, s=20, label=f'Hard Negative ({neg_mask.sum()})')
        ax1.set_title('t-SNE (hard negatives)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    except Exception as e:
        ax1.text(0.5, 0.5, f't-SNE failed:\n{str(e)[:50]}', ha='center', va='center')
        ax1.set_title('t-SNE Error')

    # 2. UMAP
    print("Применение UMAP...")
    ax2 = fig.add_subplot(2, 3, 2)
    try:
        n_neighbors_val = min(25, len(embeddings_pca) - 1)
        reducer = umap.UMAP(n_components=2, min_dist=0.3,
                            random_state=42, n_neighbors=n_neighbors_val)
        embeddings_umap = reducer.fit_transform(embeddings_pca)

        pos_mask = all_labels == 1
        neg_mask = all_labels == 0

        ax2.scatter(embeddings_umap[pos_mask, 0], embeddings_umap[pos_mask, 1],
                    c='green', alpha=0.6, s=20, label=f'Positive ({pos_mask.sum()})')
        ax2.scatter(embeddings_umap[neg_mask, 0], embeddings_umap[neg_mask, 1],
                    c='red', alpha=0.6, s=20, label=f'Hard Negative ({neg_mask.sum()})')
        ax2.set_title('UMAP (hard negatives)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f'UMAP failed:\n{str(e)[:50]}', ha='center', va='center')
        ax2.set_title('UMAP Error')

    # 3. PCA 2D
    ax3 = fig.add_subplot(2, 3, 3)
    pca_2d = PCA(n_components=2)
    embeddings_pca_2d = pca_2d.fit_transform(embeddings_scaled)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    ax3.scatter(embeddings_pca_2d[pos_mask, 0], embeddings_pca_2d[pos_mask, 1],
                c='green', alpha=0.6, s=20, label=f'Positive ({pos_mask.sum()})')
    ax3.scatter(embeddings_pca_2d[neg_mask, 0], embeddings_pca_2d[neg_mask, 1],
                c='red', alpha=0.6, s=20, label=f'Hard Negative ({neg_mask.sum()})')
    ax3.set_title('PCA 2D (hard negatives)', fontsize=12)
    ax3.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
    ax3.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Распределение сходств
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(pos_sims, bins=30, alpha=0.6, color='green',
             label=f'Positive (μ={np.mean(pos_sims):.3f})', density=True)
    ax4.hist(neg_sims, bins=30, alpha=0.6, color='red',
             label=f'Hard Negative (μ={np.mean(neg_sims):.3f})', density=True)
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Density')
    ax4.set_title('Similarity Distribution (Test Data)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Box plot
    ax5 = fig.add_subplot(2, 3, 5)
    box_data = [pos_sims, neg_sims]
    bp = ax5.boxplot(box_data, labels=['Positive', 'Hard Negative'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax5.set_ylabel('Cosine Similarity')
    ax5.set_title('Similarity Statistics')
    ax5.grid(True, alpha=0.3)

    # 6. 3D PCA (если размерность позволяет)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    if embeddings_scaled.shape[1] >= 3:
        pca_3d = PCA(n_components=3)
        embeddings_pca_3d = pca_3d.fit_transform(embeddings_scaled[:3000])  # ограничиваем для 3D
        labels_3d = all_labels[:3000]

        pos_mask_3d = labels_3d == 1
        neg_mask_3d = labels_3d == 0

        ax6.scatter(embeddings_pca_3d[pos_mask_3d, 0],
                    embeddings_pca_3d[pos_mask_3d, 1],
                    embeddings_pca_3d[pos_mask_3d, 2],
                    c='green', alpha=0.5, s=10, label='Positive')
        ax6.scatter(embeddings_pca_3d[neg_mask_3d, 0],
                    embeddings_pca_3d[neg_mask_3d, 1],
                    embeddings_pca_3d[neg_mask_3d, 2],
                    c='red', alpha=0.5, s=10, label='Hard Negative')
        ax6.set_title('PCA 3D (first 3000 points)')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, '3D PCA not available', ha='center', va='center')
        ax6.set_title('PCA 3D (insufficient dimensions)')

    plt.suptitle('CORRECT Embedding Visualization (using test_loader hard negatives)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Визуализация сохранена: {save_path}")

    return {
        'embeddings': all_embeddings,
        'labels': all_labels,
        'pos_sims': pos_sims,
        'neg_sims': neg_sims
    }


def analyze_results_correct(model, data_loader, device):
    """
    ПРАВИЛЬНЫЙ анализ - только hard negatives из теста
    """
    model.eval()

    all_pos_sims = []
    all_neg_sims = []
    all_pos_sims_raw = []
    all_neg_sims_raw = []

    with torch.no_grad():
        for batch in data_loader:
            anchor_apt = batch['anchor_apt'].to(device)
            positive_smi = batch['positive_smi'].to(device)
            negative_smis = batch['negative_smis'].to(device)

            outputs = model(anchor_apt, positive_smi, negative_smis)

            # Positive similarities (anchor - positive из одной пары)
            pos_sim = F.cosine_similarity(outputs['z_anchor'], outputs['z_positive'], dim=1)
            all_pos_sims.extend(pos_sim.cpu().numpy())
            all_pos_sims_raw.append(pos_sim.cpu().numpy())

            # Negative similarities (anchor - hard negatives)
            batch_size, n_neg, _ = outputs['z_negatives'].shape
            for i in range(batch_size):
                neg_sims = F.cosine_similarity(
                    outputs['z_anchor'][i:i + 1].expand(n_neg, -1),
                    outputs['z_negatives'][i]
                )
                all_neg_sims.extend(neg_sims.cpu().numpy())
                all_neg_sims_raw.append(neg_sims.cpu().numpy())

    all_pos_sims = np.array(all_pos_sims)
    all_neg_sims = np.array(all_neg_sims)

    # Accuracy: для каждого anchor, позитивный должен быть ближе всех hard negatives
    # Упрощенная метрика: позитивный similarity > среднего негативного
    accuracy = (all_pos_sims > all_neg_sims.mean()).mean()

    # Более точная метрика: для каждого sample отдельно
    accuracies = []
    pos_idx = 0
    for batch_raw in all_pos_sims_raw:
        for pos_sim in batch_raw:
            # Берем соответствующие негативы
            if pos_idx < len(all_neg_sims):
                neg_sims_for_this = all_neg_sims[pos_idx * n_neg:(pos_idx + 1) * n_neg]
                if len(neg_sims_for_this) > 0:
                    if pos_sim > neg_sims_for_this.max():  # позитивный лучше всех негативов
                        accuracies.append(1)
                    else:
                        accuracies.append(0)
            pos_idx += 1

    precise_accuracy = np.mean(accuracies) if accuracies else accuracy

    return {
        'pos_mean': np.mean(all_pos_sims),
        'neg_mean': np.mean(all_neg_sims),
        'pos_std': np.std(all_pos_sims),
        'neg_std': np.std(all_neg_sims),
        'separation': np.mean(all_pos_sims) - np.mean(all_neg_sims),
        'accuracy': precise_accuracy,
        'simple_accuracy': accuracy,
        'all_pos_sims': all_pos_sims,
        'all_neg_sims': all_neg_sims
    }


def visualize_embeddings_2d_simple(model, test_loader, device, save_path='embedding_2d_simple.png'):
    """
    Простая 2D визуализация через PCA
    """
    print("\n" + "=" * 60)
    print("ПРОСТАЯ 2D ВИЗУАЛИЗАЦИЯ")
    print("=" * 60)

    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            anchor_apt = batch['anchor_apt'].to(device)
            positive_smi = batch['positive_smi'].to(device)
            negative_smis = batch['negative_smis'].to(device)

            z_anchor = model.encode_aptamer(anchor_apt)
            z_positive = model.encode_molecule(positive_smi)

            batch_size = z_anchor.size(0)

            for i in range(batch_size):
                # Anchor и positive как одна точка? Нет, лучше раздельно
                all_embeddings.append(z_anchor[i].cpu().numpy())
                all_labels.append(1)

                all_embeddings.append(z_positive[i].cpu().numpy())
                all_labels.append(1)

                # Hard negatives
                for j in range(negative_smis.size(1)):
                    z_neg = model.encode_molecule(negative_smis[i, j].unsqueeze(0))
                    all_embeddings.append(z_neg[0].cpu().numpy())
                    all_labels.append(0)

            if len(all_embeddings) > 10000:
                break

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    plt.figure(figsize=(12, 10))

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    plt.scatter(embeddings_2d[pos_mask, 0], embeddings_2d[pos_mask, 1],
                c='green', alpha=0.5, s=15, label=f'Positive ({pos_mask.sum()})')
    plt.scatter(embeddings_2d[neg_mask, 0], embeddings_2d[neg_mask, 1],
                c='red', alpha=0.5, s=15, label=f'Hard Negative ({neg_mask.sum()})')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Embedding Space Visualization (PCA) - Hard Negatives from Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"✅ Простая визуализация сохранена: {save_path}")