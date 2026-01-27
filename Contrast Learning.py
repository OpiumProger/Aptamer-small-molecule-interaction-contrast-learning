import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

warnings.filterwarnings('ignore')

# Добавляем для Windows multiprocessing
if __name__ == '__main__':
    print("=" * 60)
    print("INFO-NCE С УЛУЧШЕНИЯМИ ДЛЯ КЛАСТЕРИЗАЦИИ")
    print("=" * 60)

    # ==================== 1. ЗАГРУЗКА ЕДИНОГО ФАЙЛА ====================
    print("\n📥 Загрузка единого файла с эмбеддингами...")


    def load_unified_embeddings(file_path):
        """
        Загружает единый файл с эмбеддингами sequence и SMILES
        и разделяет их на positive и negative
        """
        print(f"  Загрузка: {os.path.basename(file_path)}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return None

        print(f"✅ Файл загружен. Размер: {df.shape}")
        print(f"📋 Колонки: {list(df.columns)}")

        # Находим колонки с эмбеддингами
        seq_emb_cols = [col for col in df.columns if col.startswith('seq_emb_')]
        smi_emb_cols = [col for col in df.columns if col.startswith('smi_emb_')]

        if not seq_emb_cols or not smi_emb_cols:
            print("Не найдены колонки с эмбеддингами seq_emb_* или smi_emb_*")
            print("   Проверьте, что файл содержит колонки seq_emb_0... и smi_emb_0...")
            return None

        print(f"Найдено колонок seq_emb: {len(seq_emb_cols)}")
        print(f"Найдено колонок smi_emb: {len(smi_emb_cols)}")

        # Проверяем колонки label
        if 'label' not in df.columns:
            print(" Колонка 'label' не найдена в файле")
            print(" Файл должен содержать колонку 'label' со значениями 0 или 1")
            return None

        # Проверяем значения label
        unique_labels = sorted(df['label'].unique())
        print(f"📊 Уникальные значения label: {unique_labels}")

        # Разделяем на positive и negative
        pos_mask = df['label'] == 1
        neg_mask = df['label'] == 0

        print(f"\ Разделение по label:")
        print(f"   • Positive (label=1): {pos_mask.sum()} строк")
        print(f"   • Negative (label=0): {neg_mask.sum()} строк")

        if pos_mask.sum() == 0:
            print("❌ Нет positive данных (label=1)")
            return None

        if neg_mask.sum() == 0:
            print(" Нет negative данных (label=0)")
            print("Будет использована только positive часть")

        # Извлекаем эмбеддинги
        apt_pos_emb = df.loc[pos_mask, seq_emb_cols].values.astype(np.float32)
        smi_pos_emb = df.loc[pos_mask, smi_emb_cols].values.astype(np.float32)

        if neg_mask.sum() > 0:
            apt_neg_emb = df.loc[neg_mask, seq_emb_cols].values.astype(np.float32)
            smi_neg_emb = df.loc[neg_mask, smi_emb_cols].values.astype(np.float32)
        # else:
        #     # Если нет negative, создаем пустые массивы
        #     apt_neg_emb = np.array([], dtype=np.float32).reshape(0, len(seq_emb_cols))
        #     smi_neg_emb = np.array([], dtype=np.float32).reshape(0, len(smi_emb_cols))

        print(f"\nЭмбеддинги извлечены:")
        print(f"   • Positive аптамеры: {apt_pos_emb.shape}")
        print(f"   • Positive SMILES: {smi_pos_emb.shape}")
        print(f"   • Negative аптамеры: {apt_neg_emb.shape}")
        print(f"   • Negative SMILES: {smi_neg_emb.shape}")

        return {
            'apt_pos': apt_pos_emb,
            'smi_pos': smi_pos_emb,
            'apt_neg': apt_neg_emb,
            'smi_neg': smi_neg_emb,
            'df': df,
            'seq_dim': len(seq_emb_cols),
            'smi_dim': len(smi_emb_cols)
        }


    # Загружаем файл (укажите путь к вашему файлу)
    input_file = "AptaBench_dataset_v2_with_embeddings.csv"  # ИЗМЕНИТЕ НА ВАШ ФАЙЛ

    # Загружаем данные
    data = load_unified_embeddings(input_file)

    if data is None:
        print("Не удалось загрузить данные")
        exit()

    # Извлекаем данные
    apt_pos_emb = data['apt_pos']
    smi_pos_emb = data['smi_pos']
    apt_neg_emb = data['apt_neg']
    smi_neg_emb = data['smi_neg']
    seq_dim = data['seq_dim']
    smi_dim = data['smi_dim']

    print(f"\nДанные подготовлены:")
    print(f"   • Positive пары: {apt_pos_emb.shape[0]}")
    print(f"   • Negative пары: {apt_neg_emb.shape[0]}")
    print(f"   • Размерность sequence: {seq_dim}")
    print(f"   • Размерность SMILES: {smi_dim}")

    # ==================== 2. ПРОСТОЙ ДАТАСЕТ ====================
    print("\nСоздание датасета...")


    class SimpleContrastiveDataset(Dataset):
        """Простой датасет с positive и negative парами"""

        def __init__(self, apt_pos, smi_pos, apt_neg, smi_neg):
            # Positive пары
            n_pos = min(len(apt_pos), len(smi_pos))
            self.apt_pos = torch.FloatTensor(apt_pos[:n_pos])
            self.smi_pos = torch.FloatTensor(smi_pos[:n_pos])

            # Negative пары
            n_neg = min(len(apt_neg), len(smi_neg))
            self.apt_neg = torch.FloatTensor(apt_neg[:n_neg])
            self.smi_neg = torch.FloatTensor(smi_neg[:n_neg])

            # Создаем пары
            self.pairs = []
            self.labels = []

            # Positive пары
            for i in range(n_pos):
                self.pairs.append(('pos', i))
                self.labels.append(1.0)

            # Negative пары (столько же сколько positive, если negative достаточно)
            n_neg_to_use = min(n_pos, n_neg)  # Берем минимум
            for i in range(n_neg_to_use):
                self.pairs.append(('neg', i))
                self.labels.append(0.0)

            print(f"   Создано пар: {len(self.pairs)}")
            print(f"     • Positive: {n_pos}")
            print(f"     • Negative: {n_neg_to_use}")

            self.n_pos = n_pos
            self.n_neg = n_neg_to_use

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            pair_type, pair_idx = self.pairs[idx]

            if pair_type == 'pos':
                apt_emb = self.apt_pos[pair_idx]
                smi_emb = self.smi_pos[pair_idx]
                label = 1.0
            else:  # 'neg'
                apt_emb = self.apt_neg[pair_idx]
                smi_emb = self.smi_neg[pair_idx]
                label = 0.0

            return apt_emb, smi_emb, torch.tensor(label, dtype=torch.float)

        def get_stats(self):
            """Возвращает статистику датасета"""
            return {
                'total': len(self),
                'positive': self.n_pos,
                'negative': self.n_neg,
                'pos_ratio': self.n_pos / len(self) if len(self) > 0 else 0
            }


    # Создаем датасет
    dataset = SimpleContrastiveDataset(
        apt_pos_emb, smi_pos_emb,
        apt_neg_emb, smi_neg_emb
    )

    stats = dataset.get_stats()
    print(f"    Статистика датасета:")
    print(f"   • Всего пар: {stats['total']}")
    print(f"   • Positive: {stats['positive']} ({stats['pos_ratio']:.1%})")
    print(f"   • Negative: {stats['negative']} ({1 - stats['pos_ratio']:.1%})")

    # Разделяем на train/test
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        print(" Слишком мало данных для разделения")
        print(" Нужно больше данных или используйте весь датасет для обучения")
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Убираем num_workers для Windows
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"📊 Размеры:")
    print(f"   • Train: {len(train_dataset)}")
    print(f"   • Val: {len(val_dataset)}")

    # ==================== 3. УЛУЧШЕННАЯ МОДЕЛЬ ====================
    print("\n   Создание улучшенной модели...")


    class ImprovedProjector(nn.Module):
        def __init__(self, input_dim, output_dim=128):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(256, output_dim),
            )

        def forward(self, x):
            return F.normalize(self.network(x), p=2, dim=1)  # L2 нормализация


    class ContrastiveModel(nn.Module):
        def __init__(self, sequence_dim, smiles_dim):
            super().__init__()
            self.sequence_proj = ImprovedProjector(sequence_dim)
            self.smiles_proj = ImprovedProjector(smiles_dim)

        def forward(self, sequence_emb, smiles_emb):
            z_seq = self.sequence_proj(sequence_emb)
            z_smi = self.smiles_proj(smiles_emb)

            return z_seq, z_smi


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Модель создана на {device}")

    model = ContrastiveModel(seq_dim, smi_dim).to(device)
    print(f"   Параметров: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 4.ФУНКЦИЯ ПОТЕРЬ ====================
    print("\n   Настройка обучения...")
    class SimpleContrastiveLoss(nn.Module):
        def __init__(self, pos_threshold=0.6, neg_threshold=0.0, neg_weight=4.0):
            super().__init__()
            self.pos_threshold = pos_threshold  # Positive должны быть > этого
            self.neg_threshold = neg_threshold  # Negative должны быть < этого
            self.neg_weight = neg_weight  # Насколько negative важнее

            print(f"    SimpleContrastiveLoss:")
            print(f"   • Positive > {pos_threshold}")
            print(f"   • Negative < {neg_threshold}")
            print(f"   • Negative weight: {neg_weight}")

        def forward(self, z_seq, z_smi, labels):
            cos_sim = F.cosine_similarity(z_seq, z_smi, dim=-1)

            # Positive: штраф если < threshold
            pos_loss = F.relu(self.pos_threshold - cos_sim) * labels

            # Negative: штраф если > threshold
            neg_loss = F.relu(cos_sim - self.neg_threshold) * (1 - labels)

            return pos_loss.mean() + self.neg_weight * neg_loss.mean()


    # Инициализация:
    loss_fn = SimpleContrastiveLoss(
        pos_threshold=0.6,
        neg_threshold=0.0,
        neg_weight=5.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    if len(dataset) > 100:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)


    print(f"   Positive threshold: {loss_fn.pos_threshold}")
    print(f"   Negative threshold: {loss_fn.neg_threshold}")
    print(f"   Negative weight: {loss_fn.neg_weight}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Weight decay: {optimizer.param_groups[0]['weight_decay']}")

    # ==================== 5. ОБУЧЕНИЕ ====================
    print("\n   Начинаем обучение...")

    # Адаптивное количество эпох
    if len(dataset) < 100:
        epochs = 20
    elif len(dataset) < 1000:
        epochs = 30
    else:
        epochs = 40

    print(f"    Количество эпох: {epochs}")

    history = {
        'train_loss': [], 'val_loss': [],
        'train_separation': [], 'val_separation': [],
        'train_auc': [], 'val_auc': [],
        'train_pos_mean': [], 'train_neg_mean': [],
        'val_pos_mean': [], 'val_neg_mean': []
    }

    best_val_auc = 0.0
    best_val_separation = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_pos_sim = []
        train_neg_sim = []
        train_preds = []
        train_labels = []

        if len(train_loader) > 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
            for seq_batch, smi_batch, labels in pbar:
                seq_batch = seq_batch.to(device)
                smi_batch = smi_batch.to(device)
                labels = labels.to(device)

                # Прямой проход
                z_seq, z_smi = model(seq_batch, smi_batch)

                # Вычисляем loss
                loss = loss_fn(z_seq, z_smi, labels)

                # Вычисляем сходства для статистики
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(z_seq, z_smi, dim=-1)

                    # Сохраняем предсказания и метки для AUC
                    train_preds.extend(cos_sim.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())

                    pos_mask = labels == 1
                    neg_mask = labels == 0

                    if pos_mask.sum() > 0:
                        train_pos_sim.extend(cos_sim[pos_mask].cpu().numpy())
                    if neg_mask.sum() > 0:
                        train_neg_sim.extend(cos_sim[neg_mask].cpu().numpy())

                # Оптимизация
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_separation = np.mean(train_pos_sim) - np.mean(train_neg_sim) if train_pos_sim and train_neg_sim else 0
            train_pos_mean = np.mean(train_pos_sim) if train_pos_sim else 0
            train_neg_mean = np.mean(train_neg_sim) if train_neg_sim else 0

            # Вычисляем AUC для train
            if len(train_preds) > 0 and len(set(train_labels)) > 1:
                train_auc = roc_auc_score(train_labels, train_preds)
            else:
                train_auc = 0.5
        else:
            avg_train_loss = 0
            train_separation = 0
            train_auc = 0.5
            train_pos_mean = 0
            train_neg_mean = 0

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        val_pos_sim = []
        val_neg_sim = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for seq_batch, smi_batch, labels in val_loader:
                seq_batch = seq_batch.to(device)
                smi_batch = smi_batch.to(device)
                labels = labels.to(device)

                z_seq, z_smi = model(seq_batch, smi_batch)
                loss = loss_fn(z_seq, z_smi, labels)
                val_loss += loss.item()

                cos_sim = F.cosine_similarity(z_seq, z_smi, dim=-1)

                # Сохраняем предсказания и метки для AUC
                val_preds.extend(cos_sim.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                pos_mask = labels == 1
                neg_mask = labels == 0

                if pos_mask.sum() > 0:
                    val_pos_sim.extend(cos_sim[pos_mask].cpu().numpy())
                if neg_mask.sum() > 0:
                    val_neg_sim.extend(cos_sim[neg_mask].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_separation = np.mean(val_pos_sim) - np.mean(val_neg_sim) if val_pos_sim and val_neg_sim else 0
        val_pos_mean = np.mean(val_pos_sim) if val_pos_sim else 0
        val_neg_mean = np.mean(val_neg_sim) if val_neg_sim else 0

        # Вычисляем AUC для validation
        if len(val_preds) > 0 and len(set(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_preds)
        else:
            val_auc = 0.5

        # ===== СОХРАНЕНИЕ ИСТОРИИ =====
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_separation'].append(train_separation)
        history['val_separation'].append(val_separation)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_pos_mean'].append(train_pos_mean)
        history['train_neg_mean'].append(train_neg_mean)
        history['val_pos_mean'].append(val_pos_mean)
        history['val_neg_mean'].append(val_neg_mean)

        # ===== ВЫВОД ИНФОРМАЦИИ =====
        print(f"\nEpoch {epoch + 1:02d}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        print(f"  Train Separation: {train_separation:.4f} | Val Separation: {val_separation:.4f}")
        print(f"  Train Pos Mean: {train_pos_mean:.4f} | Train Neg Mean: {train_neg_mean:.4f}")
        print(f"  Val Pos Mean: {val_pos_mean:.4f} | Val Neg Mean: {val_neg_mean:.4f}")

        # ===== СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ =====
        # Сохраняем по AUC и separation
        if val_auc > best_val_auc or (val_auc == best_val_auc and val_separation > best_val_separation):
            if val_auc > best_val_auc:
                best_val_auc = val_auc
            if val_separation > best_val_separation:
                best_val_separation = val_separation
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_auc': val_auc,
                'val_separation': val_separation,
                'val_pos_mean': val_pos_mean,
                'val_neg_mean': val_neg_mean,
                'history': history
            }, 'best_model_improved.pth')
            print(f"  ✓ Сохранена лучшая модель (AUC: {val_auc:.4f}, Sep: {val_separation:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping на эпохе {epoch + 1}")
                break

        scheduler.step()

    print("\n   Обучение завершено!")


    # ==================== DBSCAN ФУНКЦИИ ====================

    def analyze_with_dbscan(embeddings, labels, eps=0.3, min_samples=5):
        """
        Применяет DBSCAN к эмбеддингам для анализа структуры
        """
        from sklearn.cluster import DBSCAN

        print(f"\n  DBSCAN анализ (eps={eps}, min_samples={min_samples}):")

        # DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)

        # Статистика
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"   • Найдено кластеров: {n_clusters}")
        print(f"   • Выбросов (noise): {n_noise}")
        print(f"   • Всего точек: {len(embeddings)}")

        # Анализ по классам
        unique_labels = np.unique(labels)

        for label_val in unique_labels:
            mask = labels == label_val
            class_points = embeddings[mask]
            class_clusters = cluster_labels[mask]

            n_class_clusters = len(set(class_clusters)) - (1 if -1 in class_clusters else 0)
            n_class_noise = list(class_clusters).count(-1)

            print(f"\n   Класс {'Positive' if label_val == 1 else 'Negative'}:")
            print(f"     • Кластеров: {n_class_clusters}")
            print(f"     • Выбросов: {n_class_noise}")
            print(f"     • В кластерах: {len(class_points) - n_class_noise}")

        return cluster_labels, clustering


    def find_problematic_negatives_with_dbscan(model, apt_neg_emb, smi_neg_emb,
                                               threshold=0.5, eps=0.4):
        """
        Находит проблемные negative пары с помощью DBSCAN
        """
        from sklearn.cluster import DBSCAN

        model.eval()

        with torch.no_grad():
            # Получаем эмбеддинги
            apt_tensor = torch.FloatTensor(apt_neg_emb).to(device)
            smi_tensor = torch.FloatTensor(smi_neg_emb).to(device)

            z_apt, z_smi = model(apt_tensor, smi_tensor)

            # Вычисляем similarities
            similarities = F.cosine_similarity(z_apt, z_smi, dim=-1).cpu().numpy()

            # Объединяем эмбеддинги для кластеризации
            combined_emb = np.hstack([z_apt.cpu().numpy(), z_smi.cpu().numpy()])

            # 1. Находим high similarity negative
            high_sim_mask = similarities > threshold
            high_sim_indices = np.where(high_sim_mask)[0]

            print(f"\n🔍 Проблемные negative (similarity > {threshold}): {len(high_sim_indices)}")

            if len(high_sim_indices) == 0:
                return [], [], None

            # 2. DBSCAN на проблемных negative
            problem_embeddings = combined_emb[high_sim_indices]

            dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine')
            cluster_labels = dbscan.fit_predict(problem_embeddings)

            # 3. Анализ кластеров
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            print(f"   DBSCAN обнаружил {n_clusters} кластера проблемных negative")

            # Собираем информацию по кластерам
            clusters_info = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    continue  # Пропускаем шум

                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                original_indices = high_sim_indices[cluster_indices]

                cluster_sims = similarities[original_indices]

                clusters_info.append({
                    'cluster_id': cluster_id,
                    'size': len(cluster_indices),
                    'indices': original_indices,
                    'mean_similarity': cluster_sims.mean(),
                    'max_similarity': cluster_sims.max(),
                    'min_similarity': cluster_sims.min()
                })

            # Сортируем по размеру кластера
            clusters_info.sort(key=lambda x: x['size'], reverse=True)

            for i, cluster in enumerate(clusters_info[:5]):  # Топ-5 кластеров
                print(f"\n   Кластер #{cluster['cluster_id']}:")
                print(f"     • Размер: {cluster['size']} пар")
                print(f"     • Среднее similarity: {cluster['mean_similarity']:.3f}")
                print(f"     • Максимальное: {cluster['max_similarity']:.3f}")
                print(f"     • Минимальное: {cluster['min_similarity']:.3f}")

            return high_sim_indices, clusters_info, dbscan




    # ==================== 6. ГРАФИК ОБУЧЕНИЯ ====================
    print("\n   График обучения...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3, linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3, linewidth=2)
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss во время обучения')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Separation
    axes[0, 1].plot(history['train_separation'], label='Train Separation', marker='o', markersize=3, linewidth=2)
    axes[0, 1].plot(history['val_separation'], label='Val Separation', marker='s', markersize=3, linewidth=2)
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Разделение (Positive - Negative)')
    axes[0, 1].set_title('Разделение классов во время обучения')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. AUC
    axes[0, 2].plot(history['train_auc'], label='Train AUC', marker='o', markersize=3, linewidth=2)
    axes[0, 2].plot(history['val_auc'], label='Val AUC', marker='s', markersize=3, linewidth=2)
    axes[0, 2].set_xlabel('Эпоха')
    axes[0, 2].set_ylabel('ROC-AUC')
    axes[0, 2].set_title('ROC-AUC во время обучения')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Positive Mean
    axes[1, 0].plot(history['train_pos_mean'], label='Train Pos Mean', marker='o', markersize=3, linewidth=2,
                    color='green')
    axes[1, 0].plot(history['val_pos_mean'], label='Val Pos Mean', marker='s', markersize=3, linewidth=2,
                    color='darkgreen')
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Среднее сходство')
    axes[1, 0].set_title('Positive пары - среднее сходство')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # 5. Negative Mean
    axes[1, 1].plot(history['train_neg_mean'], label='Train Neg Mean', marker='o', markersize=3, linewidth=2,
                    color='red')
    axes[1, 1].plot(history['val_neg_mean'], label='Val Neg Mean', marker='s', markersize=3, linewidth=2,
                    color='darkred')
    axes[1, 1].set_xlabel('Эпоха')
    axes[1, 1].set_ylabel('Среднее сходство')
    axes[1, 1].set_title('Negative пары - среднее сходство')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # 6. Combined
    axes[1, 2].plot(history['train_pos_mean'], label='Train Pos', marker='o', markersize=3, linewidth=2, color='green')
    axes[1, 2].plot(history['train_neg_mean'], label='Train Neg', marker='o', markersize=3, linewidth=2, color='red')
    axes[1, 2].plot(history['val_pos_mean'], label='Val Pos', marker='s', markersize=3, linewidth=2, color='darkgreen',
                    linestyle='--')
    axes[1, 2].plot(history['val_neg_mean'], label='Val Neg', marker='s', markersize=3, linewidth=2, color='darkred',
                    linestyle='--')
    axes[1, 2].set_xlabel('Эпоха')
    axes[1, 2].set_ylabel('Среднее сходство')
    axes[1, 2].set_title('Сравнение Positive и Negative')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves_improved.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" График сохранен: training_curves_improved.png")

    # ==================== 7. СОХРАНЕНИЕ АДАПТИРОВАННЫХ ЭМБЕДДИНГОВ ====================
    print("\n   Получение адаптированных эмбеддингов...")


    def get_adapted_embeddings(model, seq_emb, smi_emb, batch_size=32):
        model.eval()

        # Убедимся, что размерности совпадают
        n_samples = min(len(seq_emb), len(smi_emb))

        if n_samples == 0:
            return np.array([]), np.array([])

        adapted_seq = []
        adapted_smi = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)

                seq_batch = torch.FloatTensor(seq_emb[i:end_idx]).to(device)
                smi_batch = torch.FloatTensor(smi_emb[i:end_idx]).to(device)

                z_seq, z_smi = model(seq_batch, smi_batch)

                adapted_seq.append(z_seq.cpu().numpy())
                adapted_smi.append(z_smi.cpu().numpy())

        if adapted_seq:
            adapted_seq = np.vstack(adapted_seq)
            adapted_smi = np.vstack(adapted_smi)
            print(f"   Преобразовано: {n_samples} пар")
        else:
            adapted_seq = np.array([])
            adapted_smi = np.array([])
            print(f"   Нет данных для преобразования")
        return adapted_seq, adapted_smi

    print(" Загрузка лучшей модели...")
    try:
        checkpoint = torch.load('best_model_improved.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"    Модель эпохи {checkpoint['epoch'] + 1} загружена")
        print(f"   Val AUC: {checkpoint['val_auc']:.4f}")
        print(f"   Val Separation: {checkpoint['val_separation']:.4f}")
        print(f"   Val Pos Mean: {checkpoint['val_pos_mean']:.4f}")
        print(f"   Val Neg Mean: {checkpoint['val_neg_mean']:.4f}")
    except:
        print(" Не удалось загрузить модель, используем текущую")

    # Получаем адаптированные эмбеддинги
    print("\n🔧 Адаптация эмбеддингов...")

    if len(apt_pos_emb) > 0 and len(smi_pos_emb) > 0:
        print("  Positive пары...")
        seq_pos_adapted, smi_pos_adapted = get_adapted_embeddings(model, apt_pos_emb, smi_pos_emb)

        if len(seq_pos_adapted) > 0:
            # Сохраняем positive
            np.save('seq_pos_adapted_improved.npy', seq_pos_adapted)
            np.save('smi_pos_adapted_improved.npy', smi_pos_adapted)
            print(f"  Сохранено: {seq_pos_adapted.shape}")

    if len(apt_neg_emb) > 0 and len(smi_neg_emb) > 0:
        print("  Negative пары...")
        seq_neg_adapted, smi_neg_adapted = get_adapted_embeddings(model, apt_neg_emb, smi_neg_emb)

        if len(seq_neg_adapted) > 0:
            # Сохраняем negative
            np.save('seq_neg_adapted_improved.npy', seq_neg_adapted)
            np.save('smi_neg_adapted_improved.npy', smi_neg_adapted)
            print(f"    Сохранено: {seq_neg_adapted.shape}")

    print(f"\n  Адаптированные эмбеддинги сохранены:")
    saved_files = []
    for fname in ['seq_pos_adapted_improved.npy', 'smi_pos_adapted_improved.npy',
                  'seq_neg_adapted_improved.npy', 'smi_neg_adapted_improved.npy']:
        if os.path.exists(fname):
            data = np.load(fname, allow_pickle=True)
            if hasattr(data, 'shape'):
                print(f"   • {fname}: {data.shape}")
                saved_files.append(fname)

    if not saved_files:
        print("     Файлы не созданы (возможно, нет данных)")

    # ==================== 8. АНАЛИЗ РЕЗУЛЬТАТОВ ====================
    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)


    def compute_similarities(seq_emb, smi_emb):
        """Вычисляет косинусные сходства между парами"""
        if len(seq_emb) == 0 or len(smi_emb) == 0:
            return np.array([])

        similarities = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(seq_emb), batch_size):
                end_idx = min(i + batch_size, len(seq_emb))

                seq_batch = torch.FloatTensor(seq_emb[i:end_idx])
                smi_batch = torch.FloatTensor(smi_emb[i:end_idx])

                cos_sim = F.cosine_similarity(seq_batch, smi_batch, dim=-1)
                similarities.extend(cos_sim.numpy())

        return np.array(similarities)


    # Вычисляем сходства если есть адаптированные эмбеддинги
    if os.path.exists('seq_pos_adapted_improved.npy') and os.path.exists('smi_pos_adapted_improved.npy'):
        print("\n   Вычисление косинусных сходств...")

        seq_pos_adapted = np.load('seq_pos_adapted_improved.npy')
        smi_pos_adapted = np.load('smi_pos_adapted_improved.npy')

        pos_similarities = compute_similarities(seq_pos_adapted, smi_pos_adapted)

        if len(pos_similarities) > 0:
            print(f"\n  POSITIVE пары:")
            print(f"   • Количество: {len(pos_similarities)}")
            print(f"   • Среднее: {pos_similarities.mean():.4f}")
            print(f"   • Медиана: {np.median(pos_similarities):.4f}")
            print(f"   • Min-Max: {pos_similarities.min():.4f} - {pos_similarities.max():.4f}")

        if os.path.exists('seq_neg_adapted_improved.npy') and os.path.exists('smi_neg_adapted_improved.npy'):
            seq_neg_adapted = np.load('seq_neg_adapted_improved.npy')
            smi_neg_adapted = np.load('smi_neg_adapted_improved.npy')

            neg_similarities = compute_similarities(seq_neg_adapted, smi_neg_adapted)

            if len(neg_similarities) > 0:
                print(f"\n  NEGATIVE пары:")
                print(f"   • Количество: {len(neg_similarities)}")
                print(f"   • Среднее: {neg_similarities.mean():.4f}")
                print(f"   • Медиана: {np.median(neg_similarities):.4f}")
                print(f"   • Min-Max: {neg_similarities.min():.4f} - {neg_similarities.max():.4f}")

            # Разделение если есть оба типа
            if len(pos_similarities) > 0 and len(neg_similarities) > 0:
                separation = pos_similarities.mean() - neg_similarities.mean()
                print(f"\n  РАЗДЕЛЕНИЕ (separation): {separation:.4f}")

                # ==================== ROC-AUC КРИВАЯ ====================
                print("\n   Построение ROC-AUC кривой...")

                # Объединяем все предсказания и метки
                all_preds = np.concatenate([pos_similarities, neg_similarities])
                all_labels = np.concatenate([np.ones_like(pos_similarities),
                                             np.zeros_like(neg_similarities)])

                # Вычисляем ROC-AUC
                auc_score = roc_auc_score(all_labels, all_preds)
                print(f"   ROC-AUC Score: {auc_score:.4f}")

                # Вычисляем ROC-кривую
                fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

                # Визуализация ROC-кривой
                plt.figure(figsize=(10, 8))

                plt.subplot(2, 2, 1)
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                         label=f'ROC curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)

                # Визуализация распределения сходств
                plt.subplot(2, 2, 2)
                plt.hist(pos_similarities, bins=50, alpha=0.6, color='green',
                         label=f'Positive (n={len(pos_similarities)}, μ={pos_similarities.mean():.3f})',
                         density=True)
                plt.hist(neg_similarities, bins=50, alpha=0.6, color='red',
                         label=f'Negative (n={len(neg_similarities)}, μ={neg_similarities.mean():.3f})',
                         density=True)

                plt.xlabel('Косинусное сходство')
                plt.ylabel('Плотность вероятности')
                plt.title(f'Распределение сходств (AUC: {auc_score:.3f})')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Добавляем вертикальные линии для средних
                plt.axvline(x=pos_similarities.mean(), color='darkgreen', linestyle='--', linewidth=2)
                plt.axvline(x=neg_similarities.mean(), color='darkred', linestyle='--', linewidth=2)

                # ==================== КОНКАТЕНИРОВАННЫЕ ЭМБЕДДИНГИ ====================
                print("\n   Подготовка конкатенированных эмбеддингов для визуализации...")

                # Конкатенируем sequence и smiles эмбеддинги
                pos_combined = np.hstack([seq_pos_adapted, smi_pos_adapted])
                neg_combined = np.hstack([seq_neg_adapted, smi_neg_adapted])

                # Объединяем все эмбеддинги
                all_embeddings = np.vstack([pos_combined, neg_combined])
                all_labels_combined = np.concatenate([np.ones(len(pos_combined)),
                                                      np.zeros(len(neg_combined))])

                # Нормализуем данные
                scaler = StandardScaler()
                all_embeddings_scaled = scaler.fit_transform(all_embeddings)

                # Проверяем количество компонент для PCA
                n_components = min(50, all_embeddings_scaled.shape[1], all_embeddings_scaled.shape[0])

                # Применяем PCA для уменьшения размерности
                print(f"   Применение PCA для уменьшения размерности до {n_components} компонент...")
                pca = PCA(n_components=n_components)
                embeddings_pca = pca.fit_transform(all_embeddings_scaled)

                print(f"   Объясненная дисперсия PCA: {pca.explained_variance_ratio_.sum():.3f}")

                # ==================== t-SNE ВИЗУАЛИЗАЦИЯ ====================
                print("\n   Применение t-SNE для 2D визуализации...")

                # Используем только если достаточно данных
                if len(all_embeddings) > 10:
                    perplexity_value = min(30, len(all_embeddings) - 1)

                    try:
                        tsne = TSNE(n_components=2, perplexity=perplexity_value,
                                    random_state=42, n_iter=1000, verbose=0)
                        embeddings_tsne = tsne.fit_transform(embeddings_pca)

                        plt.subplot(2, 2, 3)

                        # Разделяем точки по классам
                        pos_indices = all_labels_combined == 1
                        neg_indices = all_labels_combined == 0

                        if pos_indices.sum() > 0:
                            plt.scatter(embeddings_tsne[pos_indices, 0],
                                        embeddings_tsne[pos_indices, 1],
                                        c='green', alpha=0.6, s=30,
                                        label=f'Positive ({pos_indices.sum()})',
                                        edgecolors='black', linewidth=0.5)

                        if neg_indices.sum() > 0:
                            plt.scatter(embeddings_tsne[neg_indices, 0],
                                        embeddings_tsne[neg_indices, 1],
                                        c='red', alpha=0.6, s=30,
                                        label=f'Negative ({neg_indices.sum()})',
                                        edgecolors='black', linewidth=0.5)

                        plt.xlabel('t-SNE 1')
                        plt.ylabel('t-SNE 2')
                        plt.title('t-SNE 2D проекция')
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                    except Exception as e:
                        print(f"     Ошибка t-SNE: {e}")

                # ==================== UMAP ВИЗУАЛИЗАЦИЯ ====================
                print("📊 Применение UMAP для 2D визуализации...")

                if len(all_embeddings) > 10:
                    try:
                        reducer = umap.UMAP(n_components=2, min_dist=0.3, random_state=42,
                                            n_neighbors=min(25, len(all_embeddings) - 1))
                        embeddings_umap = reducer.fit_transform(embeddings_pca)

                        plt.subplot(2, 2, 4)

                        # Разделяем точки по классам
                        if pos_indices.sum() > 0:
                            plt.scatter(embeddings_umap[pos_indices, 0],
                                        embeddings_umap[pos_indices, 1],
                                        c='green', alpha=0.6, s=30,
                                        label=f'Positive ({pos_indices.sum()})',
                                        edgecolors='black', linewidth=0.5)

                        if neg_indices.sum() > 0:
                            plt.scatter(embeddings_umap[neg_indices, 0],
                                        embeddings_umap[neg_indices, 1],
                                        c='red', alpha=0.6, s=30,
                                        label=f'Negative ({neg_indices.sum()})',
                                        edgecolors='black', linewidth=0.5)

                        plt.xlabel('UMAP 1')
                        plt.ylabel('UMAP 2')
                        plt.title('UMAP 2D проекция')
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                    except Exception as e:
                        print(f"    Ошибка UMAP: {e}")

                plt.tight_layout()
                plt.savefig('analysis_results_improved.png', dpi=150, bbox_inches='tight')
                plt.show()
                print("✓ График сохранен: analysis_results_improved.png")

    # ==================== DBSCAN АНАЛИЗ ====================
    print("\n" + "=" * 60)
    print("🔍 DBSCAN АНАЛИЗ КЛАСТЕРОВ")
    print("=" * 60)

    # 1. Получаем все эмбеддинги
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        # Positive
        if len(apt_pos_emb) > 0:
            z_seq_pos, z_smi_pos = model(
                torch.FloatTensor(apt_pos_emb[:1000]).to(device),  # Ограничим для скорости
                torch.FloatTensor(smi_pos_emb[:1000]).to(device)
            )
            combined_pos = torch.cat([z_seq_pos, z_smi_pos], dim=-1).cpu().numpy()
            all_embeddings.append(combined_pos)
            all_labels.extend([1] * len(combined_pos))

        # Negative
        if len(apt_neg_emb) > 0:
            z_seq_neg, z_smi_neg = model(
                torch.FloatTensor(apt_neg_emb[:1000]).to(device),
                torch.FloatTensor(smi_neg_emb[:1000]).to(device)
            )
            combined_neg = torch.cat([z_seq_neg, z_smi_neg], dim=-1).cpu().numpy()
            all_embeddings.append(combined_neg)
            all_labels.extend([0] * len(combined_neg))

    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)

        # Запускаем DBSCAN анализ
        cluster_labels, clustering = analyze_with_dbscan(
            all_embeddings, all_labels, eps=0.4, min_samples=5
        )

        # Визуализация
        if len(all_embeddings) > 10:
            from sklearn.manifold import TSNE

            # t-SNE для визуализации
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            embeddings_2d = tsne.fit_transform(all_embeddings)

            plt.figure(figsize=(12, 10))

            # Цвет по DBSCAN кластерам
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                  c=cluster_labels, cmap='tab20', alpha=0.7, s=30)

            # Обводим проблемные negative (high similarity)
            neg_mask = all_labels == 0
            if neg_mask.any():
                neg_points = embeddings_2d[neg_mask]
                plt.scatter(neg_points[:, 0], neg_points[:, 1],
                            facecolors='none', edgecolors='red', s=100,
                            linewidth=1.5, label='Negative пары')

            plt.colorbar(scatter, label='DBSCAN Cluster')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('DBSCAN кластеризация эмбеддингов')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('dbscan_clusters.png', dpi=150)
            plt.show()


    problem_indices, clusters_info, dbscan = find_problematic_negatives_with_dbscan(
        model, apt_neg_emb, smi_neg_emb, threshold=0.5, eps=0.4
    )



    if clusters_info:
        print("   1. Крупные кластеры (>10 пар) - возможно, ошибки разметки")
        print("   2. Маленькие кластеры (2-5 пар) - сложные случаи для модели")
        print("   3. Выбросы (noise) - аномалии, возможно шум в данных")

