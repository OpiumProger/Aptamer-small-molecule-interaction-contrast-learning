import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')



class FinalContrastiveDataset(Dataset):

    def __init__(self, apt_pos, smi_pos, apt_neg, smi_neg, negative_ratio=3):
        self.apt_pos = torch.FloatTensor(apt_pos)
        self.smi_pos = torch.FloatTensor(smi_pos)
        self.apt_neg = torch.FloatTensor(apt_neg)
        self.smi_neg = torch.FloatTensor(smi_neg)

        self.n_pos = len(self.apt_pos)
        self.n_neg = len(self.apt_neg)
        self.negative_ratio = min(negative_ratio, self.n_neg) if self.n_neg > 0 else 2

        print(f"📊 Final dataset:")
        print(f"   • Positive pairs: {self.n_pos}")
        print(f"   • Negative pairs: {self.n_neg}")
        print(f"   • Negative ratio: {self.negative_ratio}")

    def __len__(self):
        return self.n_pos

    def __getitem__(self, idx):
        anchor_apt = self.apt_pos[idx]
        positive_smi = self.smi_pos[idx]

        # Аугментация
        if torch.rand(1).item() > 0.5:  # 50% chance
            noise = torch.randn_like(anchor_apt) * 0.05
            anchor_apt = anchor_apt + noise

        # Случайные negatives
        if self.n_neg > 0:
            neg_indices = np.random.choice(self.n_neg, size=self.negative_ratio, replace=False)
            negative_smis = self.smi_neg[neg_indices]
        else:
            neg_indices = np.random.choice(self.n_pos, size=self.negative_ratio, replace=False)
            neg_indices = neg_indices[neg_indices != idx]
            negative_smis = self.smi_pos[neg_indices]

        return {
            'anchor_apt': anchor_apt,
            'positive_smi': positive_smi,
            'negative_smis': negative_smis
        }


class MicroContrastiveModel(nn.Module):

    def __init__(self, apt_dim, mol_dim, latent_dim=32, projection_dim=16):
        super().__init__()

        self.apt_encoder = nn.Sequential(
            nn.Linear(apt_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.4),  # ВЫСОКИЙ dropout
            nn.Linear(64, latent_dim)
        )

        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.4),  # ВЫСОКИЙ dropout
            nn.Linear(64, latent_dim)
        )

        self.apt_projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.Dropout(0.3)
        )

        self.mol_projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.Dropout(0.3)
        )

        print(f"    Micro model created:")
        print(f"   • Latent dim: {latent_dim}")
        print(f"   • Projection dim: {projection_dim}")
        print(f"   • Total params: {sum(p.numel() for p in self.parameters()):,}")

    def encode_aptamer(self, x):
        h = self.apt_encoder(x)
        z = self.apt_projection(h)
        return F.normalize(z, p=2, dim=-1)

    def encode_molecule(self, x):
        h = self.mol_encoder(x)
        z = self.mol_projection(h)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, anchor_apt, positive_smi, negative_smis=None):
        z_anchor = self.encode_aptamer(anchor_apt)
        z_positive = self.encode_molecule(positive_smi)

        if negative_smis is not None and len(negative_smis) > 0:
            batch_size, n_neg, mol_dim = negative_smis.shape
            negatives_flat = negative_smis.view(-1, mol_dim)
            z_negatives = self.encode_molecule(negatives_flat)
            z_negatives = z_negatives.view(batch_size, n_neg, -1)

            return {
                'z_anchor': z_anchor,
                'z_positive': z_positive,
                'z_negatives': z_negatives
            }

        return {
            'z_anchor': z_anchor,
            'z_positive': z_positive
        }


#Loss

class TemperatureScaledLoss(nn.Module):
    """Loss с адаптивной температурой"""

    def __init__(self, init_temperature=0.2):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        print(f"🔥 Initial temperature: {init_temperature}")

    def forward(self, z_anchor, z_positive, z_negatives):
        batch_size = z_anchor.size(0)
        n_neg = z_negatives.size(1)

        z_anchor = F.normalize(z_anchor, p=2, dim=1)
        z_positive = F.normalize(z_positive, p=2, dim=1)
        z_negatives = F.normalize(z_negatives.view(-1, z_negatives.size(-1)), p=2, dim=1)
        z_negatives = z_negatives.view(batch_size, n_neg, -1)

        # Positive similarity
        pos_sim = torch.sum(z_anchor * z_positive, dim=1) / self.temperature

        # Negative similarities
        z_anchor_expanded = z_anchor.unsqueeze(1).expand(-1, n_neg, -1)
        neg_sims = torch.sum(z_anchor_expanded * z_negatives, dim=2) / self.temperature

        # Logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)

        # Labels
        labels = torch.zeros(batch_size, dtype=torch.long).to(z_anchor.device)

        # Loss
        loss = F.cross_entropy(logits, labels)

        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()

            # Top-3 accuracy
            sorted_indices = torch.argsort(logits, dim=1, descending=True)
            top3_acc = (sorted_indices[:, :3] == 0).any(dim=1).float().mean()

        metrics = {
            'accuracy': accuracy.item(),
            'top3_acc': top3_acc.item(),
            'temperature': self.temperature.item(),
            'loss': loss.item()
        }

        return loss, metrics



class FinalTrainer:

    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        self.criterion = TemperatureScaledLoss(init_temperature=0.25)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=5e-3
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_top3': [], 'val_top3': [],
            'temperature': [],
            'train_val_gap': []
        }

        self.best_val_acc = 0
        self.patience_counter = 0
        self.patience = 5
        self.best_model_state = None

    def compute_l2_reg(self):
        l2_reg = 0
        for param in self.model.parameters():
            l2_reg += torch.norm(param, p=2)
        return l2_reg

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_top3 = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            # Move to device
            anchor_apt = batch['anchor_apt'].to(self.device)
            positive_smi = batch['positive_smi'].to(self.device)
            negative_smis = batch['negative_smis'].to(self.device)

            # Forward pass
            outputs = self.model(anchor_apt, positive_smi, negative_smis)

            # Compute loss
            loss, metrics = self.criterion(
                outputs['z_anchor'],
                outputs['z_positive'],
                outputs['z_negatives']
            )

            l2_reg = self.compute_l2_reg()
            total_loss_value = loss + 0.01 * l2_reg

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_value.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            total_acc += metrics['accuracy']
            total_top3 += metrics['top3_acc']
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'temp': f"{metrics['temperature']:.3f}"
            })

        # Average metrics
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_acc = total_acc / n_batches if n_batches > 0 else 0
        avg_top3 = total_top3 / n_batches if n_batches > 0 else 0

        return avg_loss, avg_acc, avg_top3

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_top3 = 0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                anchor_apt = batch['anchor_apt'].to(self.device)
                positive_smi = batch['positive_smi'].to(self.device)
                negative_smis = batch['negative_smis'].to(self.device)

                # Forward pass
                outputs = self.model(anchor_apt, positive_smi, negative_smis)

                # Compute loss
                loss, metrics = self.criterion(
                    outputs['z_anchor'],
                    outputs['z_positive'],
                    outputs['z_negatives']
                )

                # Update statistics
                total_loss += loss.item()
                total_acc += metrics['accuracy']
                total_top3 += metrics['top3_acc']
                n_batches += 1

        # Average metrics
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_acc = total_acc / n_batches if n_batches > 0 else 0
        avg_top3 = total_top3 / n_batches if n_batches > 0 else 0

        return avg_loss, avg_acc, avg_top3

    def train(self, n_epochs=15, save_path='final_best_model.pth'):
        print(f"\n🚀 Starting final training (max {n_epochs} epochs)...")

        for epoch in range(1, n_epochs + 1):
            # Train
            train_loss, train_acc, train_top3 = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_top3 = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Save current temperature
            current_temp = self.criterion.temperature.item()

            # Calculate gap
            gap = train_acc - val_acc

            # Save to history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_top3'].append(train_top3)
            self.history['val_top3'].append(val_top3)
            self.history['temperature'].append(current_temp)
            self.history['train_val_gap'].append(gap)

            # Print epoch summary
            print(f"\n  Epoch {epoch:03d}/{n_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | Top3: {train_top3:.3f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | Top3: {val_top3:.3f}")
            print(f"  Gap: {gap:.3f} | Temp: {current_temp:.3f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Проверка на сильное переобучение
            if train_acc > 0.85 and val_acc < 0.4:
                print(f"CRITICAL OVERFITTING! Stopping...")
                break

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_top3': val_top3,
                    'train_acc': train_acc,
                    'temperature': current_temp,
                    'history': self.history
                }, save_path)

                print(f"  💾 Saved best model (Val Acc: {val_acc:.3f})")
            else:
                self.patience_counter += 1
                print(f"  ⏳ No improvement: {self.patience_counter}/{self.patience}")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                print(f"   Best Val Accuracy: {self.best_val_acc:.3f}")
                break

        # Берём лучшую модель
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"\n✅ Training completed!")
        print(f"   Best validation accuracy: {self.best_val_acc:.3f}")
        print(f"   Final train-val gap: {self.history['train_val_gap'][-1]:.3f}")

        return self.history



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

def visualize_embeddings(model, apt_pos, smi_pos, apt_neg, smi_neg, device,
                         save_path='embedding_visualization.png'):
    """
    Визуализация эмбеддингов с помощью PCA, t-SNE и UMAP
    """
    print("\n🎨 Визуализация эмбеддингов...")

    model.eval()

    # Получаем все эмбеддинги
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        # Positive пары
        if len(apt_pos) > 0:
            apt_pos_tensor = torch.FloatTensor(apt_pos).to(device)
            smi_pos_tensor = torch.FloatTensor(smi_pos).to(device)

            z_apt_pos = model.encode_aptamer(apt_pos_tensor)
            z_smi_pos = model.encode_molecule(smi_pos_tensor)

            # Конкатенируем эмбеддинги аптамеров и молекул
            pos_combined = torch.cat([z_apt_pos, z_smi_pos], dim=1)
            all_embeddings.append(pos_combined.cpu().numpy())
            all_labels.extend([1] * len(pos_combined))

        # Negative пары
        if len(apt_neg) > 0:
            apt_neg_tensor = torch.FloatTensor(apt_neg).to(device)
            smi_neg_tensor = torch.FloatTensor(smi_neg).to(device)

            z_apt_neg = model.encode_aptamer(apt_neg_tensor)
            z_smi_neg = model.encode_molecule(smi_neg_tensor)

            neg_combined = torch.cat([z_apt_neg, z_smi_neg], dim=1)
            all_embeddings.append(neg_combined.cpu().numpy())
            all_labels.extend([0] * len(neg_combined))

    if not all_embeddings:
        print("   Нет данных для визуализации")
        return

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    print(f"   Всего эмбеддингов: {len(all_embeddings)}")
    print(f"   Размерность: {all_embeddings.shape[1]}")

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

    # Создаем фигуру для визуализации
    fig = plt.figure(figsize=(15, 10))

    # ==================== t-SNE ВИЗУАЛИЗАЦИЯ ====================
    print("\n   Применение t-SNE для 2D визуализации...")

    if len(all_embeddings) > 10:
        perplexity_value = min(30, len(all_embeddings) - 1)

        try:
            tsne = TSNE(n_components=2, perplexity=perplexity_value,
                        random_state=42, n_iter=1000, verbose=0)
            embeddings_tsne = tsne.fit_transform(embeddings_pca)

            ax1 = plt.subplot(2, 2, 1)

            # Разделяем точки по классам
            pos_indices = all_labels == 1
            neg_indices = all_labels == 0

            if pos_indices.sum() > 0:
                ax1.scatter(embeddings_tsne[pos_indices, 0],
                            embeddings_tsne[pos_indices, 1],
                            c='green', alpha=0.6, s=30,
                            label=f'Positive ({pos_indices.sum()})',
                            edgecolors='black', linewidth=0.5)

            if neg_indices.sum() > 0:
                ax1.scatter(embeddings_tsne[neg_indices, 0],
                            embeddings_tsne[neg_indices, 1],
                            c='red', alpha=0.6, s=30,
                            label=f'Negative ({neg_indices.sum()})',
                            edgecolors='black', linewidth=0.5)

            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            ax1.set_title('t-SNE 2D проекция')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        except Exception as e:
            print(f"     Ошибка t-SNE: {e}")
            # Создаем пустой subplot если t-SNE не сработал
            ax1 = plt.subplot(2, 2, 1)
            ax1.text(0.5, 0.5, 't-SNE failed', ha='center', va='center')
            ax1.set_title('t-SNE (failed)')

    # ==================== UMAP ВИЗУАЛИЗАЦИЯ ====================
    print("   Применение UMAP для 2D визуализации...")

    if len(all_embeddings) > 10:
        try:
            n_neighbors_value = min(25, len(all_embeddings) - 1)
            reducer = umap.UMAP(n_components=2, min_dist=0.3, random_state=42,
                                n_neighbors=n_neighbors_value)
            embeddings_umap = reducer.fit_transform(embeddings_pca)

            ax2 = plt.subplot(2, 2, 2)

            # Разделяем точки по классам
            if pos_indices.sum() > 0:
                ax2.scatter(embeddings_umap[pos_indices, 0],
                            embeddings_umap[pos_indices, 1],
                            c='green', alpha=0.6, s=30,
                            label=f'Positive ({pos_indices.sum()})',
                            edgecolors='black', linewidth=0.5)

            if neg_indices.sum() > 0:
                ax2.scatter(embeddings_umap[neg_indices, 0],
                            embeddings_umap[neg_indices, 1],
                            c='red', alpha=0.6, s=30,
                            label=f'Negative ({neg_indices.sum()})',
                            edgecolors='black', linewidth=0.5)

            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.set_title('UMAP 2D проекция')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        except Exception as e:
            print(f"     Ошибка UMAP: {e}")
            # Создаем пустой subplot если UMAP не сработал
            ax2 = plt.subplot(2, 2, 2)
            ax2.text(0.5, 0.5, 'UMAP failed', ha='center', va='center')
            ax2.set_title('UMAP (failed)')

    plt.suptitle('Embedding Visualizations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ График сохранен: {save_path}")

    return {
        'embeddings': all_embeddings,
        'labels': all_labels,
        'pca': embeddings_pca,
        'tsne': embeddings_tsne if 'embeddings_tsne' in locals() else None,
        'umap': embeddings_umap if 'embeddings_umap' in locals() else None
    }



def main():
    print("=" * 70)
    print("CONTRASTIVE LEARNING WITH VISUALIZATION")
    print("=" * 70)

    # Data load
    data_file = "AptaBench_dataset_v2_with_embeddings.csv"
    try:
        apt_pos, smi_pos, apt_neg, smi_neg, seq_dim, smi_dim = load_data(data_file)
    except Exception as e:
        print(f" Error loading data: {e}")
        return

    # Create Dataset
    dataset = FinalContrastiveDataset(
        apt_pos, smi_pos, apt_neg, smi_neg,
        negative_ratio=2
    )

    # train/val/test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Collate function
    def collate_fn(batch):
        return {
            'anchor_apt': torch.stack([b['anchor_apt'] for b in batch]),
            'positive_smi': torch.stack([b['positive_smi'] for b in batch]),
            'negative_smis': torch.stack([b['negative_smis'] for b in batch])
        }

    # DataLoaders with batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"\n  Dataset sizes:")
    print(f"    Train: {len(train_dataset)}")
    print(f"    Validation: {len(val_dataset)}")
    print(f"    Test: {len(test_dataset)}")
    print(f"    Batch size: 8 (very small!)")

    print("Creating mini model...")
    model = MicroContrastiveModel(
        apt_dim=seq_dim,
        mol_dim=smi_dim,
        latent_dim=32,
        projection_dim=16
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")

    trainer = FinalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # training
    print("\n Starting training...")
    history = trainer.train(n_epochs=12, save_path='final_micro_model.pth')

    plot_final_results(history)

    print("\n Analyzing on test set...")
    test_results = analyze_results(model, test_loader, device)

    print(f"\nFINAL TEST RESULTS:")
    print(f"    Accuracy: {test_results['accuracy']:.3f}")
    print(f"    Separation: {test_results['separation']:.3f}")
    print(f"    Positive mean similarity: {test_results['pos_mean']:.3f}")
    print(f"    Negative mean similarity: {test_results['neg_mean']:.3f}")

    plot_similarity_distributions(test_results)


#Load the best model

    try:
        checkpoint = torch.load('final_micro_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(" Загружена лучшая модель для визуализации")
    except:
        print(" Используется текущая модель")

    visualize_embeddings(
        model=model,
        apt_pos=apt_pos,
        smi_pos=smi_pos,
        apt_neg=apt_neg,
        smi_neg=smi_neg,
        device=device,
        save_path='embedding_visualization.png'
    )

    with torch.no_grad():
        if len(apt_pos) > 0:
            apt_pos_tensor = torch.FloatTensor(apt_pos[:1000]).to(device)
            smi_pos_tensor = torch.FloatTensor(smi_pos[:1000]).to(device)

            z_apt = model.encode_aptamer(apt_pos_tensor)
            z_smi = model.encode_molecule(smi_pos_tensor)

            pos_similarities = F.cosine_similarity(z_apt, z_smi, dim=1)
            print(f"     Positive среднее сходство: {pos_similarities.mean().item():.3f}")
            print(f"     Positive std сходство: {pos_similarities.std().item():.3f}")

        if len(apt_neg) > 0:
            apt_neg_tensor = torch.FloatTensor(apt_neg[:1000]).to(device)
            smi_neg_tensor = torch.FloatTensor(smi_neg[:1000]).to(device)

            z_apt = model.encode_aptamer(apt_neg_tensor)
            z_smi = model.encode_molecule(smi_neg_tensor)

            neg_similarities = F.cosine_similarity(z_apt, z_smi, dim=1)
            print(f"   • Negative среднее сходство: {neg_similarities.mean().item():.3f}")
            print(f"   • Negative std сходство: {neg_similarities.std().item():.3f}")

            if len(apt_pos) > 0:
                separation = pos_similarities.mean().item() - neg_similarities.mean().item()
                print(f"    separation: {separation:.3f}")

    #save embeddings
    save_embeddings(model, apt_pos, smi_pos, apt_neg, smi_neg, device)

    return model, history, test_results


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

    # Top-3 Accuracy
    axes[0, 2].plot(history['train_top3'], label='Train', linewidth=2, marker='o', markersize=4)
    axes[0, 2].plot(history['val_top3'], label='Val', linewidth=2, marker='s', markersize=4)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Top-3 Accuracy')
    axes[0, 2].set_title('Top-3 Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Temperature
    axes[1, 0].plot(history['temperature'], label='Temperature', linewidth=2, color='red', marker='o', markersize=4)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].set_title('Temperature during training')
    axes[1, 0].grid(True, alpha=0.3)

    # Train-Val Gap
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

    summary_text = (
        f"Final Results:\n\n"
        f"Best Validation: {best_val_acc:.3f}\n"
        f"Best Training: {best_train_acc:.3f}\n"
        f"Final Gap: {final_gap:.3f}\n\n"
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
    model, history, test_results = main()

