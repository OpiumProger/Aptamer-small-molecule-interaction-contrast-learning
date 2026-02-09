import numpy as np
import ot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')


class OptimalTransportAnalyzer:
    def __init__(self, embeddings_pos, embeddings_neg):

        self.embeddings_pos = embeddings_pos
        self.embeddings_neg = embeddings_neg
        self.N_pos = len(embeddings_pos)
        self.N_neg = len(embeddings_neg)

        print(f"Optimal Transport Analyzer initialized:")
        print(f"Positive samples: {self.N_pos}")
        print(f"Negative samples: {self.N_neg}")
        print(f"Embedding dimension: {embeddings_pos.shape[1]}")

    def compute_wasserstein_distance(self, reg=0.1, metric='euclidean'):

        # Матрица стоимости
        M = ot.dist(self.embeddings_pos, self.embeddings_neg, metric=metric)
        M = M ** 2  # Для W2 расстояния

        # Равномерные веса
        a = np.ones(self.N_pos) / self.N_pos
        b = np.ones(self.N_neg) / self.N_neg

        # Вычисляем оптимальный план транспортировки с регуляризацией
        gamma = ot.sinkhorn(a, b, M, reg=reg)

        # Расстояние Вассерштейна
        wasserstein_dist = np.sum(gamma * M)

        results = {
            'wasserstein_distance': np.sqrt(wasserstein_dist),
            'transport_plan': gamma,
            'cost_matrix': M,
            'reg': reg
        }

        return results

    def identify_outliers(self, gamma, cost_matrix=None, threshold_percentile=95):

        if cost_matrix is None:
            cost_matrix = ot.dist(self.embeddings_pos, self.embeddings_neg) ** 2

        # Для каждой позитивной точки: средняя стоимость транспортировки
        transport_costs_pos = []
        for i in range(self.N_pos):
            nonzero_idx = np.where(gamma[i, :] > 1e-10)[0]
            if len(nonzero_idx) > 0:
                avg_cost = np.sum(gamma[i, nonzero_idx] * cost_matrix[i, nonzero_idx])
                avg_cost /= np.sum(gamma[i, nonzero_idx])
                transport_costs_pos.append(avg_cost)
            else:
                transport_costs_pos.append(0)

        transport_costs_pos = np.array(transport_costs_pos)
        pos_threshold = np.percentile(transport_costs_pos, threshold_percentile)
        pos_outliers = np.where(transport_costs_pos > pos_threshold)[0]

        # Для негативных точек
        transport_costs_neg = []
        for j in range(self.N_neg):
            nonzero_idx = np.where(gamma[:, j] > 1e-10)[0]
            if len(nonzero_idx) > 0:
                avg_cost = np.sum(gamma[nonzero_idx, j] * cost_matrix[nonzero_idx, j])
                avg_cost /= np.sum(gamma[nonzero_idx, j])
                transport_costs_neg.append(avg_cost)
            else:
                transport_costs_neg.append(0)

        transport_costs_neg = np.array(transport_costs_neg)
        neg_threshold = np.percentile(transport_costs_neg, threshold_percentile)
        neg_outliers = np.where(transport_costs_neg > neg_threshold)[0]

        return {
            'pos_outlier_indices': pos_outliers,
            'neg_outlier_indices': neg_outliers,
            'pos_transport_costs': transport_costs_pos,
            'neg_transport_costs': transport_costs_neg,
            'pos_threshold': pos_threshold,
            'neg_threshold': neg_threshold
        }

    def find_ambiguous_points(self, gamma, cost_matrix=None,
                              distance_percentile=50, mass_threshold=0.3):

        if cost_matrix is None:
            cost_matrix = ot.dist(self.embeddings_pos, self.embeddings_neg) ** 2

        # Определяем пороговое расстояние
        if distance_percentile == 50:
            threshold_dist = np.median(cost_matrix[gamma > 1e-10])
        else:
            threshold_dist = np.percentile(cost_matrix[gamma > 1e-10], distance_percentile)

        ambiguous_indices = []
        ambiguity_scores = []

        for i in range(self.N_pos):
            # Находим транспортировки на короткие расстояния
            close_mask = cost_matrix[i, :] < threshold_dist
            if np.any(close_mask):
                close_mass = np.sum(gamma[i, close_mask])
                total_mass = np.sum(gamma[i, :])

                if total_mass > 0:
                    close_ratio = close_mass / total_mass
                    if close_ratio > mass_threshold:
                        ambiguous_indices.append(i)
                        ambiguity_scores.append(close_ratio)

        return {
            'ambiguous_indices': np.array(ambiguous_indices),
            'ambiguity_scores': np.array(ambiguity_scores),
            'threshold_distance': threshold_dist,
            'mass_threshold': mass_threshold
        }

    def create_transport_visualization(self, gamma, save_path='transport_plan.png'):

        """Визуализация плана транспортировки"""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Heatmap
        im = axes[0].imshow(np.log1p(gamma), cmap='viridis', aspect='auto')
        axes[0].set_xlabel('Negative samples')
        axes[0].set_ylabel('Positive samples')
        axes[0].set_title('Transport Plan (log scale)')
        plt.colorbar(im, ax=axes[0])

        # Гистограмма
        axes[1].hist(gamma[gamma > 1e-10].flatten(), bins=50,
                     alpha=0.7, color='purple', density=True)
        axes[1].set_xlabel('Transport Mass')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Transport Masses')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def create_clustering_visualization(self, save_path='ot_clustering.png'):

        # Объединяем все эмбеддинги
        all_embeddings = np.vstack([self.embeddings_pos, self.embeddings_neg])

        # Применяем t-SNE для визуализации
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Разделяем обратно
        pos_2d = embeddings_2d[:self.N_pos]
        neg_2d = embeddings_2d[self.N_pos:]

        # Вычисляем OT метрики
        ot_results = self.compute_wasserstein_distance()
        outlier_results = self.identify_outliers(ot_results['transport_plan'])
        ambiguous_results = self.find_ambiguous_points(ot_results['transport_plan'])

        # Создаем визуализацию
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 1. Исходная кластеризация
        axes[0].scatter(pos_2d[:, 0], pos_2d[:, 1],
                        c='green', alpha=0.6, s=30, label='Positive')
        axes[0].scatter(neg_2d[:, 0], neg_2d[:, 1],
                        c='red', alpha=0.6, s=30, label='Negative')
        axes[0].set_title(f'Original Clustering\nWasserstein Distance: {ot_results["wasserstein_distance"]:.3f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 2. С маркировкой OT анализа
        # Обычные точки
        normal_pos = np.ones(self.N_pos, dtype=bool)
        normal_pos[outlier_results['pos_outlier_indices']] = False
        normal_pos[ambiguous_results['ambiguous_indices']] = False

        axes[1].scatter(pos_2d[normal_pos, 0], pos_2d[normal_pos, 1],
                        c='lightgray', alpha=0.4, s=20, label='Normal')

        # Выбросы
        if len(outlier_results['pos_outlier_indices']) > 0:
            axes[1].scatter(pos_2d[outlier_results['pos_outlier_indices'], 0],
                            pos_2d[outlier_results['pos_outlier_indices'], 1],
                            c='orange', s=80, marker='*',
                            label=f'Outliers ({len(outlier_results["pos_outlier_indices"])})',
                            edgecolors='black', linewidth=1)

        # Неоднозначные точки
        if len(ambiguous_results['ambiguous_indices']) > 0:
            axes[1].scatter(pos_2d[ambiguous_results['ambiguous_indices'], 0],
                            pos_2d[ambiguous_results['ambiguous_indices'], 1],
                            c='blue', s=60, alpha=0.7,
                            label=f'Ambiguous ({len(ambiguous_results["ambiguous_indices"])})',
                            edgecolors='black', linewidth=0.5)

        # Негативные точки
        axes[1].scatter(neg_2d[:, 0], neg_2d[:, 1],
                        c='red', alpha=0.3, s=20, label='Negative')

        axes[1].set_title('OT Analysis Results\n(Outliers & Ambiguous Points)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return {
            'figure': fig,
            'ot_results': ot_results,
            'outlier_results': outlier_results,
            'ambiguous_results': ambiguous_results
        }

    def get_summary(self):
        """Возвращает текстовое резюме анализа"""

        ot_results = self.compute_wasserstein_distance()
        outlier_results = self.identify_outliers(ot_results['transport_plan'])
        ambiguous_results = self.find_ambiguous_points(ot_results['transport_plan'])

        summary = f"""
        OPTIMAL TRANSPORT ANALYSIS SUMMARY
        {'=' * 50}

        Basic Statistics:
          Positive samples: {self.N_pos}
          Negative samples: {self.N_neg}
          Embedding dimension: {self.embeddings_pos.shape[1]}

        Separation Metrics:
          Wasserstein Distance: {ot_results['wasserstein_distance']:.4f}

        Outlier Detection (95th percentile):
          Positive outliers: {len(outlier_results['pos_outlier_indices'])} ({len(outlier_results['pos_outlier_indices']) / self.N_pos * 100:.1f}%)
          Negative outliers: {len(outlier_results['neg_outlier_indices'])} ({len(outlier_results['neg_outlier_indices']) / self.N_neg * 100:.1f}%)

        Ambiguous Points (transport > 30% at median distance):
          Ambiguous positive points: {len(ambiguous_results['ambiguous_indices'])} ({len(ambiguous_results['ambiguous_indices']) / self.N_pos * 100:.1f}%)
        """

        return summary
def get_embeddings_from_model(model, apt_pos, smi_pos, apt_neg, smi_neg, device):
    """Извлекает эмбеддинги из модели"""

    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        # Позитивные пары
        apt_pos_tensor = torch.FloatTensor(apt_pos).to(device)
        smi_pos_tensor = torch.FloatTensor(smi_pos).to(device)

        z_apt_pos = model.encode_aptamer(apt_pos_tensor).cpu().numpy()
        z_smi_pos = model.encode_molecule(smi_pos_tensor).cpu().numpy()

        # Конкатенируем для полных пар
        embeddings_pos = np.concatenate([z_apt_pos, z_smi_pos], axis=1)

        # Негативные пары
        if len(apt_neg) > 0:
            apt_neg_tensor = torch.FloatTensor(apt_neg).to(device)
            smi_neg_tensor = torch.FloatTensor(smi_neg).to(device)

            z_apt_neg = model.encode_aptamer(apt_neg_tensor).cpu().numpy()
            z_smi_neg = model.encode_molecule(smi_neg_tensor).cpu().numpy()

            embeddings_neg = np.concatenate([z_apt_neg, z_smi_neg], axis=1)
        else:
            embeddings_neg = np.array([])

    return embeddings_pos, embeddings_neg


def analyze_model_with_ot(model, apt_pos, smi_pos, apt_neg, smi_neg, device):
    """Полный анализ модели с оптимальным транспортом"""

    print("\n" + "=" * 60)
    print("OPTIMAL TRANSPORT ANALYSIS")
    print("=" * 60)

    # 1. Получаем эмбеддинги
    print("\nExtracting embeddings from model...")
    embeddings_pos, embeddings_neg = get_embeddings_from_model(
        model, apt_pos, smi_pos, apt_neg, smi_neg, device
    )

    print(f"  • Positive embeddings: {embeddings_pos.shape}")
    print(f"  • Negative embeddings: {embeddings_neg.shape}")

    # 2. Создаем анализатор
    analyzer = OptimalTransportAnalyzer(embeddings_pos, embeddings_neg)

    # 3. Выполняем полный анализ
    print("\nRunning optimal transport analysis...")

    # Вычисляем расстояние Вассерштейна
    ot_results = analyzer.compute_wasserstein_distance()
    print(f"  Wasserstein Distance: {ot_results['wasserstein_distance']:.4f}")

    # Идентифицируем выбросы
    outlier_results = analyzer.identify_outliers(ot_results['transport_plan'])
    print(f"  • Positive outliers: {len(outlier_results['pos_outlier_indices'])}")
    print(f"  • Negative outliers: {len(outlier_results['neg_outlier_indices'])}")

    # Находим неоднозначные точки
    ambiguous_results = analyzer.find_ambiguous_points(ot_results['transport_plan'])
    print(f"  • Ambiguous points: {len(ambiguous_results['ambiguous_indices'])}")

    # 4. Создаем визуализации
    print("\nCreating visualizations...")

    # Визуализация плана транспортировки
    analyzer.create_transport_visualization(
        ot_results['transport_plan'],
        save_path='ot_transport_plan.png'
    )

    # Визуализация кластеризации с OT анализом
    viz_results = analyzer.create_clustering_visualization(
        save_path='ot_clustering_analysis.png'
    )

    # 5. Выводим резюме
    print("\n" + analyzer.get_summary())

    return {
        'analyzer': analyzer,
        'embeddings_pos': embeddings_pos,
        'embeddings_neg': embeddings_neg,
        'ot_results': ot_results,
        'outlier_results': outlier_results,
        'ambiguous_results': ambiguous_results,
        'visualization': viz_results
    }