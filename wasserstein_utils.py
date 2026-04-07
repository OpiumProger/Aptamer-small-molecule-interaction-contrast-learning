import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


class PairDistanceAnalyzer:

    def __init__(self, pos_distances, neg_distances, pos_similarities=None, neg_similarities=None):
        """
        Параметры:
            pos_distances: array [N_pos] - расстояния для позитивных пар (1 - cosine_similarity)
            neg_distances: array [N_neg] - расстояния для негативных пар
        """
        self.pos_distances = np.array(pos_distances).flatten()
        self.neg_distances = np.array(neg_distances).flatten()
        self.pos_similarities = np.array(pos_similarities).flatten() if pos_similarities is not None else None
        self.neg_similarities = np.array(neg_similarities).flatten() if neg_similarities is not None else None

        self.N_pos = len(self.pos_distances)
        self.N_neg = len(self.neg_distances)

        print(f"Pair Distance Analyzer initialized:")
        print(f"  Positive pairs: {self.N_pos}")
        print(f"  Negative pairs: {self.N_neg}")
        print(f"  Mean positive distance: {self.pos_distances.mean():.4f} ± {self.pos_distances.std():.4f}")
        print(f"  Mean negative distance: {self.neg_distances.mean():.4f} ± {self.neg_distances.std():.4f}")
        print(f"  Separation (neg_mean - pos_mean): {self.neg_distances.mean() - self.pos_distances.mean():.4f}")

    def compute_wasserstein_distance(self, reg=0.1, metric='euclidean'):
        """
        Вычисляет расстояние Вассерштейна между распределениями
        расстояний позитивных и негативных пар.
        """
        # Превращаем расстояния в "точки" в 1D пространстве
        pos_points = self.pos_distances.reshape(-1, 1)
        neg_points = self.neg_distances.reshape(-1, 1)

        # Матрица стоимости
        M = ot.dist(pos_points, neg_points, metric=metric)

        # Равномерные веса
        a = np.ones(self.N_pos) / self.N_pos
        b = np.ones(self.N_neg) / self.N_neg

        # Вычисляем оптимальный план транспортировки
        gamma = ot.sinkhorn(a, b, M, reg=reg)

        # Расстояние Вассерштейна
        wasserstein_dist = np.sum(gamma * M)

        results = {
            'wasserstein_distance': np.sqrt(wasserstein_dist),
            'transport_plan': gamma,
            'cost_matrix': M,
            'reg': reg,
            'pos_points': pos_points,
            'neg_points': neg_points
        }

        return results

    def compute_emd(self):
        """
        Вычисляет Earth Mover's Distance (без регуляризации)
        """
        pos_points = self.pos_distances.reshape(-1, 1)
        neg_points = self.neg_distances.reshape(-1, 1)

        M = ot.dist(pos_points, neg_points, metric='euclidean')

        a = np.ones(self.N_pos) / self.N_pos
        b = np.ones(self.N_neg) / self.N_neg

        emd_distance = ot.emd2(a, b, M)

        return emd_distance

    def identify_outliers(self, gamma=None, threshold_percentile=95):
        """
        Идентифицирует выбросы на основе стоимости транспортировки.
        """
        if gamma is None:
            ot_results = self.compute_wasserstein_distance()
            gamma = ot_results['transport_plan']

        # Для позитивных точек
        transport_costs_pos = []
        for i in range(self.N_pos):
            nonzero_idx = np.where(gamma[i, :] > 1e-10)[0]
            if len(nonzero_idx) > 0:
                avg_cost = np.sum(gamma[i, nonzero_idx] *
                                  (self.pos_distances[i] - self.neg_distances[nonzero_idx]) ** 2)
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
                avg_cost = np.sum(gamma[nonzero_idx, j] *
                                  (self.pos_distances[nonzero_idx] - self.neg_distances[j]) ** 2)
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

    def find_ambiguous_pairs(self, method='quantile', pos_quantile=75, neg_quantile=25):
        """
        Находит "неоднозначные" пары - те, чьи расстояния попадают в зону перекрытия
        распределений позитивных и негативных пар.

        Параметры:
            method: 'quantile'  или 'overlap'
            pos_quantile: верхний квантиль для позитивных расстояний (по умолчанию 75%)
            neg_quantile: нижний квантиль для негативных расстояний (по умолчанию 25%)
        """
        if method == 'quantile':
            # Исправленный метод: используем квантили для определения зоны неопределенности

            # Верхняя граница для позитивных (например, 75-й перцентиль)
            pos_upper = np.percentile(self.pos_distances, pos_quantile)

            # Нижняя граница для негативных (например, 25-й перцентиль)
            neg_lower = np.percentile(self.neg_distances, neg_quantile)

            # Зона неопределенности: где позитивные > pos_upper ИЛИ негативные < neg_lower
            ambiguous_threshold_low = min(pos_upper, neg_lower)
            ambiguous_threshold_high = max(pos_upper, neg_lower)

            # Находим позитивные пары в зоне неопределенности
            ambiguous_pos = np.where(
                (self.pos_distances >= ambiguous_threshold_low) &
                (self.pos_distances <= ambiguous_threshold_high)
            )[0]

            # Находим негативные пары в зоне неопределенности
            ambiguous_neg = np.where(
                (self.neg_distances >= ambiguous_threshold_low) &
                (self.neg_distances <= ambiguous_threshold_high)
            )[0]

        else:
            # Альтернативный метод: используем точку пересечения KDE
            try:
                # Создаем KDE для обоих распределений
                x_grid = np.linspace(0, 2, 200)

                # Ограничиваем данные для KDE (избегаем вырожденных случаев)
                pos_data = self.pos_distances[self.pos_distances > 0]
                neg_data = self.neg_distances[self.neg_distances > 0]

                if len(pos_data) > 10 and len(neg_data) > 10:
                    kde_pos = gaussian_kde(pos_data)
                    kde_neg = gaussian_kde(neg_data)

                    kde_pos_vals = kde_pos(x_grid)
                    kde_neg_vals = kde_neg(x_grid)

                    # Находим точку пересечения
                    diff = kde_pos_vals - kde_neg_vals
                    sign_changes = np.where(np.diff(np.sign(diff)))[0]

                    if len(sign_changes) > 0:
                        intersection_idx = sign_changes[0]
                        intersection_point = x_grid[intersection_idx]
                    else:
                        # Если нет пересечения, берем среднюю точку
                        intersection_point = (self.pos_distances.mean() + self.neg_distances.mean()) / 2
                else:
                    intersection_point = (self.pos_distances.mean() + self.neg_distances.mean()) / 2

                # Зона неопределенности вокруг точки пересечения (±0.2)
                ambiguous_threshold_low = max(0, intersection_point - 0.2)
                ambiguous_threshold_high = min(2, intersection_point + 0.2)

                ambiguous_pos = np.where(
                    (self.pos_distances >= ambiguous_threshold_low) &
                    (self.pos_distances <= ambiguous_threshold_high)
                )[0]

                ambiguous_neg = np.where(
                    (self.neg_distances >= ambiguous_threshold_low) &
                    (self.neg_distances <= ambiguous_threshold_high)
                )[0]

            except Exception as e:
                print(f"  Warning: KDE method failed ({e}), falling back to quantile method")
                # Fallback to quantile method
                pos_upper = np.percentile(self.pos_distances, 75)
                neg_lower = np.percentile(self.neg_distances, 25)
                ambiguous_threshold_low = min(pos_upper, neg_lower)
                ambiguous_threshold_high = max(pos_upper, neg_lower)

                ambiguous_pos = np.where(
                    (self.pos_distances >= ambiguous_threshold_low) &
                    (self.pos_distances <= ambiguous_threshold_high)
                )[0]

                ambiguous_neg = np.where(
                    (self.neg_distances >= ambiguous_threshold_low) &
                    (self.neg_distances <= ambiguous_threshold_high)
                )[0]

        return {
            'ambiguous_pos_indices': ambiguous_pos,
            'ambiguous_neg_indices': ambiguous_neg,
            'threshold_low': ambiguous_threshold_low,
            'threshold_high': ambiguous_threshold_high,
            'method': method
        }

    def compute_accuracy_at_threshold(self, threshold=None):
        """
        Вычисляет accuracy при заданном пороге расстояния.
        """
        if threshold is None:
            # Оптимальный порог - где точность максимальна
            thresholds = np.linspace(0, 2, 100)
            best_acc = 0
            best_threshold = 0.5

            for t in thresholds:
                pos_correct = np.sum(self.pos_distances <= t)
                neg_correct = np.sum(self.neg_distances > t)
                acc = (pos_correct + neg_correct) / (self.N_pos + self.N_neg)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = t

            return best_acc, best_threshold
        else:
            pos_correct = np.sum(self.pos_distances <= threshold)
            neg_correct = np.sum(self.neg_distances > threshold)
            acc = (pos_correct + neg_correct) / (self.N_pos + self.N_neg)
            return acc, threshold

    def create_distance_distribution_plot(self, save_path='distance_distributions.png'):
        """
        Визуализация распределений расстояний для позитивных и негативных пар.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Гистограмма распределений
        bins = np.linspace(0, 2, 50)

        axes[0].hist(self.pos_distances, bins=bins, alpha=0.6,
                     color='green', label=f'Positive (μ={self.pos_distances.mean():.3f})',
                     density=True, edgecolor='darkgreen')
        axes[0].hist(self.neg_distances, bins=bins, alpha=0.6,
                     color='red', label=f'Negative (μ={self.neg_distances.mean():.3f})',
                     density=True, edgecolor='darkred')
        axes[0].set_xlabel('Distance (1 - Cosine Similarity)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distance Distributions')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Добавляем вертикальную линию оптимального порога
        best_acc, best_threshold = self.compute_accuracy_at_threshold()
        axes[0].axvline(x=best_threshold, color='blue', linestyle='--',
                        label=f'Optimal threshold (acc={best_acc:.3f})')
        axes[0].legend()

        # 2. Box plot
        data_to_plot = [self.pos_distances, self.neg_distances]
        bp = axes[1].boxplot(data_to_plot, labels=['Positive', 'Negative'],
                             patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1].set_ylabel('Distance')
        axes[1].set_title('Distance Statistics')
        axes[1].grid(alpha=0.3)

        # Добавляем средние значения
        axes[1].scatter([1, 2], [self.pos_distances.mean(), self.neg_distances.mean()],
                        color='blue', s=100, zorder=5, label='Mean', marker='D')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def create_transport_visualization(self, gamma, save_path='transport_plan.png'):
        """
        Визуализация плана транспортировки.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Heatmap (ограничиваем размер для производительности)
        gamma_display = gamma[:min(500, self.N_pos), :min(500, self.N_neg)]
        im = axes[0].imshow(np.log1p(gamma_display), cmap='viridis', aspect='auto')
        axes[0].set_xlabel('Negative pairs')
        axes[0].set_ylabel('Positive pairs')
        axes[0].set_title(f'Transport Plan (log scale, first {gamma_display.shape[0]}x{gamma_display.shape[1]})')
        plt.colorbar(im, ax=axes[0])

        # Гистограмма транспортных масс
        masses = gamma[gamma > 1e-10].flatten()
        if len(masses) > 0:
            axes[1].hist(masses, bins=50, alpha=0.7, color='purple', density=True)
        axes[1].set_xlabel('Transport Mass')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Transport Masses')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def get_summary(self):
        """
        Возвращает текстовое резюме анализа.
        """
        ot_results = self.compute_wasserstein_distance()
        outlier_results = self.identify_outliers()
        ambiguous_results = self.find_ambiguous_pairs(method='quantile')
        best_acc, best_threshold = self.compute_accuracy_at_threshold()

        separation = self.neg_distances.mean() - self.pos_distances.mean()

        summary = f"""
        ==================================================
        PAIR DISTANCE ANALYSIS SUMMARY
        ==================================================

        BASIC STATISTICS:
          Positive pairs: {self.N_pos}
          Negative pairs: {self.N_neg}

          Positive distances: mean={self.pos_distances.mean():.4f}, std={self.pos_distances.std():.4f}
          Negative distances: mean={self.neg_distances.mean():.4f}, std={self.neg_distances.std():.4f}

          Class separation: {separation:.4f}

        CLASSIFICATION METRICS:
          Optimal threshold: {best_threshold:.4f}
          Theoretical accuracy: {best_acc:.4f} ({best_acc * 100:.1f}%)

        WASSERSTEIN DISTANCE:
          W2 distance: {ot_results['wasserstein_distance']:.4f}
          EMD distance: {self.compute_emd():.4f}

        OUTLIER DETECTION (95th percentile):
          Positive outliers: {len(outlier_results['pos_outlier_indices'])} ({len(outlier_results['pos_outlier_indices']) / self.N_pos * 100:.1f}%)
          Negative outliers: {len(outlier_results['neg_outlier_indices'])} ({len(outlier_results['neg_outlier_indices']) / self.N_neg * 100:.1f}%)

        AMBIGUOUS PAIRS (quantile method - 75th/25th):
          Ambiguous positive pairs: {len(ambiguous_results['ambiguous_pos_indices'])} ({len(ambiguous_results['ambiguous_pos_indices']) / self.N_pos * 100:.1f}%)
          Ambiguous negative pairs: {len(ambiguous_results['ambiguous_neg_indices'])} ({len(ambiguous_results['ambiguous_neg_indices']) / self.N_neg * 100:.1f}%)
          Overlap region: [{ambiguous_results['threshold_low']:.4f}, {ambiguous_results['threshold_high']:.4f}]

        ==================================================
        """

        return summary


def get_pair_distances_from_model(model, apt_pos, smi_pos, apt_neg, smi_neg, device):
    """
    Извлекает расстояния между аптамерами и молекулами для позитивных и негативных пар.
    """
    model.eval()

    with torch.no_grad():
        # Позитивные пары
        apt_pos_tensor = torch.FloatTensor(apt_pos).to(device)
        smi_pos_tensor = torch.FloatTensor(smi_pos).to(device)

        z_apt_pos = model.encode_aptamer(apt_pos_tensor)
        z_smi_pos = model.encode_molecule(smi_pos_tensor)

        # Косинусные сходства и расстояния для позитивных пар
        pos_similarities = F.cosine_similarity(z_apt_pos, z_smi_pos, dim=1)
        pos_distances = 1 - pos_similarities

        # Негативные пары
        if len(apt_neg) > 0 and len(smi_neg) > 0:
            apt_neg_tensor = torch.FloatTensor(apt_neg).to(device)
            smi_neg_tensor = torch.FloatTensor(smi_neg).to(device)

            z_apt_neg = model.encode_aptamer(apt_neg_tensor)
            z_smi_neg = model.encode_molecule(smi_neg_tensor)

            neg_similarities = F.cosine_similarity(z_apt_neg, z_smi_neg, dim=1)
            neg_distances = 1 - neg_similarities
        else:
            neg_distances = np.array([])
            neg_similarities = np.array([])

    # Конвертируем в numpy
    pos_distances = pos_distances.cpu().numpy()
    pos_similarities = pos_similarities.cpu().numpy()
    neg_distances = neg_distances.cpu().numpy() if len(neg_distances) > 0 else np.array([])
    neg_similarities = neg_similarities.cpu().numpy() if len(neg_similarities) > 0 else np.array([])

    return pos_distances, neg_distances, pos_similarities, neg_similarities


def analyze_model_pair_distances(model, apt_pos, smi_pos, apt_neg, smi_neg, device):
    """
    Полный анализ модели с использованием расстояний между парами.
    """
    print("\n" + "=" * 60)
    print("PAIR DISTANCE ANALYSIS (Optimal Transport on Pair Distances)")
    print("=" * 60)

    # 1. Получаем расстояния между парами
    print("\nExtracting pair distances from model...")
    pos_distances, neg_distances, pos_similarities, neg_similarities = get_pair_distances_from_model(
        model, apt_pos, smi_pos, apt_neg, smi_neg, device
    )

    print(f"  • Positive pairs: {len(pos_distances)}")
    print(f"    - Mean distance: {pos_distances.mean():.4f}")
    print(f"    - Mean similarity: {pos_similarities.mean():.4f}")
    print(f"  • Negative pairs: {len(neg_distances)}")
    print(f"    - Mean distance: {neg_distances.mean():.4f}")
    print(f"    - Mean similarity: {neg_similarities.mean():.4f}")
    print(f"  • Separation: {neg_distances.mean() - pos_distances.mean():.4f}")

    # 2. Создаем анализатор
    analyzer = PairDistanceAnalyzer(pos_distances, neg_distances, pos_similarities, neg_similarities)

    # 3. Вычисляем Wasserstein расстояние
    print("\nComputing Wasserstein distance...")
    ot_results = analyzer.compute_wasserstein_distance()
    print(f"  Wasserstein Distance (W2): {ot_results['wasserstein_distance']:.4f}")
    print(f"  Earth Mover's Distance (EMD): {analyzer.compute_emd():.4f}")

    # 4. Идентифицируем выбросы
    print("\nIdentifying outliers...")
    outlier_results = analyzer.identify_outliers()
    print(
        f"  Positive outliers: {len(outlier_results['pos_outlier_indices'])} ({len(outlier_results['pos_outlier_indices']) / len(pos_distances) * 100:.1f}%)")
    print(
        f"  Negative outliers: {len(outlier_results['neg_outlier_indices'])} ({len(outlier_results['neg_outlier_indices']) / len(neg_distances) * 100:.1f}%)")

    # 5. Находим неоднозначные пары (исправленный метод)
    print("\nFinding ambiguous pairs...")
    ambiguous_results = analyzer.find_ambiguous_pairs(method='quantile', pos_quantile=75, neg_quantile=25)
    print(
        f"  Ambiguous positive pairs: {len(ambiguous_results['ambiguous_pos_indices'])} ({len(ambiguous_results['ambiguous_pos_indices']) / len(pos_distances) * 100:.1f}%)")
    print(
        f"  Ambiguous negative pairs: {len(ambiguous_results['ambiguous_neg_indices'])} ({len(ambiguous_results['ambiguous_neg_indices']) / len(neg_distances) * 100:.1f}%)")
    print(f"  Overlap region: [{ambiguous_results['threshold_low']:.4f}, {ambiguous_results['threshold_high']:.4f}]")

    # 6. Теоретическая accuracy
    best_acc, best_threshold = analyzer.compute_accuracy_at_threshold()
    print(f"\nTheoretical best accuracy: {best_acc:.4f} ({best_acc * 100:.1f}%) at threshold {best_threshold:.4f}")

    # 7. Создаем визуализации
    print("\nCreating visualizations...")

    # Распределения расстояний
    analyzer.create_distance_distribution_plot(save_path='pair_distance_distributions.png')

    # План транспортировки
    analyzer.create_transport_visualization(ot_results['transport_plan'], save_path='pair_transport_plan.png')

    # 8. Выводим резюме
    print(analyzer.get_summary())

    return {
        'analyzer': analyzer,
        'pos_distances': pos_distances,
        'neg_distances': neg_distances,
        'pos_similarities': pos_similarities,
        'neg_similarities': neg_similarities,
        'ot_results': ot_results,
        'outlier_results': outlier_results,
        'ambiguous_results': ambiguous_results,
        'best_accuracy': best_acc,
        'best_threshold': best_threshold
    }


def create_pair_distance_heatmap(pos_distances, neg_distances, save_path='pair_distance_heatmap.png'):
    """
    Создает heatmap сравнения расстояний позитивных и негативных пар.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Violin plot
    data = [pos_distances, neg_distances]
    parts = axes[0].violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)

    colors = ['lightgreen', 'lightcoral']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(['Positive', 'Negative'])
    axes[0].set_ylabel('Distance (1 - Cosine Similarity)')
    axes[0].set_title('Pair Distance Distribution')
    axes[0].grid(alpha=0.3)

    # 2. ROC-like curve (cumulative distributions)
    sorted_pos = np.sort(pos_distances)
    sorted_neg = np.sort(neg_distances)

    x = np.linspace(0, 2, 100)
    cdf_pos = np.array([np.mean(pos_distances <= xi) for xi in x])
    cdf_neg = np.array([np.mean(neg_distances <= xi) for xi in x])

    axes[1].plot(x, cdf_pos, 'g-', linewidth=2, label='Positive')
    axes[1].plot(x, cdf_neg, 'r-', linewidth=2, label='Negative')
    axes[1].fill_between(x, cdf_pos, cdf_neg, where=(cdf_neg > cdf_pos),
                         color='red', alpha=0.3, label='Overlap')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution Functions')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig
