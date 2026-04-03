import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings

from wasserstein_utils import analyze_model_with_ot
warnings.filterwarnings('ignore')

from Loss import TemperatureScaledLoss
from FinalTrainer import FinalTrainer
from DataPrepare import FinalContrastiveDataset
from Model import MicroContrastiveModel
from load_data_and_visual_data import load_data, analyze_results, analyze_results_correct, visualize_embeddings_correct, visualize_embeddings_2d_simple





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

    np.random.seed(42)

    # Разделяем ПОЗИТИВНЫЕ пары
    n_pos = len(apt_pos)
    pos_indices = np.random.permutation(n_pos)

    train_pos_size = int(0.7 * n_pos)
    val_pos_size = int(0.15 * n_pos)

    train_pos_idx = pos_indices[:train_pos_size]
    val_pos_idx = pos_indices[train_pos_size:train_pos_size + val_pos_size]
    test_pos_idx = pos_indices[train_pos_size + val_pos_size:]

    # Разделяем НЕГАТИВНЫЕ пары (отдельно!)
    n_neg = len(apt_neg)
    neg_indices = np.random.permutation(n_neg)

    train_neg_size = int(0.7 * n_neg)
    val_neg_size = int(0.15 * n_neg)

    train_neg_idx = neg_indices[:train_neg_size]
    val_neg_idx = neg_indices[train_neg_size:train_neg_size + val_neg_size]
    test_neg_idx = neg_indices[train_neg_size + val_neg_size:]

    print(f"\nComplete pair-based split (positives AND negatives separated):")
    print(f"  Total positive pairs: {n_pos}")
    print(f"    Train pos: {len(train_pos_idx)}")
    print(f"    Val pos: {len(val_pos_idx)}")
    print(f"    Test pos: {len(test_pos_idx)}")
    print(f"  Total negative pairs: {n_neg}")
    print(f"    Train neg: {len(train_neg_idx)}")
    print(f"    Val neg: {len(val_neg_idx)}")
    print(f"    Test neg: {len(test_neg_idx)}")

    train_dataset = FinalContrastiveDataset(
        apt_pos[train_pos_idx],
        smi_pos[train_pos_idx],
        apt_neg[train_neg_idx],
        smi_neg[train_neg_idx]
    )

    val_dataset = FinalContrastiveDataset(
        apt_pos[val_pos_idx],
        smi_pos[val_pos_idx],
        apt_neg[val_neg_idx],
        smi_neg[val_neg_idx]
    )

    test_dataset = FinalContrastiveDataset(
        apt_pos[test_pos_idx],
        smi_pos[test_pos_idx],
        apt_neg[test_neg_idx],
        smi_neg[test_neg_idx]
    )

    # DataLoaders
    def collate_fn(batch):
        return {
            'anchor_apt': torch.stack([b['anchor_apt'] for b in batch]),
            'positive_smi': torch.stack([b['positive_smi'] for b in batch]),
            'negative_smis': torch.stack([b['negative_smis'] for b in batch])
        }

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

    print(f"\n📊 Final dataset sizes:")
    print(f"    Train: {len(train_dataset)} positive pairs, {len(train_neg_idx)} negative pairs")
    print(f"    Validation: {len(val_dataset)} positive pairs, {len(val_neg_idx)} negative pairs")
    print(f"    Test: {len(test_dataset)} positive pairs, {len(test_neg_idx)} negative pairs")

    print("Creating mini model...")
    model = MicroContrastiveModel(
        input_dim=seq_dim,
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

    visualize_embeddings_correct(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path='embedding_visualization_correct.png'
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
    ot_analysis = analyze_model_with_ot(
        model=model,
        apt_pos=apt_pos,
        smi_pos=smi_pos,
        apt_neg=apt_neg,
        smi_neg=smi_neg,
        device=device
    )

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




