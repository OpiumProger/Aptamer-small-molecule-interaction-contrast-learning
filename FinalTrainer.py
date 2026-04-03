from Loss import TemperatureScaledLoss
import torch
from tqdm import tqdm


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
        print(f"\nStarting final training (max {n_epochs} epochs)...")

        for epoch in range(1, n_epochs + 1):
            # Train
            train_loss, train_acc, train_top3 = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_top3 = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Save current temperature
            current_temp = torch.exp(self.criterion.log_temperature).item()

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

                print(f"Saved best model (Val Acc: {val_acc:.3f})")
            else:
                self.patience_counter += 1
                print(f"No improvement: {self.patience_counter}/{self.patience}")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"   Best Val Accuracy: {self.best_val_acc:.3f}")
                break

        # Берём лучшую модель
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"\nTraining completed!")
        print(f"   Best validation accuracy: {self.best_val_acc:.3f}")
        print(f"   Final train-val gap: {self.history['train_val_gap'][-1]:.3f}")

        return self.history
