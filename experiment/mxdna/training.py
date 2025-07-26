import json
import os
import warnings
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from experiment.mxdna.mxdna import MxDNAForPhageClassification, encode_dna_sequence

warnings.filterwarnings('ignore')


# Import your MxDNA model (assuming it's in the same directory)
# from mxdna_model import MxDNAForPhageClassification, encode_dna_sequence, create_attention_mask

class PhageDataset(Dataset):
    """Dataset class for phage sequences"""

    def __init__(self, sequences: List[str], labels: List[int], max_length: int = 4096):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Encode DNA sequence
        encoded = encode_dna_sequence(sequence)

        # Truncate or pad to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded = encoded + [0] * (self.max_length - len(encoded))

        # Create attention mask
        attention_mask = [1] * min(len(sequence), self.max_length) + [0] * max(0, self.max_length - len(sequence))

        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class PhageClassificationTrainer:
    """Trainer class for phage classification with MxDNA"""

    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader = None,
            learning_rate: float = 3e-5,
            weight_decay: float = 0.01,
            warmup_steps: int = 1000,
            max_steps: int = 10000,
            eval_steps: int = 500,
            save_steps: int = 1000,
            output_dir: str = "./phage_mxdna_results",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_steps = max_steps

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0) if step < warmup_steps
            else (max_steps - step) / (max_steps - warmup_steps)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.global_step = 0

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_tokenization_loss=True
            )

            # Calculate loss
            classification_loss = self.criterion(outputs['logits'], labels)
            balancing_loss = outputs['balancing_loss']
            total_loss_batch = classification_loss + balancing_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += total_loss_batch.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'cls_loss': f'{classification_loss.item():.4f}',
                'bal_loss': f'{balancing_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            # Evaluation
            if self.global_step % self.eval_steps == 0:
                val_loss, val_accuracy = self.evaluate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

                print(f"\nStep {self.global_step}: Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

                # Save best model
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.save_model("best_model.pt")
                    print(f"New best model saved with accuracy: {val_accuracy:.4f}")

                self.model.train()

            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # Stop if max steps reached
            if self.global_step >= self.max_steps:
                break

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_tokenization_loss=False
                )

                loss = self.criterion(outputs['logits'], labels)
                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return avg_loss, accuracy

    def test(self) -> Dict:
        """Test model on test set"""
        if self.test_dataloader is None:
            print("No test dataloader provided")
            return {}

        # Load best model
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for testing")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_tokenization_loss=False
                )

                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(outputs['logits'].cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, target_names=['Temperate', 'Virulent'])
        cm = confusion_matrix(all_labels, all_predictions)

        # Save results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'logits': all_logits
        }

        # Save to file
        with open(os.path.join(self.output_dir, "test_results.json"), 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }, f, indent=2)

        print(f"Test Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")

        return results

    def save_model(self, filename: str):
        """Save model state dict"""
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, filename))

    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, os.path.join(self.output_dir, filename))

    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        checkpoint_path = os.path.join(self.output_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.best_val_accuracy = checkpoint['best_val_accuracy']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
            print(f"Loaded checkpoint from step {self.global_step}")
        else:
            print(f"Checkpoint {filename} not found")

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot validation accuracy
        axes[1].plot(self.val_accuracies, label='Val Accuracy', color='green')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_xlabel('Evaluation Step')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Total training steps: {self.max_steps}")
        print(f"Evaluation every {self.eval_steps} steps")
        print(f"Save checkpoint every {self.save_steps} steps")

        epoch = 0
        while self.global_step < self.max_steps:
            print(f"\nEpoch {epoch + 1}")
            epoch_loss = self.train_epoch()
            self.train_losses.append(epoch_loss)

            print(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss:.4f}")

            if self.global_step >= self.max_steps:
                break

            epoch += 1

        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")

        # Final evaluation
        print("\nRunning final evaluation...")
        final_val_loss, final_val_acc = self.evaluate()
        print(f"Final validation accuracy: {final_val_acc:.4f}")

        # Test evaluation
        if self.test_dataloader is not None:
            print("\nRunning test evaluation...")
            test_results = self.test()

        # Plot training history
        self.plot_training_history()

        return self.best_val_accuracy


def create_data_loaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        batch_size: int = 8,
        max_length: int = 4096,
        random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    print(f"Train: {train_df.shape[0]}, Val: {val_df.shape[0]}")

    # Create datasets
    train_dataset = PhageDataset(train_seqs, train_labels, max_length)
    val_dataset = PhageDataset(val_seqs, val_labels, max_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, None


def main():
    """Main training script"""
    # Configuration
    config = {
        'data_file': 'phage_data.csv',  # Update this path
        'output_dir': './phage_mxdna_results',
        'max_length': 4096,
        'batch_size': 8,
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'max_steps': 10000,
        'eval_steps': 500,
        'save_steps': 1000,
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42,

        # Model configuration
        'vocab_size': 4,
        'hidden_dim': 512,
        'num_layers': 22,
        'num_attention_heads': 16,
        'intermediate_size': 2048,
        'max_position_embeddings': 4096,
        'num_experts': 10,
        'num_classes': 2,
        'dropout_prob': 0.1,
        'tokenization_layer_position': 5
    }

    # Save configuration
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load data
    print("Loading data...")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences, labels,
        test_size=config['test_size'],
        val_size=config['val_size'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        random_state=config['random_state']
    )

    # Create model
    print("Creating model...")
    model = MxDNAForPhageClassification(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        max_position_embeddings=config['max_position_embeddings'],
        num_experts=config['num_experts'],
        num_classes=config['num_classes'],
        dropout_prob=config['dropout_prob'],
        tokenization_layer_position=config['tokenization_layer_position']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = PhageClassificationTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        eval_steps=config['eval_steps'],
        save_steps=config['save_steps'],
        output_dir=config['output_dir']
    )

    # Train model
    print("Starting training...")
    best_accuracy = trainer.train()

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
