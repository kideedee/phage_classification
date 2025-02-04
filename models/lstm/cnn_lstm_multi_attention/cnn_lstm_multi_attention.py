import os
import time
from typing import Tuple, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import utils

# Configuration
CONFIG = {
    'num_workers': 8,
    'input_size': 4,
    'hidden_size': 128,
    'num_layers': 2,
    'num_heads': 4,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'bidirectional': False,
    'dropout': 0.3,
    'weight_decay': 1e-4,
    'lr_patience': 5,
    'patience': 20
}


class BacteriaDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.tensor(self.labels[idx].item(), dtype=torch.float32)
        return sequence, label


class EnhancedLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        # Multi-kernel CNN
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2)
            ) for k in [3, 5, 7]
        ])

        # LSTM layer - không dùng bidirectional
        self.lstm = nn.LSTM(
            hidden_size * len(self.conv_layers),
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention - điều chỉnh hidden size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        # Output layers
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-kernel convolution
        x = x.transpose(1, 2)  # (batch, 4, seq_len)
        conv_outputs = [conv(x) for conv in self.conv_layers]
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Multi-head attention
        lstm_out = lstm_out.transpose(0, 1)
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output.transpose(0, 1)

        # Get final output
        context = attn_output[:, -1, :]
        output = self.fc(context)

        return output.squeeze(-1), attn_weights


def load_fold_datasets(group: str, length: int, fold: int, log_file: str) -> Tuple[BacteriaDataset, BacteriaDataset]:
    utils.log_and_append_file(log_file, f"Loading dee_phage_data for group {group} fold {fold}")

    root_data_dir = os.path.join("../../../data", group)
    train_dir = os.path.join(root_data_dir, "train")
    test_dir = os.path.join(root_data_dir, "test")

    train_sequence_file = f'P_train_ds_{group}_{fold}.mat'
    train_label_file = f'T_train_ds_{group}_{fold}.mat'
    test_sequence_file = f'P_test_{group}_{fold}.mat'
    test_label_file = f'label_{group}_{fold}.mat'

    with h5py.File(os.path.join(train_dir, train_sequence_file), 'r') as f:
        train_matrix = f['P_train_ds'][:].transpose()
    with h5py.File(os.path.join(train_dir, train_label_file), 'r') as f:
        train_label = f['T_train_ds'][:].transpose()
    with h5py.File(os.path.join(test_dir, test_sequence_file), 'r') as f:
        test_matrix = f['P_test'][:].transpose()
    with h5py.File(os.path.join(test_dir, test_label_file), 'r') as f:
        test_label = f['T_test'][:].transpose()

    train_matrix = train_matrix.reshape(-1, length, 4)
    test_matrix = test_matrix.reshape(-1, length, 4)

    utils.log_and_append_file(log_file, f"Train dataset shape: {train_matrix.shape}")
    utils.log_and_append_file(log_file, f"Test dataset shape: {test_matrix.shape}")

    return BacteriaDataset(train_matrix, train_label), BacteriaDataset(test_matrix, test_label)


def plot_attention_weights(sequence: torch.Tensor, attention_weights: torch.Tensor,
                           save_path: str, index: int = 0) -> None:
    plt.figure(figsize=(15, 5))
    attention = attention_weights[index].cpu().detach().numpy()
    plt.plot(attention, label='Attention Weights')
    plt.title('Attention Weights Distribution')
    plt.xlabel('Sequence Position')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(history: dict, plot_file: str) -> None:
    plt.figure(figsize=(20, 6))  # Tăng chiều rộng

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'], label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.legend()

    plt.subplots_adjust(wspace=0.3)  # Tăng khoảng cách giữa các subplot
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()


def process_epoch(model: nn.Module, data_loader: DataLoader,
                  criterion: nn.Module, optimizer: optim.Optimizer,
                  device: torch.device, is_training: bool, scaler: GradScaler) -> Tuple[float, List, List]:
    total_loss = 0
    predictions = []
    true_labels = []

    desc = "Training" if is_training else "Validation"
    pbar = tqdm(data_loader, desc=desc)
    with torch.set_grad_enabled(is_training):
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Sử dụng AMP
            with autocast(device_type='cuda'):
                outputs, _ = model(sequences)
                loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            predictions.extend((outputs > 0.5).float().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

            # Clear cache định kỳ
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    return total_loss / len(data_loader), predictions, true_labels


def train_and_evaluate(model: nn.Module, train_loader: DataLoader,
                       val_loader: DataLoader, criterion: nn.Module,
                       optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.ReduceLROnPlateau,
                       device: torch.device, num_epochs: int, log_file: str,
                       plot_file: str, scaler: GradScaler) -> None:
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    patience = CONFIG['patience']  # Add patience from CONFIG
    no_improve = 0

    attention_dir = os.path.join(os.path.dirname(plot_file), 'attention_plots')
    os.makedirs(attention_dir, exist_ok=True)

    for epoch in range(num_epochs):
        start = time.time()

        # Training phase
        model.train()
        train_loss, train_preds, train_labels = process_epoch(
            model, train_loader, criterion, optimizer, device, True, scaler
        )
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

        # Validation phase
        model.eval()
        val_loss, val_preds, val_labels = process_epoch(
            model, val_loader, criterion, optimizer, device, False, scaler
        )
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # Log the current learning rate
        utils.log_and_append_file(log_file, f'Current learning rate: {current_lr}')

        # Save attention plots
        if epoch % 5 == 0:
            with torch.no_grad():
                sequences = next(iter(val_loader))[0].to(device)
                _, attention_weights = model(sequences)
                plot_attention_weights(
                    sequences,
                    attention_weights,
                    os.path.join(attention_dir, f'attention_epoch_{epoch}.png')
                )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, f'{log_file}_best_model.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                utils.log_and_append_file(log_file, "Early stopping triggered")
                break

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)

        # Log results
        utils.log_and_append_file(log_file, f'\nEpoch [{epoch + 1}/{num_epochs}]')
        utils.log_and_append_file(log_file,
                                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                                  f'LR: {current_lr:.6f}'
                                  )
        utils.log_and_append_file(log_file, '\nTraining Performance:')
        utils.log_and_append_file(log_file, str(classification_report(train_labels, train_preds)))
        utils.log_and_append_file(log_file, '\nValidation Performance:')
        utils.log_and_append_file(log_file, str(classification_report(val_labels, val_preds)))

        utils.log_and_append_file(log_file, f'Epoch time: {time.time() - start:.2f} seconds')

    plot_training_history(history, plot_file)


def main():
    max_length = [100, 400, 800, 1200, 1800]
    num_fold = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(1, 5):
        length = max_length[i]
        group = f"{max_length[i - 1]}_{length}"

        if max_length[i] == 400:
            CONFIG["batch_size"] = 256
            CONFIG["learning_rate"] = 0.001
        elif max_length[i] == 800:
            CONFIG["batch_size"] = 128
            CONFIG["learning_rate"] = 0.001
        elif max_length[i] == 1200:
            CONFIG["batch_size"] = 64
            CONFIG["learning_rate"] = 0.001
        elif max_length[i] == 1800:
            CONFIG["batch_size"] = 32
            CONFIG["learning_rate"] = 0.001
        else:
            raise ValueError("Invalid length")

        for fold in range(num_fold):
            # if fold < 4: continue
            # Setup directories
            experiment_name = "01"
            result_dir = f"../../../results/results/{experiment_name}/group_{group}/fold_{fold + 1}"
            os.makedirs(result_dir, exist_ok=True)

            log_file = os.path.join(result_dir, f"log_{group}.txt")
            plot_file = os.path.join(result_dir, f"training_plot_{group}.png")

            # Log configuration
            utils.log_and_append_file(log_file, f"Training LSTM model for group {group} fold {fold + 1}")
            utils.log_and_append_file(log_file, f"Start time: {time.ctime()}")
            utils.log_and_append_file(log_file, "Configuration:")
            for key, value in CONFIG.items():
                utils.log_and_append_file(log_file, f"{key}: {value}")

            # Load datasets
            train_dataset, test_dataset = load_fold_datasets(group, length, fold + 1, log_file)
            train_loader = DataLoader(
                train_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                num_workers=CONFIG['num_workers'],
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                num_workers=CONFIG['num_workers'],
                pin_memory=True,
                persistent_workers=True
            )

            # Initialize model and training components
            model = EnhancedLSTMClassifier(
                CONFIG['input_size'],
                CONFIG['hidden_size'],
                CONFIG['num_layers'],
                CONFIG['num_heads'],
                CONFIG['dropout']
            ).cuda()

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=CONFIG['learning_rate'],
                weight_decay=CONFIG['weight_decay']
            )

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=CONFIG['lr_patience'],
                min_lr=1e-6
            )

            # Initialize model and training components
            scaler = GradScaler()  # Add this line

            # Train and evaluate
            start_time = time.time()
            train_and_evaluate(
                model, train_loader, test_loader, criterion,
                optimizer, scheduler, device, CONFIG['num_epochs'],
                log_file, plot_file, scaler
            )

            # Cleanup
            torch.cuda.empty_cache()
            utils.log_and_append_file(log_file, f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
