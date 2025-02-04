import os
import time
from typing import Tuple, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.python.data import Dataset
from torch import nn
from torch.amp import autocast
from torch.cpu.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import utils
from common.env_config import config
from common.utils import log_and_append_file


class ProteinEmbeddingDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, is_training: bool = True):
        # Robust standardization using median and IQR
        median = np.median(sequences)
        q1 = np.percentile(sequences, 25)
        q3 = np.percentile(sequences, 75)
        iqr = q3 - q1

        # Clip extreme values and normalize
        sequences_clipped = np.clip(sequences, median - 3 * iqr, median + 3 * iqr)
        self.sequences = (sequences_clipped - median) / (iqr + 1e-8)
        self.sequences = self.sequences.reshape(-1, 1024, 1)
        self.labels = labels
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.tensor(self.labels[idx].item(), dtype=torch.float32)

        if self.is_training:
            # Add small random noise for regularization
            noise = torch.randn_like(sequence) * 0.01
            sequence = sequence + noise

        return sequence, label


class ProteinClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Input normalization layer
        self.input_bn = nn.BatchNorm1d(1)

        # First conv block - capture local patterns
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)

        # Second conv block - increase receptive field
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)

        # Third conv block - higher-level features
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        # Pooling layers
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers with batch norm
        self.fc1 = nn.Linear(256, 128)  # 256 from concatenated pooling
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

        # Regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input shape: (batch_size, 1024, 1)
        x = x.transpose(1, 2)
        x = self.input_bn(x)

        # First conv block with residual
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)  # GELU instead of ReLU
        x = x + F.interpolate(identity, size=x.shape[2]) if x.shape[1] == identity.shape[1] else x
        x = self.dropout(x)

        # Second conv block
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = x + F.interpolate(identity, size=x.shape[2]) if x.shape[1] == identity.shape[1] else x
        x = self.dropout(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Global pooling
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        x = torch.cat([max_pooled, avg_pooled], dim=1).squeeze(-1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


def analyze_data(train_dataset, test_dataset):
    # Kiểm tra phân phối của embedding values
    train_mean = np.mean(train_dataset.sequences)
    train_std = np.std(train_dataset.sequences)
    print(f"Train data - Mean: {train_mean:.4f}, Std: {train_std:.4f}")

    # Kiểm tra labels
    train_pos = (train_dataset.labels == 1).sum()
    train_neg = (train_dataset.labels == 0).sum()
    print(f"Train labels - Positive: {train_pos}, Negative: {train_neg}")

    # Kiểm tra ranges
    print(f"Train data range: [{np.min(train_dataset.sequences):.4f}, {np.max(train_dataset.sequences):.4f}]")

    # Kiểm tra missing values
    print(f"NaN values in train: {np.isnan(train_dataset.sequences).sum()}")

    # Visualize embeddings distribution
    plt.figure(figsize=(10, 5))
    plt.hist(train_dataset.sequences.flatten(), bins=50)
    plt.title('Distribution of embedding values')
    plt.show()


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
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            # Sử dụng AMP
            with autocast(device_type='cuda'):
                outputs = model(sequences)
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
    patience = config.patience
    no_improve = 0

    # attention_dir = os.path.join(os.path.dirname(plot_file), 'attention_plots')
    # os.makedirs(attention_dir, exist_ok=True)

    for epoch in range(num_epochs):
        start = time.time()

        # Training phase
        model.train()
        train_loss, train_preds, train_labels = process_epoch(
            model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device,
            is_training=True, scaler=scaler
        )
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

        # Validation phase
        model.eval()
        val_loss, val_preds, val_labels = process_epoch(
            model=model, data_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device,
            is_training=False, scaler=scaler
        )
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # Log the current learning rate
        utils.log_and_append_file(log_file, f'Current learning rate: {current_lr}')

        # Early stopping check
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'val_loss': val_loss,
        #     }, f'{log_file}_best_model.pt')
        #     no_improve = 0
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         utils.log_and_append_file(log_file, "Early stopping triggered")
        #         break

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


def load_fold_datasets(group: str, length: int, fold: int, log_file: str) -> Tuple[
    ProteinEmbeddingDataset, ProteinEmbeddingDataset]:
    utils.log_and_append_file(log_file, f"Loading dee_phage_data for group {group} fold {fold}")

    root_data_dir = config.root_data_dir
    train_dir = os.path.join(root_data_dir, "train", group)
    test_dir = os.path.join(root_data_dir, "test", group)

    train_file = f'train_{group}_{fold}.h5'
    print(train_file)
    test_file = f'{group}_{fold}_test.h5'
    print(test_file)

    with h5py.File(os.path.join(train_dir, train_file), 'r') as f:
        train_matrix = f['embeddings'][:].transpose()
        train_label = f['labels'][:].transpose()
    with h5py.File(os.path.join(test_dir, test_file), 'r') as f:
        test_matrix = f['embeddings'][:].transpose()
        test_label = f['labels'][:].transpose()

    train_matrix = train_matrix.reshape(-1, 1024, 1)
    test_matrix = test_matrix.reshape(-1, 1024, 1)

    # analyze_data(ProtBertEmbeddingDataset(train_matrix, train_label), ProtBertEmbeddingDataset(test_matrix, test_label))

    log_and_append_file(log_file, f"Train dataset shape: {train_matrix.shape}")
    log_and_append_file(log_file, f"Test dataset shape: {test_matrix.shape}")

    return ProteinEmbeddingDataset(train_matrix, train_label), ProteinEmbeddingDataset(test_matrix, test_label)


def main():
    max_length = [100, 400, 800, 1200, 1800]
    num_fold = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(1, 5):
        length = max_length[i]
        group = f"{max_length[i - 1]}_{length}"

        if max_length[i] == 400:
            config.batch_size = 256
            config.learning_rate = 0.001
        elif max_length[i] == 800:
            config.batch_size = 128
            config.learning_rate = 0.001
        elif max_length[i] == 1200:
            config.batch_size = 64
            config.learning_rate = 0.001
        elif max_length[i] == 1800:
            config.batch_size = 32
            config.learning_rate = 0.001
        else:
            raise ValueError("Invalid length")

        for fold in range(num_fold):
            experiment_name = "01"
            result_dir = f"../../results/results/{experiment_name}/group_{group}/fold_{fold + 1}"
            os.makedirs(result_dir, exist_ok=True)

            log_file = os.path.join(result_dir, f"log_{group}.txt")
            plot_file = os.path.join(result_dir, f"training_plot_{group}.png")

            # Log configuration
            utils.log_and_append_file(log_file, f"Training LSTM model for group {group} fold {fold + 1}")
            utils.log_and_append_file(log_file, f"Start time: {time.ctime()}")
            utils.log_and_append_file(log_file, "Configuration:")
            for key, value in config.__dict__.items():
                utils.log_and_append_file(log_file, f"{key}: {value}")

            # Load datasets
            train_dataset, test_dataset = load_fold_datasets(group, length, fold + 1, log_file)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=True
            )

            model = ProteinClassifier().to(device)
            # Thay BCEWithLogitsLoss
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.0005,  # Smaller learning rate
                weight_decay=0.01
            )
            scaler = torch.cuda.amp.GradScaler()
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,  # Giảm 50% learning rate khi plateau
                patience=5,
                min_lr=1e-6
            )

            # Train and evaluate
            start_time = time.time()
            train_and_evaluate(
                model=model, train_loader=train_loader, val_loader=test_loader, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=config.num_epochs,
                log_file=log_file, plot_file=plot_file, scaler=scaler
            )

            # Cleanup
            torch.cuda.empty_cache()
            utils.log_and_append_file(log_file, f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
