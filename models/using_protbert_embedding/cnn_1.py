import os
import time
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch import nn
from torch.amp import autocast
from torch.cpu.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import utils
from common.env_config import config


class ProtBertEmbeddingDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        # Normalize data
        self.sequences = (sequences - np.mean(sequences)) / np.std(sequences)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.tensor(self.labels[idx].item(), dtype=torch.float32)
        return sequence, label


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.squeeze(-1)  # Remove last dimension
        x = self.fc(x)
        return x


def load_fold_datasets(group: str, length: int, fold: int, log_file: str) -> Tuple[
    ProtBertEmbeddingDataset, ProtBertEmbeddingDataset]:
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

    utils.log_and_append_file(log_file, f"Train dataset shape: {train_matrix.shape}")
    utils.log_and_append_file(log_file, f"Test dataset shape: {test_matrix.shape}")

    return ProtBertEmbeddingDataset(train_matrix, train_label), ProtBertEmbeddingDataset(test_matrix, test_label)


def process_epoch(model, data_loader, criterion, optimizer, device, is_training, scaler):
    total_loss = 0
    predictions = []
    true_labels = []

    # Log gradients và weights
    if is_training:
        grad_norms = []
        weight_norms = []

    for sequences, labels in tqdm(data_loader):
        sequences = sequences.to(device)
        labels = labels.to(device).view(-1, 1)

        with autocast(device_type='cuda'):
            outputs = model(sequences)
            loss = criterion(outputs, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Log gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            # Log weight norms
            total_weight_norm = 0
            for p in model.parameters():
                param_norm = p.data.norm(2)
                total_weight_norm += param_norm.item() ** 2
            total_weight_norm = total_weight_norm ** 0.5
            weight_norms.append(total_weight_norm)

            scaler.step(optimizer)
            scaler.update()

        predictions.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

    if is_training:
        print(f"Average gradient norm: {np.mean(grad_norms):.4f}")
        print(f"Average weight norm: {np.mean(weight_norms):.4f}")

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

            # Test với model đơn giản
            model = SimpleModel()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train vài epochs và xem kết quả

            # Train and evaluate
            start_time = time.time()
            train_and_evaluate(
                model=model, train_loader=train_loader, val_loader=test_loader, criterion=criterion,
                optimizer=optimizer, scheduler=None, device=device, num_epochs=config.num_epochs,
                log_file=log_file, plot_file=plot_file, scaler=None
            )

            # Cleanup
            torch.cuda.empty_cache()
            utils.log_and_append_file(log_file, f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
