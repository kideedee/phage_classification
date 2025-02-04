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


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.flatten = nn.Flatten()
        # Single linear layer with sigmoid activation (built into loss function)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.squeeze(-1)  # Remove last dimension (1024, 1) -> (1024)
        x = self.flatten(x)  # Ensure input is flattened
        x = self.linear(x)
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


def process_epoch(model, data_loader, criterion, optimizer, device, is_training):
    total_loss = 0
    predictions = []
    true_labels = []

    for sequences, labels in tqdm(data_loader):
        sequences = sequences.to(device)
        labels = labels.to(device).view(-1, 1)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        predictions.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()

    return total_loss / len(data_loader), predictions, true_labels


def plot_training_history(history: dict, plot_file: str) -> None:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()


def train_and_evaluate(model: nn.Module, train_loader: DataLoader,
                       val_loader: DataLoader, criterion: nn.Module,
                       optimizer: optim.Optimizer, device: torch.device,
                       num_epochs: int, log_file: str, plot_file: str) -> None:
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    patience = config.patience
    no_improve = 0

    for epoch in range(num_epochs):
        start = time.time()

        # Training phase
        model.train()
        train_loss, train_preds, train_labels = process_epoch(
            model=model, data_loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, is_training=True
        )
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_preds, val_labels = process_epoch(
                model=model, data_loader=val_loader, criterion=criterion,
                optimizer=optimizer, device=device, is_training=False
            )
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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

        # Log results
        utils.log_and_append_file(log_file, f'\nEpoch [{epoch + 1}/{num_epochs}]')
        utils.log_and_append_file(log_file,
                                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
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

        # Adjust batch size based on sequence length
        if max_length[i] == 400:
            config.batch_size = 256
        elif max_length[i] == 800:
            config.batch_size = 128
        elif max_length[i] == 1200:
            config.batch_size = 64
        elif max_length[i] == 1800:
            config.batch_size = 32
        else:
            raise ValueError("Invalid length")

        for fold in range(num_fold):
            experiment_name = "logistic_regression"
            result_dir = f"../../results/results/{experiment_name}/group_{group}/fold_{fold + 1}"
            os.makedirs(result_dir, exist_ok=True)

            log_file = os.path.join(result_dir, f"log_{group}.txt")
            plot_file = os.path.join(result_dir, f"training_plot_{group}.png")

            # Log configuration
            utils.log_and_append_file(log_file, f"Training Logistic Regression model for group {group} fold {fold + 1}")
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

            # Initialize logistic regression model
            model = LogisticRegressionModel().to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train and evaluate
            start_time = time.time()
            train_and_evaluate(
                model=model, train_loader=train_loader, val_loader=test_loader,
                criterion=criterion, optimizer=optimizer, device=device,
                num_epochs=config.num_epochs, log_file=log_file, plot_file=plot_file
            )

            # Cleanup
            torch.cuda.empty_cache()
            utils.log_and_append_file(log_file, f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()