import os
import shutil
import time
from typing import Tuple, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import utils

# Configuration
CONFIG = {
    'input_size': 4,
    'hidden_size': 128,
    'num_layers': 2,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'num_epochs': 20,
    'bidirectional': True,
    'dropout': 0.3
}


class BacteriaDataset(Dataset):
    """Dataset class for bacterial sequences and their labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.tensor(self.labels[idx].item(), dtype=torch.float32)
        return sequence, label


class LSTMClassifier(nn.Module):
    """LSTM model for sequence classification."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 bidirectional: bool, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization
        self.batch_norm = nn.BatchNorm1d(input_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_output_size),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, features = x.size()

        # Apply batch normalization
        x = x.reshape(-1, features)
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, features)

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Calculate attention weights
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, seq_len)

        # Apply attention weights
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            lstm_out  # (batch_size, seq_len, hidden_size * num_directions)
        )  # (batch_size, 1, hidden_size * num_directions)
        context_vector = context_vector.squeeze(1)

        # Final prediction
        output = self.fc(context_vector)

        return output.squeeze(-1), attention_weights


def load_fold_datasets(group: str, length: int, fold: int, log_file: str) -> Tuple[
    BacteriaDataset, BacteriaDataset]:
    """Load and prepare datasets from fold."""
    utils.log_and_append_file(log_file, f"Loading dee_phage_data for group {group} fold {fold}")
    train_datasets = []
    test_datasets = []

    root_data_dir = os.path.join("../../../data", group)
    train_dir = os.path.join(root_data_dir, "train")
    test_dir = os.path.join(root_data_dir, "test")

    # Find fold-specific files
    train_sequence_file = f'P_train_ds_{group}_{fold}.mat'
    train_label_file = f'T_train_ds_{group}_{fold}.mat'
    test_sequence_file = f'P_test_{group}_{fold}.mat'
    # test_label_file = f'{group}_{fold}_test_label.csv'
    test_label_file = f'label_{group}_{fold}.mat'
    print(train_sequence_file, train_label_file, test_sequence_file, test_label_file)
    # train_sequence_file = next(f for f in os.listdir(train_dir) if f'P_train_ds_{group}_{fold}.mat' == f)
    # train_label_file = next(f for f in os.listdir(train_dir) if f'T_train_ds_{group}_{fold}.mat' == f)
    # test_sequence_file = next(f for f in os.listdir(test_dir) if f'P_test_{group}_{fold}.mat' == f)
    # test_label_file = next(f for f in os.listdir(test_dir) if f'{group}_{fold}_test_label.csv' == f)

    # Load dee_phage_data with context managers
    with h5py.File(os.path.join(train_dir, train_sequence_file), 'r') as f:
        train_matrix = f['P_train_ds'][:].transpose()
    with h5py.File(os.path.join(train_dir, train_label_file), 'r') as f:
        train_label = f['T_train_ds'][:].transpose()
    with h5py.File(os.path.join(test_dir, test_sequence_file), 'r') as f:
        test_matrix = f['P_test'][:].transpose()
    # test_label = pd.read_csv(os.path.join(test_dir, test_label_file), header=None).to_numpy()
    with h5py.File(os.path.join(test_dir, test_label_file), 'r') as f:
        test_label = f['T_test'][:].transpose()

    # Reshape matrices
    train_num = train_label.shape[0]
    train_matrix = train_matrix.reshape(train_num, length, 4)
    test_num = test_label.shape[0]
    test_matrix = test_matrix.reshape(test_num, length, 4)

    utils.log_and_append_file(log_file, f"Train dataset shape: {train_matrix.shape}")
    utils.log_and_append_file(log_file, f"Test dataset size: {test_matrix.shape}")
    utils.log_and_append_file(log_file, f"Loading dee_phage_data for group {group} fold {fold} successful")

    return BacteriaDataset(train_matrix, train_label), BacteriaDataset(test_matrix, test_label)


def plot_attention_weights(sequence: torch.Tensor, attention_weights: torch.Tensor,
                           save_path: str, index: int = 0) -> None:
    """Plot attention weights visualization."""
    plt.figure(figsize=(15, 5))

    # Plot attention weights
    attention = attention_weights[index].cpu().detach().numpy()
    plt.plot(attention, label='Attention Weights')
    plt.title('Attention Weights Distribution')
    plt.xlabel('Sequence Position')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(history: dict, plot_file: str) -> None:
    """
    Plot training and validation metrics history.

    Args:
        history: Dictionary containing training history metrics
        plot_file: Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def log_epoch_results(epoch: int, num_epochs: int,
                      train_loss: float, train_acc: float,
                      val_loss: float, val_acc: float,
                      train_preds: list, train_labels: list,
                      val_preds: list, val_labels: list,
                      log_file: str) -> None:
    """
    Log training and validation results for an epoch.

    Args:
        epoch: Current epoch number
        num_epochs: Total number of epochs
        train_loss: Training loss for the epoch
        train_acc: Training accuracy for the epoch
        val_loss: Validation loss for the epoch
        val_acc: Validation accuracy for the epoch
        train_preds: Training predictions
        train_labels: Training true labels
        val_preds: Validation predictions
        val_labels: Validation true labels
        log_file: Path to log file
    """
    # Log training results
    utils.log_and_append_file(log_file, f'\nEpoch [{epoch + 1}/{num_epochs}]')
    utils.log_and_append_file(log_file, f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}')
    utils.log_and_append_file(log_file, 'Training Classification Report:')
    utils.log_and_append_file(log_file, str(classification_report(train_labels, train_preds)))
    utils.log_and_append_file(log_file, 'Training Confusion Matrix:')
    utils.log_and_append_file(log_file, str(confusion_matrix(train_labels, train_preds)))

    # Log validation results
    utils.log_and_append_file(log_file, f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}')
    utils.log_and_append_file(log_file, 'Validation Classification Report:')
    utils.log_and_append_file(log_file, str(classification_report(val_labels, val_preds)))
    utils.log_and_append_file(log_file, 'Validation Confusion Matrix:')
    utils.log_and_append_file(log_file, str(confusion_matrix(val_labels, val_preds)))
    utils.log_and_append_file(log_file, '-' * 50)


def train_and_evaluate(model: nn.Module, train_loader: DataLoader,
                       val_loader: DataLoader, criterion: nn.Module,
                       optimizer: optim.Optimizer, device: torch.device,
                       num_epochs: int, log_file: str,
                       plot_file: str) -> None:
    """Train the model and evaluate its performance."""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # Create directory for attention plots
    attention_dir = os.path.join(os.path.dirname(plot_file), 'attention_plots')
    if os.path.exists(attention_dir):
        shutil.rmtree(attention_dir)
    os.makedirs(attention_dir, exist_ok=False)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch_idx, (sequences, labels) in tqdm(enumerate(train_loader), desc="Training attention weights"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs, attention_weights = model(sequences)

            # Save attention weights periodically
            if batch_idx % 5000 == 0:  # Save every 50 batches
                attention_plot_file = os.path.join(
                    attention_dir,
                    f'attention_epoch_{epoch}_batch_{batch_idx}.png'
                )
                plot_attention_weights(sequences, attention_weights, attention_plot_file)

        train_loss, train_preds, train_labels = process_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

        # Validation phase
        model.eval()
        val_loss, val_preds, val_labels = process_epoch(
            model, val_loader, criterion, optimizer, device, is_training=False
        )
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Log results
        log_epoch_results(epoch, num_epochs, train_loss, train_acc,
                          val_loss, val_acc, train_preds, train_labels,
                          val_preds, val_labels, log_file)

    # Plot training history
    plot_training_history(history, plot_file)


def process_epoch(model: nn.Module, data_loader: DataLoader,
                  criterion: nn.Module, optimizer: optim.Optimizer,
                  device: torch.device, is_training: bool) -> Tuple[float, List, List]:
    """Process one epoch of training or validation."""
    total_loss = 0
    predictions = []
    true_labels = []

    desc = "Training" if is_training else "Validation"
    with torch.set_grad_enabled(is_training):
        for sequences, labels in tqdm(data_loader, desc=desc):
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs, attention_weights = model(sequences)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predictions.extend((outputs > 0.5).float().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

    return total_loss / len(data_loader), predictions, true_labels


def main(start_group=1):
    """Main execution function."""
    max_length = [100, 400, 800, 1200, 1800]
    num_fold = 5
    # data_dir = "../../dee_phage_data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(start_group, len(max_length)):
        if max_length == 2:
            break
        for fold in range(num_fold):
            # if fold > 0:
            #     break
            length = max_length[i]
            group = f"{max_length[i - 1]}_{length}"
            # train_dir = os.path.join(data_dir, f"{group}/train")
            # test_dir = os.path.join(data_dir, f"{group}/test")

            # Setup directories
            experiment_day = "26-12-2024"
            if not os.path.exists(experiment_day):
                os.makedirs(experiment_day)
            # else:
            #     raise ValueError("Experiment day directory already exists")

            group_dir = os.path.join(experiment_day, f"{group}")
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)
            # else:
            #     raise ValueError("Group directory already exists")

            fold_dir = os.path.join(group_dir, f"fold_{fold + 1}")
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            # else:
            #     raise ValueError("Fold directory already exists")

            result_dir = os.path.join(
                fold_dir,
                f"hid_size_{CONFIG['hidden_size']}_n_layers_{CONFIG['num_layers']}_lr_{CONFIG['learning_rate']}_epo_{CONFIG['num_epochs']}_bidi_{CONFIG['bidirectional']}_0"
            )
            if not os.path.exists(result_dir):
                # shutil.rmtree(result_dir)
                os.makedirs(result_dir)
            else:
                f = [f for f in os.listdir(fold_dir) if f.__contains__(f"hid_size_{CONFIG['hidden_size']}")]
                count = len(f) + 1
                result_dir = os.path.join(
                    fold_dir,
                    f"hid_size_{CONFIG['hidden_size']}_n_layers_{CONFIG['num_layers']}_lr_{CONFIG['learning_rate']}_epo_{CONFIG['num_epochs']}_bidi_{CONFIG['bidirectional']}_{count}"
                )
                os.makedirs(result_dir)

            log_file = os.path.join(result_dir, f"log_{group}.txt")
            plot_file = os.path.join(result_dir, f"training_plot_{group}.png")

            # Log starting time
            utils.log_and_append_file(log_file, f"Training LSTM model for group {group} fold {fold + 1}")
            utils.log_and_append_file(log_file, f"Start time: {time.ctime()}")

            # Log configuration
            utils.log_and_append_file(log_file, "Configuration:")
            for key, value in CONFIG.items():
                utils.log_and_append_file(log_file, f"{key}: {value}")

            # Load datasets
            train_dataset, test_dataset = load_fold_datasets(group, length, fold + 1, log_file)
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

            # Initialize model and training components
            model = LSTMClassifier(
                CONFIG['input_size'], CONFIG['hidden_size'],
                CONFIG['num_layers'], CONFIG['bidirectional'],
                CONFIG['dropout']
            ).to(device)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

            # Train and evaluate
            start_time = time.time()
            train_and_evaluate(
                model, train_loader, test_loader, criterion,
                optimizer, device, CONFIG['num_epochs'],
                log_file, plot_file
            )

            # Cleanup
            torch.cuda.empty_cache()
            utils.log_and_append_file(log_file, f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
    # print(datetime.date.today())
    # print(datetime.datetime.now())
    # print(time.ctime())