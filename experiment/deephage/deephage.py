import gc
import os
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from common.env_config import config
from logger.phg_cls_log import setup_logger

log = setup_logger(__file__)


def classification_report_csv(report, path_save, c):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        # Skip empty lines
        if not line.strip():
            continue

        row = {}
        row_data = line.split()

        # Skip lines that don't have enough data
        if len(row_data) < 5:
            continue

        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    if c == 0:
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(os.path.join(path_save, "classification_report.csv"), index=False)
    else:
        log.info('train')
    return report_data


# Calculate sensitivity (recall) and specificity
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # For binary classification, confusion matrix is [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'f1_score': f1
    }


# Define the model
class DeePhage(nn.Module):
    def __init__(self, max_length):
        super(DeePhage, self).__init__()

        # Conv1D layer with 64 filters, kernel size 6, ReLU activation
        self.conv = nn.Conv1d(4, 64, kernel_size=6, padding='same')
        self.relu = nn.ReLU()

        # MaxPooling1D with pool size 3
        self.pool = nn.MaxPool1d(3)

        # BatchNormalization
        self.bn1 = nn.BatchNorm1d(64)

        # Dropout (0.3)
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        # Calculate feature size after pooling
        self.feature_size = max_length // 3  # After pooling with size 3

        self.fc1 = nn.Linear(64, 64)  # Using GlobalAveragePooling instead of flattening
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # PyTorch Conv1d expects [batch, channels, length] format
        # Input is [batch, length, channels], so transpose dimensions
        x = x.permute(0, 2, 1)

        # Apply convolution and activation
        x = self.relu(self.conv(x))

        # Apply pooling
        x = self.pool(x)

        # Apply batch normalization and dropout
        x = self.bn1(x)
        x = self.dropout(x)

        # Global average pooling (equivalent to GlobalAveragePooling1D in Keras)
        x = torch.mean(x, dim=2)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)

        return x


# History tracking class (similar to Keras' LossHistory callback)
class TrainingHistory:
    def __init__(self):
        self.train_loss = {'batch': [], 'epoch': []}
        self.train_acc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.val_auc = {'epoch': []}
        self.sensitivity = {'epoch': []}
        self.specificity = {'epoch': []}
        self.val_sensitivity = {'epoch': []}
        self.val_specificity = {'epoch': []}

    def update_batch(self, loss, acc, val_loss=None, val_acc=None):
        self.train_loss['batch'].append(loss)
        self.train_acc['batch'].append(acc)
        if val_loss is not None:
            self.val_loss['batch'].append(val_loss)
        if val_acc is not None:
            self.val_acc['batch'].append(val_acc)

    def update_epoch(self, train_loss, train_acc, val_loss, val_acc, sensitivity=None, specificity=None,
                     val_sensitivity=None, val_specificity=None, val_auc=None):
        self.train_loss['epoch'].append(train_loss)
        self.train_acc['epoch'].append(train_acc)
        self.val_loss['epoch'].append(val_loss)
        self.val_acc['epoch'].append(val_acc)

        if sensitivity is not None:
            self.sensitivity['epoch'].append(sensitivity)
        if specificity is not None:
            self.specificity['epoch'].append(specificity)
        if val_sensitivity is not None:
            self.val_sensitivity['epoch'].append(val_sensitivity)
        if val_specificity is not None:
            self.val_specificity['epoch'].append(val_specificity)
        if val_auc is not None:
            self.val_auc['epoch'].append(val_auc)

    def history_plot(self, loss_type, path_save, max_length, lr_rate, b_size):

        # Original loss/accuracy plot
        plt.figure(figsize=(16, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.train_loss['epoch'], label='Train Loss')
        plt.plot(self.val_loss['epoch'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.train_acc['epoch'], label='Train Accuracy')
        plt.plot(self.val_acc['epoch'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.val_sensitivity['epoch'], label='Val Sensitivity')
        plt.plot(self.val_specificity['epoch'], label='Val Specificity')
        plt.title('Sensitivity - Specificity')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()

        # Plot ROC AUC
        plt.subplot(2, 2, 4)
        plt.plot(self.val_auc['epoch'], 'r-', label='ROC AUC')
        plt.title('ROC AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(path_save, f"{max_length}_{lr_rate}_{b_size}.png"))
        plt.show()
        plt.close()


# Binary accuracy calculation
def binary_accuracy(y_pred, y_true):
    y_pred_sigmoid = torch.sigmoid(y_pred)  # Apply sigmoid here
    y_pred_tag = (y_pred_sigmoid > 0.5).float()
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum / y_true.shape[0]
    return acc.item()


def train_step(model, train_loader, optimizer, criterion, scaler, device):
    """Perform one training epoch step"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    train_preds = []
    train_labels = []

    # Use tqdm for progress bar
    train_loop = tqdm(train_loader, desc=f"[Train]", leave=False)
    for inputs, labels in train_loop:
        inputs = inputs.to(device)
        labels = labels.view(-1, 1).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy
        acc = binary_accuracy(outputs, labels)

        # Collect predictions and labels for metrics
        score = torch.sigmoid(outputs)
        preds = (score > 0.5).float().cpu().detach().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())

        # Update batch statistics
        running_loss += loss.detach().item()
        running_acc += acc

        # Update progress bar
        train_loop.set_postfix(loss=running_loss, acc=acc)

    # Calculate epoch-level training metrics
    train_loss = running_loss / len(train_loader)
    train_acc = running_acc / len(train_loader)

    # Calculate sensitivity and specificity for training data
    train_metrics = calculate_metrics(np.array(train_labels), np.array(train_preds))

    return {
        'loss': train_loss,
        'acc': train_acc,
        'sensitivity': train_metrics['sensitivity'],
        'specificity': train_metrics['specificity'],
        'preds': train_preds,
        'labels': train_labels
    }


def eval_step(model, test_loader, criterion, device):
    """Perform one evaluation epoch step"""
    model.eval()
    val_running_loss = 0.0
    val_running_acc = 0.0
    val_preds = []
    val_labels = []

    # Use tqdm for progress bar
    val_loop = tqdm(test_loader, desc=f"[Val]", leave=False)
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            acc = binary_accuracy(outputs, labels)

            # Collect predictions and labels for metrics
            score = torch.sigmoid(outputs)
            preds = (score > 0.5).float().cpu().detach().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

            val_running_loss += loss.detach().item()
            val_running_acc += acc

            # Update progress bar
            val_loop.set_postfix(loss=val_running_loss, acc=acc)

    val_loss = val_running_loss / len(test_loader)
    val_acc = val_running_acc / len(test_loader)

    # Calculate sensitivity and specificity for validation data
    val_metrics = calculate_metrics(np.array(val_labels), np.array(val_preds))

    return {
        'loss': val_loss,
        'acc': val_acc,
        'sensitivity': val_metrics['sensitivity'],
        'specificity': val_metrics['specificity'],
        'preds': val_preds,
        'labels': val_labels
    }


def start_executing(device, model, train_loader, test_loader, num_epochs, optimizer, criterion, scaler, model_save_path,
                    history,
                    is_training=True):
    """
    Run training or evaluation based on is_training flag

    Args:
        device: The device to run on (CPU or GPU)
        model: The model to train/evaluate
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing/validation data
        num_epochs: Number of epochs to train for
        optimizer: Optimizer for training
        criterion: Loss function
        scaler: GradScaler for mixed precision training
        model_save_path: Path to save the model
        history: History save model history
        is_training: If True, run training phase; if False, run evaluation only

    Returns:
        history: Training history object
    """

    # Training loop
    for epoch in range(num_epochs):
        epoch_desc = f"Epoch {epoch + 1}/{num_epochs}"
        log.info(f"Starting {epoch_desc}")

        # Training phase
        if is_training:
            train_results = train_step(model, train_loader, optimizer, criterion, scaler, device)
            train_loss = train_results['loss']
            train_acc = train_results['acc']
            train_sensitivity = train_results['sensitivity']
            train_specificity = train_results['specificity']
        else:
            # If not training, use placeholder values
            train_loss, train_acc = 0.0, 0.0
            train_sensitivity, train_specificity = 0.0, 0.0

        # Validation/Evaluation phase (always run)
        val_results = eval_step(model, test_loader, criterion, device)
        val_loss = val_results['loss']
        val_acc = val_results['acc']
        val_sensitivity = val_results['sensitivity']
        val_specificity = val_results['specificity']

        # Update epoch statistics
        history.update_epoch(
            train_loss, train_acc, val_loss, val_acc,
            train_sensitivity, train_specificity,
            val_sensitivity, val_specificity
        )

        # Print epoch summary
        log.info(f'{epoch_desc} - '
                 f'Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, Sens={train_sensitivity:.4f}, Spec={train_specificity:.4f} | '
                 f'Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Sens={val_sensitivity:.4f}, Spec={val_specificity:.4f}')

        # Save best model (optional) - only if training
        if is_training and epoch > 0 and val_acc > max(history.val_acc['epoch'][:-1]):
            torch.save(model.state_dict(), model_save_path.replace('.pt', '_best.pt'))
            log.info(f"Saved new best model with validation accuracy: {val_acc:.4f}")

    return history


def last_evaluation(device, model, train_loader, test_loader, path_save, predict_save_path, model_save_path):
    # Final evaluation
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_train_predictions = []
    all_train_true_labels = []

    log.info("Performing final evaluation...")
    with torch.no_grad():
        # Test predictions
        test_loop = tqdm(test_loader, desc="Evaluating test data")
        for inputs, labels in test_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

        # Train predictions
        train_loop = tqdm(train_loader, desc="Evaluating train data")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_train_predictions.append(outputs.cpu().numpy())
            all_train_true_labels.append(labels.cpu().numpy())

    # Concatenate predictions and true labels
    predict = np.concatenate(all_predictions).reshape(-1)
    true_labels = np.concatenate(all_true_labels).reshape(-1)
    predict_train = np.concatenate(all_train_predictions).reshape(-1)
    true_train_labels = np.concatenate(all_train_true_labels).reshape(-1)

    # Save predictions
    np.savetxt(predict_save_path, predict)

    # Save model
    torch.save(model.state_dict(), model_save_path)
    log.info(f"Model saved to {model_save_path}")

    # Generate binary predictions
    predict_binary = (predict > 0.5).astype(int)
    predict_train_binary = (predict_train > 0.5).astype(int)

    # Calculate final metrics
    test_metrics = calculate_metrics(true_labels, predict_binary)
    train_metrics = calculate_metrics(true_train_labels, predict_train_binary)

    # Print detailed metrics
    log.info("\nFinal Test Metrics:")
    log.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    log.info(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
    log.info(f"Specificity: {test_metrics['specificity']:.4f}")
    log.info(f"Precision: {test_metrics['precision']:.4f}")
    log.info(f"F1 Score: {test_metrics['f1_score']:.4f}")

    log.info("\nFinal Train Metrics:")
    log.info(f"Accuracy: {train_metrics['accuracy']:.4f}")
    log.info(f"Sensitivity: {train_metrics['sensitivity']:.4f}")
    log.info(f"Specificity: {train_metrics['specificity']:.4f}")
    log.info(f"Precision: {train_metrics['precision']:.4f}")
    log.info(f"F1 Score: {train_metrics['f1_score']:.4f}")

    # Generate classification reports
    report_test = classification_report(true_labels, predict_binary, output_dict=False)
    log.info('\nDetailed Test Classification Report:')
    log.info(report_test)
    report_dic_test = classification_report_csv(report_test, path_save, 0)
    temp_acc, viru_acc = report_dic_test[0].get('recall'), report_dic_test[1].get('recall')

    report_train = classification_report(true_train_labels, predict_train_binary, output_dict=False)
    log.info('\nDetailed Train Classification Report:')
    log.info(report_train)
    report_dic_train = classification_report_csv(report_train, path_save, 1)
    train_temp_acc, train_viru_acc = report_dic_train[0].get('recall'), report_dic_train[1].get('recall')

    # Create confusion matrix
    test_cm = confusion_matrix(true_labels, predict_binary)
    train_cm = confusion_matrix(true_train_labels, predict_train_binary)

    log.info("\nTest Confusion Matrix:")
    log.info(test_cm)
    log.info("\nTrain Confusion Matrix:")
    log.info(train_cm)

    # Save confusion matrices to CSV
    pd.DataFrame(test_cm).to_csv(os.path.join(path_save, "test_confusion_matrix.csv"))
    pd.DataFrame(train_cm).to_csv(os.path.join(path_save, "train_confusion_matrix.csv"))

    # Save additional metrics to CSV
    metrics_df = pd.DataFrame({
        'dataset': ['test', 'train'],
        'accuracy': [test_metrics['accuracy'], train_metrics['accuracy']],
        'sensitivity': [test_metrics['sensitivity'], train_metrics['sensitivity']],
        'specificity': [test_metrics['specificity'], train_metrics['specificity']],
        'precision': [test_metrics['precision'], train_metrics['precision']],
        'f1_score': [test_metrics['f1_score'], train_metrics['f1_score']]
    })
    metrics_df.to_csv(os.path.join(path_save, "additional_metrics.csv"), index=False)


def run(device):
    min_length = 1200
    max_length = 1800
    group = f"{min_length}_{max_length}"
    lr_rate = 0.0001
    b_size = 32
    num_epochs = 5
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    data_dir = os.path.join(config.MY_DATA_DIR, f"one_hot/{group}")

    num_fold = 1
    for r in range(num_fold):
        fold = r + 1
        log.info(f"\n{'=' * 50}")
        log.info(f"Starting fold {fold}/5")
        log.info(f"{'=' * 50}")

        # Path definitions
        path_save = os.path.join(config.RESULT_DIR, f"dee_phage/{group}/fold_{fold}/{timestamp}")
        os.makedirs(path_save, exist_ok=True)  # Ensure directory exists

        predict_save_path = os.path.join(path_save, f"{max_length}_{lr_rate}_{b_size}_prediction.csv")
        model_save_path = os.path.join(path_save, f"{max_length}_{lr_rate}_{b_size}_model.pt")

        # Load data from .mat files
        log.info('Loading data...')
        train_matrix = np.load(os.path.join(data_dir, f'one_hot_{group}_train_vector.npy'))
        train_label = np.load(os.path.join(data_dir, f'y_train.npy'))
        test_matrix = np.load(os.path.join(data_dir, f'one_hot_{group}_val_vector.npy'))
        test_label = np.load(os.path.join(data_dir, f'y_val.npy'))

        log.info(f"train_matrix shape: {train_matrix.shape}")
        log.info(f"train_label shape: {train_label.shape}")
        log.info(f"test_matrix shape: {test_matrix.shape}")
        log.info(f"test_label shape: {test_label.shape}")

        log.info(f"Training data distribution")
        label_counts = Counter(train_label)
        for label, count in label_counts.items():
            log.info(f"Label {label}: {count} samples")

        train_num = train_label.shape[0]
        test_num = test_label.shape[0]

        log.info(f"Train samples: {train_num}, Test samples: {test_num}")

        train_matrix = train_matrix.reshape(train_num, max_length, 4)
        test_matrix = test_matrix.reshape(test_num, max_length, 4)

        # Convert to PyTorch tensors
        train_matrix_tensor = torch.FloatTensor(train_matrix)
        train_label_tensor = torch.FloatTensor(train_label)
        test_matrix_tensor = torch.FloatTensor(test_matrix)
        test_label_tensor = torch.FloatTensor(test_label)

        # Create datasets and dataloaders
        train_dataset = TensorDataset(train_matrix_tensor, train_label_tensor)
        test_dataset = TensorDataset(test_matrix_tensor, test_label_tensor)

        train_loader = DataLoader(train_dataset, batch_size=b_size, num_workers=4, prefetch_factor=4, shuffle=True,
                                  persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=b_size, num_workers=4, prefetch_factor=4, shuffle=False,
                                 persistent_workers=True)

        # Initialize model
        model = DeePhage(max_length)
        model.to(device)

        # Print model summary
        log.info(f"Model architecture: {model}")
        log.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)
        scaler = GradScaler('cuda')

        # Initialize history logger
        history = TrainingHistory()
        start_executing(device, model, train_loader, test_loader, num_epochs, optimizer, criterion, scaler,
                        model_save_path, history, is_training=True)

        start_executing(device, model, train_loader, test_loader, 1, optimizer, criterion, scaler,
                        model_save_path, history, is_training=False)

        last_evaluation(device, model, train_loader, test_loader, path_save, predict_save_path, model_save_path)

        # Plot and save results
        history.history_plot('epoch', path_save, max_length, lr_rate, b_size)

    log.info("\nAll folds completed!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check GPU availability (since you have RTX 5070Ti)
    if torch.cuda.is_available():
        # Monitor initial GPU memory
        initial_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        log.info(f"Initial GPU memory usage: {initial_mem:.2f} GB")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        # Clear memory and cache before training
        gc.collect()
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.deterministic = True

        # Log CUDA information
        log.info(f"CUDA Version: {torch.version.cuda}")
        log.info(
            f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")

        # Blackwell optimized kernel tuning
        try:
            # Set environment variables for Blackwell
            os.environ['CUDA_AUTO_TUNE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            log.info("Set optimized kernel autotuning for Blackwell")
        except:
            pass
    else:
        log.info("No GPU available, training on CPU")

    run(device=device)
