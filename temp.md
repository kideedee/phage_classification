import gc
import os
import time
from collections import Counter

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
from torch import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import logging
log = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt', log=None):
        """
        Args:
            patience (int): How long to wait after last improvement.
                            Default: 10
            verbose (bool): If True, prints a message for each improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.log = log

    def __call__(self, val_loss, model):
        score = -val_loss  # We want to maximize the negative loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            self.print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
        print('train')
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

    result = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'recall': sensitivity,
        'f1_score': f1
    }

    return result


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
        self.lr_rates = {'batch': [], 'epoch': []}
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

        # plt.subplot(2, 3, 5)  # Add a 5th subplot for learning rate
        # plt.plot(self.lr_rates['epoch'], 'g-', label='Learning Rate')
        # plt.title('Learning Rate')
        # plt.xlabel('Epoch')
        # plt.ylabel('Learning Rate')
        # plt.yscale('log')  # Use log scale for better visualization
        # plt.legend()

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


def train_step(model, train_loader, optimizer, criterion, scaler, scheduler, device):
    """Perform one training epoch step"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    train_preds = []
    train_labels = []
    current_lr = optimizer.param_groups[0]['lr']  # Track current learning rate

    # Use tqdm for progress bar
    train_loop = tqdm(train_loader, desc=f"[Train] LR={current_lr:.6f}", leave=False)
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

        # Step the scheduler if it's a batch-based scheduler like OneCycleLR
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']  # Update current LR

        # # Forward pass (không sử dụng autocast)
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        #
        # # Backward pass and optimize (không sử dụng scaler)
        # loss.backward()
        # optimizer.step()

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
        'labels': train_labels,
        'lr': current_lr
    }


def eval_step(model, test_loader, criterion, device):
    """Perform one evaluation epoch step"""
    model.eval()
    val_running_loss = 0.0
    val_running_acc = 0.0
    val_preds = []
    val_labels = []
    val_scores = []  # Add this to collect raw probabilities

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
            val_scores.extend(score.cpu().detach().numpy())  # Add this line
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

    # Calculate ROC AUC score
    val_auc = roc_auc_score(np.array(val_labels), np.array(val_scores))

    return {
        'loss': val_loss,
        'acc': val_acc,
        'precision': val_metrics['precision'],
        'recall': val_metrics['recall'],
        'f1': val_metrics['f1_score'],
        'sensitivity': val_metrics['sensitivity'],
        'specificity': val_metrics['specificity'],
        'preds': val_preds,
        'labels': val_labels,
        'auc': val_auc  # Add AUC to the return dictionary
    }


def start_executing(device, model, train_loader, test_loader, num_epochs, optimizer, criterion, scaler, model_save_path,
                    history,
                    early_stopping,
                    scheduler,
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

    # Initialize best metric tracking
    best_metrics = {
        'val_loss': float('inf'),  # Start with infinity for loss
        'val_acc': 0.0,
        'val_auc': 0.0,
        'val_sensitivity': 0.0,
        'val_specificity': 0.0,
        'val_f1': 0.0,
        'epoch': 0
    }

    # Training loop
    for epoch in range(num_epochs):
        epoch_desc = f"Epoch {epoch + 1}/{num_epochs}"
        print(f"Starting {epoch_desc}")

        # Training phase
        if is_training:
            train_results = train_step(model, train_loader, optimizer, criterion, scaler, scheduler, device)
            train_loss = train_results['loss']
            train_acc = train_results['acc']
            train_sensitivity = train_results['sensitivity']
            train_specificity = train_results['specificity']
            current_lr = train_results['lr']
        else:
            # If not training, use placeholder values
            train_loss, train_acc = 0.0, 0.0
            train_sensitivity, train_specificity = 0.0, 0.0
            current_lr = optimizer.param_groups[0]['lr']

        # Validation/Evaluation phase (always run)
        val_results = eval_step(model, test_loader, criterion, device)
        val_loss = val_results['loss']
        val_acc = val_results['acc']
        val_sensitivity = val_results['sensitivity']
        val_specificity = val_results['specificity']
        val_f1 = val_results['f1']
        val_precision = val_results['precision']
        val_recall = val_results['recall']
        val_auc = val_results['auc']  # Get AUC value

        # Step scheduler if it's epoch-based like ReduceLROnPlateau
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        # Calculate F1 score
        # val_precision = val_results.get('precision', 0.0)
        # if val_sensitivity > 0 and val_precision > 0:
        #     val_f1 = 2 * (val_precision * val_sensitivity) / (val_precision + val_sensitivity)
        # else:
        #     val_f1 = 0.0

        # Update epoch statistics
        history.update_epoch(
            train_loss, train_acc, val_loss, val_acc,
            train_sensitivity, train_specificity,
            val_sensitivity, val_specificity,
            val_auc  # Add AUC to the history update
        )

        # Print epoch summary with AUC and learning rate
        print(f'{epoch_desc} - LR={current_lr:.6f} - '
                 f'Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, Sens={train_sensitivity:.4f}, Spec={train_specificity:.4f} | '
                 f'Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Sens={val_sensitivity:.4f}, Spec={val_specificity:.4f}, AUC={val_auc:.4f}')

        # Save best model (optional) - only if training
        # if is_training and epoch > 0 and val_acc > max(history.val_acc['epoch'][:-1]):
        #     torch.save(model.state_dict(), model_save_path.replace('.pt', '_best.pt'))
        #     print(f"Saved new best model with validation accuracy: {val_acc:.4f}")

        # Save model and metrics only if validation loss improved
        if is_training and val_loss < best_metrics['val_loss']:
            best_metrics['val_loss'] = val_loss
            best_metrics['val_acc'] = val_acc
            best_metrics['val_auc'] = val_auc
            best_metrics['val_sensitivity'] = val_sensitivity
            best_metrics['val_specificity'] = val_specificity
            best_metrics['val_f1'] = val_f1
            best_metrics['val_precision'] = val_precision
            best_metrics['val_recall'] = val_recall
            best_metrics['epoch'] = epoch

            # Save best model
            best_model_path = model_save_path.replace('.pt', '_best.pt')
            torch.save(model.state_dict(), best_model_path)

            # Save all metrics to CSV file
            metrics_df = pd.DataFrame({
                'epoch': [epoch],
                'val_loss': [val_loss],
                'val_accuracy': [val_acc],
                'val_auc': [val_auc],
                'val_f1': [val_f1],
                'val_sensitivity': [val_sensitivity],
                'val_specificity': [val_specificity],
                'train_loss': [train_loss],
                'train_accuracy': [train_acc],
                'train_sensitivity': [train_sensitivity],
                'train_specificity': [train_specificity],
                'val_precision': [val_precision],
                'val_recall': [val_recall],
                'learning_rate': [current_lr]  # Add learning rate to metrics
            })

            metrics_file = os.path.join(os.path.dirname(model_save_path), 'best_metrics.csv')
            metrics_df.to_csv(metrics_file, index=False)

            print(f"New best model saved with validation loss: {val_loss:.4f}")
            print(
                f"Metrics at best epoch - Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, Sens: {val_sensitivity:.4f}, Spec: {val_specificity:.4f}")

        if is_training:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # At the end of training, report the best metrics
    if is_training:
        print("\nBest Model Summary (Lowest Validation Loss):")
        print(f"Best Epoch: {best_metrics['epoch']}")
        print(f"Best Val Loss: {best_metrics['val_loss']:.4f}")
        print(f"Acc at Best: {best_metrics['val_acc']:.4f}")
        print(f"AUC at Best: {best_metrics['val_auc']:.4f}")
        print(f"F1 at Best: {best_metrics['val_f1']:.4f}")
        print(f"Sensitivity at Best: {best_metrics['val_sensitivity']:.4f}")
        print(f"Specificity at Best: {best_metrics['val_specificity']:.4f}")

    return history


def last_evaluation(device, model, test_loader, path_save, predict_save_path, model_save_path, group, fold):
    # Final evaluation
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_scores = []  # Add this line to collect raw probabilities

    print("Performing final evaluation...")
    with torch.no_grad():
        # Test predictions
        test_loop = tqdm(test_loader, desc="Evaluating test data")
        for inputs, labels in test_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)  # Get raw probabilities
            all_scores.append(scores.cpu().numpy())  # Store raw probabilities
            all_predictions.append(outputs.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

        # Train predictions
        # train_loop = tqdm(train_loader, desc="Evaluating train data")
        # for inputs, labels in train_loop:
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     outputs = model(inputs)
        #     scores = torch.sigmoid(outputs)  # Get raw probabilities
        #     all_train_scores.append(scores.cpu().numpy())  # Store raw probabilities
        #     all_train_predictions.append(outputs.cpu().numpy())
        #     all_train_true_labels.append(labels.cpu().numpy())

    # Concatenate predictions and true labels
    predict = np.concatenate(all_predictions).reshape(-1)
    true_labels = np.concatenate(all_true_labels).reshape(-1)
    scores = np.concatenate(all_scores).reshape(-1)  # Raw probabilities for AUC

    # Save predictions
    np.savetxt(predict_save_path, predict)

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Generate binary predictions
    predict_binary = (predict > 0.5).astype(int)

    # Calculate final metrics
    test_metrics = calculate_metrics(true_labels, predict_binary)
    test_metrics['group'] = group
    test_metrics['fold'] = fold
    df = pd.DataFrame([test_metrics])

    csv_file = "./result_deephage_3.csv"
    file_exists = os.path.exists(csv_file)

    # Save to CSV with append mode
    df.to_csv(csv_file,
              mode='a',  # Append mode
              header=not file_exists,  # Write header only if file doesn't exist
              index=False)

    # Calculate ROC AUC
    test_auc = roc_auc_score(true_labels, scores)

    # Print detailed metrics
    print("\nFinal Test Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"ROC AUC: {test_auc:.4f}")  # Add ROC AUC to output

    # Generate classification reports
    report_test = classification_report(true_labels, predict_binary, output_dict=False)
    print('\nDetailed Test Classification Report:')
    print(report_test)
    report_dic_test = classification_report_csv(report_test, path_save, 0)
    temp_acc, viru_acc = report_dic_test[0].get('recall'), report_dic_test[1].get('recall')

    # Create confusion matrix
    test_cm = confusion_matrix(true_labels, predict_binary)

    print("\nTest Confusion Matrix:")
    print(test_cm)

    # Save confusion matrices to CSV
    pd.DataFrame(test_cm).to_csv(os.path.join(path_save, "test_confusion_matrix.csv"))

    # Save additional metrics to CSV including ROC AUC
    metrics_df = pd.DataFrame({
        'dataset': ['test'],
        'accuracy': [test_metrics['accuracy']],
        'sensitivity': [test_metrics['sensitivity']],
        'specificity': [test_metrics['specificity']],
        'precision': [test_metrics['precision']],
        'f1_score': [test_metrics['f1_score']],
        'roc_auc': [test_auc]  # Add ROC AUC to metrics dataframe
    })
    metrics_df.to_csv(os.path.join(path_save, "additional_metrics.csv"), index=False)

    # # Also save the raw probabilities for later ROC curve plotting
    # np.savetxt(os.path.join(path_save, "test_probabilities.csv"), scores)
    # np.savetxt(os.path.join(path_save, "test_true_labels.csv"), true_labels)
    # np.savetxt(os.path.join(path_save, "train_probabilities.csv"), train_scores)
    # np.savetxt(os.path.join(path_save, "train_true_labels.csv"), true_train_labels)


def load_my_data(group, data_path, predict_save_path, model_save_path, fold, max_length, b_size):
    data_dir = os.path.join(data_path, f"{group}/fold_{fold}")
    print(f"Data dir: {data_dir}")
    print(f"Predict save path: {predict_save_path}")
    print(f"Model save path: {model_save_path}")

    # Load data from .mat files
    print('Loading data...')
    x_train_path = os.path.join(data_dir, f'train/vectors.npy')
    y_train_path = os.path.join(data_dir, f'train/labels.npy')
    x_val_path = os.path.join(data_dir, f'test/vectors.npy')
    y_val_path = os.path.join(data_dir, f'test/labels.npy')
    print(f"x_train_path: {x_train_path}")
    print(f"y_train_path: {y_train_path}")
    print(f"x_val_path: {x_val_path}")
    print(f"y_val_path: {y_val_path}")

    train_matrix = np.load(x_train_path)
    train_label = np.load(y_train_path)
    test_matrix = np.load(x_val_path)
    test_label = np.load(y_val_path)

    print(f"train_matrix shape: {train_matrix.shape}")
    print(f"train_label shape: {train_label.shape}")
    print(f"test_matrix shape: {test_matrix.shape}")
    print(f"test_label shape: {test_label.shape}")

    print(f"Training data distribution")
    label_counts = Counter(train_label)
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

    train_num = train_label.shape[0]
    test_num = test_label.shape[0]

    print(f"Train samples: {train_num}, Test samples: {test_num}")

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

    return train_loader, test_loader

def run(device):
    lr_rate = 0.01
    b_size = 6144
    num_epochs = 100
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_result_dir = "dee_phage_after_fix_bug_contig_" + timestamp
    result_dir = "./result"
    data_path = "/content/drive/MyDrive/data"

    for i in range(1):

        if i == 0:
            min_length = 100
            max_length = 400
        elif i == 1:
            min_length = 400
            max_length = 800
        elif i == 2:
            min_length = 800
            max_length = 1200
        elif i == 3:
            min_length = 1200
            max_length = 1800
        elif i == 4:
            min_length = 50
            max_length = 100
            overlap = 10
        elif i == 5:
            min_length = 100
            max_length = 200
            overlap = 10
        elif i == 6:
            min_length = 200
            max_length = 300
            overlap = 10
        elif i == 7:
            min_length = 300
            max_length = 400
            overlap = 10
        else:
            raise ValueError(f"Invalid group: {i + 1}")

        group = f"{min_length}_{max_length}"
        for r in range(5):
            fold = r + 1
            print(f"\n{'=' * 50}")
            print(f"Starting fold {fold}/5")
            print(f"{'=' * 50}")

            # Path definitions
            path_save = os.path.join(result_dir, f"{exp_result_dir}/{group}/fold_{fold}/{timestamp}")
            os.makedirs(path_save, exist_ok=True)  # Ensure directory exists

            predict_save_path = os.path.join(path_save, f"{max_length}_{lr_rate}_{b_size}_prediction.csv")
            model_save_path = os.path.join(path_save, f"{max_length}_{lr_rate}_{b_size}_model.pt")

            train_loader, test_loader = load_my_data(group, data_path, predict_save_path, model_save_path, fold, max_length,
                                                     b_size)
            # train_loader, test_loader = load_dee_phage_data(group, fold, max_length, b_size)

            # Initialize model
            model = DeePhage(max_length)
            model.to(device)

            # Print model summary
            print(f"Model architecture: {model}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

            # Define loss function and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr_rate)
            scaler = GradScaler('cuda')
            # scaler = None

            # Create a learning rate scheduler
            total_steps = len(train_loader) * num_epochs
            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr_rate,
                total_steps=total_steps,
                pct_start=0.3,  # Spend 30% of training time increasing LR, 70% decreasing
                div_factor=25,  # initial_lr/div_factor = starting lr
                final_div_factor=1000,  # final_lr = initial_lr/final_div_factor
                anneal_strategy='cos'  # Use cosine annealing
            )

            # Initialize history logger
            history = TrainingHistory()
            early_stopping = EarlyStopping(patience=20, verbose=True, path=model_save_path.replace('.pt', '_best.pt'),
                                           log=log)
            start_executing(device, model, train_loader, test_loader, num_epochs, optimizer, criterion, scaler,
                            model_save_path, history, early_stopping, scheduler, is_training=True)

            # start_executing(device, model, train_loader, test_loader, 1, optimizer, criterion, scaler,
            #                 model_save_path, history, is_training=False)

            last_evaluation(device, model, test_loader, path_save, predict_save_path, model_save_path, group, fold)

            # Plot and save results
            history.history_plot('epoch', path_save, max_length, lr_rate, b_size)

        print("\nAll folds completed!")
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    run(device=device)
