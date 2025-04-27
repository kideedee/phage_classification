#!/usr/bin/env python
import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Import tqdm for progress bars

from common.env_config import config
from logger.phg_cls_log import setup_logger

log = setup_logger(inspect.getfile(inspect.currentframe()))


def classification_report_csv(report, path_save, c):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()
        if len(row_data) < 5:  # Skip empty lines
            continue

        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)

    if c == 0:
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(path_save + 'classification_report.csv', index=False)
    else:
        log.info('train')

    return report_data


# Function to calculate sensitivity and specificity
def calculate_sensitivity_specificity(y_true, y_pred):
    """
    Calculate sensitivity and specificity from true labels and predictions.

    Args:
        y_true: True labels (ground truth)
        y_pred: Predicted labels

    Returns:
        sensitivity, specificity
    """
    # Create confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also known as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity


# Define the CNN model in PyTorch
class DeePhageModel(nn.Module):
    def __init__(self, max_length):
        super(DeePhageModel, self).__init__()

        # Convolutional layer
        self.conv1 = nn.Conv1d(4, 64, kernel_size=6, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)

        # Dense layers
        self.fc1 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # PyTorch uses (batch_size, channels, seq_length) format
        # while Keras used (batch_size, seq_length, channels)
        # so we need to transpose
        x = x.permute(0, 2, 1)

        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.dropout(x)

        # Global Average Pooling (equivalent to Keras GlobalAveragePooling1D)
        x = torch.mean(x, dim=2)

        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


# Updated Training history class to include sensitivity and specificity
class TrainingHistory:
    def __init__(self):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'epoch': []}
        # Add new metrics
        self.sensitivity = {'epoch': []}
        self.specificity = {'epoch': []}

    def add_batch_history(self, loss, acc):
        self.losses['batch'].append(loss)
        self.accuracy['batch'].append(acc)

    def add_epoch_history(self, loss, acc, val_loss, val_acc, sensitivity=None, specificity=None):
        self.losses['epoch'].append(loss)
        self.accuracy['epoch'].append(acc)
        self.val_loss['epoch'].append(val_loss)
        self.val_acc['epoch'].append(val_acc)

        # Add sensitivity and specificity if provided
        if sensitivity is not None:
            self.sensitivity['epoch'].append(sensitivity)
        if specificity is not None:
            self.specificity['epoch'].append(specificity)

    def loss_plot(self, loss_type, accuracy, viru_acc, temp_acc, train_viru_acc, train_temp_acc,
                  path_save, max_length, lr_rate, b_size):
        iters = range(len(self.losses[loss_type]))

        # Create figure with 2 subplots (one for acc/loss, one for sensitivity/specificity)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot accuracy and loss
        ax1.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        ax1.plot(iters, self.losses[loss_type], 'g', label='train loss')

        if loss_type == 'epoch':
            ax1.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            ax1.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        ax1.grid(True)
        ax1.set_ylim((0, 2))
        ax1.set_xlabel(loss_type)
        ax1.set_ylabel('acc-loss')
        ax1.set_title('%s test_acc: %s \n test--- viru: %s temp: %s \ntrain---viru: %s temp: %s'
                      % (str(max_length), accuracy, viru_acc, temp_acc,
                         train_viru_acc, train_temp_acc))
        ax1.legend(loc="upper right")

        # Plot sensitivity and specificity if available
        if hasattr(self, 'sensitivity') and loss_type == 'epoch' and len(self.sensitivity['epoch']) > 0:
            ax2.plot(iters, self.sensitivity['epoch'], 'b', label='sensitivity')
            ax2.plot(iters, self.specificity['epoch'], 'r', label='specificity')
            ax2.grid(True)
            ax2.set_ylim((0, 1.1))
            ax2.set_xlabel(loss_type)
            ax2.set_ylabel('value')
            ax2.set_title('Sensitivity and Specificity')
            ax2.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '.png')
        plt.close()

    # New method to plot sensitivity and specificity history
    def sensitivity_specificity_plot(self, path_save, max_length, lr_rate, b_size):
        if len(self.sensitivity['epoch']) > 0:
            epochs = range(1, len(self.sensitivity['epoch']) + 1)

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.sensitivity['epoch'], 'b', label='Sensitivity')
            plt.plot(epochs, self.specificity['epoch'], 'r', label='Specificity')

            plt.grid(True)
            plt.ylim((0, 1.1))
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.title(f'Sensitivity and Specificity Over Epochs (Length: {max_length})')
            plt.legend(loc="lower right")

            # Save the plot
            plt.savefig(path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '_sens_spec.png')
            plt.close()


# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).float()
    correct = (y_pred_binary == y_true).sum().item()
    return correct / len(y_true)


def main():
    # GPU setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Configuration parameters
    max_length = 400  # You can adjust this list for different sequence lengths
    lr_rate = 0.0001
    b_size = 32

    # Path for saving results
    path_save = './results/'
    os.makedirs(path_save, exist_ok=True)

    # Path to your .npy files
    data_path = os.path.join(config.DATA_DIR, "my_data/one_hot/100_400")  # Change this to your data directory

    predict_save_path = path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '_prediction.csv'
    model_save_path = path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '_model.pt'
    # Add a new path for saving metrics history
    metrics_save_path = path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '_metrics.csv'

    # Load data from NPY files
    log.info('Loading training data...')
    x_train = np.load(os.path.join(data_path, 'one_hot_train_vector.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))

    log.info('Loading validation data...')
    x_val = np.load(os.path.join(data_path, 'one_hot_val_vector.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))

    # Print shapes for debugging
    log.info(f"x_train shape: {x_train.shape}")
    log.info(f"y_train shape: {y_train.shape}")
    log.info(f"x_val shape: {x_val.shape}")
    log.info(f"y_val shape: {y_val.shape}")
    log.info(f"train distribution: {np.bincount(y_train.flatten())}")
    log.info(f"y_train distribution: {np.bincount(y_train.flatten())}")

    index_array = np.arange(x_train.shape[0]).reshape(-1, 1)  # Generate indices and reshape to 2D array
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    index_resampled, y_resampled = undersampler.fit_resample(index_array, y_train)
    X_resampled = x_train[index_resampled.flatten()]
    log.info(f"X_resampled shape: {X_resampled.shape}")
    log.info(f"y_resampled shape: {y_resampled.shape}")
    log.info(f"Distribution of classes in resampled dataset: {pd.Series(y_resampled).value_counts()}")

    # Ensure y_train and y_val are 2D (for BCE loss)
    if len(y_resampled.shape) == 1:
        y_resampled = y_resampled.reshape(-1, 1)
    if len(y_resampled.shape) == 1:
        y_resampled = y_resampled.reshape(-1, 1)

    # Extract dataset sizes
    train_num = y_resampled.shape[0]
    test_num = y_val.shape[0]

    # Ensure the shapes are correct for our model
    if len(X_resampled.shape) == 3 and X_resampled.shape[2] == 4:
        # Data is already in the right shape (samples, seq_length, features)
        log.info("Data is already correctly shaped")
    else:
        log.info("Reshaping data...")
        # Reshape if needed - adjust this based on your actual data shape
        if len(X_resampled.shape) == 2:  # If data is flat
            X_resampled = X_resampled.reshape(train_num, max_length, 4)
            x_val = x_val.reshape(test_num, max_length, 4)

    # Convert data to PyTorch tensors
    train_matrix_tensor = torch.FloatTensor(X_resampled)
    train_label_tensor = torch.FloatTensor(y_resampled)
    test_matrix_tensor = torch.FloatTensor(x_val)
    test_label_tensor = torch.FloatTensor(y_val)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(train_matrix_tensor, train_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, num_workers=6, prefetch_factor=6)

    # Create DataLoader for validation data to avoid memory issues
    val_dataset = TensorDataset(test_matrix_tensor, test_label_tensor)
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False, num_workers=6, prefetch_factor=6)

    # Create the model
    model = DeePhageModel(max_length).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # Training history
    history = TrainingHistory()

    # Create dataframe to store metrics
    metrics_df = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'sensitivity', 'specificity'])

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Add a tqdm progress bar for tracking batches within each epoch
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                  bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = model(data)

                # Calculate loss
                loss = criterion(output, target)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track statistics
                train_loss += loss.detach().item()
                pred = (output > 0.5).float()
                batch_correct = (pred == target).sum().item()
                batch_total = target.size(0)
                train_correct += batch_correct
                train_total += batch_total

                # Add batch history
                batch_acc = train_correct / train_total
                history.add_batch_history(loss.item(), batch_acc)

                # Update progress bar with batch info
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'batch_acc': f'{batch_correct / batch_total:.4f}',
                    'avg_loss': f'{train_loss / (batch_idx + 1):.4f}',
                    'avg_acc': f'{train_correct / train_total:.4f}'
                })
                pbar.update(1)

        # Evaluate on validation set using batches to avoid memory issues
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            log.info("Evaluating on validation set...")
            # Use validation data loader to process in batches
            val_preds = []
            val_targets = []

            with tqdm(total=len(val_loader), desc="Validation", bar_format='{l_bar}{bar:30}{r_bar}') as val_pbar:
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)

                    # Forward pass
                    output = model(data)

                    # Calculate batch loss
                    batch_loss = criterion(output, target.unsqueeze(-1)).item()
                    val_loss += batch_loss * target.size(0)  # Weight by batch size

                    # Calculate batch accuracy
                    pred = (output > 0.5).float()
                    batch_correct = (pred == target).sum().item()
                    batch_total = target.size(0)
                    val_correct += batch_correct
                    val_total += batch_total

                    # Store predictions and targets for later classification report
                    val_preds.append(pred.cpu())
                    val_targets.append(target.cpu())

                    # Update progress bar
                    val_pbar.set_postfix({
                        'batch_loss': f'{batch_loss:.4f}',
                        'batch_acc': f'{batch_correct / batch_total:.4f}',
                        'avg_loss': f'{val_loss / val_total:.4f}',
                        'avg_acc': f'{val_correct / val_total:.4f}'
                    })
                    val_pbar.update(1)

            # Normalize the total validation loss
            val_loss = val_loss / val_total

            # Calculate sensitivity and specificity for the epoch
            all_val_preds = torch.cat(val_preds, dim=0).numpy().flatten()
            all_val_targets = torch.cat(val_targets, dim=0).numpy().flatten()

            sensitivity, specificity = calculate_sensitivity_specificity(
                all_val_targets,
                (all_val_preds > 0.5).astype(int)
            )

        # Add epoch history
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        history.add_epoch_history(
            train_loss / len(train_loader),
            train_acc,
            val_loss,
            val_acc,
            sensitivity,
            specificity
        )

        # Create a new DataFrame for the current epoch
        new_metrics = pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }])

        # Drop empty or all-NA columns before concatenation
        new_metrics = new_metrics.dropna(how='all', axis=1)

        # Concatenate with the existing metrics DataFrame
        metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)

        log.info(f'Epoch {epoch + 1}/{num_epochs}, '
                 f'Train Loss: {train_loss / len(train_loader):.4f}, '
                 f'Train Acc: {train_acc:.4f}, '
                 f'Val Loss: {val_loss:.4f}, '
                 f'Val Acc: {val_acc:.4f}, '
                 f'Sensitivity: {sensitivity:.4f}, '
                 f'Specificity: {specificity:.4f}')

    # Save metrics history
    metrics_df.to_csv(metrics_save_path, index=False)
    log.info(f"Metrics history saved to {metrics_save_path}")

    # Final evaluation using batches to avoid memory issues
    model.eval()
    log.info("Performing final evaluation...")
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_test_preds = []
    all_test_outputs = []
    all_test_targets = []

    with torch.no_grad():
        # Process validation data in batches
        with tqdm(total=len(val_loader), desc="Final Evaluation", bar_format='{l_bar}{bar:30}{r_bar}') as test_pbar:
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)

                # Calculate batch loss
                batch_loss = criterion(output, target).item()
                test_loss += batch_loss * target.size(0)

                # Calculate batch accuracy
                pred = (output > 0.5).float()
                batch_correct = (pred == target).sum().item()
                batch_total = target.size(0)
                test_correct += batch_correct
                test_total += batch_total

                # Store predictions, outputs and targets for later use
                all_test_preds.append(pred.cpu())
                all_test_outputs.append(output.cpu())
                all_test_targets.append(target.cpu())

                # Update progress bar
                test_pbar.set_postfix({
                    'batch_loss': f'{batch_loss:.4f}',
                    'batch_acc': f'{batch_correct / batch_total:.4f}',
                    'avg_loss': f'{test_loss / test_total:.4f}',
                    'avg_acc': f'{test_correct / test_total:.4f}'
                })
                test_pbar.update(1)

        # Calculate overall test accuracy and loss
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total

        # Combine all predictions and outputs
        all_test_outputs = torch.cat(all_test_outputs, dim=0).numpy()
        all_test_preds = torch.cat(all_test_preds, dim=0).numpy()
        all_test_targets = torch.cat(all_test_targets, dim=0).numpy()

        # Calculate final sensitivity and specificity
        test_pred_binary = (all_test_preds > 0.5).astype(int)
        final_sensitivity, final_specificity = calculate_sensitivity_specificity(
            all_test_targets.flatten(),
            test_pred_binary.flatten()
        )

        log.info(f"Final Test Results:")
        log.info(f"Accuracy: {test_acc:.4f}")
        log.info(f"Sensitivity: {final_sensitivity:.4f}")
        log.info(f"Specificity: {final_specificity:.4f}")

        # Save predictions
        np.savetxt(predict_save_path, all_test_outputs)

        # Save model
        torch.save(model.state_dict(), model_save_path)

        # Generate classification reports
        # Convert to 1D arrays for classification report if needed
        y_val_flat = y_val.flatten() if len(y_val.shape) > 1 else y_val
        test_pred_flat = test_pred_binary.flatten() if len(test_pred_binary.shape) > 1 else test_pred_binary

        report_test = classification_report(y_val_flat, test_pred_flat)
        log.info('test')
        log.info(report_test)
        report_dic_test = classification_report_csv(report_test, path_save, 0)

        # Handle possible differences in report structure
        if len(report_dic_test) >= 2:
            temp_acc, viru_acc = report_dic_test[0].get('recall'), report_dic_test[1].get('recall')
        else:
            # If there's only one class in the report
            temp_acc = report_dic_test[0].get('recall')
            viru_acc = temp_acc  # Default to the same value

        # Also get train predictions for report using batches
        model.eval()
        all_train_preds = []
        all_train_targets = []

        with torch.no_grad():
            log.info("Evaluating on training set...")
            # Process training data in batches to avoid memory issues
            with tqdm(total=len(train_loader), desc="Train Evaluation",
                      bar_format='{l_bar}{bar:30}{r_bar}') as train_eval_pbar:
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)

                    # Forward pass
                    output = model(data)

                    # Calculate batch predictions
                    pred = (output > 0.5).float()

                    # Store predictions and targets
                    all_train_preds.append(pred.cpu())
                    all_train_targets.append(target.cpu())

                    train_eval_pbar.update(1)

            # Combine all predictions and targets
            all_train_preds = torch.cat(all_train_preds, dim=0).numpy()
            all_train_targets = torch.cat(all_train_targets, dim=0).numpy()

            # Convert to binary predictions for classification report
            train_pred_binary = all_train_preds.astype(int)

            # Convert to 1D arrays for classification report if needed
            y_train_flat = all_train_targets.flatten() if len(all_train_targets.shape) > 1 else all_train_targets
            train_pred_flat = train_pred_binary.flatten() if len(train_pred_binary.shape) > 1 else train_pred_binary

            report_train = classification_report(y_train_flat, train_pred_flat)
            log.info('train')
            log.info(report_train)
            report_dic_train = classification_report_csv(report_train, path_save, 1)

            # Handle possible differences in report structure
            if len(report_dic_train) >= 2:
                train_temp_acc, train_viru_acc = report_dic_train[0].get('recall'), report_dic_train[1].get(
                    'recall')
            else:
                # If there's only one class in the report
                train_temp_acc = report_dic_train[0].get('recall')
                train_viru_acc = train_temp_acc  # Default to the same value

        # Plot training history
        history.loss_plot('epoch', test_acc, viru_acc, temp_acc, train_viru_acc, train_temp_acc,
                          path_save, max_length, lr_rate, b_size)

        # Also plot the sensitivity/specificity history
        history.sensitivity_specificity_plot(path_save, max_length, lr_rate, b_size)

        log.info(f"Training completed. Model saved to {model_save_path}")
        log.info(f"Predictions saved to {predict_save_path}")


if __name__ == "__main__":
    main()
