import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from logger.phg_cls_log import log


# Existing DeePhage class remains unchanged
class DeePhage(nn.Module):
    def __init__(self, input_size=300, kernel_size=6, filters=64, pool_size=3, pool_stride=3, dropout_rate=0.3,
                 dense_units=64):
        """
        DeePhage model based on the paper, adjusted for pre-extracted features

        Args:
            input_size: Size of input features (300 in your case)
            kernel_size: Size of convolutional kernel
            filters: Number of convolutional filters
            pool_size: Size of max pooling window
            pool_stride: Stride of max pooling window
            dropout_rate: Dropout rate
            dense_units: Number of units in dense layer
        """
        super(DeePhage, self).__init__()

        # Convert 1D input to 2D for Conv1D by reshaping to (batch, 1, input_size)
        self.reshape = lambda x: x.unsqueeze(1)

        # Conv1D layer
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu_conv = nn.ReLU()

        # MaxPooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

        # BatchNorm1 layer
        self.bn1 = nn.BatchNorm1d(filters)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the size after conv+maxpool
        self.pool_output_size = (input_size // pool_stride)

        # GlobalPooling layer
        self.globalpool = nn.AdaptiveAvgPool1d(1)

        # Dense1 layer
        self.dense1 = nn.Linear(filters, dense_units)
        self.relu_dense = nn.ReLU()

        # BatchNorm2 layer
        self.bn2 = nn.BatchNorm1d(dense_units)

        # Dense2 layer (output)
        self.dense2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input for Conv1D
        x = self.reshape(x)

        # Conv1D layer
        x = self.conv1d(x)
        x = self.relu_conv(x)

        # MaxPooling layer
        x = self.maxpool(x)

        # BatchNorm1 layer
        x = self.bn1(x)

        # Dropout layer
        x = self.dropout(x)

        # GlobalPooling layer
        x = self.globalpool(x)
        x = x.squeeze(-1)  # Remove the last dimension

        # Dense1 layer
        x = self.dense1(x)
        x = self.relu_dense(x)

        # BatchNorm2 layer
        x = self.bn2(x)

        # Dense2 layer (output)
        x = self.dense2(x)
        x = self.sigmoid(x)

        return x


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix given true and predicted labels

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)

    Returns:
        Confusion matrix as numpy array
    """
    # Ensure they are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return cm


def plot_confusion_matrix(cm, epoch, save_path=None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        epoch: Current epoch number
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Temperate', 'Virulent'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - Epoch {epoch}')

    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix_epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()


def train_deephage(x_train, y_train, x_test, y_test, batch_size=32, epochs=10, learning_rate=0.0001, save_path=None):
    """
    Train the DeePhage model with confusion matrix tracking

    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Testing features
        y_test: Testing labels
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_path: Path to save confusion matrix plots (optional)

    Returns:
        Trained model and training history
    """
    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Print GPU information if available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3  # Convert to GB
        log.info(f"Using GPU: {gpu_name}")
        log.info(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
        log.info(f"GPU Memory Reserved: {memory_reserved:.2f} GB")

    model = DeePhage().to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'confusion_matrices': [],  # Add storage for confusion matrices
        'sensitivity': [],  # Add storage for sensitivity/recall
        'specificity': [],  # Add storage for specificity
        'roc_auc': []  # Add storage for AUC values
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Add progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]',
                          leave=False, ncols=100)

        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * inputs.size(0)
            train_loss += batch_loss

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            batch_correct = (predicted == targets).sum().item()
            train_total += targets.size(0)
            train_correct += batch_correct

            # Update progress bar
            batch_acc = batch_correct / inputs.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.4f}'})

        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        # Add progress bar for validation
        val_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs} [Valid]',
                        leave=False, ncols=100)

        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_loss = loss.item() * inputs.size(0)
                val_loss += batch_loss

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                batch_correct = (predicted == targets).sum().item()
                val_total += targets.size(0)
                val_correct += batch_correct

                # Collect predictions and targets for confusion matrix
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

                # Update progress bar
                batch_acc = batch_correct / inputs.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.4f}'})

        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(test_loader.dataset)
        val_acc = val_correct / val_total

        # Convert to numpy arrays for metrics calculation
        all_val_targets_np = np.array(all_val_targets).flatten()
        all_val_preds_np = np.array(all_val_preds).flatten().round()
        all_val_probs_np = np.array(all_val_preds).flatten()  # Raw probabilities for ROC

        # Calculate and log confusion matrix
        cm = calculate_confusion_matrix(all_val_targets_np, all_val_preds_np)

        # Store confusion matrix in history
        history['confusion_matrices'].append(cm)

        # Log confusion matrix values
        tn, fp, fn, tp = cm.ravel()
        log.info(f"\nConfusion Matrix (Epoch {epoch + 1}):")
        log.info(f"TN: {tn}, FP: {fp}")
        log.info(f"FN: {fn}, TP: {tp}")

        # Calculate precision, recall/sensitivity, specificity and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall for positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # recall for negative class
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_val_targets_np, all_val_probs_np)
        roc_auc = auc(fpr, tpr)

        log.info(f"Precision: {precision:.4f}")
        log.info(f"Sensitivity/Recall: {sensitivity:.4f}")
        log.info(f"Specificity: {specificity:.4f}")
        log.info(f"F1 Score: {f1:.4f}")
        log.info(f"AUC: {roc_auc:.4f}")

        # Plot confusion matrix
        plot_confusion_matrix(cm, epoch + 1, save_path)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - Epoch {epoch + 1}')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(f"{save_path}/roc_curve_epoch_{epoch + 1}.png")
            plt.close()
        else:
            plt.show()

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['sensitivity'] = history.get('sensitivity', []) + [sensitivity]
        history['specificity'] = history.get('specificity', []) + [specificity]
        history['roc_auc'] = history.get('roc_auc', []) + [roc_auc]

        log.info(f'Epoch {epoch + 1}/{epochs} - '
                 f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                 f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return model, history


# The rest of the functions remain the same
def evaluate_model(model, x_test, y_test, batch_size=32):
    """
    Evaluate the trained model

    Args:
        model: Trained DeePhage model
        x_test: Testing features
        y_test: Testing labels
        batch_size: Batch size for evaluation

    Returns:
        Accuracy, precision, recall, F1 score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print GPU information if available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3  # Convert to GB
        log.info(f"Using GPU: {gpu_name}")
        log.info(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
        log.info(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
    else:
        log.info("Using CPU for evaluation")

    # Convert numpy arrays to PyTorch tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create dataset and dataloader
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    # Collect all predictions and targets
    all_preds = []
    all_targets = []

    # Add progress bar for evaluation
    eval_pbar = tqdm(test_loader, desc='Evaluating', ncols=100)

    with torch.no_grad():
        for inputs, targets in eval_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Store predictions and targets
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)

    # Calculate binary predictions
    binary_preds = (all_preds > 0.5).astype(int)

    # Calculate metrics
    accuracy = np.mean(binary_preds == all_targets)

    # Calculate sensitivity (recall for positive class)
    positive_indices = all_targets == 1
    sensitivity = np.mean(binary_preds[positive_indices] == all_targets[positive_indices]) if np.any(
        positive_indices) else 0

    # Calculate specificity (recall for negative class)
    negative_indices = all_targets == 0
    specificity = np.mean(binary_preds[negative_indices] == all_targets[negative_indices]) if np.any(
        negative_indices) else 0

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    log.info(f"\nFinal Confusion Matrix:")
    log.info(f"TN: {tn}, FP: {fp}")
    log.info(f"FN: {fn}, TP: {tp}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Temperate', 'Virulent'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Final Confusion Matrix')
    plt.show()

    log.info(f"Accuracy: {accuracy:.4f}")
    log.info(f"Sensitivity (Recall for virulent): {sensitivity:.4f}")
    log.info(f"Specificity (Recall for temperate): {specificity:.4f}")
    log.info(f"AUC: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy, sensitivity, specificity, roc_auc


def plot_training_history(history):
    """
    Plot training and validation loss and accuracy

    Args:
        history: Training history dictionary
    """
    # Plot training & validation loss and accuracy
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot sensitivity and specificity
    plt.subplot(2, 2, 3)
    plt.plot(history['sensitivity'], 'g-', label='Sensitivity')
    plt.plot(history['specificity'], 'b-', label='Specificity')
    plt.title('Sensitivity & Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    # Plot ROC AUC
    plt.subplot(2, 2, 4)
    plt.plot(history['roc_auc'], 'r-', label='ROC AUC')
    plt.title('ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot confusion matrices evolution
    if 'confusion_matrices' in history:
        # Create a grid of subplots based on the number of epochs
        num_epochs = len(history['confusion_matrices'])
        num_cols = min(5, num_epochs)  # Max 5 columns
        num_rows = (num_epochs + num_cols - 1) // num_cols

        plt.figure(figsize=(16, 3 * num_rows))

        for i, cm in enumerate(history['confusion_matrices']):
            plt.subplot(num_rows, num_cols, i + 1)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['Temperate', 'Virulent'])
            disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=False)
            plt.title(f'Epoch {i + 1}')
            plt.tight_layout()

        plt.suptitle('Confusion Matrix Evolution', fontsize=16)
        plt.subplots_adjust(top=0.92, wspace=0.3, hspace=0.3)
        plt.show()


def save_model(model, path='deephage_model.pth'):
    """
    Save the trained model

    Args:
        model: Trained DeePhage model
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)
    log.info(f"Model saved to {path}")


def load_model(path='deephage_model.pth', input_size=300):
    """
    Load a saved model

    Args:
        path: Path to the saved model
        input_size: Input size for the model

    Returns:
        Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeePhage(input_size=input_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


if __name__ == '__main__':
    # Create directories for saving plots
    import os

    os.makedirs("confusion_matrices", exist_ok=True)
    os.makedirs("roc_curves", exist_ok=True)

    X_train = np.load("../word2vec_train_vector.npy")
    y_train = np.load("../y_train.npy")
    X_val = np.load("../word2vec_val_vector.npy")
    y_val = np.load("../y_val.npy")

    log.info(f"Train set shape: {X_train.shape}")
    log.info(f"Train labels shape: {y_train.shape}")
    log.info(f"Validation set shape: {X_val.shape}")
    log.info(f"Validation labels shape: {y_val.shape}")

    # Train the model with confusion matrix tracking
    model, history = train_deephage(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=32,
        epochs=10,
        learning_rate=0.0001,
        save_path="confusion_matrices"
    )

    # Plot training history including confusion matrices and metrics
    plot_training_history(history)

    # Save model
    save_model(model)

    log.info("Complete!")
