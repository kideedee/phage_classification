import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from experiment.custom_dataset import FCGRDataset
from experiment.resnet.resnet_model import ResNetBinaryClassifier
from logger.phg_cls_log import log


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_model_wts = model.state_dict()

    # Tính tổng số batch cho toàn bộ quá trình training
    total_batches = num_epochs * (len(train_loader) + len(val_loader))
    current_batch = 0

    # Single progress bar cho toàn bộ quá trình
    pbar = tqdm(total=total_batches, desc="Training", unit="batch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    log.info(f"\n{'=' * 80}")
    log.info(f"Starting Training - {num_epochs} epochs")
    log.info(f"{'=' * 80}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Detach loss để tránh warning
            running_loss += loss.detach().item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            current_batch += 1

            # Update progress bar
            pbar.set_postfix({
                'Epoch': f'{epoch + 1}/{num_epochs}',
                'Phase': 'Train',
                'Loss': f'{loss.detach().item():.4f}',
                'Batch': f'{batch_idx + 1}/{len(train_loader)}'
            })
            pbar.update(1)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                # Detach loss để tránh warning
                val_running_loss += loss.detach().item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                current_batch += 1

                # Update progress bar
                pbar.set_postfix({
                    'Epoch': f'{epoch + 1}/{num_epochs}',
                    'Phase': 'Val',
                    'Loss': f'{loss.detach().item():.4f}',
                    'Batch': f'{batch_idx + 1}/{len(val_loader)}'
                })
                pbar.update(1)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = val_running_corrects.double() / len(val_loader.dataset)

        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        log.info("Confusion Matrix Values:")
        log.info(f"True Negatives (TN):  {tn}")
        log.info(f"False Positives (FP): {fp}")
        log.info(f"False Negatives (FN): {fn}")
        log.info(f"True Positives (TP):  {tp}")
        log.info(f"Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)")
        log.info(f"Sensitivity:  {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
        log.info(f"Specificity:  {specificity:.4f} ({specificity * 100:.2f}%)")
        log.info(f"Precision:    {precision:.4f} ({precision * 100:.2f}%)")
        log.info(f"F1-Score:     {f1_score:.4f} ({f1_score * 100:.2f}%)")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = model.state_dict()

        # Save metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc.cpu().numpy())
        val_accs.append(epoch_val_acc.cpu().numpy())

        # Print epoch summary - Đây là phần chính để in metrics sau mỗi epoch
        log.info(f"\n--- Epoch {epoch + 1}/{num_epochs} Summary ---")
        log.info(
            f"Train Loss: {epoch_train_loss:.6f} | Train Acc: {epoch_train_acc:.4f} ({epoch_train_acc * 100:.2f}%)")
        log.info(f"Val Loss:   {epoch_val_loss:.6f} | Val Acc:   {epoch_val_acc:.4f} ({epoch_val_acc * 100:.2f}%)")

        # Hiển thị improvement
        if epoch_val_acc > best_val_acc:
            log.info(
                f"🎉 New best validation accuracy: {epoch_val_acc:.4f} (improved by {(epoch_val_acc - best_val_acc):.4f})")
        else:
            log.info(f"Best Val Acc so far: {best_val_acc:.4f}")

        # Learning rate info
        current_lr = optimizer.param_groups[0]['lr']
        log.info(f"Learning Rate: {current_lr:.2e}")
        log.info(f"{'-' * 50}")

        scheduler.step()

        # Update progress bar với thông tin tổng kết epoch
        pbar.set_postfix({
            'Epoch': f'{epoch + 1}/{num_epochs}',
            'Train_Loss': f'{epoch_train_loss:.4f}',
            'Train_Acc': f'{epoch_train_acc:.4f}',
            'Val_Loss': f'{epoch_val_loss:.4f}',
            'Val_Acc': f'{epoch_val_acc:.4f}',
            'Best_Val': f'{best_val_acc:.4f}'
        })

    pbar.close()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Print final results
    log.info(f"{'=' * 80}")
    log.info(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    log.info(f"Final train accuracy: {train_accs[-1]:.4f} ({train_accs[-1] * 100:.2f}%)")
    log.info(f"Final train loss: {train_losses[-1]:.6f}")
    log.info(f"Final val loss: {val_losses[-1]:.6f}")
    log.info(f"{'=' * 80}")

    # Optional: Log to file as well
    if 'log' in globals():
        log.info(f"Training completed!")
        log.info(f"Best validation accuracy: {best_val_acc:.4f}")
        log.info(f"Final metrics - Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    log.info(f'Test Accuracy: {accuracy:.4f}')
    log.info(f'Test Precision: {precision:.4f}')
    log.info(f'Test Recall: {recall:.4f}')
    log.info(f'Test F1-score: {f1:.4f}')

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir):
    import os

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    log.info(f"Training history plot saved to: {save_path}")

    plt.show()


# Data transforms
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def init_data_loader(data_path, batch_size):
    dataset = FCGRDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True,
                        prefetch_factor=4)
    return loader


def run(data_dir, output_data_dir):
    BATCH_SIZE = 512
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    MODEL_NAME = 'resnet101'  # resnet18, resnet34, resnet50, resnet101

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')
    if torch.cuda.is_available():
        log.info(f'GPU: {torch.cuda.get_device_name(0)}')

    log.info(f"Loading data from {data_dir}")

    train_path = os.path.join(data_dir, "train/data.h5")
    test_path = os.path.join(data_dir, "test/data.h5")

    train_loader = init_data_loader(train_path, BATCH_SIZE)
    test_loader = init_data_loader(test_path, BATCH_SIZE)

    model = ResNetBinaryClassifier(
        model_name=MODEL_NAME,
        pretrained=True,
        freeze_backbone=False  # Set True để freeze backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training
    log.info("Starting training...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device
    )

    log.info("\nEvaluating...")
    metrics = evaluate_model(model, test_loader, device)
    df = pd.DataFrame([metrics])
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    df.to_csv(os.path.join(output_data_dir, "result.csv"), index=False)

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, output_data_dir)


if __name__ == "__main__":
    # root_data_dir = config.HDFS_FCGR_EMBEDDING_OUTPUT_DIR
    root_data_dir = "C:\\Users\Admin\Temp"
    for i in range(4):
        group = i + 1
        if group == 1:
            min_size = 100
            max_size = 400
            overlap = 10
        elif group == 2:
            min_size = 400
            max_size = 800
            overlap = 10
        elif group == 3:
            min_size = 800
            max_size = 1200
            overlap = 30
        else:
            min_size = 1200
            max_size = 1800
            overlap = 30

        for j in range(5):
            fold = j + 1

            if group == 3:
                data_dir = os.path.join(root_data_dir, f"{min_size}_{max_size}/fold_{fold}")
                output_data_dir = f"./{min_size}_{max_size}/fold_{fold}"
                run(data_dir, output_data_dir)
            else:
                continue
