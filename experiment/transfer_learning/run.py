import gc
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, AutoTokenizer

from common.env_config import config
from experiment.transfer_learning.models import create_custom_config, Dnabert2CnnModelNoK3, Dnabert2PoolingOnly, \
    Dnabert2CnnAblationNoK5, Dnabert2CnnNoK7Model
from logger.phg_cls_log import transfer_learning_dnabert_2_log as log

warnings.filterwarnings('ignore')


def train_two_phase_transfer_learning(model, tokenizer, tokenized_train, tokenized_val, device,
                                      warmup_epochs=10, finetune_epochs=5,
                                      warmup_lr=1e-3, finetune_lr=1e-5,
                                      patience=5, batch_size=32):
    """
    Optimized two-phase transfer learning with speed improvements
    """
    # Use AMP for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    criterion = nn.CrossEntropyLoss()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dataloader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_dataloader = DataLoader(
        tokenized_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Enhanced training history
    history = {
        'train_loss': [], 'train_acc': [], 'val_acc': [],
        'val_f1': [], 'val_precision': [], 'val_recall': [],
        'val_specificity': [], 'val_sensitivity': [], 'val_g_mean': [],
        'phase': [], 'lr': []
    }

    # ===============================
    # PHASE 1: WARM-UP (Freeze DNABERT-2)
    # ===============================
    log.info("PHASE 1: WARM-UP - Training CNN layers only")

    # Freeze DNABERT-2
    model.freeze_dnabert()

    # Optimizer for CNN parameters only
    cnn_params = [param for name, param in model.named_parameters()
                  if 'dnabert' not in name and param.requires_grad]

    optimizer_warmup = torch.optim.AdamW(
        cnn_params,
        lr=warmup_lr,
        weight_decay=1e-4,
        eps=1e-6,
        betas=(0.9, 0.999)
    )

    # Simple scheduler for single epoch
    scheduler_warmup = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_warmup,
        max_lr=warmup_lr,
        steps_per_epoch=len(train_dataloader),
        epochs=warmup_epochs,
        pct_start=0.3,
        div_factor=5,
        final_div_factor=10
    )

    for epoch in range(warmup_epochs):
        # Training
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        desc = f"Warm-up Epoch {epoch + 1}/{warmup_epochs}"
        progress_bar = tqdm(train_dataloader, desc=desc, leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer_warmup.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_warmup)
                torch.nn.utils.clip_grad_norm_(cnn_params, max_norm=1.0)
                scaler.step(optimizer_warmup)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn_params, max_norm=1.0)
                optimizer_warmup.step()

            scheduler_warmup.step()

            # Accumulate stats
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct_predictions / total_predictions:.4f}'
                })

        # Single validation at end of warmup
        val_acc, val_loss, val_metrics = validate_model_optimized(
            model, val_dataloader, device, criterion, desc, scaler
        )

        train_acc = correct_predictions / total_predictions
        avg_loss = total_loss / len(train_dataloader)
        current_lr = scheduler_warmup.get_last_lr()[0]

        # Save history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_g_mean'].append(val_metrics['g_mean'])
        history['phase'].append('warmup')
        history['lr'].append(current_lr)

        # Simple logging
        log.info(f"Warm-up Epoch {epoch + 1}/{warmup_epochs}")
        log.info(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        log.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        log.info(f"  Learning Rate: {current_lr:.6f}")

        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        log.info("-" * 50)

    # Initialize best metrics after warmup
    best_val_acc = val_acc
    best_metrics = val_metrics.copy()
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    log.info(f"Warm-up phase completed. Accuracy: {val_metrics['accuracy']:.4f}")

    # ===============================
    # PHASE 2: FINE-TUNE (Unfreeze DNABERT-2)
    # ===============================
    log.info("PHASE 2: FINE-TUNE - Training entire model")

    model.unfreeze_dnabert()

    # Optimized optimizer with differential learning rates
    dnabert_params = [param for name, param in model.named_parameters() if 'dnabert' in name]
    cnn_params = [param for name, param in model.named_parameters() if 'dnabert' not in name]

    optimizer_finetune = torch.optim.AdamW([
        {'params': dnabert_params, 'lr': finetune_lr, 'weight_decay': 1e-5},
        {'params': cnn_params, 'lr': finetune_lr * 10, 'weight_decay': 1e-4}
    ], eps=1e-6, betas=(0.9, 0.999))

    scheduler_finetune = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_finetune,
        max_lr=[finetune_lr, finetune_lr * 10],
        steps_per_epoch=len(train_dataloader),
        epochs=finetune_epochs,
        pct_start=0.1,
        div_factor=5,
        final_div_factor=50
    )

    patience_counter = 0

    for epoch in range(finetune_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        desc = f"Fine-tune Epoch {epoch + 1}/{finetune_epochs}"
        progress_bar = tqdm(train_dataloader, desc=desc, leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer_finetune.zero_grad(set_to_none=True)

            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_finetune)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_finetune)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_finetune.step()

            scheduler_finetune.step()

            # Stats
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct_predictions / total_predictions:.4f}'
                })

        # Full validation
        val_acc, val_loss, val_metrics = validate_model_optimized(
            model, val_dataloader, device, criterion, desc, scaler
        )

        train_acc = correct_predictions / total_predictions
        avg_loss = total_loss / len(train_dataloader)

        # Save history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_g_mean'].append(val_metrics['g_mean'])
        history['phase'].append('finetune')
        history['lr'].append(scheduler_finetune.get_last_lr())

        log.info(f"Fine-tune Epoch {epoch + 1}/{finetune_epochs}")
        log.info(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        log.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        log.info(f"  DNABERT-2 LR: {scheduler_finetune.get_last_lr()[0]:.6f}")
        log.info(f"  CNN LR: {scheduler_finetune.get_last_lr()[1]:.6f}")

        print_detailed_metrics(val_metrics, "Fine-tune", epoch + 1)

        # Best model tracking
        if val_metrics['accuracy'] > best_metrics['accuracy']:
            best_val_acc = val_acc
            best_metrics = val_metrics.copy()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            log.info(f"New best accuracy: {val_metrics['accuracy']:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping triggered (patience={patience})")
                break

        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        log.info("-" * 50)

    # Load best model
    if best_model_state is not None:
        # Move state dict back to device
        device_state_dict = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(device_state_dict)
        log.info(f"Loaded best model:")
        print_detailed_metrics(best_metrics, "Best Model")

    log.info("Two-phase transfer learning completed!")

    return {
        'best_val_acc': best_val_acc,
        'best_metrics': best_metrics,
        'history': history,
        'model_state': best_model_state
    }


def save_best_metrics_to_csv(results, config_run, csv_file='training_results.csv'):
    """
    Save best metrics to CSV file with append mode
    """
    import pandas as pd
    import os
    from datetime import datetime

    # Prepare data row
    best_metrics = results['best_metrics']

    data_row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'DNABERT2_CNN_Model',
        'train_sample': config_run['train_sample'],
        'val_sample': config_run['val_sample'],
        'n_classes': 2,
        'group': config_run['group'],
        'fold': config_run['fold'],
        'batch_size': config_run['batch_size'],
        'warmup_epochs': config_run['warmup_epochs'],
        'finetune_epochs': config_run['finetune_epochs'],
        'warmup_lr': config_run['warmup_lr'],
        'finetune_lr': config_run['finetune_lr'],
        'best_val_acc': results['best_val_acc'],
        'best_accuracy': best_metrics['accuracy'],
        'best_f1': best_metrics['f1'],
        'best_precision': best_metrics['precision'],
        'best_recall': best_metrics['recall'],
        'best_specificity': best_metrics['specificity'],
        'best_sensitivity': best_metrics['sensitivity'],
        'best_g_mean': best_metrics['g_mean'],
        'roc_auc': best_metrics['roc_auc'],
        'tn': best_metrics['tn'],
        'fp': best_metrics['fp'],
        'fn': best_metrics['fn'],
        'tp': best_metrics['tp'],
        'total_epochs': len(results['history']['train_loss']),
        'warmup_epochs_done': len([p for p in results['history']['phase'] if p == 'warmup']),
        'finetune_epochs_done': len([p for p in results['history']['phase'] if p == 'finetune'])
    }

    # Convert to DataFrame
    df_new = pd.DataFrame([data_row])

    # Check if file exists
    if os.path.exists(csv_file):
        # File exists, append without header
        df_new.to_csv(csv_file, mode='a', header=False, index=False)
        log.info(f"Metrics appended to existing file: {csv_file}")
    else:
        # File doesn't exist, create with header
        df_new.to_csv(csv_file, mode='w', header=True, index=False)
        log.info(f"New CSV file created: {csv_file}")

    log.info(f"Best metrics saved: F1={best_metrics['f1']:.4f}, Acc={best_metrics['accuracy']:.4f}")


def print_detailed_metrics(metrics, phase="", epoch=None):
    """
    In ra detailed metrics một cách có tổ chức

    Args:
        metrics: dict chứa các chỉ số đánh giá
        phase: phase name (warmup/finetune)
        epoch: epoch number
    """
    if epoch is not None:
        log.info(f"{phase} Epoch {epoch} - Detailed Metrics:")
    else:
        log.info(f"{phase} - Detailed Metrics:")

    log.info("-" * 50)
    log.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    log.info(f"  Precision:   {metrics['precision']:.4f}")
    log.info(f"  Recall:      {metrics['recall']:.4f}")
    log.info(f"  F1-Score:    {metrics['f1']:.4f}")
    log.info(f"  Specificity: {metrics['specificity']:.4f}")
    log.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    log.info(f"  G-Mean:      {metrics['g_mean']:.4f}")
    log.info(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    log.info("-" * 50)
    log.info("  Confusion Matrix:")
    log.info(f"    TN: {int(metrics['tn']):<8} FP: {int(metrics['fp'])}")
    log.info(f"    FN: {int(metrics['fn']):<8} TP: {int(metrics['tp'])}")
    log.info("-" * 50)


def calculate_binary_metrics(all_labels, all_predictions, all_probabilities=None):
    # Đảm bảo labels và predictions là numpy arrays
    if not isinstance(all_labels, np.ndarray):
        all_labels = np.array(all_labels, dtype=np.int32)
    if not isinstance(all_predictions, np.ndarray):
        all_predictions = np.array(all_predictions, dtype=np.int32)
    if all_probabilities is not None and not isinstance(all_probabilities, np.ndarray):
        all_probabilities = np.array(all_probabilities, dtype=np.float32)

    # Flatten arrays
    all_labels = all_labels.flatten()
    all_predictions = all_predictions.flatten()
    if all_probabilities is not None:
        all_probabilities = all_probabilities.flatten()

    # Tính confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Xử lý trường hợp chỉ có 1 class trong predictions
    if cm.shape == (1, 1):
        if all_labels[0] == 0:  # Chỉ có class 0
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:  # Chỉ có class 1
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()

    # Tính các chỉ số
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )

    acc = accuracy_score(all_labels, all_predictions)

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Sensitivity = Recall = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # G-mean = sqrt(Sensitivity * Specificity)
    gmean = np.sqrt(sensitivity * specificity) if (sensitivity > 0 and specificity > 0) else 0

    # Calculate ROC AUC if probabilities are provided
    roc_auc = 0.0
    if all_probabilities is not None:
        try:
            # Check if we have both classes in labels
            unique_labels = np.unique(all_labels)
            if len(unique_labels) > 1:
                roc_auc = roc_auc_score(all_labels, all_probabilities)
            else:
                # If only one class is present, ROC AUC is undefined
                roc_auc = 0.0
        except ValueError:
            # Handle any edge cases where ROC AUC cannot be computed
            roc_auc = 0.0

    result = {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'g_mean': float(gmean),
        'roc_auc': float(roc_auc),
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'tp': float(tp)
    }

    return result


def validate_model_optimized(model, val_dataloader, device, criterion, desc, scaler=None):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc=f"Validate, {desc}", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predictions and probabilities
            _, predicted = torch.max(outputs.data, 1)
            probabilities = F.softmax(outputs, dim=1)[:, 1]

            # Move to CPU and convert to numpy for efficiency
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / len(val_dataloader)
    detailed_metrics = calculate_binary_metrics(all_labels, all_predictions, all_probabilities)

    return detailed_metrics['accuracy'], avg_loss, detailed_metrics


def plot_training_history(history, save_path=None):
    """Plot training curves and optionally save to file"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(len(history['train_loss']))

    # Training & Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Training & Validation Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'g-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Phase visualization
    phases = history['phase']
    warmup_epochs = [i for i, p in enumerate(phases) if p == 'warmup']
    finetune_epochs = [i for i, p in enumerate(phases) if p == 'finetune']

    # Validation accuracy with phase colors
    axes[1, 0].plot([epochs[i] for i in warmup_epochs],
                    [history['val_acc'][i] for i in warmup_epochs],
                    'bo-', label='Warm-up', linewidth=2, markersize=6)
    axes[1, 0].plot([epochs[i] for i in finetune_epochs],
                    [history['val_acc'][i] for i in finetune_epochs],
                    'ro-', label='Fine-tune', linewidth=2, markersize=6)
    axes[1, 0].set_title('Validation Accuracy by Phase', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Learning rate schedule
    if len(warmup_epochs) > 0 and len(finetune_epochs) > 0:
        axes[1, 1].semilogy([epochs[i] for i in warmup_epochs],
                            [history['lr'][i] for i in warmup_epochs],
                            'bo-', label='Warm-up LR', linewidth=2, markersize=6)
        # For fine-tune, we have multiple LRs, so take the first one
        finetune_lrs = [lr[0] if isinstance(lr, list) else lr for i, lr in enumerate(history['lr']) if
                        i in finetune_epochs]
        axes[1, 1].semilogy([epochs[i] for i in finetune_epochs],
                            finetune_lrs,
                            'ro-', label='Fine-tune LR (DNABERT)', linewidth=2, markersize=6)

    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate (log scale)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()

    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")

    plt.show()


def main():
    log.info("=" * 60)
    log.info("DNABERT-2 + CNN Transfer Learning Pipeline")

    # Configuration
    config_run = {
        'warmup_epochs': 1,
        'finetune_epochs': 10,
        'warmup_lr': 2e-3,
        'finetune_lr': 1e-5,
        'random_state': 42
    }

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_folder = f"./result/cnn/{timestamp}"

    for i in range(0, 4):
        if i == 0:
            min_size = 100
            max_size = 400
            batch_size = 64
        elif i == 1:
            min_size = 400
            max_size = 800
            batch_size = 48
        elif i == 2:
            min_size = 800
            max_size = 1200
            batch_size = 16
        elif i == 3:
            min_size = 1200
            max_size = 1800
            batch_size = 16
        elif i == 4:
            min_size = 50
            max_size = 100
            batch_size = 192
        elif i == 5:
            min_size = 100
            max_size = 200
            batch_size = 96
        elif i == 6:
            min_size = 200
            max_size = 300
            batch_size = 64
        elif i == 7:
            min_size = 300
            max_size = 400
            batch_size = 64
        else:
            raise ValueError

        group = f"{min_size}_{max_size}"
        if max_size == 1800:
            tokenizer_max_length = 512
        else:
            tokenizer_max_length = max_size / 4

        for j in range(5):
            fold = j + 1

            # if i < 2 or (i == 2 and fold < 3):
            #     continue

            data_dir = os.path.join(config.DNA_BERT_2_OUTPUT_DIR, f"{group}/fold_{fold}")
            result_dir = f"{result_folder}/{group}/fold_{fold}"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            log.info("+" * 60)
            log.info(f"Start experiment with group {group} fold {fold}")

            # Create datasets
            log.info("Loading datasets...")
            tokenized_train = load_from_disk(os.path.join(data_dir, "processed_train_dataset"))
            tokenized_val = load_from_disk(os.path.join(data_dir, "processed_val_dataset"))

            config_run['batch_size'] = batch_size
            config_run['train_sample'] = tokenized_train.num_rows
            config_run['val_sample'] = tokenized_val.num_rows
            config_run['group'] = group
            config_run['fold'] = fold
            config_run['patience'] = 10
            log.info("-" * 60)
            log.info("Configuration:")
            for key, value in config_run.items():
                log.info(f"{key}: {value}")
            log.info("-" * 60)

            model_config = create_custom_config()
            model = Dnabert2CnnNoK7Model(
                num_classes=2,
                dropout_rate=0.1,
                config=model_config
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                max_length=tokenizer_max_length,
            )
            model.to(device)

            # Print initial model info
            # print_model_trainable_params(model)

            # Two-phase transfer learning
            log.info("Starting two-phase transfer learning...")
            results = train_two_phase_transfer_learning(
                model=model,
                tokenizer=tokenizer,
                tokenized_train=tokenized_train,
                tokenized_val=tokenized_val,
                device=device,
                warmup_epochs=config_run['warmup_epochs'],
                finetune_epochs=config_run['finetune_epochs'],
                warmup_lr=config_run['warmup_lr'],
                finetune_lr=config_run['finetune_lr'],
                patience=config_run['patience'],
                batch_size=batch_size
            )

            # Plot training history
            log.info("Plotting training history...")
            plot_training_history(results['history'], save_path=f"./{result_dir}/fig.png")

            # Final evaluation
            # log.info("🔬 Final model evaluation on test set...")
            # class_names = [f'Class_{i}' for i in range(config['n_classes'])]
            # test_results = evaluate_model(model, test_loader, device, class_names)

            # Summary
            log.info("TRAINING SUMMARY")
            log.info(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
            # log.info(f"🧪 Test Accuracy: {test_results['accuracy']:.4f}")
            log.info(f"Total Training Epochs: {len(results['history']['train_loss'])}")

            warmup_epochs_done = len([p for p in results['history']['phase'] if p == 'warmup'])
            finetune_epochs_done = len([p for p in results['history']['phase'] if p == 'finetune'])
            log.info(f"Warm-up Epochs: {warmup_epochs_done}")
            log.info(f"Fine-tune Epochs: {finetune_epochs_done}")

            # Model size info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"Total Parameters: {total_params:,}")
            log.info(f"Trainable Parameters: {trainable_params:,}")

            log.info("Saving best metrics to CSV...")
            save_best_metrics_to_csv(results, config_run, f'./{result_dir}/training_results.csv')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                log.info(f"GPU memory after training: {current_mem:.2f} GB")

            save_model_and_results(model, results, result_dir)
            log.info("-" * 60)


def save_model_and_results(model, results, save_path='dnabert2_cnn_model'):
    """
    Save trained model và training results
    """
    import pickle
    import os

    log.info(f"Saving model and results to {save_path}...")

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), f'{save_path}/model_weights.pth')

    # Save model architecture info
    model_info = {
        'num_classes': model.classifier[-1].out_features,
        'dropout_rate': 0.1,  # Default value used
        'model_name': 'zhihan1996/DNABERT-2-117M'
    }

    with open(f'{save_path}/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    # Save training results
    with open(f'{save_path}/training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save training history as CSV for easy analysis
    history_df = pd.DataFrame(results['history'])
    history_df.to_csv(f'{save_path}/training_history.csv', index=False)

    log.info(f"Model weights saved to {save_path}/model_weights.pth")
    log.info(f"Model info saved to {save_path}/model_info.pkl")
    log.info(f"Training results saved to {save_path}/training_results.pkl")
    log.info(f"Training history saved to {save_path}/training_history.csv")


def load_trained_model(save_path='dnabert2_cnn_model', device=None):
    """
    Load trained model từ saved files
    """
    import pickle

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info(f"📂 Loading model from {save_path}...")

    # Load model info
    with open(f'{save_path}/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)

    # Initialize model
    config = create_custom_config()
    model = Dnabert2CnnModelNoK3(
        num_classes=model_info['num_classes'],
        # Will be set appropriately
        dropout_rate=model_info['dropout_rate'],
        config=config,
    )

    # Load weights
    model.load_state_dict(torch.load(f'{save_path}/model_weights.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load training results
    with open(f'{save_path}/training_results.pkl', 'rb') as f:
        results = pickle.load(f)

    log.info(f"  ✅ Model loaded successfully!")
    log.info(f"  📊 Best validation accuracy: {results['best_val_acc']:.4f}")

    return model, results


if __name__ == "__main__":
    # Run main pipeline
    main()

    # comparison_results = compare_training_strategies(
    #     sequences=pipeline_results['sequences'],
    #     labels=pipeline_results['labels'],
    #     config=pipeline_results['config']
    # )

    # Optional: Save model
    # log.info("💾 Saving trained model and results...")
    # save_model_and_results(
    #     model=pipeline_results['model'],
    #     results=pipeline_results['results']
    # )
    #
    # log.info("🎉 Pipeline completed successfully!")
    # log.info("To use the trained model later, run:")
    # log.info("model, results = load_trained_model('dnabert2_cnn_model')")
