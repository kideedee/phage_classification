# import math
# import os
# from datetime import datetime
#
# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import timm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
#
# from common.env_config import config
# from logger.phg_cls_log import log
#
#
# class CustomDataset(Dataset):
#     def __init__(self, h5_path, vectors_key='vectors', labels_key='labels',
#                  transform=None, batch_size=1000, preload_all=False):
#         """
#         Args:
#             h5_path: Path to HDF5 file
#             vectors_key: Key for vectors in HDF5 file
#             labels_key: Key for labels in HDF5 file
#             transform: Optional transform to apply to data
#             batch_size: Number of samples to load per batch
#             preload_all: If True, load entire dataset into memory at initialization
#         """
#         self.h5_path = h5_path
#         self.vectors_key = vectors_key
#         self.labels_key = labels_key
#         self.transform = transform
#         self.batch_size = batch_size
#
#         # Get dataset length and shape info
#         with h5py.File(h5_path, 'r') as f:
#             self.length = len(f[labels_key])
#             self.vector_shape = f[vectors_key].shape[1:]  # Shape excluding batch dimension
#
#         # Calculate number of batches
#         self.num_batches = math.ceil(self.length / self.batch_size)
#
#         # Memory cache for current batch
#         self.current_batch_idx = -1
#         self.cached_vectors = None
#         self.cached_labels = None
#         self.batch_start_idx = 0
#         self.batch_end_idx = 0
#
#     def _preload_all_data(self):
#         with h5py.File(self.h5_path, 'r') as f:
#             self.all_vectors = torch.from_numpy(
#                 f[self.vectors_key][:].astype(np.float32)
#             )
#             self.all_labels = torch.from_numpy(
#                 f[self.labels_key][:].astype(np.int64)
#             )
#         self.preloaded = True
#
#     def _get_batch_idx(self, idx):
#         return idx // self.batch_size
#
#     def _load_batch(self, batch_idx):
#         start_idx = batch_idx * self.batch_size
#         end_idx = min(start_idx + self.batch_size, self.length)
#
#         with h5py.File(self.h5_path, 'r') as f:
#             vectors = f[self.vectors_key][start_idx:end_idx].astype(np.float32)
#             labels = f[self.labels_key][start_idx:end_idx].astype(np.int64)
#
#         # Convert to tensors
#         self.cached_vectors = torch.from_numpy(vectors)
#         self.cached_labels = torch.from_numpy(labels)
#
#         # Update batch tracking
#         self.current_batch_idx = batch_idx
#         self.batch_start_idx = start_idx
#         self.batch_end_idx = end_idx
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         if idx >= self.length:
#             raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
#
#         # Determine which batch this index belongs to
#         required_batch_idx = self._get_batch_idx(idx)
#
#         # Load batch if not already cached or if different batch needed
#         if self.current_batch_idx != required_batch_idx:
#             self._load_batch(required_batch_idx)
#
#         # Get data from cached batch
#         batch_relative_idx = idx - self.batch_start_idx
#         vector = self.cached_vectors[batch_relative_idx]
#         label = self.cached_labels[batch_relative_idx]
#
#         # Apply reshaping
#         reshaped_vector = vector.view(1, vector.shape[0], vector.shape[1])
#
#         # Apply transform if provided
#         if self.transform:
#             reshaped_vector = self.transform(reshaped_vector)
#
#         return reshaped_vector, label
#
#
# def calculate_sensitivity_specificity(predictions, targets):
#     """Calculate sensitivity and specificity for binary classification"""
#     # Convert to numpy for easier computation
#     if isinstance(predictions, torch.Tensor):
#         predictions = predictions.cpu().numpy()
#     if isinstance(targets, torch.Tensor):
#         targets = targets.cpu().numpy()
#
#     # Calculate confusion matrix components
#     cm = confusion_matrix(targets, predictions)
#     tn, fp, fn, tp = cm.ravel()
#
#     # Calculate sensitivity and specificity
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#
#     return sensitivity, specificity, tp, tn, fp, fn
#
#
# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience):
#     """
#     Training loop for the model with early stopping
#
#     Args:
#         patience: Number of epochs to wait for improvement before stopping
#     """
#     train_losses = []
#     val_losses = []
#     train_accuracies = []
#     val_accuracies = []
#
#     best_val_acc = 0.0
#     best_model_state = None
#     epochs_without_improvement = 0
#     early_stopped = False
#
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         correct_predictions = 0
#         total_samples = 0
#
#         # Store all predictions and targets for sensitivity/specificity calculation
#         all_train_predictions = []
#         all_train_targets = []
#
#         train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
#         for batch_idx, (data, target) in enumerate(train_pbar):
#             data, target = data.to(device), target.to(device)
#
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.detach().item()
#             _, predicted = torch.max(output.data, 1)
#             total_samples += target.size(0)
#             correct_predictions += (predicted == target).sum().item()
#
#             # Store predictions and targets for sensitivity/specificity
#             all_train_predictions.extend(predicted.cpu().numpy())
#             all_train_targets.extend(target.cpu().numpy())
#
#             # Update progress bar
#             train_pbar.set_postfix({
#                 'Loss': f'{loss.detach().item():.4f}',
#                 'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
#             })
#
#         train_loss = running_loss / len(train_loader)
#         train_acc = 100. * correct_predictions / total_samples
#         train_losses.append(train_loss)
#         train_accuracies.append(train_acc)
#
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         correct_predictions = 0
#         total_samples = 0
#
#         # Store all predictions and targets for sensitivity/specificity calculation
#         all_val_predictions = []
#         all_val_targets = []
#
#         with torch.no_grad():
#             val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
#             for data, target in val_pbar:
#                 data, target = data.to(device), target.to(device)
#                 output = model(data)
#                 loss = criterion(output, target)
#
#                 val_loss += loss.item()
#                 _, predicted = torch.max(output.data, 1)
#                 total_samples += target.size(0)
#                 correct_predictions += (predicted == target).sum().item()
#
#                 # Store predictions and targets for sensitivity/specificity
#                 all_val_predictions.extend(predicted.cpu().numpy())
#                 all_val_targets.extend(target.cpu().numpy())
#
#                 val_pbar.set_postfix({
#                     'Loss': f'{loss.item():.4f}',
#                     'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
#                 })
#
#         val_loss /= len(val_loader)
#         val_acc = 100. * correct_predictions / total_samples
#         val_losses.append(val_loss)
#         val_accuracies.append(val_acc)
#
#         # Update learning rate
#         scheduler.step()
#
#         # Early stopping logic
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model_state = model.state_dict().copy()
#             epochs_without_improvement = 0
#             log.info(f'New best validation accuracy: {val_acc:.2f}%')
#         else:
#             epochs_without_improvement += 1
#             log.info(f'No improvement for {epochs_without_improvement} epoch(s)')
#
#         # Enhanced logging
#         log.info(f'Epoch {epoch + 1}/{num_epochs}:')
#         log.info(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
#         log.info(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
#         log.info(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
#         log.info(f'Best Val Acc: {best_val_acc:.2f}%')
#
#         # Check for early stopping
#         if epochs_without_improvement >= patience:
#             log.info(f'Early stopping triggered after {epoch + 1} epochs')
#             log.info(f'No improvement for {patience} consecutive epochs')
#             early_stopped = True
#             break
#
#         log.info('-' * 80)
#
#     # Load best model
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#         log.info(f'Loaded best model with validation accuracy: {best_val_acc:.2f}%')
#
#     # Final summary
#     total_epochs_trained = epoch + 1 if early_stopped else num_epochs
#     log.info(f'Training completed after {total_epochs_trained} epochs')
#     if early_stopped:
#         log.info(f'Training stopped early due to no improvement for {patience} epochs')
#
#     return {
#         'train_losses': train_losses,
#         'val_losses': val_losses,
#         'train_accuracies': train_accuracies,
#         'val_accuracies': val_accuracies,
#         'best_val_acc': best_val_acc,
#         'epochs_trained': total_epochs_trained,
#         'early_stopped': early_stopped
#     }
#
#
# def evaluate_model(model, test_loader, device, class_names=['Class 0', 'Class 1'], save_path=None):
#     """
#     Evaluate model on test set with detailed metrics
#
#     Args:
#         model: PyTorch model to evaluate
#         test_loader: DataLoader for test data
#         device: Device to run evaluation on
#         class_names: List of class names for visualization
#         save_path: Path to save the evaluation plots. Can be:
#                   - None: Only display plots
#                   - String ending with image extension: Save with specific filename
#                   - Directory path: Save with auto-generated filename
#     """
#     import os
#     from datetime import datetime
#
#     model.eval()
#     all_predictions = []
#     all_targets = []
#
#     with torch.no_grad():
#         for data, target in tqdm(test_loader, desc='Testing'):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             _, predicted = torch.max(output, 1)
#
#             all_predictions.extend(predicted.cpu().numpy())
#             all_targets.extend(target.cpu().numpy())
#
#     # Calculate basic metrics
#     accuracy = accuracy_score(all_targets, all_predictions)
#
#     # Calculate confusion matrix
#     cm = confusion_matrix(all_targets, all_predictions)
#
#     tn, fp, fn, tp = cm.ravel()
#
#     # Sensitivity (True Positive Rate/Recall) = TP / (TP + FN)
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#
#     # Specificity (True Negative Rate) = TN / (TN + FP)
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#     # Precision = TP / (TP + FP)
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#
#     # F1-Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
#     f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
#
#     log.info("=" * 60)
#     log.info("DETAILED CLASSIFICATION METRICS")
#     log.info("=" * 60)
#     log.info(f"Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)")
#     log.info(f"Sensitivity:  {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
#     log.info(f"Specificity:  {specificity:.4f} ({specificity * 100:.2f}%)")
#     log.info(f"Precision:    {precision:.4f} ({precision * 100:.2f}%)")
#     log.info(f"F1-Score:     {f1_score:.4f} ({f1_score * 100:.2f}%)")
#     log.info("=" * 60)
#
#     log.info("\nConfusion Matrix Values:")
#     log.info(f"True Negatives (TN):  {tn}")
#     log.info(f"False Positives (FP): {fp}")
#     log.info(f"False Negatives (FN): {fn}")
#     log.info(f"True Positives (TP):  {tp}")
#     log.info("-" * 40)
#
#     # Store metrics for return
#     detailed_metrics = {
#         'accuracy': accuracy,
#         'sensitivity': sensitivity,
#         'specificity': specificity,
#         'precision': precision,
#         'f1_score': f1_score,
#         'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
#     }
#
#     log.info("\nClassification Report:")
#     log.info(classification_report(all_targets, all_predictions, target_names=class_names))
#
#     # Enhanced Confusion Matrix visualization
#     plt.figure(figsize=(10, 8))
#
#     # Create subplot for confusion matrix
#     plt.subplot(2, 2, 1)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#
#     # Create metrics bar plot if binary classification
#     if len(class_names) == 2:
#         plt.subplot(2, 2, 2)
#         metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
#         metrics_values = [accuracy, sensitivity, specificity, precision, f1_score]
#         colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
#
#         bars = plt.bar(metrics_names, metrics_values, color=colors)
#         plt.title('Classification Metrics')
#         plt.ylabel('Score')
#         plt.ylim(0, 1.1)
#
#         # Add value labels on bars
#         for bar, value in zip(bars, metrics_values):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{value:.3f}', ha='center', va='bottom')
#
#         plt.xticks(rotation=45)
#
#         # Create ROC-style visualization (without actual ROC curve)
#         plt.subplot(2, 2, 3)
#         plt.scatter([1 - specificity], [sensitivity], s=100, c='red', marker='o')
#         plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         plt.xlabel('1 - Specificity (False Positive Rate)')
#         plt.ylabel('Sensitivity (True Positive Rate)')
#         plt.title('ROC Point')
#         plt.grid(True, alpha=0.3)
#
#         # Add metrics text box
#         plt.subplot(2, 2, 4)
#         plt.axis('off')
#         metrics_text = f"""
#         Detailed Metrics:
#
#         Accuracy:    {accuracy:.4f}
#         Sensitivity: {sensitivity:.4f}
#         Specificity: {specificity:.4f}
#         Precision:   {precision:.4f}
#         F1-Score:    {f1_score:.4f}
#
#         Confusion Matrix:
#         TN: {tn}  FP: {fp}
#         FN: {fn}  TP: {tp}
#         """
#         plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
#                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
#
#     plt.tight_layout()
#
#     # Handle saving the plot
#     if save_path:
#         # Create directory if it doesn't exist
#         if os.path.dirname(save_path):
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#         # If save_path is just a directory or doesn't have an extension, create a filename
#         if os.path.isdir(save_path) or save_path.endswith('/') or '.' not in os.path.basename(save_path):
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             filename = f'evaluation_results_{timestamp}.png'
#             if save_path.endswith('/') or os.path.isdir(save_path):
#                 save_path = os.path.join(save_path, filename)
#             else:
#                 save_path = f"{save_path}_{filename}"
#
#         # Save with high DPI for better quality
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         log.info(f"Evaluation plot saved to: {save_path}")
#
#     plt.show()
#
#     return detailed_metrics, all_predictions, all_targets
#
#
# # Helper function to save metrics to CSV
# # def save_metrics_to_csv(detailed_metrics, csv_path=None):
# #     """
# #     Save detailed metrics to CSV file
# #
# #     Args:
# #         detailed_metrics: Dictionary containing evaluation metrics
# #         csv_path: Path to save CSV file. If None, creates default filename
# #     """
# #     import pandas as pd
# #     import os
# #     from datetime import datetime
# #
# #     if csv_path is None:
# #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# #         csv_path = f'evaluation_metrics_{timestamp}.csv'
# #
# #     # Create directory if it doesn't exist
# #     if os.path.dirname(csv_path):
# #         os.makedirs(os.path.dirname(csv_path), exist_ok=True)
# #
# #     # Add timestamp to metrics
# #     detailed_metrics_with_time = detailed_metrics.copy()
# #     detailed_metrics_with_time['evaluation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #
# #     # Convert to DataFrame and save
# #     df = pd.DataFrame([detailed_metrics_with_time])
# #     df.to_csv(csv_path, index=False)
# #
# #     log.info(f"Metrics saved to: {csv_path}")
# #     return csv_path
#
#
# def plot_training_history(history, save_path=None):
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
#
#     # Plot losses
#     ax1.plot(history['train_losses'], label='Train Loss', color='blue', linewidth=2)
#     ax1.plot(history['val_losses'], label='Validation Loss', color='red', linewidth=2)
#     ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
#
#     # Plot accuracies
#     ax2.plot(history['train_accuracies'], label='Train Accuracy', color='blue', linewidth=2)
#     ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='red', linewidth=2)
#     ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy (%)')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#
#     plt.tight_layout(pad=3.0)
#
#     # Save the plot if save_path is provided
#     if save_path:
#         # Create directory if it doesn't exist
#         if os.path.dirname(save_path):
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#         # If save_path is just a directory, create a filename
#         if os.path.isdir(save_path) or save_path.endswith('/'):
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             filename = f'training_history_{timestamp}.png'
#             save_path = os.path.join(save_path, filename)
#
#         # Save with high DPI for better quality
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         log.info(f"Training history plot saved to: {save_path}")
#
#     plt.show()
#
#     # Print final metrics summary
#     log.info("\n" + "=" * 80)
#     log.info("FINAL TRAINING METRICS SUMMARY")
#     log.info("=" * 80)
#     log.info(f"{'Metric':<20} {'Train (Final)':<15} {'Validation (Final)':<20} {'Best Val':<15}")
#     log.info("-" * 80)
#     log.info(
#         f"{'Accuracy (%)':<20} {history['train_accuracies'][-1]:<15.2f} {history['val_accuracies'][-1]:<20.2f} {history['best_val_acc']:<15.2f}")
#     log.info(f"{'Loss':<20} {history['train_losses'][-1]:<15.4f} {history['val_losses'][-1]:<20.4f} {'-':<15}")
#     log.info("=" * 80)
#
#
# def init_model(CONFIG):
#     log.info("Loading datasets...")
#     model = timm.create_model(
#         CONFIG['model_name'],
#         pretrained=CONFIG['pretrained'],
#         num_classes=CONFIG['num_classes']
#     )
#     original_conv = model.stem[0]
#     new_conv = nn.Conv2d(
#         in_channels=1,  # thay vì 3
#         out_channels=original_conv.out_channels,
#         kernel_size=original_conv.kernel_size,
#         stride=original_conv.stride,
#         padding=original_conv.padding,
#         bias=original_conv.bias is not None
#     )
#     if hasattr(original_conv, 'weight'):
#         new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
#         if original_conv.bias is not None:
#             new_conv.bias.data = original_conv.bias.data
#
#     model.stem[0] = new_conv
#     model = model.to(CONFIG['device'])
#     return model
#
#
# # Main training script
# def run(data_dir, output_data_dir):
#     # Configuration
#     CONFIG = {
#         'train_path': os.path.join(data_dir, "train/data.h5"),
#         'test_path': os.path.join(data_dir, 'test/data.h5'),
#         'batch_size': 128,
#         'num_epochs': 100,
#         'learning_rate': 1e-4,
#         'weight_decay': 1e-4,
#         'num_classes': 2,
#         'model_name': 'convnext_base',  # Options: convnext_tiny, convnext_small, convnext_base
#         'pretrained': True,
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#         'patience': 10,
#     }
#
#     log.info(f"Using device: {CONFIG['device']}")
#     log.info(f"Model: {CONFIG['model_name']}")
#
#     # Load datasets
#     train_dataset = CustomDataset(CONFIG['train_path'])
#     test_dataset = CustomDataset(CONFIG['test_path'])
#
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4,
#                               persistent_workers=True, prefetch_factor=8)
#     test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4,
#                              persistent_workers=True, prefetch_factor=8)
#
#     model = init_model(CONFIG)
#
#     # Print model info
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     log.info(f"Total parameters: {total_params:,}")
#     log.info(f"Trainable parameters: {trainable_params:,}")
#
#     # Loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(),
#                             lr=CONFIG['learning_rate'],
#                             weight_decay=CONFIG['weight_decay'])
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
#
#     # Train model
#     log.info("Starting training...")
#     history = train_model(model, train_loader, test_loader, criterion, optimizer,
#                           scheduler, CONFIG['num_epochs'], CONFIG['device'], patience=CONFIG['patience'])
#
#     # Plot training history
#     plot_training_history(history, save_path=output_data_dir)
#     log.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
#
#     # Evaluate on test set
#     log.info("Evaluating on test set...")
#     detailed_metrics, predictions, targets = evaluate_model(model, test_loader, CONFIG['device'],
#                                                             save_path=output_data_dir)
#
#     # Save model with detailed metrics
#     # torch.save({
#     #     'model_state_dict': model.state_dict(),
#     #     'config': CONFIG,
#     #     'history': history,
#     #     'detailed_metrics': detailed_metrics,
#     #     'predictions': predictions,
#     #     'targets': targets
#     # }, f'convnext_model_{CONFIG["model_name"]}.pth')
#
#     # log.info(f"Model saved as 'convnext_model_{CONFIG['model_name']}.pth'")
#     df = pd.DataFrame([detailed_metrics])
#     if not os.path.exists(output_data_dir):
#         os.makedirs(output_data_dir)
#     df.to_csv(os.path.join(output_data_dir, "result.csv"), index=False)
#
#     if 'accuracy' in detailed_metrics:
#         log.info(f"Test accuracy: {detailed_metrics['accuracy'] * 100:.2f}%")
#     if 'sensitivity' in detailed_metrics:
#         log.info(f"Sensitivity: {detailed_metrics['sensitivity'] * 100:.2f}%")
#     if 'specificity' in detailed_metrics:
#         log.info(f"Specificity: {detailed_metrics['specificity'] * 100:.2f}%")
#
#
# if __name__ == "__main__":
#     # root_data_dir = config.HDFS_FCGR_EMBEDDING_OUTPUT_DIR
#     root_data_dir = "C:\\Users\Admin\Temp"
#     for i in range(4):
#         group = i + 1
#         if group == 1:
#             min_size = 100
#             max_size = 400
#             overlap = 10
#         elif group == 2:
#             min_size = 400
#             max_size = 800
#             overlap = 10
#         elif group == 3:
#             min_size = 800
#             max_size = 1200
#             overlap = 30
#         else:
#             min_size = 1200
#             max_size = 1800
#             overlap = 30
#
#         for j in range(5):
#             fold = j + 1
#
#             if group == 1 and fold <= 4:
#                 continue
#
#             if group == 1:
#                 data_dir = os.path.join(root_data_dir, f"{min_size}_{max_size}/fold_{fold}")
#                 output_data_dir = f"./{min_size}_{max_size}/fold_{fold}"
#                 run(data_dir, output_data_dir)
