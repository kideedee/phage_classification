import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from common.env_config import config

warnings.filterwarnings('ignore')
torch.cuda.set_per_process_memory_fraction(1.0)  # Dùng 95% dedicated memory
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:False'
torch.backends.cuda.enable_flash_sdp(True)


class CustomImageDataset(Dataset):
    """Custom Dataset class cho ViT fine-tuning"""

    def __init__(self, data, labels, processor, transform=None, data_type='paths'):
        self.data = data  # Could be image_paths or numpy arrays
        self.labels = labels
        self.processor = processor
        self.transform = transform
        self.data_type = data_type  # 'paths' or 'numpy'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == 'paths':
            # Original path-based loading
            image_path = self.data[idx]
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return a blank image if loading fails
                image = Image.new('RGB', (224, 224), color='black')


        elif self.data_type == 'numpy':

            # Load from numpy array

            image_array = self.data[idx]

            # ENHANCED PREPROCESSING FOR DENSITY MAPS

            if len(image_array.shape) == 3 and image_array.shape[2] == 1:

                density_map = image_array[:, :, 0]

            elif len(image_array.shape) == 2:

                density_map = image_array

            else:

                density_map = image_array[:, :, 0] if len(image_array.shape) == 3 else image_array

            # IMPROVED NORMALIZATION

            # Method 1: Scale to full 0-255 range

            if density_map.max() > density_map.min():

                # Min-max normalization to [0, 255]

                normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min())

                normalized = (normalized * 255).astype(np.uint8)

            else:

                # If all values are the same

                normalized = np.zeros_like(density_map, dtype=np.uint8)

            # Convert to 3-channel RGB

            if len(normalized.shape) == 2:
                image_array = np.stack([normalized] * 3, axis=-1)

            image = Image.fromarray(image_array, 'RGB')

        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

        # Apply custom transforms if any
        if self.transform:
            image = self.transform(image)

        # Process image for ViT
        encoding = self.processor(image, return_tensors="pt")

        # Remove batch dimension and add label
        pixel_values = encoding['pixel_values'].squeeze()

        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class ViTFineTuner:
    """Complete ViT Fine-tuning Pipeline"""

    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=None, fold=None, group=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fold = fold
        self.group = group

        # Initialize processor
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # Will be initialized later
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None

        print(f"🚀 ViT Fine-tuner initialized")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {model_name}")

    def prepare_data_from_numpy(self, data_dir, class_names=None):

        # Load numpy arrays
        try:
            train_image_arrays = np.load(os.path.join(data_dir, "train/fcgr_vectors.npy"))
            train_image_labels = np.load(os.path.join(data_dir, "train/fcgr_labels.npy"))
            val_image_arrays = np.load(os.path.join(data_dir, "test/fcgr_vectors.npy"))
            val_image_labels = np.load(os.path.join(data_dir, "test/fcgr_labels.npy"))
        except Exception as e:
            raise ValueError(f"Error loading numpy files: {e}")

        print(f"📊 Train data info:")
        print(f"   Image arrays shape: {train_image_arrays.shape}")
        print(f"   Labels shape: {train_image_labels.shape}")
        print(f"   Data type: {train_image_arrays.dtype}")
        print(f"   Value range: [{train_image_arrays.min():.3f}, {train_image_arrays.max():.3f}]")
        print(f"📊 Val data info:")
        print(f"   Image arrays shape: {val_image_arrays.shape}")
        print(f"   Labels shape: {val_image_labels.shape}")
        print(f"   Data type: {val_image_arrays.dtype}")
        print(f"   Value range: [{val_image_arrays.min():.3f}, {val_image_arrays.max():.3f}]")

        # Validate data
        if len(train_image_arrays) != len(train_image_labels):
            raise ValueError(f"Mismatch: {len(train_image_arrays)} images vs {len(train_image_labels)} labels")
        if len(val_image_arrays) != len(val_image_labels):
            raise ValueError(f"Mismatch: {len(val_image_labels)} images vs {len(val_image_labels)} labels")

        # Handle labels
        # Handle labels
        unique_labels = np.unique(train_image_labels)
        self.num_classes = len(unique_labels)

        print(f"📋 Found {self.num_classes} classes: {unique_labels}")

        # Create label mappings - assume labels are already 0, 1, 2, ...
        if class_names:
            if len(class_names) != self.num_classes:
                raise ValueError(f"Provided {len(class_names)} class names but found {self.num_classes} unique labels")
            self.label2id = {name: idx for idx, name in enumerate(class_names)}
            self.id2label = {idx: name for idx, name in enumerate(class_names)}
        else:
            self.label2id = {f"class_{idx}": idx for idx in range(self.num_classes)}
            self.id2label = {idx: f"class_{idx}" for idx in range(self.num_classes)}

        # Verify labels are in correct format (0, 1, 2, ...)
        expected_labels = set(range(self.num_classes))
        actual_labels = set(unique_labels)
        if actual_labels != expected_labels:
            raise ValueError(f"Labels should be 0 to {self.num_classes - 1}, but found: {unique_labels}")

        # Print class distribution
        unique, counts = np.unique(train_image_labels, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = self.id2label[label]
            print(f"   Train data, {class_name}: {count} samples")

        unique, counts = np.unique(val_image_labels, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = self.id2label[label]
            print(f"   Val data, {class_name}: {count} samples")

        print(f"📈 Data split:")
        print(f"   Train: {len(train_image_arrays)} images")
        print(f"   Validation: {len(val_image_arrays)} images")

        # train_image_arrays, _, train_image_labels, _ = train_test_split(
        #     train_image_arrays, train_image_labels, test_size=0.99, random_state=42, stratify=train_image_labels
        # )
        # val_image_arrays, _, val_image_labels, _ = train_test_split(
        #     val_image_arrays, val_image_labels, test_size=0.9, random_state=42, stratify=val_image_labels
        # )

        # Create datasets
        self.train_dataset = CustomImageDataset(train_image_arrays, train_image_labels, self.processor,
                                                data_type='numpy')
        self.val_dataset = CustomImageDataset(val_image_arrays, val_image_labels, self.processor, data_type='numpy')

        return self.train_dataset, self.val_dataset

    def prepare_data_from_folder(self, data_dir, test_size=0.2, val_size=0.1):
        """
        Chuẩn bị dữ liệu từ folder structure:
        data_dir/
        ├── class1/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── class2/
        │   ├── img3.jpg
        │   └── img4.jpg
        """
        print(f"📂 Loading data from {data_dir}")

        image_paths = []
        labels = []
        class_names = []

        # Get all class directories
        class_dirs = [d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]
        class_dirs.sort()

        print(f"📋 Found {len(class_dirs)} classes: {class_dirs}")

        # Create label mapping
        self.label2id = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        self.id2label = {idx: class_name for class_name, idx in self.label2id.items()}
        self.num_classes = len(class_dirs)

        # Collect all images
        for class_name in class_dirs:
            class_dir = os.path.join(data_dir, class_name)
            class_images = [f for f in os.listdir(class_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            print(f"   {class_name}: {len(class_images)} images")

            for img_name in class_images:
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(self.label2id[class_name])

        print(f"📊 Total images: {len(image_paths)}")

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=test_size,
            stratify=labels, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (1 - test_size),
            stratify=y_temp, random_state=42
        )

        print(f"📈 Data split:")
        print(f"   Train: {len(X_train)} images")
        print(f"   Validation: {len(X_val)} images")
        print(f"   Test: {len(X_test)} images")

        # Create datasets
        self.train_dataset = CustomImageDataset(X_train, y_train, self.processor, data_type='paths')
        self.val_dataset = CustomImageDataset(X_val, y_val, self.processor, data_type='paths')
        self.test_dataset = CustomImageDataset(X_test, y_test, self.processor, data_type='paths')

        return self.train_dataset, self.val_dataset, self.test_dataset

    def prepare_data_from_lists(self, train_paths, train_labels, val_paths, val_labels,
                                test_paths=None, test_labels=None, class_names=None):
        """Chuẩn bị dữ liệu từ lists"""

        # Create label mappings
        if class_names:
            self.label2id = {name: idx for idx, name in enumerate(class_names)}
            self.id2label = {idx: name for name, idx in self.label2id.items()}
        else:
            unique_labels = sorted(set(train_labels + val_labels))
            self.label2id = {f"class_{label}": label for label in unique_labels}
            self.id2label = {label: f"class_{label}" for label in unique_labels}

        self.num_classes = len(self.label2id)

        # Create datasets
        self.train_dataset = CustomImageDataset(train_paths, train_labels, self.processor, data_type='paths')
        self.val_dataset = CustomImageDataset(val_paths, val_labels, self.processor, data_type='paths')

        if test_paths and test_labels:
            self.test_dataset = CustomImageDataset(test_paths, test_labels, self.processor, data_type='paths')

        print(f"📊 Dataset prepared:")
        print(f"   Classes: {self.num_classes}")
        print(f"   Train: {len(self.train_dataset)}")
        print(f"   Val: {len(self.val_dataset)}")
        if self.test_dataset:
            print(f"   Test: {len(self.test_dataset)}")

    def initialize_model(self):
        """Initialize model với số classes phù hợp"""
        if self.num_classes is None:
            raise ValueError("Number of classes not set. Prepare data first.")

        print(f"🔧 Initializing model for {self.num_classes} classes")

        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        # Move to device
        self.model.to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"📊 Model parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")

    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
        labels = pred.label_ids
        logits = pred.predictions

        # Process in batches if predictions are large
        if isinstance(logits, np.ndarray) and logits.size > 1e6:  # Threshold for batch processing
            batch_size = 1000
            preds = []
            for i in range(0, len(logits), batch_size):
                batch_preds = np.argmax(logits[i:i + batch_size], axis=-1)
                preds.extend(batch_preds)
            preds = np.array(preds)
        else:
            preds = np.argmax(logits, axis=-1)

        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)
        print(f"Confusion Matrix:\n{cm}")

        # Log detailed stats from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}")
        print(f"False Negatives: {fn}, True Positives: {tp}")

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        gmean = np.sqrt(sensitivity * specificity) if (sensitivity > 0 and specificity > 0) else 0

        # Safely compute ROC AUC with batching for large datasets
        # try:
        #     if isinstance(logits, np.ndarray) and logits.shape[0] > 1e6:
        #         # Calculate ROC AUC in batches
        #         batch_size = 1000
        #         y_scores = []
        #         for i in range(0, len(logits), batch_size):
        #             y_scores.extend(logits[i:i + batch_size, 1])
        #         roc_auc = roc_auc_score(labels, np.array(y_scores))
        #     else:
        #         roc_auc = roc_auc_score(labels, logits[:, 1])
        # except:
        #     roc_auc = 0

        # Explicitly free memory
        del logits, preds

        result = {
            'fold': self.fold,
            'group': self.group,
            'acc': acc,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'g_mean': gmean,  # Thêm G-Mean metric
            'tn': float(tn),
            'fp': float(fp),
            'fn': float(fn),
            'tp': float(tp)
        }

        csv_file = "result_1.csv"
        file_exists = os.path.isfile(csv_file)
        df = pd.DataFrame([result])
        if file_exists:
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode='w', header=True, index=False)

        return result

    def setup_training(self, output_dir="./vit-finetuned", **training_kwargs):
        """Setup training arguments và trainer"""

        default_args = {
            "output_dir": output_dir,
            "num_train_epochs": 10,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
            "logging_dir": f"{output_dir}/logs",
            "logging_steps": 50,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "report_to": None,

            # DataLoader optimizations
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,

            # Memory optimizations
            "gradient_accumulation_steps": 2,
            "eval_accumulation_steps": 8,

            # Mixed precision
            "fp16": True,
            "fp16_full_eval": False,

            "optim": "adamw_torch",
            "learning_rate": 5e-5,
        }

        # Update with user provided arguments
        default_args.update(training_kwargs)

        self.training_args = TrainingArguments(**default_args)

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=None,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print(f"⚙️ Training setup complete")
        print(f"   Output directory: {output_dir}")
        print(f"   Epochs: {self.training_args.num_train_epochs}")
        print(f"   Batch size: {self.training_args.per_device_train_batch_size}")
        print(f"   Learning rate: {self.training_args.learning_rate}")
        print(f"   Mixed precision: {self.training_args.fp16}")

    def train(self):
        """Start training"""
        if self.trainer is None:
            raise ValueError("Setup training first using setup_training()")

        print("🚀 Starting training...")

        # Train the model
        train_result = self.trainer.train()

        # Save the model
        self.trainer.save_model()

        print("✅ Training completed!")
        print(f"📊 Final train loss: {train_result.training_loss:.4f}")

        return train_result

    def evaluate(self, dataset=None):
        """Evaluate model"""
        if dataset is None:
            dataset = self.val_dataset

        print("📊 Evaluating model...")

        # Evaluate
        eval_result = self.trainer.evaluate(eval_dataset=dataset)

        print("📈 Evaluation results:")
        for key, value in eval_result.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")

        return eval_result

    def test(self):
        """Test on test set"""
        if self.test_dataset is None:
            print("❌ No test dataset available")
            return None

        print("🧪 Testing model...")

        # Predict on test set
        predictions = self.trainer.predict(self.test_dataset)

        # Compute metrics
        test_metrics = self.compute_metrics((predictions.predictions, predictions.label_ids))

        print("🎯 Test results:")
        for key, value in test_metrics.items():
            print(f"   {key}: {value:.4f}")

        return test_metrics, predictions

    def predict_single_image(self, image_path, top_k=3):
        """Predict single image"""
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get top-k predictions
        top_predictions = torch.topk(predictions[0], top_k)

        results = []
        for score, idx in zip(top_predictions.values, top_predictions.indices):
            label = self.id2label[idx.item()]
            confidence = score.item()
            results.append((label, confidence))

        return results

    def plot_training_history(self):
        """Plot training history"""
        if self.trainer is None:
            print("❌ No training history available")
            return

        # Get training logs
        logs = self.trainer.state.log_history

        # Extract metrics
        train_loss = []
        eval_loss = []
        eval_accuracy = []
        eval_f1 = []
        steps = []

        for log in logs:
            if 'loss' in log and 'eval_loss' not in log:
                train_loss.append(log['loss'])
                steps.append(log['step'])
            elif 'eval_loss' in log:
                eval_loss.append(log['eval_loss'])
                eval_accuracy.append(log.get('eval_accuracy', 0))
                eval_f1.append(log.get('eval_f1', 0))

        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        if train_loss:
            ax1.plot(steps, train_loss, 'b-', label='Training Loss')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

        # Validation loss
        if eval_loss:
            eval_steps = [log['step'] for log in logs if 'eval_loss' in log]
            ax2.plot(eval_steps, eval_loss, 'r-', label='Validation Loss')
            ax2.set_title('Validation Loss')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

        # Accuracy
        if eval_accuracy:
            eval_steps = [log['step'] for log in logs if 'eval_accuracy' in log]
            ax3.plot(eval_steps, eval_accuracy, 'g-', label='Validation Accuracy')
            ax3.set_title('Validation Accuracy')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True)

        # F1 Score
        if eval_f1:
            eval_steps = [log['step'] for log in logs if 'eval_f1' in log]
            ax4.plot(eval_steps, eval_f1, 'm-', label='Validation F1')
            ax4.set_title('Validation F1 Score')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('F1 Score')
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def save_model(self, save_path):
        """Save fine-tuned model"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        # Save label mappings
        with open(os.path.join(save_path, 'label_mappings.json'), 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, indent=2)

        print(f"💾 Model saved to {save_path}")

    def load_model(self, model_path):
        """Load fine-tuned model"""
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.processor = ViTImageProcessor.from_pretrained(model_path)

        # Load label mappings
        with open(os.path.join(model_path, 'label_mappings.json'), 'r') as f:
            mappings = json.load(f)
            self.label2id = mappings['label2id']
            self.id2label = {int(k): v for k, v in mappings['id2label'].items()}

        self.model.to(self.device)
        print(f"📁 Model loaded from {model_path}")

    def visualize_random_sample(self, dataset_type='train', show_processed=True, use_colormap='viridis'):
        """
        Hiển thị ảnh mật độ ngẫu nhiên từ dataset

        Args:
            dataset_type: 'train', 'val', hoặc 'test'
            show_processed: True để show ảnh sau khi qua processor, False để show ảnh gốc
            use_colormap: colormap cho density visualization ('viridis', 'plasma', 'hot', 'jet', 'gray')
        """
        import matplotlib.pyplot as plt
        import random

        # Chọn dataset
        if dataset_type == 'train':
            dataset = self.train_dataset
            dataset_name = "Training"
        elif dataset_type == 'val':
            dataset = self.val_dataset
            dataset_name = "Validation"
        elif dataset_type == 'test':
            dataset = self.test_dataset
            dataset_name = "Test"
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")

        if dataset is None:
            print(f"❌ {dataset_name} dataset not available")
            return

        # Chọn index ngẫu nhiên
        random_idx = random.randint(0, len(dataset) - 1)

        # Lấy sample
        sample = dataset[random_idx]
        pixel_values = sample['pixel_values']
        label = sample['labels'].item()

        # Lấy ảnh gốc từ dataset.data
        original_array = dataset.data[random_idx]

        # Tạo figure
        if show_processed:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            ((ax1, ax2), (ax3, ax4)) = axes
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Hiển thị ảnh density gốc
        if len(original_array.shape) == 3 and original_array.shape[2] == 1:
            density_map = original_array[:, :, 0]
        elif len(original_array.shape) == 2:
            density_map = original_array
        else:
            # Nếu có nhiều channel, lấy channel đầu tiên
            density_map = original_array[:, :, 0] if len(original_array.shape) == 3 else original_array

        im1 = ax1.imshow(density_map, cmap=use_colormap, interpolation='nearest')
        ax1.set_title(f'Original Density Map\n{dataset_name} Set - Index: {random_idx}')
        ax1.axis('off')

        # Colorbar cho density map
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Density Value', rotation=270, labelpad=15)

        # Thông tin ảnh
        class_name = self.id2label[label]
        shape_info = f"Shape: {original_array.shape}"
        dtype_info = f"Dtype: {original_array.dtype}"
        range_info = f"Range: [{original_array.min():.4f}, {original_array.max():.4f}]"
        mean_density = f"Mean: {original_array.mean():.4f}"
        std_density = f"Std: {original_array.std():.4f}"
        nonzero_pixels = f"Non-zero pixels: {np.count_nonzero(original_array)}"

        info_text = f"Label: {label} ({class_name})\n{shape_info}\n{dtype_info}\n{range_info}\n{mean_density}\n{std_density}\n{nonzero_pixels}"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                 fontsize=9)

        # 2. Histogram của density values
        ax2.hist(original_array.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Density Values Distribution')
        ax2.set_xlabel('Density Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # Thêm statistics lên histogram
        stats_text = f"Min: {original_array.min():.4f}\nMax: {original_array.max():.4f}\nMean: {original_array.mean():.4f}\nStd: {original_array.std():.4f}"
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                 fontsize=9)

        if show_processed:
            # 3. Ảnh sau khi qua ViT processor (3 channels)
            processed_array = pixel_values.permute(1, 2, 0).numpy()  # CHW -> HWC

            # Denormalize cho visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            processed_display = (processed_array * std + mean)
            processed_display = np.clip(processed_display, 0, 1)

            im3 = ax3.imshow(processed_display)
            ax3.set_title('Processed for ViT\n(3-channel RGB representation)')
            ax3.axis('off')

            # Processed info
            processed_info = f"Shape: {pixel_values.shape}\nChannels: 3 (RGB)\nNormalized for ViT\nRange: [0, 1]"
            ax3.text(0.02, 0.98, processed_info, transform=ax3.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                     fontsize=9)

            # 4. So sánh từng channel sau processing
            ax4.plot(processed_array[:, :, 0].flatten()[:1000], 'r-', alpha=0.7, label='Red channel', linewidth=0.8)
            ax4.plot(processed_array[:, :, 1].flatten()[:1000], 'g-', alpha=0.7, label='Green channel', linewidth=0.8)
            ax4.plot(processed_array[:, :, 2].flatten()[:1000], 'b-', alpha=0.7, label='Blue channel', linewidth=0.8)
            ax4.set_title('RGB Channels After Processing\n(First 1000 pixels)')
            ax4.set_xlabel('Pixel Index')
            ax4.set_ylabel('Normalized Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # In thông tin chi tiết
        print(f"🗺️  Density map analysis from {dataset_name} dataset:")
        print(f"   Index: {random_idx}")
        print(f"   Label: {label} → {class_name}")
        print(f"   Original shape: {original_array.shape}")
        print(f"   Original dtype: {original_array.dtype}")
        print(f"   Density range: [{original_array.min():.6f}, {original_array.max():.6f}]")
        print(f"   Mean density: {original_array.mean():.6f}")
        print(f"   Std density: {original_array.std():.6f}")
        print(f"   Non-zero pixels: {np.count_nonzero(original_array)} / {original_array.size}")
        print(f"   Sparsity: {(1 - np.count_nonzero(original_array) / original_array.size) * 100:.2f}%")
        if show_processed:
            print(f"   Processed shape: {pixel_values.shape}")
        print(f"   Dataset size: {len(dataset)} samples")

    def check_density_distribution(self, dataset_type='train', num_samples=100):
        """
        Phân tích phân bố density values trong dataset
        """
        import matplotlib.pyplot as plt
        import random

        # Chọn dataset
        if dataset_type == 'train':
            dataset = self.train_dataset
            dataset_name = "Training"
        elif dataset_type == 'val':
            dataset = self.val_dataset
            dataset_name = "Validation"
        elif dataset_type == 'test':
            dataset = self.test_dataset
            dataset_name = "Test"
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")

        if dataset is None:
            print(f"❌ {dataset_name} dataset not available")
            return

        # Sample random indices
        sample_size = min(num_samples, len(dataset))
        random_indices = random.sample(range(len(dataset)), sample_size)

        # Collect statistics
        all_densities = []
        class_stats = {}

        for idx in random_indices:
            sample = dataset[idx]
            label = sample['labels'].item()
            density_map = dataset.data[idx]

            # Flatten density values
            flat_density = density_map.flatten()
            all_densities.extend(flat_density)

            # Per-class statistics
            if label not in class_stats:
                class_stats[label] = {
                    'min': [], 'max': [], 'mean': [], 'std': [],
                    'nonzero_ratio': [], 'total_density': []
                }

            class_stats[label]['min'].append(density_map.min())
            class_stats[label]['max'].append(density_map.max())
            class_stats[label]['mean'].append(density_map.mean())
            class_stats[label]['std'].append(density_map.std())
            class_stats[label]['nonzero_ratio'].append(np.count_nonzero(density_map) / density_map.size)
            class_stats[label]['total_density'].append(density_map.sum())

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall density distribution
        ax1.hist(all_densities, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'Overall Density Distribution\n({sample_size} samples from {dataset_name})')
        ax1.set_xlabel('Density Value')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale('log')  # Log scale for better visibility
        ax1.grid(True, alpha=0.3)

        # 2. Per-class mean density
        classes = sorted(class_stats.keys())
        class_names = [self.id2label[c] for c in classes]
        mean_densities = [np.mean(class_stats[c]['mean']) for c in classes]
        std_densities = [np.std(class_stats[c]['mean']) for c in classes]

        bars = ax2.bar(class_names, mean_densities, yerr=std_densities,
                       capsize=5, alpha=0.7, color=['orange', 'green'][:len(classes)])
        ax2.set_title('Mean Density per Class')
        ax2.set_ylabel('Mean Density Value')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, val in zip(bars, mean_densities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + std_densities[classes.index(classes[0])] * 0.1,
                     f'{val:.4f}', ha='center', va='bottom')

        # 3. Sparsity comparison
        sparsity_ratios = [1 - np.mean(class_stats[c]['nonzero_ratio']) for c in classes]
        ax3.bar(class_names, sparsity_ratios, alpha=0.7, color=['red', 'blue'][:len(classes)])
        ax3.set_title('Sparsity per Class')
        ax3.set_ylabel('Sparsity Ratio (0 = dense, 1 = sparse)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)

        # Add percentage labels
        for i, (name, ratio) in enumerate(zip(class_names, sparsity_ratios)):
            ax3.text(i, ratio + 0.02, f'{ratio * 100:.1f}%', ha='center', va='bottom')

        # 4. Total density per class
        total_densities = [np.mean(class_stats[c]['total_density']) for c in classes]
        ax4.bar(class_names, total_densities, alpha=0.7, color=['purple', 'brown'][:len(classes)])
        ax4.set_title('Total Density per Class')
        ax4.set_ylabel('Total Density Sum')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # Print detailed statistics
        print(f"📊 Density Distribution Analysis ({sample_size} samples):")
        print("=" * 60)

        for class_id in classes:
            class_name = self.id2label[class_id]
            stats = class_stats[class_id]

            print(f"\n🏷️  Class {class_id} ({class_name}):")
            print(f"   Samples analyzed: {len(stats['mean'])}")
            print(f"   Mean density: {np.mean(stats['mean']):.6f} ± {np.std(stats['mean']):.6f}")
            print(f"   Max density: {np.mean(stats['max']):.6f} ± {np.std(stats['max']):.6f}")
            print(f"   Sparsity: {(1 - np.mean(stats['nonzero_ratio'])) * 100:.2f}%")
            print(f"   Total density: {np.mean(stats['total_density']):.4f} ± {np.std(stats['total_density']):.4f}")


# Example usage
def run():
    print("=" * 50)

    # 2A. Prepare data from numpy arrays (NEW!)
    for i in range(5):
        fold = i + 1
        if fold < 3:
            continue

        for j in range(4):
            if j == 0:
                group = "100_400"
            elif j == 1:
                group = "400_800"
            elif j == 2:
                group = "800_1200"
            else:
                group = "1200_1800"

            finetuner = ViTFineTuner(model_name="google/vit-base-patch16-224", num_classes=2, fold=fold, group=group)

            finetuner.prepare_data_from_numpy(
                data_dir=os.path.join(config.FCGR_DATA_DIR, f"{group}/fold_{fold}"),
                class_names=["temperate", "virulent"]  # Optional
            )

            # 2B. Prepare data from folder structure (original)
            # finetuner.prepare_data_from_folder("path/to/your/dataset")

            # finetuner.visualize_random_sample('train', show_processed=True, use_colormap='viridis')
            # finetuner.visualize_random_sample('train', use_colormap='hot')  # Đỏ-vàng
            # finetuner.visualize_random_sample('train', use_colormap='plasma')  # Tím-hồng
            # finetuner.visualize_random_sample('train', use_colormap='jet')  # Cầu vồng
            # finetuner.check_density_distribution('train', num_samples=200)

            # 3. Initialize model
            finetuner.initialize_model()

            # 4. Setup training with RTX 5070ti optimizations
            finetuner.setup_training(
                output_dir="./my-vit-model",
                num_train_epochs=10,
                per_device_train_batch_size=48,  # Higher batch size for RTX 5070ti
                learning_rate=3e-5,
                fp16=True,  # Mixed precision
                save_strategy="epoch",
            )

            # 5. Train
            train_result = finetuner.train()

            # 6. Evaluate
            # eval_result = finetuner.evaluate()

            # 7. Test
            # test_metrics, predictions = finetuner.test()

            # 8. Save model
            # finetuner.save_model("./my-fine-tuned-vit")

            # 9. Predict single image
            # results = finetuner.predict_single_image("path/to/image.jpg")
            # print("Predictions:", results)

            # 10. Plot training history
            finetuner.plot_training_history()

            # print("\n📝 CODE TEMPLATE FOR NUMPY DATA:")
            # print("""
            #     # Quick start with numpy arrays
            #     finetuner = ViTFineTuner()
            #     finetuner.prepare_data_from_numpy(
            #         vectors_path="vectors.npy",
            #         labels_path="label.npy",
            #         class_names=["class1", "class2", "class3"]  # Optional
            #     )
            #     finetuner.initialize_model()
            #     finetuner.setup_training(output_dir="./my-model", num_train_epochs=5)
            #     finetuner.train()
            #     finetuner.evaluate()
            #     finetuner.save_model("./my-fine-tuned-vit")
            # """)
            #
            # print("\n📝 CODE TEMPLATE FOR FOLDER DATA:")
            # print("""
            #     # Quick start with folder structure
            #     finetuner = ViTFineTuner()
            #     finetuner.prepare_data_from_folder("your_dataset_folder")
            #     finetuner.initialize_model()
            #     finetuner.setup_training(output_dir="./my-model", num_train_epochs=5)
            #     finetuner.train()
            #     finetuner.evaluate()
            #     finetuner.save_model("./my-fine-tuned-vit")
            # """)


if __name__ == "__main__":
    run()
