import gc
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
# Sử dụng Focal Loss từ thư viện có sẵn
from torchvision.ops import sigmoid_focal_loss
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Wrapper cho Focal Loss từ torchvision để phù hợp với trường hợp sử dụng của chúng ta
class FocalLoss(nn.Module):
    """
    Focal Loss wrapper sử dụng torchvision.ops.sigmoid_focal_loss
    Hỗ trợ cả binary và multi-class classification.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tham số cân bằng lớp
        self.gamma = gamma  # Tham số focusing
        self.reduction = reduction

    def forward(self, logits, targets):
        # Xử lý binary classification
        if logits.shape[1] == 2:
            # Chuyển đổi targets sang định dạng phù hợp
            if targets.dim() == 1:
                # Nếu targets là indices, chuyển sang định dạng one-hot
                targets_oh = F.one_hot(targets, num_classes=2).float()
            else:
                targets_oh = targets

            # Sử dụng sigmoid_focal_loss từ torchvision
            # Chú ý: torchvision chỉ hỗ trợ sigmoid focal loss, không phải softmax
            loss = sigmoid_focal_loss(
                logits,
                targets_oh,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction
            )
            return loss
        else:
            # Xử lý multi-class case (nếu cần)
            if targets.dim() == 1:
                targets_oh = F.one_hot(targets, num_classes=logits.shape[1]).float()
            else:
                targets_oh = targets

            # Sử dụng sigmoid_focal_loss cho từng lớp và tổng hợp
            loss = sigmoid_focal_loss(
                logits,
                targets_oh,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction
            )
            return loss


# Custom trainer với focal loss
class FocalLossTrainer(Trainer):
    """
    Custom Trainer class với Focal Loss cho dữ liệu mất cân bằng.
    """

    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
        super(FocalLossTrainer, self).__init__(**kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Ghi đè phương thức compute_loss để sử dụng focal loss
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Áp dụng focal loss
        focal_loss_fct = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        loss = focal_loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        """
        Custom prediction step để giảm sử dụng bộ nhớ trong quá trình đánh giá.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)

        # Xử lý khác nhau cho _prepare_inputs trong transformers 4.28.0
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if prediction_loss_only:
                    loss = outputs[0].mean().detach()
                    return (loss, None, None)
                else:
                    loss = outputs[0].mean().detach()
                    logits = outputs[1]
                    labels = tuple(inputs.get(name).detach().cpu() for name in self.label_names)
                    if len(labels) == 1:
                        labels = labels[0]
            else:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                loss = None
                if self.args.past_index >= 0:
                    logits = outputs[0]
                else:
                    logits = outputs
                labels = None

        # Di chuyển logits về CPU để tiết kiệm bộ nhớ GPU
        if not prediction_loss_only:
            logits = logits.detach().cpu()

        return (loss, logits, labels)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred):
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

    # Safely compute ROC AUC with batching for large datasets
    try:
        if isinstance(logits, np.ndarray) and logits.shape[0] > 1e6:
            # Calculate ROC AUC in batches
            batch_size = 1000
            y_scores = []
            for i in range(0, len(logits), batch_size):
                y_scores.extend(logits[i:i + batch_size, 1])
            roc_auc = roc_auc_score(labels, np.array(y_scores))
        else:
            roc_auc = roc_auc_score(labels, logits[:, 1])
    except:
        roc_auc = 0

    # Explicitly free memory
    del logits, preds

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'roc_auc': roc_auc,
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'tp': float(tp)
    }


def main():
    # Set seed for reproducibility
    set_seed()

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Monitor initial GPU memory
        initial_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"Initial GPU memory usage: {initial_mem:.2f} GB")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        # Clear memory and cache before training
        gc.collect()
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.deterministic = True

        # Log CUDA information
        print(f"CUDA Version: {torch.version.cuda}")
        print(
            f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")

        # RTX 5070 Ti optimized kernel tuning
        try:
            # Set environment variables
            os.environ['CUDA_AUTO_TUNE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            print("Set optimized kernel autotuning for RTX 5070 Ti")
        except:
            pass
    else:
        print("No GPU available, training on CPU")

    # Load datasets
    print("Loading datasets...")
    tokenized_train = load_from_disk("processed_train_dataset")
    tokenized_val = load_from_disk("processed_val_dataset")

    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        num_labels=2,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimize model for RTX 5070 Ti memory constraints
    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Apply gradient checkpointing for memory efficiency
    if hasattr(model.bert, "encoder") and hasattr(model.bert.encoder, "gradient_checkpointing"):
        model.bert.encoder.gradient_checkpointing = True
        print("Enabled gradient checkpointing for encoder")

    # Log memory savings
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_after:,}")

    # Tính toán tỷ lệ các lớp để tối ưu hóa alpha cho focal loss
    # Trong trường hợp không có thông tin, dùng giá trị mặc định
    # Ideally, calculate this from your dataset
    try:
        # Đếm số lượng mẫu trong mỗi lớp (0 và 1)
        label_counts = {0: 0, 1: 0}
        for item in tokenized_train:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        # Tính tỉ lệ lớp thiểu số (giả sử lớp 1 là lớp positive và là lớp thiểu số)
        total_samples = sum(label_counts.values())
        pos_ratio = label_counts[1] / total_samples

        # Thiết lập alpha dựa trên tỉ lệ lớp
        # alpha nên là tỉ lệ nghịch với tần suất của lớp (giá trị cao hơn cho lớp ít phổ biến)
        focal_alpha = 1 - pos_ratio  # alpha cao hơn cho lớp positive nếu nó là lớp thiểu số
        focal_gamma = 2.0  # Giá trị mặc định từ bài báo gốc

        print(f"Class distribution - Negative: {label_counts[0]}, Positive: {label_counts[1]}")
        print(f"Using dynamic Focal Loss with alpha={focal_alpha:.4f}, gamma={focal_gamma}")
    except Exception as e:
        # Sử dụng giá trị mặc định nếu có lỗi
        focal_alpha = 0.25  # Giá trị mặc định từ bài báo gốc
        focal_gamma = 2.0
        print(f"Using default Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")

    training_args = TrainingArguments(
        output_dir="output",
        # Learning rate and optimization
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Modern optimizer settings
        optim="adamw_torch_fused",  # Optimized for CUDA

        # Batch size and epochs
        per_device_train_batch_size=24,  # Thiết lập cho RTX 5070 Ti
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=2,  # Accumulate for effective larger batch
        num_train_epochs=5,

        # Precision settings - modern approach
        bf16=True,  # Better than fp16 on RTX 5070 Ti
        fp16=False,  # Use either bf16 OR fp16, not both

        # Memory optimization
        gradient_checkpointing=True,  # Trades compute for memory savings

        # Evaluation and saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # Keep best and last
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Quan trọng cho dữ liệu mất cân bằng
        greater_is_better=True,

        # Warmup and scheduling
        lr_scheduler_type="cosine",  # Better convergence
        warmup_ratio=0.1,

        # System utilization
        dataloader_num_workers=6,  # Better CPU utilization
        dataloader_pin_memory=True,  # Faster data transfer to GPU

        # Logging
        logging_dir="logs",
        logging_steps=10,
        logging_first_step=True,
        report_to=None,

        # Hub settings
        push_to_hub=False,
    )

    # Create our custom trainer with Focal Loss
    trainer = FocalLossTrainer(
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Monitor GPU memory before training
    if torch.cuda.is_available():
        before_train_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"GPU memory before training: {before_train_mem:.2f} GB")

    # Train model
    print("Starting training with Focal Loss and RTX 5070 Ti optimizations...")
    trainer.train()

    # Free up memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        current_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"GPU memory after training: {current_mem:.2f} GB")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

    # Save model - use half precision to reduce file size
    print("Saving model in optimized format...")

    # Save model
    model_save_path = "./best_dnabert2_phage_classifier_focal_loss"
    trainer.save_model(model_save_path)

    # Log final statistics
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        final_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"Peak GPU memory usage: {peak_mem:.2f} GB")
        print(f"Final GPU memory usage: {final_mem:.2f} GB")

    print("Training completed and model saved with Focal Loss!")


if __name__ == "__main__":
    main()
