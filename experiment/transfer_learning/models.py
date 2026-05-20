import warnings

from transformers import PreTrainedModel

warnings.filterwarnings('ignore')


def create_custom_config():
    """
    Tạo config cho custom model
    """
    config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config.num_labels = 2
    config.classifier_dropout = 0.1
    config.problem_type = "single_label_classification"
    return config


class Dnabert2CnnModel(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2CnnModel, self).__init__(config)

        # Load DNABERT-2 pre-trained model
        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        self.cnn_branches = nn.ModuleList([
            # Branch 1: Small receptive field for local motifs
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            # Branch 2: Medium receptive field
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            # Branch 3: Large receptive field for longer patterns
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        ])

        # Global pooling for transformer features
        self.global_pooling = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Attention-based pooling for sequence representation
        self.attention_pooling = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Final classifier
        total_features = 128 * 3 + 128  # CNN features + global features
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Extract features từ DNABERT-2
        with torch.set_grad_enabled(self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            # Shape: [batch_size, seq_len, 768]
            # hidden_states = dnabert_outputs.last_hidden_state
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]  # Usually the first element
            else:
                hidden_states = outputs['last_hidden_state']

        # # Transpose cho Conv1D: [batch_size, seq_len, 768] -> [batch_size, 768, seq_len]
        # hidden_states = hidden_states.transpose(1, 2)
        #
        # # Pass qua CNN layers
        # conv_output = self.conv_layers(hidden_states)  # [batch_size, 64, seq_len]
        #
        # # Global pooling
        # avg_pool = self.global_avg_pool(conv_output).squeeze(-1)  # [batch_size, 64]
        # max_pool = self.global_max_pool(conv_output).squeeze(-1)  # [batch_size, 64]
        #
        # # Concatenate pooled features
        # pooled_features = torch.cat([avg_pool, max_pool], dim=1)  # [batch_size, 128]
        #
        # # Classification
        # logits = self.classifier(pooled_features)

        # Global transformer features with attention-based pooling
        attention_weights = self.attention_pooling(hidden_states)  # [batch_size, seq_len, 1]
        global_features = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, 768]
        global_features = self.global_pooling(global_features)

        # CNN features (transpose for Conv1d: [batch_size, channels, seq_len])
        cnn_input = hidden_states.transpose(1, 2)  # [batch_size, 768, seq_len]

        cnn_features = []
        for branch in self.cnn_branches:
            branch_output = branch(cnn_input)  # [batch_size, 128, 1]
            cnn_features.append(branch_output.squeeze(2))  # [batch_size, 128]

        # Combine all features
        combined_features = torch.cat([global_features] + cnn_features, dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits


class Dnabert2DenseModel(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2DenseModel, self).__init__(config)

        # Load DNABERT-2 pre-trained model
        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        # Attention-based pooling for sequence representation
        self.attention_pooling = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Single dense processing layer
        self.feature_processor = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)
        )

        # Final classifier
        classifier_dropout = getattr(config, 'classifier_dropout', dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Extract features from DNABERT-2
        with torch.set_grad_enabled(self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)

            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # Shape: [batch_size, seq_len, 768]

        # Attention-based pooling to get sequence representation
        attention_weights = self.attention_pooling(hidden_states)  # [batch_size, seq_len, 1]
        sequence_features = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, 768]

        # Process features through dense layers
        processed_features = self.feature_processor(sequence_features)  # [batch_size, 128]

        # Classification
        logits = self.classifier(processed_features)

        return logits


class Dnabert2CnnModelNoAttention(PreTrainedModel):
    """
    Ablation variant: PhaBERT-CNN without attention-based pooling.
    Attention pooling is replaced by masked mean pooling for global
    transformer feature aggregation.
    """

    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2CnnModelNoAttention, self).__init__(config)

        # Load DNABERT-2 pre-trained model
        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        self.cnn_branches = nn.ModuleList([
            # Branch 1: Small receptive field for local motifs
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            # Branch 2: Medium receptive field
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            # Branch 3: Large receptive field for longer patterns
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        ])

        # Global pooling for transformer features (NO attention pooling)
        # Mean-pooled features are projected through this layer
        self.global_pooling = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # NOTE: self.attention_pooling is REMOVED in this ablation variant.
        # Instead, masked mean pooling is used directly in forward().

        # Final classifier (same architecture as full model)
        total_features = 128 * 3 + 128  # CNN features + global features
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Extract features từ DNABERT-2
        with torch.set_grad_enabled(self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # ============================================================
        # ABLATION: Masked mean pooling (replaces attention pooling)
        # ============================================================
        if attention_mask is not None:
            # Expand mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Sum of hidden states weighted by mask
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)  # [batch_size, 768]
            # Count of non-padding tokens
            count = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # [batch_size, 1]
            global_features = sum_hidden / count  # [batch_size, 768]
        else:
            # Fallback: simple mean pooling without mask
            global_features = torch.mean(hidden_states, dim=1)  # [batch_size, 768]

        global_features = self.global_pooling(global_features)  # [batch_size, 128]

        # CNN features (same as full model)
        cnn_input = hidden_states.transpose(1, 2)  # [batch_size, 768, seq_len]

        cnn_features = []
        for branch in self.cnn_branches:
            branch_output = branch(cnn_input)  # [batch_size, 128, 1]
            cnn_features.append(branch_output.squeeze(2))  # [batch_size, 128]

        # Combine all features
        combined_features = torch.cat([global_features] + cnn_features, dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits


import warnings

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

warnings.filterwarnings('ignore')


def create_custom_config():
    config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config.num_labels = 2
    config.classifier_dropout = 0.1
    config.problem_type = "single_label_classification"
    return config


class Dnabert2CnnModelNoK3(PreTrainedModel):
    """
    Ablation variant: PhaBERT-CNN without kernel_size=3 branch.
    Retains kernel_size=5 and kernel_size=7 branches only.
    """

    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2CnnModelNoK3, self).__init__(config)

        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        # Only kernel_size=5 and kernel_size=7 branches
        self.cnn_branches = nn.ModuleList([
            # Branch: Medium receptive field (kernel_size=5)
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            # Branch: Large receptive field (kernel_size=7)
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        ])

        self.global_pooling = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.attention_pooling = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 128 * 2 (CNN) + 128 (global) = 384
        total_features = 128 * 2 + 128
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.set_grad_enabled(self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # Global features with attention pooling
        attention_weights = self.attention_pooling(hidden_states)
        global_features = torch.sum(hidden_states * attention_weights, dim=1)
        global_features = self.global_pooling(global_features)

        # CNN features
        cnn_input = hidden_states.transpose(1, 2)

        cnn_features = []
        for branch in self.cnn_branches:
            branch_output = branch(cnn_input)
            cnn_features.append(branch_output.squeeze(2))

        combined_features = torch.cat([global_features] + cnn_features, dim=1)
        logits = self.classifier(combined_features)

        return logits


class Dnabert2APModel(PreTrainedModel):
    """
    Ablation: DNABERT-2 + Attention Pooling only (no CNN).

    Architecture:
        DNABERT-2 [768-dim] -> Attention Pooling -> Linear [768->128] -> Classifier [128->2]

    Giu nguyen ten class Dnabert2CnnModel de tuong thich voi run.py
    (import interface khong thay doi).
    """

    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2APModel, self).__init__(config)

        # Load DNABERT-2 pre-trained model
        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        # --- CNN branches REMOVED ---

        # Global pooling for transformer features
        self.global_pooling = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Attention-based pooling for sequence representation
        # (Lin et al., 2017: A Structured Self-Attentive Sentence Embedding)
        self.attention_pooling = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Final classifier - input: 128-dim (attention global only)
        total_features = 128  # Only global features, no CNN
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Extract features from DNABERT-2
        with torch.set_grad_enabled(
                self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())
        ):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # Global transformer features with attention-based pooling
        # hidden_states: [batch_size, seq_len, 768]
        attention_weights = self.attention_pooling(hidden_states)  # [batch_size, seq_len, 1]
        global_features = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, 768]
        global_features = self.global_pooling(global_features)  # [batch_size, 128]

        # --- No CNN features ---

        # Classification (128-dim input)
        logits = self.classifier(global_features)

        return logits


class Dnabert2OnlyCNNModel(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2OnlyCNNModel, self).__init__(config)

        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        self.cnn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        ])

        # ĐÃ XÓA: global_pooling, attention_pooling

        total_features = 128 * 3  # Chỉ còn CNN features
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.set_grad_enabled(self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # Chỉ CNN features
        cnn_input = hidden_states.transpose(1, 2)  # [batch, 768, seq_len]

        cnn_features = []
        for branch in self.cnn_branches:
            branch_output = branch(cnn_input)       # [batch, 128, 1]
            cnn_features.append(branch_output.squeeze(2))  # [batch, 128]

        combined_features = torch.cat(cnn_features, dim=1)  # [batch, 384]
        logits = self.classifier(combined_features)

        return logits


class Dnabert2PoolingOnly(PreTrainedModel):
    """Ablation: DNABERT-2 + Attention Pooling only (no CNN branches)."""

    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2PoolingOnly, self).__init__(config)

        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        # Attention-based pooling (giữ nguyên)
        self.attention_pooling = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Global pooling (giữ nguyên)
        self.global_pooling = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Classifier — input chỉ còn 128 (không có 128*3 từ CNN)
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.set_grad_enabled(
                self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())
        ):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # Attention pooling → global features
        attention_weights = self.attention_pooling(hidden_states)  # [batch, seq_len, 1]
        global_features = torch.sum(hidden_states * attention_weights, dim=1)  # [batch, 768]
        global_features = self.global_pooling(global_features)  # [batch, 128]

        # Classification (chỉ từ pooled features, không có CNN)
        logits = self.classifier(global_features)

        return logits

class Dnabert2CnnAblationNoK5(PreTrainedModel):
    """Ablation: Bỏ CNN branch kernel_size=5, giữ k=3 và k=7."""

    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2CnnAblationNoK5, self).__init__(config)

        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        # Chỉ giữ branch k=3 và k=7, bỏ k=5
        self.cnn_branches = nn.ModuleList([
            # Branch 1: kernel_size=3
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            # Branch 2: kernel_size=7 (bỏ k=5)
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        ])

        self.global_pooling = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.attention_pooling = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # total_features: 128*2 (2 CNN branches) + 128 (global) = 384
        total_features = 128 * 2 + 128
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        with torch.set_grad_enabled(
            self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())
        ):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # Attention pooling
        attention_weights = self.attention_pooling(hidden_states)
        global_features = torch.sum(hidden_states * attention_weights, dim=1)
        global_features = self.global_pooling(global_features)

        # CNN features (chỉ k=3 và k=7)
        cnn_input = hidden_states.transpose(1, 2)

        cnn_features = []
        for branch in self.cnn_branches:
            branch_output = branch(cnn_input)
            cnn_features.append(branch_output.squeeze(2))

        combined_features = torch.cat([global_features] + cnn_features, dim=1)

        logits = self.classifier(combined_features)

        return logits

class Dnabert2CnnNoK7Model(PreTrainedModel):
    def __init__(self, config, dropout_rate=0.1, num_classes=2):
        super(Dnabert2CnnNoK7Model, self).__init__(config)

        # Load DNABERT-2 pre-trained model
        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        for param in self.dnabert.parameters():
            param.requires_grad = False

        self.cnn_branches = nn.ModuleList([
            # Branch 1: Small receptive field for local motifs
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
            # Branch 2: Medium receptive field
            nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ),
        ])

        # Global pooling for transformer features
        self.global_pooling = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Attention-based pooling for sequence representation
        self.attention_pooling = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Final classifier
        total_features = 128 * 2 + 128  # CNN (k=3, k=5) + global features = 384
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, config.num_labels)
        )

    def freeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = False

    def unfreeze_dnabert(self):
        for param in self.dnabert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Extract features từ DNABERT-2
        with torch.set_grad_enabled(self.dnabert.training and any(p.requires_grad for p in self.dnabert.parameters())):
            outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs['last_hidden_state']

        # Global transformer features with attention-based pooling
        attention_weights = self.attention_pooling(hidden_states)  # [batch_size, seq_len, 1]
        global_features = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, 768]
        global_features = self.global_pooling(global_features)

        # CNN features (transpose for Conv1d: [batch_size, channels, seq_len])
        cnn_input = hidden_states.transpose(1, 2)  # [batch_size, 768, seq_len]

        cnn_features = []
        for branch in self.cnn_branches:
            branch_output = branch(cnn_input)  # [batch_size, 128, 1]
            cnn_features.append(branch_output.squeeze(2))  # [batch_size, 128]

        # Combine: global (128) + k=3 (128) + k=5 (128) = 384
        combined_features = torch.cat([global_features] + cnn_features, dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits