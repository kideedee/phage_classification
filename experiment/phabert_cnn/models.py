"""
Custom model architectures for DNABERT-2 phage classification
Compatible with Hugging Face Transformers framework
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    PreTrainedModel,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput


def create_custom_config():
    """
    Tạo config cho custom model
    """
    config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config.num_labels = 2
    config.classifier_dropout = 0.1
    config.problem_type = "single_label_classification"
    return config


class DNABERT2_CNN_HybridForPhageClassification(PreTrainedModel):
    """
    Option 3: Hybrid model kết hợp DNABERT-2 và CNN layers
    Inspiration from DeePhage CNN approach combined with transformer features
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # DNABERT-2 backbone
        self.dnabert = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            add_pooling_layer=False
        )

        # Strategy: Freeze early layers, fine-tune later layers
        # This allows learning task-specific representations while maintaining pre-trained knowledge
        # for i, layer in enumerate(self.dnabert.encoder.layer[:8]):
        #     for param in layer.parameters():
        #         param.requires_grad = False
        # print("First 8 layers frozen, last 4 layers trainable")

        # CNN branches for local pattern detection (inspired by DeePhage)
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

        # Initialize new parameters
        for module in [self.cnn_branches, self.global_pooling, self.attention_pooling, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get DNABERT-2 outputs
        outputs = self.dnabert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Handle different output formats
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            sequence_output = outputs[0]  # Usually the first element
        else:
            sequence_output = outputs['last_hidden_state']
        # sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # Global transformer features with attention-based pooling
        attention_weights = self.attention_pooling(sequence_output)  # [batch_size, seq_len, 1]
        global_features = torch.sum(sequence_output * attention_weights, dim=1)  # [batch_size, 768]
        global_features = self.global_pooling(global_features)

        # CNN features (transpose for Conv1d: [batch_size, channels, seq_len])
        cnn_input = sequence_output.transpose(1, 2)  # [batch_size, 768, seq_len]

        cnn_features = []
        for branch in self.cnn_branches:
            branch_output = branch(cnn_input)  # [batch_size, 128, 1]
            cnn_features.append(branch_output.squeeze(2))  # [batch_size, 128]

        # Combine all features
        combined_features = torch.cat([global_features] + cnn_features, dim=1)

        # Classification
        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


# Additional utility functions
def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")

    return total_params, trainable_params


def verify_model_setup(model, sample_input_ids, sample_attention_mask=None):
    """Verify model setup with sample inputs"""
    model.eval()

    with torch.no_grad():
        if sample_attention_mask is not None:
            outputs = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
        else:
            outputs = model(input_ids=sample_input_ids)

    print(f"Model output shape: {outputs.logits.shape}")
    print(f"Sample predictions: {torch.softmax(outputs.logits, dim=-1)}")

    return outputs


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    config = create_custom_config()

    print("Testing frozen encoder model...")
    frozen_model = DNABERT2ForPhageClassification(config)
    count_parameters(frozen_model)

    print("\nTesting hybrid CNN model...")
    hybrid_model = DNABERT2_CNN_HybridForPhageClassification(config)
    count_parameters(hybrid_model)

    print("\nTesting default model...")
    default_model = load_dnabert2_for_classification()
    count_parameters(default_model)
