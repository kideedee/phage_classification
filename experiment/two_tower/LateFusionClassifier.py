import torch
from torch import nn

from experiment.two_tower.DNABert2Tower import DNABert2Tower
from experiment.two_tower.OneHotTower import OneHotTower


class BinaryLateFusionClassifier(nn.Module):
    """
    Binary classification model using late fusion of one-hot and DNABERT2 features
    """

    def __init__(self, seq_length, fusion_dim=128, dropout_rate=0.3,
                 use_attention=True, freeze_bert=True):
        super().__init__()

        self.use_attention = use_attention
        self.fusion_dim = fusion_dim

        # Initialize towers
        self.onehot_tower = OneHotTower(seq_length, hidden_dims=[512, 256, fusion_dim])
        self.dnabert2_tower = DNABert2Tower(
            dnabert2_dim=768,
            hidden_dims=[512, 256, fusion_dim],
            freeze_bert=freeze_bert
        )

        # Fusion mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

            # Additional cross-attention between modalities
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )

        # Fusion and classification layers
        fusion_input_dim = fusion_dim * 2 if not use_attention else fusion_dim * 3

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),

            # Binary classification: single output with sigmoid
            nn.Linear(fusion_dim // 2, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, onehot_seq, input_ids, attention_mask):
        # Extract features from both towers
        onehot_features = self.onehot_tower(onehot_seq)  # [batch, fusion_dim]
        dnabert2_features = self.dnabert2_tower(input_ids, attention_mask)  # [batch, fusion_dim]

        if self.use_attention:
            # Self-attention within each modality
            features = torch.stack([onehot_features, dnabert2_features], dim=1)  # [batch, 2, fusion_dim]

            # Self-attention
            attended_features, _ = self.attention(features, features, features)

            # Cross-attention between modalities
            onehot_attended = attended_features[:, 0:1, :]  # [batch, 1, fusion_dim]
            dnabert2_attended = attended_features[:, 1:2, :]  # [batch, 1, fusion_dim]

            # Cross attention: onehot queries dnabert2
            cross_onehot, _ = self.cross_attention(onehot_attended, dnabert2_attended, dnabert2_attended)
            # Cross attention: dnabert2 queries onehot
            cross_dnabert2, _ = self.cross_attention(dnabert2_attended, onehot_attended, onehot_attended)

            # Combine all representations
            fused_features = torch.cat([
                attended_features[:, 0, :],  # Self-attended onehot
                attended_features[:, 1, :],  # Self-attended dnabert2
                (cross_onehot.squeeze(1) + cross_dnabert2.squeeze(1)) / 2  # Cross-attended
            ], dim=1)
        else:
            # Simple concatenation
            fused_features = torch.cat([onehot_features, dnabert2_features], dim=1)

        # Final classification
        logits = self.fusion_layer(fused_features)  # [batch, 1]

        return logits.squeeze(-1)  # [batch] - remove last dimension for binary classification
