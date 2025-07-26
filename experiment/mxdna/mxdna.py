import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class BasicUnitNMS:
    """
    Python implementation of Non-Maximum Suppression for basic units
    Replaces the C++ implementation for simplicity
    """

    @staticmethod
    def apply(scores: torch.Tensor, kernel_sizes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (batch_size, seq_len, num_experts) - confidence scores
            kernel_sizes: (num_experts,) - kernel sizes for each expert
        Returns:
            mask: (batch_size, seq_len) - selected basic unit mask with kernel sizes
        """
        batch_size, seq_len, num_experts = scores.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=scores.device)

        for b in range(batch_size):
            # Flatten and sort all (position, expert) pairs by score
            flat_scores = scores[b].view(-1)
            sorted_indices = torch.argsort(flat_scores, descending=True)

            occupied = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)

            for idx in sorted_indices:
                pos = idx // num_experts
                expert_idx = idx % num_experts
                kernel_size = kernel_sizes[expert_idx].item()

                # Calculate region boundaries
                start = max(0, pos - kernel_size // 2)
                end = min(seq_len, pos + (kernel_size + 1) // 2)

                # Check if region is already occupied
                if not occupied[start:end].any():
                    occupied[start:end] = True
                    mask[b, pos] = kernel_size

        return mask


class MxDNAConvExpert(nn.Module):
    """Individual convolution expert for processing basic units of specific length"""

    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        # Following the paper's design with GLU and grouped convolution
        self.pointwise_in = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)

        # Fix: Calculate groups to ensure divisibility
        # Find the largest divisor of hidden_dim that is <= kernel_size
        groups = 1
        for g in range(min(hidden_dim, kernel_size), 0, -1):
            if hidden_dim % g == 0:
                groups = g
                break

        self.grouped_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pointwise_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # Pointwise convolution with GLU
        gate_input = self.pointwise_in(x)  # (B, L, 2*H)
        gate, input_proj = gate_input.chunk(2, dim=-1)  # Each (B, L, H)
        gated_input = input_proj * torch.sigmoid(gate)

        # Grouped convolution (convert to conv1d format)
        conv_input = gated_input.transpose(1, 2)  # (B, H, L)
        conv_output = self.grouped_conv(conv_input)  # (B, H, L)
        conv_output = conv_output.transpose(1, 2)  # (B, L, H)

        # Layer norm and activation
        normalized = self.layer_norm(conv_output)
        activated = F.silu(normalized)  # Swish activation

        # Final pointwise convolution
        output = self.pointwise_out(activated)

        return output


class MxDNAConvMoeBlock(nn.Module):
    """Sparse Mixture of Convolution Experts"""

    def __init__(self, hidden_dim: int, num_experts: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Expert kernel sizes from 1 to num_experts
        self.expert_kernel_sizes = torch.arange(1, num_experts + 1)

        # Router for scoring basic units
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Convolution experts
        self.experts = nn.ModuleList([
            MxDNAConvExpert(hidden_dim, k)
            for k in range(1, num_experts + 1)
        ])

        # Balancing loss weight
        self.balancing_loss_weight = 0.01

    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            training: whether in training mode
        Returns:
            basic_units: (batch_size, num_basic_units, hidden_dim)
            balancing_loss: scalar tensor
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Router scoring
        router_logits = self.router(hidden_states)  # (B, L, num_experts)

        # Add jitter noise during training for ambiguity
        if training:
            jitter = torch.empty_like(router_logits).uniform_(0.99, 1.01)
            router_logits = router_logits * jitter

        # Non-maximum suppression to select basic units
        basic_unit_mask = BasicUnitNMS.apply(
            router_logits,
            self.expert_kernel_sizes.to(hidden_states.device)
        )  # (B, L)

        # Calculate softmax weights for experts
        router_probs = F.softmax(router_logits, dim=-1)  # (B, L, num_experts)

        # Process each expert and aggregate
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)  # (B, L, H)
            # Weight by router probability
            expert_mask = (basic_unit_mask == (i + 1)).float().unsqueeze(-1)  # (B, L, 1)
            # Ensure all tensors have the same sequence length
            seq_len = min(expert_output.shape[1], expert_mask.shape[1], router_probs.shape[1])
            weighted_output = expert_output[:, :seq_len] * expert_mask[:, :seq_len] * router_probs[:, :seq_len, i:i + 1]
            expert_outputs.append(weighted_output)

        # Sum all expert outputs
        aggregated = torch.stack(expert_outputs, dim=0).sum(dim=0)  # (B, L, H)

        # Extract only selected basic units
        basic_unit_positions = (basic_unit_mask > 0)  # (B, L)
        basic_units = []
        for b in range(batch_size):
            selected = aggregated[b][basic_unit_positions[b]]  # (num_selected, H)
            basic_units.append(selected)

        # Pad to same length for batching
        max_units = max(len(units) for units in basic_units) if basic_units else 1
        if max_units == 0:
            max_units = 1  # Prevent empty tensors

        padded_basic_units = torch.zeros(
            batch_size, max_units, hidden_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for b, units in enumerate(basic_units):
            if len(units) > 0:
                padded_basic_units[b, :len(units)] = units

        # Calculate balancing loss to prevent expert collapse
        expert_usage = torch.zeros(self.num_experts, device=hidden_states.device)
        for i in range(self.num_experts):
            expert_usage[i] = (basic_unit_mask == (i + 1)).float().sum()

        # Normalize and calculate variance for balancing
        total_usage = expert_usage.sum()
        if total_usage > 0:
            expert_usage_normalized = expert_usage / total_usage
            balancing_loss = expert_usage_normalized.var() * self.balancing_loss_weight
        else:
            balancing_loss = torch.tensor(0.0, device=hidden_states.device)

        return padded_basic_units, balancing_loss


class MxDNADeformableConvBlock(nn.Module):
    """Deformable Convolution for assembling basic units into final tokens"""

    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Offset and modulation prediction
        self.offset_conv = nn.Linear(hidden_dim, kernel_size, bias=False)
        self.modulation_conv = nn.Linear(hidden_dim, kernel_size, bias=False)

        # Main convolution weights
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim, kernel_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.ones_(self.modulation_conv.weight)

    def forward(self, basic_units: torch.Tensor) -> torch.Tensor:
        """
        Args:
            basic_units: (batch_size, num_basic_units, hidden_dim)
        Returns:
            final_tokens: (batch_size, num_basic_units, hidden_dim)
        """
        batch_size, num_units, hidden_dim = basic_units.shape

        if num_units == 0:
            return basic_units

        # Predict offsets and modulation factors
        offset = self.offset_conv(basic_units)  # (B, N, kernel_size)
        modulation = torch.sigmoid(self.modulation_conv(basic_units))  # (B, N, kernel_size)

        # Apply deformable convolution
        final_tokens = torch.zeros_like(basic_units)

        for b in range(batch_size):
            for i in range(num_units):
                output = torch.zeros(hidden_dim, device=basic_units.device)

                for k in range(self.kernel_size):
                    # Calculate sampling position with offset
                    base_pos = i + k - self.kernel_size // 2
                    sampling_pos = base_pos + offset[b, i, k].item()

                    # Bilinear interpolation for fractional positions
                    if 0 <= sampling_pos < num_units:
                        floor_pos = int(math.floor(sampling_pos))
                        ceil_pos = min(floor_pos + 1, num_units - 1)
                        weight_ceil = sampling_pos - floor_pos
                        weight_floor = 1 - weight_ceil

                        # Interpolated feature
                        if floor_pos < num_units:
                            interpolated = (weight_floor * basic_units[b, floor_pos] +
                                            weight_ceil * basic_units[b, ceil_pos])

                            # Apply convolution weight and modulation
                            conv_weight = self.weight[:, :, k]  # (hidden_dim, hidden_dim)
                            modulated = interpolated * modulation[b, i, k]
                            output += torch.matmul(conv_weight, modulated)

                final_tokens[b, i] = output

        return final_tokens


class MxDNALearntTokenizationLayer(nn.Module):
    """Complete Learnt Tokenization Module combining MoE and Deformable Conv"""

    def __init__(self, hidden_dim: int, num_experts: int = 10, deformable_kernel_size: int = 3):
        super().__init__()
        self.conv_moe = MxDNAConvMoeBlock(hidden_dim, num_experts)
        self.deformable_conv = MxDNADeformableConvBlock(hidden_dim, deformable_kernel_size)

    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            training: whether in training mode
        Returns:
            final_tokens: (batch_size, num_tokens, hidden_dim)
            balancing_loss: scalar tensor
        """
        # Step 1: Basic Units Recognition
        basic_units, balancing_loss = self.conv_moe(hidden_states, training)

        # Step 2: Basic Units Assembly
        final_tokens = self.deformable_conv(basic_units)

        return final_tokens, balancing_loss


class MxDNAForPhageClassification(nn.Module):
    """Complete MxDNA model for Temperate vs Virulent Phage Classification"""

    def __init__(
            self,
            vocab_size: int = 4,  # A, T, C, G
            hidden_dim: int = 512,
            num_layers: int = 22,
            num_attention_heads: int = 16,
            intermediate_size: int = 2048,
            max_position_embeddings: int = 4096,
            num_experts: int = 10,
            num_classes: int = 2,  # temperate vs virulent
            dropout_prob: float = 0.1,
            tokenization_layer_position: int = 5  # Replace 5th transformer layer
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tokenization_layer_position = tokenization_layer_position

        # Embeddings
        self.nucleotide_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # Pre-tokenization transformer layers
        self.pre_tokenization_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout_prob,
                activation='gelu',
                batch_first=True
            ) for _ in range(tokenization_layer_position)
        ])

        # Learnt Tokenization Module
        self.learnt_tokenization = MxDNALearntTokenizationLayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            deformable_kernel_size=3
        )

        # Post-tokenization transformer layers
        self.post_tokenization_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout_prob,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers - tokenization_layer_position)
        ])

        # Cross attention for pretraining (optional)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout_prob,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert nucleotide sequence to embeddings"""
        batch_size, seq_len = input_ids.shape

        # Position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        inputs_embeds = self.nucleotide_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_tokenization_loss: bool = True
    ) -> dict:
        """
        Args:
            input_ids: (batch_size, seq_len) - nucleotide sequence (0=A, 1=T, 2=C, 3=G)
            attention_mask: (batch_size, seq_len) - attention mask
            return_tokenization_loss: whether to return balancing loss
        Returns:
            dict with 'logits', 'loss' (if training), 'balancing_loss'
        """
        batch_size, seq_len = input_ids.shape

        # Encode sequence to embeddings
        hidden_states = self.encode_sequence(input_ids)

        # Pre-tokenization transformer layers
        for layer in self.pre_tokenization_layers:
            if attention_mask is not None:
                # Convert attention mask for transformer
                hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask.bool())
            else:
                hidden_states = layer(hidden_states)

        # Store pre-tokenization states for cross-attention
        pre_tokenization_states = hidden_states

        # Learnt Tokenization
        tokenized_states, balancing_loss = self.learnt_tokenization(
            hidden_states,
            training=self.training
        )

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokenized_states = torch.cat([cls_tokens, tokenized_states], dim=1)

        # Post-tokenization transformer layers
        for layer in self.post_tokenization_layers:
            tokenized_states = layer(tokenized_states)

        # Classification using CLS token
        cls_representation = tokenized_states[:, 0]  # First token is CLS
        logits = self.classifier(cls_representation)

        outputs = {
            'logits': logits,
            'tokenized_states': tokenized_states,
            'pre_tokenization_states': pre_tokenization_states
        }

        if return_tokenization_loss:
            outputs['balancing_loss'] = balancing_loss

        return outputs


# Utility functions for data processing
def encode_dna_sequence(sequence: str) -> list:
    """Convert DNA sequence string to token IDs"""
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    return [mapping.get(nucleotide.upper(), 0) for nucleotide in sequence]


def create_attention_mask(sequences: list) -> torch.Tensor:
    """Create attention mask for padded sequences"""
    max_len = max(len(seq) for seq in sequences)
    mask = torch.zeros(len(sequences), max_len)
    for i, seq in enumerate(sequences):
        mask[i, :len(seq)] = 1
    return mask


# Example usage and training setup
def create_model_for_phage_classification():
    """Create MxDNA model for phage classification"""
    model = MxDNAForPhageClassification(
        vocab_size=4,  # A, T, C, G
        hidden_dim=512,
        num_layers=22,
        num_attention_heads=16,
        intermediate_size=2048,
        max_position_embeddings=4096,
        num_experts=10,
        num_classes=2,  # temperate vs virulent
        dropout_prob=0.1,
        tokenization_layer_position=5
    )
    return model


if __name__ == "__main__":
    # Example usage
    model = create_model_for_phage_classification()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Example DNA sequences (replace with your actual data)
    sequences = [
        "ATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTA"
    ]

    # Encode sequences
    encoded_sequences = [encode_dna_sequence(seq) for seq in sequences]

    # Pad sequences
    max_len = max(len(seq) for seq in encoded_sequences)
    padded_sequences = []
    for seq in encoded_sequences:
        padded = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded)

    # Create tensors
    input_ids = torch.tensor(padded_sequences)
    attention_mask = create_attention_mask(encoded_sequences)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Balancing loss: {outputs['balancing_loss'].item()}")
        print(f"Tokenized states shape: {outputs['tokenized_states'].shape}")
        print("Model created successfully!")