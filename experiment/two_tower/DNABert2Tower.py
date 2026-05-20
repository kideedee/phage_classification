import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class DNABERT2Tower(nn.Module):
    """Tower for processing DNABERT2 features"""

    def __init__(self, dnabert2_dim=768, hidden_dims=[512, 256, 128], freeze_bert=True):
        super().__init__()
        # Load DNABERT2
        self.dnabert2 = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")

        # Freeze DNABERT2 parameters if specified
        if freeze_bert:
            for param in self.dnabert2.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(dnabert2_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dims[1], hidden_dims[2])
        )

    def forward(self, input_ids, attention_mask):
        if self.training:
            outputs = self.dnabert2(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.dnabert2(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

        x = self.projection(pooled_output)
        return F.normalize(x, p=2, dim=1)  # L2 normalize for better fusion
