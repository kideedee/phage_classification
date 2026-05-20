from torch import nn


class OneHotTower(nn.Module):
    def __init__(self, seq_length, hidden_dims=[512, 256, 128]):
        super().__init__()
        # Input: [batch_size, seq_length, 4] (A,T,G,C)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2])
        )

    def forward(self, x):
        # x shape: [batch_size, seq_length, 4]
        x = x.transpose(1, 2)  # [batch_size, 4, seq_length]
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # [batch_size, 256]
        x = self.fc_layers(x)
        return x  # [batch_size, 128]
