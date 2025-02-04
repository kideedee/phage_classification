import torch.nn as nn


class DeePhage(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Input shape: (batch_size, 1024, 1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=6, padding='same')
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.3)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, 1024, 1)
        x = x.transpose(1, 2)  # Shape: (batch_size, 1, 1024)

        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.avg_pool(x)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
