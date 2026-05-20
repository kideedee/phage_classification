import torch
import torch.nn.functional as F
from torch import nn


class SimpleFCN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # Binary classification
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=6, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        self.fc1 = nn.Linear(64 + 768, 256)  # 64 from CNN + 768 from BERT
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, one_hot, dna_bert):
        one_hot = one_hot.permute(0, 2, 1)
        cnn_feat = self.conv_layers(one_hot)
        cnn_feat = cnn_feat.squeeze(-1)  # Remove the last dimension

        x = torch.cat((cnn_feat, dna_bert), dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        pred = F.log_softmax(x, dim=1)
        return pred
