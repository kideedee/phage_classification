import torch
import torch.nn as nn
from torchvision import models


class ResNetBinaryClassifier(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, freeze_backbone=False, input_channels=5):
        super(ResNetBinaryClassifier, self).__init__()

        self.input_channels = input_channels

        # Tải pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Modify first conv layer for 5 channels input
        if input_channels != 3:
            # Lấy conv layer đầu tiên
            original_conv1 = self.backbone.conv1

            # Tạo conv layer mới với 5 input channels
            self.backbone.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )

            # Initialize weights cho conv layer mới
            with torch.no_grad():
                if pretrained:
                    # Copy weights từ pretrained model cho 3 channels đầu
                    self.backbone.conv1.weight[:, :3, :, :] = original_conv1.weight
                    # Initialize 2 channels còn lại bằng trung bình của 3 channels đầu
                    if input_channels > 3:
                        avg_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                        for i in range(3, input_channels):
                            self.backbone.conv1.weight[:, i:i + 1, :, :] = avg_weights
                else:
                    # Xavier initialization cho toàn bộ weights
                    nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Freeze backbone nếu cần (sau khi modify conv1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Nhưng vẫn cho phép train conv1 layer để adapt với 5 channels
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True

        # Thay đổi classifier cuối cho phân loại nhị phân
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # 2 classes cho phân loại nhị phân
        )

    def forward(self, x):
        # Kiểm tra input channels
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected input with {self.input_channels} channels, got {x.size(1)}")
        return self.backbone(x)
