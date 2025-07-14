import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import timm
import numpy as np
import cv2

class EfficientNetDetector(nn.Module):
    def __init__(self, backbone_name='efficientnet_b1', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.feature_channels = self.backbone.feature_info[-1]['num_chs']

        self.detector_head = DetectionHead(
            in_channels=self.feature_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        feature_map = self.backbone(x)[-1]
        class_logits, bbox_preds = self.detector_head(feature_map)
        return class_logits, bbox_preds

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, dropout_rate=0.5):
        super().__init__()

        self.pool_cls = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        self.box_regressor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        class_logits = self.classifier(self.pool_cls(x))
        box_preds = self.box_regressor(x)
        return class_logits, box_preds


model = EfficientNetDetector()

weights_path = "detector_model_epoch_3.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model weights loaded successfully from {weights_path} to {device}.")

except FileNotFoundError:
    print(f"Error: The file '{weights_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading weights: {e}")