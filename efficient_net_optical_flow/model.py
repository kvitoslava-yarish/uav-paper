
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, dropout_rate=0.5):
        super().__init__()
        self.shared_convs = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        shared_conv_out_channels = 256 
        self.cls_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(shared_conv_out_channels * 1 * 1, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        self.box_reg_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(shared_conv_out_channels * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shared_features = self.shared_convs(x)
        class_logits = self.cls_branch(shared_features)
        box_preds = self.box_reg_branch(shared_features)
        return class_logits, box_preds

class EfficientNetDetectorOpticalFlow(nn.Module):
    def __init__(self, backbone_name='efficientnet_b1', num_classes=2):
        super().__init__()
        
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)

        original_first_conv = self.backbone.conv_stem 
        
        self.backbone.conv_stem = nn.Conv2d(
            in_channels=5,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias is not None
        )
        
        with torch.no_grad():
            avg_original_weights = original_first_conv.weight.mean(dim=1, keepdim=True)
            
            self.backbone.conv_stem.weight[:, 0:1, :, :] = avg_original_weights.clone()
            
            self.backbone.conv_stem.weight[:, 1:, :, :] = avg_original_weights.repeat(1, 4, 1, 1)

            if original_first_conv.bias is not None:
                self.backbone.conv_stem.bias.copy_(original_first_conv.bias)

        self.feature_channels = self.backbone.feature_info[-1]['num_chs']

        self.detector_head = DetectionHead(
            in_channels=self.feature_channels,
            num_classes=num_classes
        )

    def forward(self, x):
        feature_map = self.backbone(x)[-1]
        class_logits, bbox_preds = self.detector_head(feature_map)
        return class_logits, bbox_preds
