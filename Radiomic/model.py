import torch
import torch.nn as nn
from torchvision.models import resnet50

from torchvision.models import ResNet50_Weights

#model de base pour fusion radiomics et resnet
""""
class BreastMRIModel(nn.Module):
    def __init__(self, num_classes=2, feature_size=306, freeze_pretrained=True): #970 pour feature size de base
        super(BreastMRIModel, self).__init__()
        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Optionally freeze pretrained weights
        if freeze_pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
            # Then selectively unfreeze layer3, layer4 and fc
            for name, param in self.resnet.named_parameters():
                if "layer3" in name or "layer4" in name or "fc" in name:
                    param.requires_grad = True

        # Replace the fully connected (FC) layer for feature extraction
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)

        # Update Fusion Layer: Combine deep learning features (256) + radiomics features
        #self.fusion_layer = nn.Linear(256 + feature_size, 128)
        #self.output_layer = nn.Linear(128, num_classes)

        self.fusion_layer1 = nn.Linear(256 + feature_size, 256)
        self.bn1 = nn.LayerNorm(256)

        self.dropout1 = nn.Dropout(0.3)

        self.fusion_layer2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.3)

        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x, radiomics_features):
        
        Args:
            x: Tensor of shape (batch_size, 3, 224, 224) - 3-channel MRI input.
            radiomics_features: Tensor of shape (batch_size, feature_size) - Radiomic features.

        Returns:
            output: Tensor of shape (batch_size, num_classes).
        
        # Extract deep learning features using ResNet50
        deep_features = self.resnet(x)  # Shape: (batch_size, 256)

        # Fuse with radiomic features
        combined = torch.cat((deep_features, radiomics_features), dim=1)  # Shape: (batch_size, 256 + feature_size)
        #fused = torch.relu(self.fusion_layer(combined))  # Shape: (batch_size, 128)

        # Classification output
        #output = self.output_layer(fused)  # Shape: (batch_size, num_classes)

        # Make sure combined is on the same device as the model
        combined = combined.to(next(self.parameters()).device)

        # Fusion layers

        x = self.dropout1(torch.relu(self.bn1(self.fusion_layer1(combined))))
        x = self.dropout2(torch.relu(self.bn2(self.fusion_layer2(x))))

        output = self.output_layer(x)

        return output
"""



class BreastMRIModel(nn.Module):
    def __init__(self, num_classes=2, feature_size=306, freeze_pretrained=True, mode="fusion"):
        """
        mode: "radiomics", "resnet", or "fusion"
        """
        super(BreastMRIModel, self).__init__()
        self.mode = mode
        self.feature_size = feature_size

        # If using ResNet (image features)
        if mode in ["resnet", "fusion"]:
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            if freeze_pretrained:
                for param in self.resnet.parameters():
                    param.requires_grad = False
                for name, param in self.resnet.named_parameters():
                    if "layer3" in name or "layer4" in name or "fc" in name:
                        param.requires_grad = True
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)

        # Define classification layers based on the mode
        if mode == "radiomics":
            self.classifier = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        elif mode == "resnet":
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        elif mode == "fusion":
            self.fusion_layer1 = nn.Linear(256 + feature_size, 256)
            self.bn1 = nn.LayerNorm(256)
            self.dropout1 = nn.Dropout(0.3)

            self.fusion_layer2 = nn.Linear(256, 128)
            self.bn2 = nn.LayerNorm(128)
            self.dropout2 = nn.Dropout(0.3)

            self.output_layer = nn.Linear(128, num_classes)

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'radiomics', 'resnet', 'fusion'.")

    def forward(self, x=None, radiomics_features=None):
        """
        Forward method supports different input combinations depending on the mode.
        """
        if self.mode == "radiomics":
            assert radiomics_features is not None, "Radiomics features required in 'radiomics' mode."
            x = self.classifier(radiomics_features)

        elif self.mode == "resnet":
            assert x is not None, "Image input required in 'resnet' mode."
            features = self.resnet(x)
            x = self.classifier(features)

        elif self.mode == "fusion":
            assert x is not None and radiomics_features is not None, "Both image and radiomics input required in 'fusion' mode."
            deep_features = self.resnet(x)
            if radiomics_features.shape[0] * 3 == x.shape[0]:
                radiomics_features = radiomics_features.repeat_interleave(3, dim=0)

            combined = torch.cat((deep_features, radiomics_features), dim=1)
            combined = combined.to(next(self.parameters()).device)

            x = self.dropout1(torch.relu(self.bn1(self.fusion_layer1(combined))))
            x = self.dropout2(torch.relu(self.bn2(self.fusion_layer2(x))))
            x = self.output_layer(x)

        return x
