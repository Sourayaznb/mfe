import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from monai.apps import load_from_mmar, download_mmar, get_model_spec

"""
class SwinUNETRModel(nn.Module):
    def __init__(self, img_size=(128, 128, 32), in_channels=1, out_channels=1, feature_size=48,
                 pretrained=True, weight_path="/home/radiology/Mfe Souraya/Segmentation_ViT/fold1_f48_ep300_4gpu_dice0_9059/model.pt", device="cuda"):
        
        SwinUNETR model initialization with optional pretrained weights.

        :param img_size: The size of the input image (must match your dataloader dimensions).
        :param in_channels: Number of input channels (e.g., 1 for grayscale images).
        :param out_channels: Number of output classes for segmentation.
        :param feature_size: Feature size in the Swin Transformer (default is 48).
        :param pretrained: Whether to load pre-trained weights.
        :param weight_path: Path to the pretrained weights file.
        :param device: Device to use ('cuda' or 'cpu').
        
        super(SwinUNETRModel, self).__init__()
        self.device = device
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True
        )

        # Load pretrained weights if requested
        if pretrained:
            self.load_pretrained_weights(weight_path)

    def load_pretrained_weights(self, weight_path):
        Loads pretrained weights into the model.
        try:
            checkpoint = torch.load(weight_path, map_location=self.device)

            if "state_dict" in checkpoint:  # Ensure checkpoint contains the correct key
                self.swin_unetr.load_state_dict(checkpoint["state_dict"], strict=False)
                print(f"✅ Pretrained weights loaded successfully from {weight_path}")
            else:
                print(f"⚠️ Warning: 'state_dict' not found in checkpoint. Skipping pretrained weights.")
        except FileNotFoundError:
            print(f"⚠️ Warning: Pretrained weights file '{weight_path}' not found. Using randomly initialized weights.")
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRModel(nn.Module):
    def __init__(self, img_size=(128, 128, 32), in_channels=1, out_channels=1, feature_size=48, #in_channels = 1 for substraction 3 and 1
                 pretrained=True, weight_path="model_swinvit.pt", device="cuda"):
        """
        SwinUNETR model initialization with optional pretrained weights.

        :param img_size: The size of the input image (must match your dataloader dimensions).
        :param in_channels: Number of input channels (1 for grayscale images).
        :param out_channels: Number of output classes for segmentation.
        :param feature_size: Feature size in the Swin Transformer.
        :param pretrained: Whether to load pre-trained weights.
        :param weight_path: Path to the pretrained weights file.
        :param device: Device to use ('cuda' or 'cpu').
        """
        super(SwinUNETRModel, self).__init__()
        self.device = device
        self.in_channels = in_channels  # Store number of channels

        # Initialize the model
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,  # 1 channel instead of 4
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True
        )

        # Load pretrained weights if requested
        if pretrained:
            self.load_pretrained_weights(weight_path)

    def load_pretrained_weights(self, weight_path):
        """Loads pretrained weights into the model, handling channel mismatches."""
        try:
            checkpoint = torch.load(weight_path, map_location=self.device)

            if "state_dict" in checkpoint:
                pretrained_dict = checkpoint["state_dict"]
                model_dict = self.swin_unetr.state_dict()

                # Handle the first convolution layer (if input channels are different)
                first_layer_key = "swinViT.patch_embed.proj.weight"
                if first_layer_key in pretrained_dict:
                    pretrained_conv_weights = pretrained_dict[first_layer_key]  # Shape (C_out, 4, H, W, D)


                    #FOR SUBSTRACTION in_channels ==1 et on prends l'average des 4 channels
                    if pretrained_conv_weights.shape[1] == 4 and self.in_channels == 2:
                        print(f"⚠️ Adjusting first convolutional layer: 4 → 1 channel")
                        # Average the 4 channels into 1 for substraction
                        #new_conv_weights = pretrained_conv_weights.mean(dim=1,keepdim=True)  # Shape (C_out, 1, H, W, D)

                        new_conv_weights = pretrained_conv_weights[:, :2, :, :, :]
                        pretrained_dict[first_layer_key] = new_conv_weights




                # Load pretrained weights (ignoring missing keys)
                self.swin_unetr.load_state_dict(pretrained_dict, strict=False)
                print(f"✅ Pretrained weights loaded successfully from {weight_path}")
            else:
                print(f"⚠️ Warning: 'state_dict' not found in checkpoint. Skipping pretrained weights.")
        except FileNotFoundError:
            print(f"⚠️ Warning: Pretrained weights file '{weight_path}' not found. Using randomly initialized weights.")



    def forward(self, x):
        """
        Forward pass for SwinUNETR.
        :param x: Input tensor of shape [batch_size, in_channels, D, H, W].
        :return: Output tensor of shape [batch_size, out_channels, D, H, W].
        """
        return self.swin_unetr(x)


# Model instantiation example
def get_model(img_size=(128, 128, 32), in_channels=1, out_channels=1, feature_size=48, pretrained=True, device="cuda"): #in_channels=1 for substrcation
    """
    Instantiate the SwinUNETR model for transfer learning.
    :param img_size: Input image size.
    :param in_channels: Number of input channels (e.g., 1 for grayscale).
    :param out_channels: Number of output classes.
    :param feature_size: Feature size for the Swin Transformer.
    :param pretrained: Whether to load pre-trained weights.
    :param device: The device to place the model on (e.g., 'cuda' or 'cpu').
    :return: The initialized model.
    """
    model = SwinUNETRModel(img_size=img_size, in_channels=in_channels, out_channels=out_channels, feature_size=feature_size, pretrained=pretrained, device=device)
    return model.to(device)
