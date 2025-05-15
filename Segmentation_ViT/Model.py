import os
import time
from datetime import datetime
from collections import OrderedDict

import torch
from datashader.bundling import smooth
from monai.losses import DiceCELoss, TverskyLoss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter

# Import the SwinUNETR model from your network.py file
from Network import get_model
import torchio as tio
from matplotlib import pyplot as plt
import numpy as np
import csv

def safe_dice_score(pred, target, epsilon=1e-6):
    """
    Computes Dice score safely: returns 1.0 if both pred and target are empty.
    """
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 1.0  # perfect match: both empty
    else:
        return (2.0 * intersection + epsilon) / (pred_sum + target_sum + epsilon)


class Model:
    def __init__(self, expr_dir, img_size=(128, 128, 32), in_channels=1, out_channels=1,
                 feature_size=48, lr=1e-4, beta1=0.9, max_epochs=200,
                 save_epoch_freq=10, monitor_grad_norm=True, device="cuda", resume=False):
        """
        Initialize the MONAI model for training with SwinUNETR.
        """
        self.expr_dir = expr_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.lr = lr
        self.beta1 = beta1
        self.max_epochs = max_epochs
        self.save_epoch_freq = save_epoch_freq
        self.monitor_grad_norm = monitor_grad_norm
        self.device = torch.device(device)
        self.resume = resume

        self.netG = get_model(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_size=self.feature_size,
            device=device
        )

        # Loss function and optimizer
        #self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.loss = TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.7, beta=0.3, smooth_nr = 1e-6, smooth_dr= 1e-6)
        self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.metric = DiceMetric(include_background=True, reduction="mean")

        # TensorBoard
        self.time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_writer = SummaryWriter(os.path.join(self.expr_dir, 'TensorBoard', self.time))

        # Create directories
        self._create_directories()

        # Resume training if specified
        if self.resume:
            self.load_checkpoint()

    def _create_directories(self):
        """
        Create directories for saving models, logs, and visualizations.
        """
        os.makedirs(self.expr_dir, exist_ok=True)
        os.makedirs(os.path.join(self.expr_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.expr_dir, "TensorBoard"), exist_ok=True)
        os.makedirs(os.path.join(self.expr_dir, "TensorBoard", self.time, "visualizations"), exist_ok=True)

    def train(self, train_loader, val_loader):
        """
        Train the model with the given training and validation loaders.
        """
        for epoch in range(1, self.max_epochs + 1):
            print(f"Epoch {epoch}/{self.max_epochs}")
            epoch_start_time = time.time()

            # Training loop
            self.netG.train()
            epoch_loss = 0.0
            positive_loss = 0.0  # Cumulative loss weighted by the number of lesion pixels
            positive_pixel_count = 0



            for batch_data in train_loader:
                images = batch_data["image"][tio.DATA].to(self.device)
                masks = batch_data["mask"][tio.DATA].to(self.device)


                self.optimizer_G.zero_grad()
                outputs = self.netG(images)
                loss = self.loss(outputs, masks)
                loss.backward()

                # Clip gradients (optional)
                if self.monitor_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)

                self.optimizer_G.step()
                epoch_loss += loss.item()

                # Update lesion-specific loss:
                # Multiply the batch loss by the sum of lesion pixels (assuming masks are binary)
                lesion_pixels = masks.sum().item()
                positive_pixel_count += lesion_pixels
                positive_loss += loss.item() * lesion_pixels

            avg_epoch_loss = epoch_loss / len(train_loader)
            #avg_lesion_loss = positive_loss / positive_pixel_count if positive_pixel_count > 0 else 0

            print(f"Training Loss: {avg_epoch_loss:.4f}")
            #print(f"Training Loss (per lesion pixel): {avg_lesion_loss:.4f}")
            self.tensorboard_writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)

            # Validation loop
            if val_loader is not None:
                self.validate(val_loader, epoch)

            # Save model checkpoint
            if epoch % self.save_epoch_freq == 0:
                self.save_checkpoint(epoch)

            print(f"Epoch time: {time.time() - epoch_start_time:.2f} seconds\n")

    def validate(self, val_loader, epoch):
        """
        Validate the model with the given validation loader.
        """
        self.netG.eval()
        val_loss = 0.0
        dice_scores = []
        patient_dice_scores = []

        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                images = batch_data["image"][tio.DATA].to(self.device)
                masks = batch_data["mask"][tio.DATA].to(self.device)

                outputs = self.netG(images)
                loss = self.loss(outputs, masks)
                val_loss += loss.item()

                # Convert to binary masks
                binarized_outputs = (outputs > 0.5).float()
                dice_score = self.metric(y_pred=binarized_outputs, y=masks).mean().item()
                #dice_score = safe_dice_score(binarized_outputs, masks).item()

                dice_scores.append(dice_score)

                patient_id = batch_data.get("subject_id", f"Patient_{i}")
                patient_dice_scores.append((patient_id, dice_score))

                # Save visualization every 10 batches
                if i % 10 == 0:
                    self.save_visualization(images, masks, binarized_outputs, epoch, i, state="validation")
                with open(os.path.join(self.expr_dir, "dice_scores_epoch_{}.csv".format(epoch)), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["PatientID", "DiceScore"])
                    writer.writerows(patient_dice_scores)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = sum(dice_scores) / len(dice_scores)
        max_dice_score = max(dice_scores) #TO DO
        print(f"Validation Loss: {avg_val_loss:.4f}, Dice Score: {max_dice_score:.4f}") #j'ai changé à max au lieu de avergae pour éviter d'avoir des nan

        # Log to TensorBoard
        self.tensorboard_writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        self.tensorboard_writer.add_scalar("Metric/Dice", avg_dice_score, epoch)
        self.tensorboard_writer.add_scalar("Metric/Dice", avg_dice_score, epoch)

    def save_visualization(self, images, masks, predictions, epoch, batch_idx, state="train"):
        """
        Save visualizations of input images, ground truth masks, and model predictions.
        The filename now indicates whether it is a training or validation (testing) visualization.
        """
        save_path = os.path.join(self.expr_dir, "TensorBoard", self.time, "visualizations")
        os.makedirs(save_path, exist_ok=True)

        # Apply transposition to match original code (TorchIO format)
        images = images.cpu().transpose_(2, 4)  # Shape: (batch, channels, depth, height, width)
        masks = masks.cpu().transpose_(2, 4)
        predictions = predictions.cpu().transpose_(2, 4)

        for i in range(min(5, images.shape[0])):  # Save up to 5 images per batch
            for depth in range(0, images.shape[2], max(1, images.shape[2] // 5)):  # Sample a few slices
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(np.squeeze(images[i, 0, depth]), cmap="gray")
                axes[0].set_title("Input Image")

                axes[1].imshow(np.squeeze(masks[i, 0, depth]), cmap="gray")
                axes[1].set_title("Ground Truth Mask")

                axes[2].imshow(np.squeeze(predictions[i, 0, depth]), cmap="gray")
                axes[2].set_title("Model Prediction")

                for ax in axes:
                    ax.axis("off")

                # The filename now includes the state (e.g. "train" or "validation")
                filename = f"{state}_epoch_{epoch}_batch_{batch_idx}_sample_{i}_slice_{depth}.png"
                plt.savefig(os.path.join(save_path, filename))
                plt.close(fig)

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.
        """
        checkpoint_path = os.path.join(self.expr_dir, "checkpoints", f"model_epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.netG.state_dict(),
            "optimizer_state_dict": self.optimizer_G.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load model checkpoint.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.expr_dir, "checkpoints", "latest.pth")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.netG.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

class Model2:
    def __init__(self, expr_dir, img_size=(128, 128, 32), in_channels=1, out_channels=1,
                 feature_size=48, lr=1e-4, beta1=0.9, max_epochs=200,
                 save_epoch_freq=10, monitor_grad_norm=True, device="cuda", resume=False):
        """
        Initialize the MONAI model for training with SwinUNETR.
        :param expr_dir: Directory to save results.
        :param img_size: Input image size.
        :param in_channels: Number of input channels (e.g., 1 for grayscale).
        :param out_channels: Number of output segmentation classes.
        :param feature_size: Feature size for SwinUNETR.
        :param lr: Learning rate.
        :param beta1: Beta1 hyperparameter for Adam optimizer.
        :param max_epochs: Total number of training epochs.
        :param save_epoch_freq: Frequency to save model checkpoints.
        :param monitor_grad_norm: Whether to monitor gradient norm.
        :param device: Device to use ('cuda' or 'cpu').
        :param resume: Whether to resume training from a checkpoint.
        """
        self.expr_dir = expr_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.lr = lr
        self.beta1 = beta1
        self.max_epochs = max_epochs
        self.save_epoch_freq = save_epoch_freq
        self.monitor_grad_norm = monitor_grad_norm
        self.device = torch.device(device)
        self.resume = resume


        self.netG = get_model(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_size=self.feature_size,
            device=device
        )

        # Loss function and optimizer
        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.metric = DiceMetric(include_background=True, reduction="mean")

        # TensorBoard
        self.time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_writer = SummaryWriter(os.path.join(self.expr_dir, 'TensorBoard', self.time))

        # Create directories
        self._create_directories()

        # Resume training if specified
        if self.resume:
            self.load_checkpoint()

    def _create_directories(self):
        """
        Create directories for saving models, logs, and visualizations.
        """
        os.makedirs(self.expr_dir, exist_ok=True)
        os.makedirs(os.path.join(self.expr_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.expr_dir, "TensorBoard"), exist_ok=True)

    def train(self, train_loader, val_loader):
        """
        Train the model with the given training and validation loaders.
        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        """
        for epoch in range(1, self.max_epochs + 1):
            print(f"Epoch {epoch}/{self.max_epochs}")
            epoch_start_time = time.time()

            # Training loop
            self.netG.train()
            epoch_loss = 0.0
            for batch_data in train_loader:
                images = batch_data["image"][tio.DATA].to(self.device)
                masks = batch_data["mask"][tio.DATA].to(self.device)

                self.optimizer_G.zero_grad()
                outputs = self.netG(images)
                loss = self.loss(outputs, masks)
                loss.backward()

                # Clip gradients (optional)
                if self.monitor_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)

                self.optimizer_G.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Training Loss: {avg_epoch_loss:.4f}")
            self.tensorboard_writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)

            # Validation loop
            if val_loader is not None:
                self.validate(val_loader, epoch)

            # Save model checkpoint
            if epoch % self.save_epoch_freq == 0:
                self.save_checkpoint(epoch)

            print(f"Epoch time: {time.time() - epoch_start_time:.2f} seconds\n")

    def validate(self, val_loader, epoch, label="Validation"):
        """
        Validate the model with the given validation loader.
        :param val_loader: DataLoader for validation data.
        :param epoch: Current epoch number.
        """
        self.netG.eval()
        val_loss = 0.0
        dice_scores = []

        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data["image"][tio.DATA].to(self.device)
                masks = batch_data["mask"][tio.DATA].to(self.device)

                outputs = self.netG(images)
                loss = self.loss(outputs, masks)
                val_loss += loss.item()

                # Calculate Dice score
                binarized_outputs = (outputs > 0.5).float()
                dice_score = self.metric(y_pred=binarized_outputs, y=masks).mean().item() #avant c'était juste outputs pour y_pred
                dice_scores.append(dice_score)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = sum(dice_scores) / len(dice_scores)
        print(f"Validation Loss: {avg_val_loss:.4f}, Dice Score: {avg_dice_score:.4f}")

        # Log to TensorBoard
        #self.tensorboard_writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        #self.tensorboard_writer.add_scalar("Metric/Dice", avg_dice_score, epoch)

        print(f"{label} Loss: {avg_val_loss:.4f}, Dice Score: {avg_dice_score:.4f}")
        self.tensorboard_writer.add_scalar(f"Loss/{label}", avg_val_loss, epoch)
        self.tensorboard_writer.add_scalar(f"Metric/Dice_{label}", avg_dice_score, epoch)

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.
        :param epoch: Current epoch number.
        """
        checkpoint_path = os.path.join(self.expr_dir, "checkpoints", f"model_epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.netG.state_dict(),
            "optimizer_state_dict": self.optimizer_G.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load model checkpoint.
        :param checkpoint_path: Path to the checkpoint file.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.expr_dir, "checkpoints", "latest.pth")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.netG.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

    def visualize(self, visuals, affine, epoch, index, state="training"):
        substraction_data = visuals['sub_T1'].cpu().transpose_(2, 4)
        # print(f"Shape of visuals['true_seg']: {visuals['true_seg'].shape}")

        true_segmentation = visuals['true_seg'].cpu().transpose_(2, 4)

        estimated_segmentation = visuals['estimated_seg'].cpu().transpose_(2, 4)

        # substraction_data = substraction_data[:,None,:,:,:]
        # true_segmentation = true_segmentation[:,None,:,:,:]
        # estimated_segmentation = estimated_segmentation[:,None,:,:,:]

        for i in range(1):
            subject = tio.Subject(
                Raw_data=tio.ScalarImage(tensor=substraction_data[i], affine=affine[i]),
                True_seg=tio.ScalarImage(tensor=true_segmentation[i], affine=affine[i]),
                Est_seg=tio.ScalarImage(tensor=estimated_segmentation[i], affine=affine[i]),
            )

            save_path = os.path.join(self.expr_dir, 'TensorBoard', self.time, state + "_visuals")
            save_path = os.path.join(save_path, 'cycle_' + str(epoch) + '_' + str(index) + '_' + str(
                i) + '.png')
            subject.plot(show=False, output_path=save_path, figsize=(10, 10))
        plt.close('all')