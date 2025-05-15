import json

from torch.utils.data import DataLoader
from dataloader import *
from model import *
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from preprocess_dataset import PreprocessedBreastDataset



#Creation of the output directory for the results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results/run_{timestamp}"
os.makedirs(results_dir, exist_ok=True)


class FocalLoss(nn.Module): #good for imbalanced class
    def __init__(self, gamma=2, alpha=0.25, ignore_index=-1):
        """
        Focal Loss with ignore index support for no-lesion cases (-1 labels).
        Args:
            gamma (float): Focusing parameter (higher value = more focus on hard samples).
            alpha (float): Class weighting (useful for imbalanced datasets).
            ignore_index (int): Label value to ignore during loss computation.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index  # Handle ignored labels

    def forward(self, inputs, targets):
        """
        Compute focal loss.
        Args:
            inputs (Tensor): Model outputs (logits), shape [batch_size, num_classes].
            targets (Tensor): True labels, shape [batch_size].
        Returns:
            Tensor: Computed focal loss.
        """
        # Mask out ignored labels (-1)
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if targets.numel() == 0:  # If no valid labels, return zero loss
            return torch.tensor(0.0, device=inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()







def custom_collate(batch):
    """
    Custom collate function to handle variable lesion counts per patient.
    Args:
        batch (list of tuples): Each tuple contains (images, features, labels) for a patient.
    Returns:
        images_list: List of image tensors (variable-sized batches for lesions).
        features_list: List of feature tensors (variable-sized batches for lesions).
        labels_list: List of label tensors (variable-sized batches for lesions).
    """
    images_list = [item[0] for item in batch]  # Images tensors (variable-sized)
    features_list = [item[1] for item in batch]  # Radiomics features
    labels_list = [item[2] for item in batch]  # Labels
    return images_list, features_list, labels_list


# Initialize dataset
"""
dataset = BreastMRIDataset(
    patient_to_roi=patient_to_roi,
    radiomics_dir="/home/radiology/Mfe Souraya/Radiomics/radiomics_output",
    ground_truth_path="/home/radiology/Mfe Souraya/Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx",
    augment=False
)
"""

dataset = PreprocessedBreastDataset(data_dir="/home/radiology/Mfe Souraya/Radiomics/preprocessed_with_normalization_3slices")

patient_labels = []

for i in range(len(dataset)):
    _, _, labels = dataset[i]
    labels = labels[labels != -1]  # Remove no-lesion markers
    if len(labels) == 0:
        label = 0  # No lesion = benign
    else:
        label = 1 if 1 in labels else 0  # At least one malignant = malignant
    patient_labels.append(label)

"""
# Generate indices for splitting
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create subset datasets for training and testing
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=30, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=30, collate_fn=custom_collate)
"""

from sklearn.model_selection import train_test_split

indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=patient_labels,
    random_state=42
)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=30, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=30, collate_fn=custom_collate)





"""
# Limit the dataset to the first 10 participants
num_participants = 50
subset_indices = list(range(num_participants))

# Create a smaller subset dataset
small_dataset = Subset(dataset, subset_indices)

# Split the smaller dataset into training and testing sets
train_indices, test_indices = train_test_split(subset_indices, test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders for the smaller dataset
train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate)
"""


# Initialize model
model = BreastMRIModel(num_classes=2, feature_size=306) #feature_size=714
model = model.to("cuda")

# Loss function and optimizer
#criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore no-lesion cases (-1)
#optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
criterion = FocalLoss(gamma=1, alpha=0.6, ignore_index=-1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)


# Define the learning rate schedule
def lr_lambda(epoch):
    if epoch < 10:
        return 1.0
    elif epoch < 30:
        return 0.5
    elif epoch < 40:
        return 0.25
    elif epoch < 100:
        return 0.125
    else:
        return 0.0625

# Create the scheduler
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)



for idx in range(len(dataset)):
    images, features, labels = dataset[idx]
    print(f"Patient {idx}: Images Shape: {images.shape}, Labels Shape: {labels.shape}")
    if images.size(0) != labels.size(0):
        print(f"Mismatch for Patient {idx}")



# Training loop
num_epochs = 100
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    #for images_list, features_list, labels_list in train_loader:
    for images_list, features_list, labels_list in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

        for images, features, labels in zip(images_list, features_list, labels_list):
            # Skip no-lesion cases
            if labels.max().item() == -1:
                continue

            # Move tensors to GPU
            images = images.to("cuda")
            features = features.to("cuda")
            labels = labels.to("cuda")

            # Flatten images, features, and labels to process all lesions in the batch
            num_lesions, channels, height, width = images.shape
            images_flat = images.view(num_lesions, channels, height, width)  # Shape: (num_lesions, 3, 224, 224)
            features_flat = features.view(num_lesions, -1)  # Shape: (num_lesions, 714)
            labels_flat = labels.view(-1)  # Shape: (num_lesions)

            # Ensure shapes match
            assert images_flat.size(0) == labels_flat.size(0), "Mismatch between images and labels"
            assert features_flat.size(0) == labels_flat.size(0), "Mismatch between features and labels"

            # Forward pass
            outputs = model(images_flat, features_flat)  # Shape: (num_lesions, num_classes)

            # Ensure output shape matches labels
            assert outputs.size(0) == labels_flat.size(0), "Mismatch between outputs and labels"

            # Compute loss
            loss = criterion(outputs, labels_flat)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            """
            print(f"Images Flat Shape: {images_flat.shape}")
            print(f"Features Flat Shape: {features_flat.shape}")
            print(f"Labels Flat Shape: {labels_flat.shape}")
            print(f"Outputs Shape: {outputs.shape}")
            """
    # Adjust learning rate
    avg_train_loss = running_loss / len(train_loader)
    scheduler.step(avg_train_loss)
    # Epoch summary
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")



import seaborn as sns

# Evaluation loop
model.eval()
all_predictions = []
all_labels = []
all_probs = []  # Store probabilities for AUC curve

with torch.no_grad():
    for images_list, features_list, labels_list in test_loader:
        for images, features, labels in zip(images_list, features_list, labels_list):
            images = torch.tensor(images).to("cuda")
            features = torch.tensor(features).to("cuda")
            labels = torch.tensor(labels).to("cuda")

            num_lesions, channels, height, width = images.shape
            images_flat = images.view(num_lesions, channels, height, width)
            features_flat = features.view(num_lesions, -1)
            labels_flat = labels.view(-1)

            outputs = model(images_flat, features_flat)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
            predictions = torch.argmax(outputs, dim=1)

            valid_indices = labels_flat != -1
            predictions = predictions[valid_indices]
            labels_flat = labels_flat[valid_indices]
            probabilities = probabilities[valid_indices.cpu().numpy()]

            if len(labels_flat) == 0:
                continue

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels_flat.cpu().tolist())
            all_probs.extend(probabilities.tolist())

# Unique class names
unique_classes = unique_labels(all_labels)
target_names = ["Benign", "Malignant"][:len(unique_classes)]

print("Sample probs:", probabilities[:])
print("Sample preds:", predictions[:])


# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_predictions, labels=unique_classes)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(all_labels, all_predictions, target_names=target_names)
print("\nClassification Report:")
print(report)


# ROC Curve and AUC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


"""
# Evaluation loop
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images_list, features_list, labels_list in test_loader:
        # Iterate through each patient in the batch
        for images, features, labels in zip(images_list, features_list, labels_list):
            # Ensure images, features, and labels are tensors
            images = torch.tensor(images).to("cuda")  # Tensor of shape [num_lesions, 3, 224, 224]
            features = torch.tensor(features).to("cuda")  # Tensor of shape [num_lesions, total_features]
            labels = torch.tensor(labels).to("cuda")  # Tensor of shape [num_lesions]

            # Flatten batch and lesion dimensions
            num_lesions, channels, height, width = images.shape
            images_flat = images.view(num_lesions, channels, height, width)
            features_flat = features.view(num_lesions, -1)
            labels_flat = labels.view(-1)

            test_labels = [label for images_list, _, labels_list in test_loader
                           for labels in labels_list
                           for label in labels.cpu().tolist()]
            print("Test Dataset Label Distribution:", {label: test_labels.count(label) for label in set(test_labels)})

            # Forward pass
            outputs = model(images_flat, features_flat)
            predictions = torch.argmax(outputs, dim=1)

            # Filter out no-lesion cases (-1)
            valid_indices = labels_flat != -1
            predictions = predictions[valid_indices]
            labels_flat = labels_flat[valid_indices]

            if len(labels_flat) == 0:
                continue  # Skip this patient if there are no valid labels

            # Collect predictions and labels
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels_flat.cpu().tolist())




# Automatically adjust target names based on unique labels
unique_classes = unique_labels(all_labels)
target_names = ["Benign", "Malignant"][:len(unique_classes)]  # Adjust based on present classes

if len(all_labels) == 0 or len(all_predictions) == 0:
    print("No valid predictions or labels for classification report.")
else:
    # Compute confusion matrix
    print("All Labels:", all_labels)
    print("All Predictions:", all_predictions)
    print("Unique Labels:", set(all_labels))
    print("Unique Predictions:", set(all_predictions))
    benign_count = all_predictions.count(0)
    malignant_count = all_predictions.count(1)

    print(f"Benign Predictions: {benign_count}")
    print(f"Malignant Predictions: {malignant_count}")

    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=unique_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute classification report
    report = classification_report(all_labels, all_predictions, target_names=target_names)
    print("\nClassification Report:")
    print(report)

"""
