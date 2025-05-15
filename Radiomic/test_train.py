from torch.utils.data import DataLoader
from dataloader import *
from model import *
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np

def custom_collate(batch):
    images_list = [item[0] for item in batch]
    features_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]
    return images_list, features_list, labels_list

dataset = BreastMRIDataset(
    patient_to_roi=patient_to_roi,
    radiomics_dir="/home/radiology/Mfe Souraya/Radiomics/radiomics_output",
    ground_truth_path="/home/radiology/Mfe Souraya/Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx",
    augment=False
)

indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=30, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=30, collate_fn=custom_collate)

model = BreastMRIModel(num_classes=2, feature_size=714).to("cuda")
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)

def lr_lambda(epoch):
    if epoch < 10: return 1.0
    elif epoch < 30: return 0.5
    elif epoch < 40: return 0.25
    elif epoch < 100: return 0.125
    else: return 0.0625

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

num_epochs = 100
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images_list, features_list, labels_list in train_loader:
        for images, features, labels in zip(images_list, features_list, labels_list):
            if labels.max().item() == -1:
                continue
            images, features, labels = images.to("cuda"), features.to("cuda"), labels.to("cuda")
            outputs = model(images, features)
            loss = criterion(outputs, labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images_list, features_list, labels_list in test_loader:

            for images, features, labels in zip(images_list, features_list, labels_list):
                images, features, labels = images.to("cuda"), features.to("cuda"), labels.to("cuda")
                outputs = model(images, features)
                loss = criterion(outputs, labels.view(-1))
                val_loss += loss.item()
    val_losses.append(val_loss / len(test_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

model.eval()
all_predictions, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images_list, features_list, labels_list in test_loader:
        for images, features, labels in zip(images_list, features_list, labels_list):
            if labels.max().item() == -1:
                continue
            images, features, labels = images.to("cuda"), features.to("cuda"), labels.to("cuda")
            outputs = model(images, features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.show()
