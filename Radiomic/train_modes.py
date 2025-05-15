import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from preprocess_dataset import PreprocessedBreastDataset
from model import BreastMRIModel


# ========== Custom Focal Loss ========== #
class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=0.4, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        if targets.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ========== Custom Collate Function ========== #
def custom_collate(batch):
    images_list = [item[0] for item in batch]
    features_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]
    return images_list, features_list, labels_list


# ========== Load Dataset ========== #
dataset = PreprocessedBreastDataset(data_dir="/home/radiology/Mfe Souraya/Classification/preprocessed_with_normalization")
patient_labels = []

for i in range(len(dataset)):
    _, _, labels = dataset[i]
    labels = labels[labels != -1]
    label = 0 if len(labels) == 0 else int(1 in labels)
    patient_labels.append(label)

train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=patient_labels, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=30, collate_fn=custom_collate)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=30, collate_fn=custom_collate)

# ========== Set Up Models ========== #
feature_size = 306
modes = ["radiomics", "resnet", "fusion"]
models = {}
optimizers = {}
schedulers = {}
criterion = FocalLoss(gamma=2, alpha=0.6, ignore_index=-1)

for mode in modes:
    model = BreastMRIModel(num_classes=2, feature_size=feature_size, mode=mode).to("cuda")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    models[mode] = model
    optimizers[mode] = optimizer
    schedulers[mode] = scheduler

# ========== Output directory ========== #
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results/run_{timestamp}_gamma2_alpha0.6"
os.makedirs(results_dir, exist_ok=True)

# ========== Train & Evaluate Each Mode ========== #
results = {}
num_epochs = 100

for mode in modes:
    model = models[mode]
    optimizer = optimizers[mode]
    scheduler = schedulers[mode]

    print(f"\n================= MODE: {mode.upper()} =================")

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images_list, features_list, labels_list in tqdm(train_loader, desc=f"[{mode}] Epoch {epoch + 1}/{num_epochs}"):
            for images, features, labels in zip(images_list, features_list, labels_list):
                if labels.max().item() == -1:
                    continue
                images = images.to("cuda")
                features = features.to("cuda")
                labels = labels.to("cuda")

                N = labels.size(0)
                images_flat = images.view(N, 3, 224, 224)
                features_flat = features.view(N, -1)
                labels_flat = labels.view(-1)

                if mode == "radiomics":
                    outputs = model(None, features_flat)
                elif mode == "resnet":
                    outputs = model(images_flat, None)
                else:
                    outputs = model(images_flat, features_flat)

                loss = criterion(outputs, labels_flat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"[{mode}] Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images_list, features_list, labels_list in test_loader:
            for images, features, labels in zip(images_list, features_list, labels_list):
                images = images.to("cuda")
                features = features.to("cuda")
                labels = labels.to("cuda")

                N = labels.size(0)
                images_flat = images.view(N, 3, 224, 224)
                features_flat = features.view(N, -1)
                labels_flat = labels.view(-1)

                if mode == "radiomics":
                    outputs = model(None, features_flat)
                elif mode == "resnet":
                    outputs = model(images_flat, None)
                else:
                    outputs = model(images_flat, features_flat)

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1)
                valid = labels_flat != -1

                all_preds += preds[valid].cpu().tolist()
                all_labels += labels_flat[valid].cpu().tolist()
                all_probs += probs[valid.cpu().numpy()].tolist()

    # Metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"], output_dict=True)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    results[mode] = {
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
        "roc_auc": auc_score
    }

    # Confusion Matrix
    print(f"confusion matrix: {conf_matrix}")
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title(f"Confusion Matrix - {mode}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(results_dir, f"conf_matrix_{mode}.png"))
    plt.close()

    # Classification report
    print(f"Classification report: {report}")

    # ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {mode}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"roc_curve_{mode}.png"))
    plt.close()

# Save results
with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
    json.dump(results, f, indent=4)
