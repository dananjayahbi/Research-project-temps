import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CyclicLR
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Paths
data_dir = "../FBMM/Unsplitted_Ready_Sets/set_01_class_balanced_augs_applied_splitted"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")
pretrained_model_path = "./efficientnet_b2_emotion_model.pth"
model_save_path = "./models/fine_tuned_efficientnet_b2.pth"

# Configuration
batch_size = 16
num_epochs = 50
initial_lr = 1e-4
weight_decay = 1e-4
num_classes = 7
img_height, img_width = 260, 260
seed = 42
accumulation_steps = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion categories
emotion_classes = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Improved Data Augmentation for RGB Dataset
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(260, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val_test)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_val_test)

# Compute Class Weights
labels = [label for _, label in train_dataset.samples]
class_counts = np.bincount(labels, minlength=num_classes)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights /= class_weights.sum()
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Load Pretrained EfficientNet-B2 Model and Adapt to RGB
def load_pretrained_model(num_classes, pretrained_path):
    print("Loading pre-trained EfficientNet-B2 model...")

    # Initialize the model architecture
    model = models.efficientnet_b2()

    # Load the pre-trained state dictionary (ignoring size mismatches)
    pretrained_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    model_dict = model.state_dict()

    # Filter out classifier & first conv layer (these need modification)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ["classifier.1.weight", "classifier.1.bias", "features.0.0.weight"]}

    # Load compatible parameters
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)  # Allow mismatches

    # Convert First Layer to RGB (Preserve Grayscale Weights)
    with torch.no_grad():
        old_weights = model.features[0][0].weight.mean(dim=1, keepdim=True)  # Convert grayscale weights
        new_conv_layer = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        new_conv_layer.weight = nn.Parameter(old_weights.repeat(1, 3, 1, 1))  # Expand to 3 channels
        model.features[0][0] = new_conv_layer

    # Modify classifier head to match new num_classes
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

    return model.to("cuda" if torch.cuda.is_available() else "cpu")

model = load_pretrained_model(num_classes, pretrained_model_path)
print(" Model loaded and modified successfully!")

# Custom Focal Loss to Handle Class Imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

criterion = FocalLoss(gamma=2)

# Use AdamW with CyclicLR
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr, weight_decay=weight_decay)
scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode="triangular")

# Function to Update Class Weights Each Epoch Based on Performance
def update_class_weights(class_correct, class_total, num_classes):
    class_accuracies = {cls: (100 * class_correct[cls] / max(1, class_total[cls])) for cls in emotion_classes}
    
    # Compute inverse accuracy for reweighting
    class_weights = torch.zeros(num_classes, dtype=torch.float32).to(device)
    for i, cls in enumerate(emotion_classes):
        class_weights[i] = 1.0 / (class_accuracies[cls] + 1e-6)
    
    class_weights /= class_weights.sum()
    return class_weights

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_loss = np.inf

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        class_correct = {cls: 0 for cls in emotion_classes}
        class_total = {cls: 0 for cls in emotion_classes}

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                class_correct[emotion_classes[labels[i].item()]] += (preds[i] == labels[i]).item()
                class_total[emotion_classes[labels[i].item()]] += 1

        # Compute overall accuracy and loss
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation Step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Display Metrics
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        
        # Show per-class accuracy
        for cls in emotion_classes:
            acc = 100 * class_correct[cls] / max(1, class_total[cls])
            print(f"{cls} Accuracy: {acc:.2f}%")

        # Recalculate Class Weights Based on This Epoch’s Performance
        if epoch > 1:
            class_weights = update_class_weights(class_correct, class_total, num_classes)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(device)
            print(f" Updated Class Weights for Next Epoch: {class_weights.cpu().numpy()}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("✅ Best Model Saved!")

# Train Model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

# Function to Evaluate Model on Test Data
def evaluate_model(model, test_loader, model_path):
    print("\n🔍 Evaluating the Model on Test Data...")

    # Load the best saved model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    class_correct = {cls: 0 for cls in emotion_classes}
    class_total = {cls: 0 for cls in emotion_classes}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(labels.size(0)):
                class_correct[emotion_classes[labels[i].item()]] += (preds[i] == labels[i]).item()
                class_total[emotion_classes[labels[i].item()]] += 1

    # Compute per-class accuracy
    print("\n Test Results:")
    for cls in emotion_classes:
        acc = 100 * class_correct[cls] / max(1, class_total[cls])
        print(f"{cls} Accuracy: {acc:.2f}%")

    # Compute overall test accuracy
    overall_acc = correct / total
    print(f"\n Final Test Accuracy: {overall_acc:.4f}")

    # Generate Classification Report
    report = classification_report(all_labels, all_preds, target_names=emotion_classes, digits=4)
    print("\n Classification Report:\n", report)

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_classes, yticklabels=emotion_classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Run Model Evaluation After Training
evaluate_model(model, test_loader, model_save_path)

