import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

from config import DATASET_PATH, EPOCHS, LR
from dataset_handler import prepare_dataset
# Step 1: Prepare Dataset
from carmodel import ConvModel, IMG_SIZE, all_categories, encode_labels, CarModel

empty_category_images = []
multi_category_images = []

# Prepare the dataset
dataset_path = DATASET_PATH
df = prepare_dataset(dataset_path)

# Convert labels to multi-hot encoding

df["encoded_labels"] = df["labels"].apply(encode_labels)

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
# Step 2: Prepare DataLoader
BATCH_SIZE = 32


class CarDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["filename"]
        image = Image.open(img_path).convert("RGB")
        label = np.array(self.dataframe.iloc[idx]["encoded_labels"], dtype=np.float32)
        if self.transform:
            image = self.transform(image)

        return image, label


# Data augmentation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    # transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

# Create DataLoader
train_dataset = CarDataset(train_df, transform=transform)
val_dataset = CarDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 3: Build the Model


# Initialize the model
model = ConvModel(num_classes=len(all_categories))
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# Step 4: Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Step 4: Train the Model with F1-Score Tracking
from sklearn.metrics import f1_score, classification_report

# Adjust for class imbalance using class weights
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Example weights, replace with actual ones
criterion = nn.BCELoss(weight=class_weights.to(device))

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decays LR every 5 epochs

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_train_preds, all_train_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()  # Multi-label thresholding
        correct += (predicted == labels).sum().item()
        total += labels.numel()

        all_train_preds.extend(predicted.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    # Compute training metrics
    train_f1 = f1_score(all_train_labels, all_train_preds, average="micro")
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Validation phase
    model.eval()
    all_val_preds, all_val_labels = [], []
    val_loss, correct_val, total_val = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
            val_loss += criterion(outputs, labels).item()

            predicted = (outputs > 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.numel()

            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    # Compute validation metrics
    val_f1 = f1_score(all_val_labels, all_val_preds, average="micro")
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct_val / total_val

    # Print epoch summary
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, F1: {train_f1:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1: {val_f1:.4f}")

    # Step the scheduler
    scheduler.step()
model.eval()


def evaluate(loader, name=""):
    all_preds, all_true_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float()  # Multi-label thresholding
            all_preds.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_true_labels, all_preds, average='micro')
    print(f"{name} Report:")
    print(f"\tF1 Score (Micro): {f1:.4f}")
    print("\tClassification Report:")
    print("\t", classification_report(all_true_labels, all_preds, target_names=all_categories))
    return f1


evaluate(train_loader, "train_loader")
f1 = evaluate(val_loader, "val_loader")
# Save the best model based on validation F1-score
model_path = f"model/model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{EPOCHS}Epoch_{int(f1*10)}F1.pth"
torch.save(model.state_dict(), model_path)

print(f"model saved at:{model_path}")
car_model = CarModel(model_path=model_path)
all_preds, all_true_labels = [], []
count = 0
empty = 0
for file_name, labels in zip(df["filename"], df["labels"]):
    r = car_model.predict(file_name, is_decoded=False)

    if any(r):
        # print(r)
        count += 1
        all_true_labels.append(encode_labels(labels))
        all_preds.append(list(r))
        pass
    else:
        # print("Empty")
        empty += 1
print(f"predicted: {count}/{count + empty}, empty: {empty}")
f1 = f1_score(all_true_labels, all_preds, average='micro')
print(f"\tF1 Score (Micro): {f1:.4f}")
print("\tClassification Report:")
print("\t", classification_report(all_true_labels, all_preds, target_names=all_categories))
