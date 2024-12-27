# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 27/12/24
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim import Adam

# --- Config ---
from config import DATASET_PATH, CAR_PARTS_NAMES
from prepare_dataframe import process_via_dataset

# Define directories to save the model
best_model_path = "model/multimodel/best_model.pth"
last_model_path = "model/multimodel/last_model.pth"
IMG_SIZE = (256, 256)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Step 1: Detection Model for Parts ---
class PartDetectionModel(nn.Module):
    def __init__(self):
        super(PartDetectionModel, self).__init__()
        self.backbone = models.resnet18()
        self.backbone.load_state_dict(torch.load("model/resnet18-f37072fd.pth", weights_only=True))
        self.head = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, len(CAR_PARTS_NAMES) + 1)  # Output part labels
        )

    def forward(self, image):
        features = self.backbone(image)
        outputs = self.head(features)
        return outputs  # Returns bounding boxes and part labels


# --- Step 2: Multimodal Classification Model ---
class MultimodalCarClassifierWithPositions(nn.Module):
    def __init__(self, num_categories):
        super(MultimodalCarClassifierWithPositions, self).__init__()
        self.image_backbone = models.resnet18()
        self.image_backbone.load_state_dict(torch.load("model/resnet18-f37072fd.pth", weights_only=True))

        # Bounding box embedding
        self.bbox_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Part labels embedding
        self.part_embed = nn.Embedding(len(CAR_PARTS_NAMES) + 1, 64, padding_idx=len(CAR_PARTS_NAMES))

        # Fusion layer
        fused_feature_dim = 1128
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_feature_dim, 640),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification head
        self.classifier = nn.Linear(640, num_categories)

    def forward(self, image, bboxes, part_labels):
        image_features = self.image_backbone(image)
        bbox_embeddings = torch.stack([self.bbox_embed(bbox) for bbox in bboxes])
        part_label_embeddings = torch.stack([self.part_embed(label) for label in part_labels])

        fused_features = torch.cat([image_features, bbox_embeddings, part_label_embeddings], dim=1)

        if fused_features.shape[1] != 1128:
            raise ValueError(f"Expected fused_features size of (batch_size, 1128), but got {fused_features.shape}")

        fused_features = self.fusion_layer(fused_features)
        logits = self.classifier(fused_features)
        return logits


# --- Dataset Preparation ---
class CarPartsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, bboxes, labels, car_category = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, bboxes, labels, car_category


# --- Evaluation Function ---
def evaluate_model(model, val_loader, device):
    model.eval()
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for images, bboxes, labels, car_categories in val_loader:
            images = images.to(device)
            car_categories = car_categories.to(device)
            outputs = model(images, bboxes, labels)
            _, predicted = outputs.max(1)
            true_labels.extend(car_categories.tolist())
            pred_labels.extend(predicted.tolist())

    f1 = f1_score(true_labels, pred_labels, average='weighted')
    return f1


# --- Training Script ---
if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 1e-4
    batch_size = 16
    epochs = 10
    num_categories = 7

    # Dataset and DataLoader
    dataset = process_via_dataset(DATASET_PATH)
    train_data = dataset[:int(len(dataset) * 0.8)]
    val_data = dataset[int(len(dataset) * 0.8):]

    train_dataset = CarPartsDataset(train_data)
    val_dataset = CarPartsDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Models
    detection_model = PartDetectionModel()
    detection_model.to(device)

    classifier = MultimodalCarClassifierWithPositions(num_categories)
    classifier.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        classifier.train()
        epoch_loss, correct, total = 0, 0, 0

        for images, bboxes, labels, car_categories in train_loader:
            images, car_categories = images.to(device), car_categories.to(device)
            outputs = classifier(images, bboxes, labels)

            loss = criterion(outputs, car_categories)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == car_categories).sum().item()
            total += car_categories.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f}")

        # Validation
        f1 = evaluate_model(classifier, val_loader, device)
        print(f"Validation F1 Score: {f1:.4f}")
