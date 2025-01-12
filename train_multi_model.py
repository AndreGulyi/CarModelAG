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

# Define directories to save the models
BEST_DETECTION_MODEL_PATH = "model/detection/best_model.pth"
BEST_CLASSIFIER_MODEL_PATH = "model/classifier/best_model.pth"
IMG_SIZE = (256, 256)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


class CarPartsDataset(Dataset):
    def __init__(self, data, IMG_SIZE, CAR_PARTS_NAMES, stage="detection"):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])
        self.IMG_SIZE = IMG_SIZE
        self.CAR_PARTS_NAMES = CAR_PARTS_NAMES
        self.stage = stage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, bboxes, labels, car_category = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int)
        max_size = 30
        if len(bboxes) < max_size:
            pad_value = torch.tensor([0.0, 0.0, 0.0, 0.0])  # Padding for bounding boxes
            bboxes = torch.cat((bboxes, pad_value.repeat(max_size - len(bboxes), 1)), dim=0)
            pad_value_label = len(self.CAR_PARTS_NAMES)  # Padding value for labels
            padding_labels = torch.full((max_size - len(labels),), pad_value_label, dtype=labels.dtype)
            labels = torch.cat((labels, padding_labels), dim=0)
        elif len(bboxes) > max_size:
            raise ValueError(f"Number of bounding boxes exceeds max_size: {len(bboxes)} > {max_size}")
        if self.stage == "detection":
            return image, bboxes, labels
        else:
            return image, bboxes, labels, car_category


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
        return outputs


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


# --- Evaluation Function ---
def evaluate_model(model, val_loader, device, stage="classification"):
    model.eval()
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            if stage == "detection":
                images, _, labels = batch
            else:
                images, bboxes, labels, car_categories = batch

            images = images.to(device)

            if stage == "detection":
                outputs = model(images)
                predictions = outputs.argmax(dim=1).tolist()
                true_labels.extend(labels)
                pred_labels.extend(predictions)
            else:
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

    train_detection_dataset = CarPartsDataset(train_data, IMG_SIZE, CAR_PARTS_NAMES, stage="detection")
    val_detection_dataset = CarPartsDataset(val_data, IMG_SIZE, CAR_PARTS_NAMES, stage="detection")

    train_classifier_dataset = CarPartsDataset(train_data, IMG_SIZE, CAR_PARTS_NAMES, stage="classification")
    val_classifier_dataset = CarPartsDataset(val_data, IMG_SIZE, CAR_PARTS_NAMES, stage="classification")

    train_detection_loader = DataLoader(train_detection_dataset, batch_size=batch_size, shuffle=True)
    val_detection_loader = DataLoader(val_detection_dataset, batch_size=batch_size)

    train_classifier_loader = DataLoader(train_classifier_dataset, batch_size=batch_size, shuffle=True)
    val_classifier_loader = DataLoader(val_classifier_dataset, batch_size=batch_size)

    # Detection Model Training
    detection_model = PartDetectionModel().to(device)
    detection_optimizer = Adam(detection_model.parameters(), lr=learning_rate)
    detection_criterion = nn.CrossEntropyLoss()

    print("--- Training Detection Model ---")
    for epoch in range(epochs):
        detection_model.train()
        epoch_loss = 0
        for images, _, labels in train_detection_loader:
            images, labels = images.to(device), labels.to(device)

            # Ensure labels are 1D before passing to the loss function
            labels = labels.squeeze()  # Remove extra dimensions if necessary

            outputs = detection_model(images)
            loss = detection_criterion(outputs, labels)

            detection_optimizer.zero_grad()
            loss.backward()
            detection_optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        val_f1 = evaluate_model(detection_model, val_detection_loader, device, stage="detection")
        print(f"Validation F1 Score (Detection): {val_f1:.4f}")

    torch.save(detection_model.state_dict(), BEST_DETECTION_MODEL_PATH)

    # Classification Model Training
    classifier_model = MultimodalCarClassifierWithPositions(num_categories).to(device)
    classifier_optimizer = Adam(classifier_model.parameters(), lr=learning_rate)
    classifier_criterion = nn.CrossEntropyLoss()

    print("--- Training Classification Model ---")
    for epoch in range(epochs):
        classifier_model.train()
        epoch_loss, correct, total = 0, 0, 0

        for images, bboxes, labels, car_categories in train_classifier_loader:
            images, car_categories = images.to(device), car_categories.to(device)
            outputs = classifier_model(images, bboxes, labels)

            loss = classifier_criterion(outputs, car_categories)
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == car_categories).sum().item()
            total += car_categories.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f}")

        val_f1 = evaluate_model(classifier_model, val_classifier_loader, device)
        print(f"Validation F1 Score (Classification): {val_f1:.4f}")

    torch.save(classifier_model.state_dict(), BEST_CLASSIFIER_MODEL_PATH)
