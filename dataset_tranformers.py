# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 27/12/24
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
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
