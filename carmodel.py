# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 21/12/24
import datetime
import sys

from torch import nn
import torch
from torchvision import transforms
import logging
from PIL import Image

from config import MODEL_PATH

logging.basicConfig(filename=f"log/{datetime.datetime.now()}.log")
logging.basicConfig(
    filename=f"log/eval/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.StreamHandler(sys.stdout)
logging.debug("Test carmodel message")
all_categories = ["front", "frontleft", "rearleft", "rear", "rearright", "frontright"]
IMG_SIZE = (128, 128)
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
invalid_combinations = [
    {'rear', 'front'},  # Rear and front shouldn't appear together
    {'rearleft', 'rearight'},  # Rear left and right shouldn't appear together
    {'fronleft', 'frontright'},  # Front left and right shouldn't appear together
]


def encode_labels(labels):
    encoding = [True if category in labels else False for category in all_categories]
    return encoding


def decode_labels(labels):
    return list(category for category, label in zip(all_categories, labels) if label)


# Define category mapping
category_map = {
    'front': ['frontbumper',
              # 'frontbumpercladding','headlightwasher',
              'rightheadlamp',
              'leftheadlamp',
              'frontws', 'licenseplate', 'bonnet'],
    'frontleft': ['leftfrontdoor',
                  # 'leftfrontdoorglass',
                  'leftfender',
                  'leftheadlamp',
                  # 'leftfoglamp'
                  ],
    'rearleft': ['leftreardoor',
                 # 'leftreardoorglass',
                 'lefttaillamp', 'leftrearventglass', 'leftqpanel'],
    'rear': ['rearbumper', 'rearws', 'tailgate', 'licenseplate'],
    'rearright': ['rightreardoor', 'rightreardoorglass', 'righttaillamp', 'rightrearventglass', 'rightqpanel'],
    'frontright': ['rightfrontdoor',
                   # 'rightfrontdoorglass',
                   'rightfender', 'rightheadlamp',
                   # 'rightfoglamp'
                   ]
}


class ConvModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if IMG_SIZE[0] == 128:
            self.fc1 = nn.Linear(128 * 16 * 16, 128)
        elif IMG_SIZE[0] == 640:
            self.fc1 = nn.Linear(128 * 80 * 80, 128)  # Updated for 640x640 input size

        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()  # Multi-label classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Multi-label classification output
        return x


class CarModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path=MODEL_PATH
        self.model = ConvModel(num_classes=len(all_categories))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])
        # self.transform = transforms.Compose([
        #     transforms.Resize(IMG_SIZE),
        #     # transforms.RandomRotation(15),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     # transforms.RandomHorizontalFlip(),
        #     # transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        #     transforms.ToTensor(),
        # ])

    def predict(self, image_path, is_decoded=True):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(image)

        predicted_classes = (output > 0.5).cpu().numpy()[0]  # Multi-label thresholding
        if is_decoded:
            return decode_labels(predicted_classes)
        return predicted_classes



