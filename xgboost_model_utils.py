# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 05/01/25
import numpy as np

import torch
import math

from torch.utils.data import Dataset


def prepare_features_and_labels(dataset):
    features = []
    labels = []
    for feature, label in dataset:
        features.append(feature.view(-1).numpy())  # Flatten feature array
        labels.append(label)
    return np.array(features), np.array(labels)

# Function to compute the centroid of a polygon
def compute_centroid(polygon):
    if len(polygon) == 0:
        return (0,0)
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    centroid_x = sum(x_coords) / len(polygon)
    centroid_y = sum(y_coords) / len(polygon)
    return (centroid_x, centroid_y)

# Function to compute relative features between two polygons
def compute_relative_features(poly1, poly2):
    centroid1 = compute_centroid(poly1)
    centroid2 = compute_centroid(poly2)

    # Compute distance
    distance = math.sqrt((centroid2[0] - centroid1[0]) ** 2 + (centroid2[1] - centroid1[1]) ** 2)

    # Compute angle
    delta_x = centroid2[0] - centroid1[0]
    delta_y = centroid2[1] - centroid1[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))  # Angle in degrees

    return distance, angle

# Custom Dataset class
class CarPartsDataset(Dataset):
    def __init__(self, data, max_parts=20):
        self.data = data
        self.max_parts = max_parts
        self.max_nodes = self.max_parts * (self.max_parts-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, polygons, labels, category = self.data[idx]

        # Compute features for all pairs of polygons
        features = []
        for i, poly1 in enumerate(polygons):
            for j, poly2 in enumerate(polygons):
                if i != j:
                    distance, angle = compute_relative_features(poly1, poly2)
                    part_a = labels[i]
                    part_b = labels[j]

                    # Add feature as (part_a, part_b, direction)
                    features.append((part_a, part_b, angle, distance))

        # Pad or truncate features to max_parts
        if len(features) >= self.max_nodes:
            print(f"max_nodes exceded : {len(features)} , parts No: {len(labels)}")
        while len(features) < self.max_nodes:
            features.append((-1, -1, -1, -1))
        features = features[:self.max_nodes]

        # Prepare numerical category label
        category_label = category
        try:
            return torch.tensor(features, dtype=torch.int), category_label
        except Exception as e:
            print(e)