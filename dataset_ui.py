# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 26/12/24
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import DATASET_PATH


def load_annotations(json_path):
    """Load annotations from a VIA JSON file."""
    with open(json_path, 'r') as file:
        annotations = json.load(file)
    return annotations


def draw_polygons(image_path, regions):
    """Draw polygons on the image."""
    image = cv2.imread(image_path)
    for region in regions:
        points = np.array(
            list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y'])),
            dtype=np.int32
        )
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return image


def visualize_annotations(group_folder):
    """Visualize annotated images from a group folder."""
    json_path = os.path.join(group_folder, 'via_region_data.json')
    annotations = load_annotations(json_path)

    for file_key, annotation in annotations.items():
        image_path = os.path.join(group_folder, annotation['filename'])
        if not os.path.exists(image_path):
            continue
        image = draw_polygons(image_path, annotation['regions'])
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(annotation['filename'])
        plt.axis('off')
        plt.show()


# Specify the dataset group folder
group_folder = DATASET_PATH + "610910057eb77b4a469ebb37/"
visualize_annotations(group_folder)
