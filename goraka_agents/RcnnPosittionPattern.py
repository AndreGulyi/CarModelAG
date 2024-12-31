# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 31/12/24
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from yolov5 import YOLOv5

from positional_pattern_model import calculate_centroid, pad_sequence, pad_labels, pad_positions
from prepare_dataframe import process_via_dataset
from tensorflow.keras.utils import to_categorical

# Load the YOLO model
yolo_model = YOLOv5("path_to_yolov5_model.pt")

# Load the classification model
model = load_model("car_parts_model.h5")

# Example input image for YOLO detection
image = cv2.imread("image_path.jpg")

# Run YOLO prediction
yolo_results = yolo_model.predict(image)

# Extract the segmentation masks or bounding boxes from YOLO results
masks = yolo_results.masks.numpy()  # Assuming masks are part of the output
positions = [calculate_centroid(mask) for mask in masks]

# Process the YOLO results (you may need to pad or resize masks, positions, and labels)
input_masks = np.array([pad_sequence(masks, max_parts=20, height=640, width=640)])
input_positions = np.array([pad_positions(positions, max_parts=20)])
input_labels = np.array([pad_labels(yolo_results.names, max_parts=20, num_part_types=5)])

# Predict the category using the position-based classification model
predictions = model.predict([input_masks, input_positions, input_labels])

# Output the predicted category
predicted_category = np.argmax(predictions, axis=-1)
print(f'Predicted category index: {predicted_category}')
