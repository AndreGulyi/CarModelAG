# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 31/12/24
import numpy as np
import cv2
import xgboost as xgb
from tensorflow.keras.models import load_model
# from prepare_dataframe import process_via_dataset

from config import MAJOR_PARTS_NO_DIRECTION
from model.xgboost_model_utils import CarPartsDataset, prepare_features_and_labels
from positional_pattern_model import calculate_centroid, pad_sequence, pad_labels, pad_positions
from tensorflow.keras.utils import to_categorical
from keras.utils import custom_object_scope
import tensorflow as tf
from ultralytics import YOLO

image_shape = (640, 640)
# Load the YOLO model
yolo_model = YOLO("/Users/yarramsettinaresh/PycharmProjects/CarModel/yolo_model/carparts_poly3/weights/best.pt", )

# Load the classification model
# custom_objects = {"Cast":  tf.cast}

model = load_model("/Users/yarramsettinaresh/PycharmProjects/CarModel/model/car_parts_model_v2.keras")
bst_loaded = xgb.Booster()
model_filename = 'model/car_parts_xgboost_model.json'

bst_loaded.load_model(model_filename)
print(model.summary())

def yolo_to_normalized_polygon(masks, image_shape):
  """
  Converts YOLOv8 segmentation prediction results to normalized polygons.

  Args:
    results: YOLOv8 prediction results object.
    image_shape: Shape of the input image (height, width).

  Returns:
    A list of normalized polygons, where each polygon is a list of
    (x, y) coordinates in the range [0, 1].
  """

  normalized_polygons = []
  mask = masks.astype(np.uint8)

  # Find contours
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Convert contours to polygons
  polygons = [contour.squeeze().tolist() for contour in contours if contour.shape[0] > 2]

  # Print polygons
  # for polygon in polygons:
  #     print(polygon)
  original_height, original_width = image_shape[:2]
  mask_height, mask_width = mask.shape[:2]

  scaling_factor_x = original_width / mask_width
  scaling_factor_y = original_height / mask_height

  # Scale the polygons
  scaled_polygons = [
      [[int(point[0] * scaling_factor_x), int(point[1] * scaling_factor_y)] for point in polygon]
      for polygon in polygons
  ]
  return scaled_polygons[0] if scaled_polygons else []
  for mask in results:
    # Convert mask to binary
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      # Approximate the contour to a polygon
      epsilon = 0.01 * cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, epsilon, True)

      # Normalize polygon coordinates
      normalized_polygon = approx.astype(np.float32) / np.array([image_shape[1], image_shape[0]])
      normalized_polygons.append(normalized_polygon)

  return normalized_polygons

def mask_to_polygons(mask):
    # Ensure the mask is binary
    binary_mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary and scale to 8-bit
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Convert the contour to a list of (x, y) coordinates
        polygon = approx.reshape(-1, 2)
        polygons.append(polygon)
    return polygons

def predict(image=None):
    if image is None:
        # Example input image for YOLO detection
        image = cv2.imread("/Users/yarramsettinaresh/PycharmProjects/CarModel/_car_parts_poly_region_dataset/images/train/0EETD2gUOZ_1648015344105.jpg")
    else:
        image = cv2.imread(image)
    resized_image = cv2.resize(image, image_shape)
    # Run YOLO prediction
    yolo_results = yolo_model.predict(resized_image)

    # Convert Masks object to NumPy arrays
    masks_data = yolo_results[0].masks.data.cpu().numpy()  # Access and convert to NumPy array



    # Calculate centroids for each mask


    max_parts = 20
    num_part_types = len(MAJOR_PARTS_NO_DIRECTION)
    # Process the YOLO results (you may need to pad or resize masks, positions, and labels)
    # input_masks = np.array([pad_sequence(masks_data, max_parts=20, height=640, width=640)])
    # input_positions = np.array([pad_positions(positions, max_parts=20)])
    # input_labels = np.array([pad_labels(yolo_results[0].boxes.numpy().cls.astype(int), max_parts=20, num_part_types=5)])
    #
    masks_data = [mask for mask in masks_data]

    if False:
        # masks = pad_sequence(masks_data, 20, image_shape[0], image_shape[1])
        positions = []
        for mask in masks_data:  # Iterate through NumPy masks
            polygons = mask_to_polygons(mask)
            for polygon in polygons:
                positions.append(calculate_centroid(polygon))
        # positions = [calculate_centroid(polygon) for polygon in polygons]
        positions = pad_positions(np.array(positions), max_parts)
        one_hot_labels = [np.eye(num_part_types)[label] for label in yolo_results[0].boxes.numpy().cls.astype(int)]
        one_hot_labels = pad_labels(one_hot_labels, max_parts, num_part_types)
        # Predict the category using the position-based classification model
        masks = np.array([masks])
        positions = np.array([positions])
        one_hot_labels = np.array([one_hot_labels])
        print(
            f"Training mask Shape: {masks.shape}, Position Shape:{positions.shape}, Labels Shape: {one_hot_labels.shape}")
        predictions = model.predict([masks, positions, one_hot_labels])

        # Output the predicted category
        predicted_category = np.argmax(predictions, axis=-1)
        print(f'Predicted category index: {predicted_category}')
        return predicted_category
    if True:
        # polygons = yolo_to_normalized_polygon( yolo_results[0],(640, 640))
        polygons = [yolo_to_normalized_polygon(mask, image_shape) for mask in masks_data]
        labels_list = yolo_results[0].boxes.numpy().cls.astype(int)
        train_dataset = CarPartsDataset([[None,polygons, labels_list,0]])
        X_train, y_train = prepare_features_and_labels(train_dataset)

        # bst_loaded.predict(X_train)
        dtest_loaded = xgb.DMatrix(X_train)  # Assuming X_test is the same input features for prediction
        y_pred_loaded = bst_loaded.predict(dtest_loaded)
        # print(y_pred_loaded)
        print(int(y_pred_loaded[0]))
        return int(y_pred_loaded[0])+1



