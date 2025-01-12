# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 05/01/25
import datetime
import logging

import tensorflow as tf
from onnxconverter_common import FloatTensorType

from config import MAJOR_PARTS_NO_DIRECTION, CATEGORY_MAPP, DATASET_PATH
from xgboost_model_utils import prepare_features_and_labels, CarPartsDataset
from prepare_dataframe import process_via_dataset
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Define major parts without direction (example list, update as needed)
DIRECTION_MAPPING = {"right": 0, "up": 1, "down": 2, "left": 3}
PART_LABEL_MAPPING = {part: idx for idx, part in enumerate(MAJOR_PARTS_NO_DIRECTION)}
PART_LABEL_MAPPING["unknown"] = -1  # Add a default label for padding



data = process_via_dataset(DATASET_PATH, is_poly=True, is_no_direction=True)

# Generate synthetic data and split into train/test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare datasets
train_dataset = CarPartsDataset(train_data)
test_dataset = CarPartsDataset(test_data)
full_dataset = CarPartsDataset(data)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X_train, y_train = prepare_features_and_labels(train_dataset)
X_test, y_test = prepare_features_and_labels(test_dataset)
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
X_val, y_val = prepare_features_and_labels(train_dataset)
y_val = label_encoder.fit_transform(y_val)

# Train an XGBoost Classifier
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': len(np.unique(y_train))+1,  # Number of classes
    'max_depth': 3,# 6:.87, 5: .87, 3: 0.90, 2: 88??
    'eta': 0.3,
    'subsample': 0.75, #.8:90 , 0.9:.894, 0.7:0.895, 0.75 ??
    'colsample_bytree': 0.8, # 0.8:0.90,0.7: 89, 0.6: 0.89, 0.9:.90, 1.0:.894, 0.95:.883 ??
    'eval_metric': 'merror',  # Multi-class error rate
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=100)

# Predict with the trained model
y_pred = bst.predict(dtest)

# Evaluate the model
accuracy = np.sum(y_pred == y_test) / len(y_test)
logging.debug(f"Test Accuracy: {accuracy:.3f}")

# Confusion Matrix, F1-Score, Precision-Recall
conf_matrix = confusion_matrix(y_test, y_pred)
logging.debug("Confusion Matrix:")
logging.debug(conf_matrix)

# F1-Score
f1 = f1_score(y_test, y_pred, average='weighted')
logging.debug(f"F1-Score: {f1:.3f}")

# Precision-Recall
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
logging.debug(f"Precision: {precision:.3f}")
logging.debug(f"Recall: {recall:.3f}")

# Visualizing Confusion Matrix
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(set(y_test)))
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# plt.show()

xgb.plot_tree(bst)  # num_trees=0 for the first tree
# plt.show()
logging.debug(f"done{datetime.datetime.now()}")

model_filename = 'model/car_parts_xgboost_model.json'
bst.save_model(model_filename)
logging.debug(f"Model saved to {model_filename}")

# For example, take the first sample from the training dataset (adjust to fit your actual dataset)
sample = train_dataset[0][0]  # Assuming dataset returns a tuple (features, labels)
print("Shape of a single training sample:", sample.shape)

# Print the length of the features (number of features used for training)
print("Number of features used for training:", sample.shape[0])  # Number of features per sample

# Now create the dummy input for ONNX conversion using the correct number of features
dummy_input = np.random.rand(1, sample.shape[0])  # 1 sample with the same number of features as the training data
print("Dummy input shape for ONNX conversion:", dummy_input.shape)

# Define the initial types based on the dummy input shape
input_type = [("input", FloatTensorType([None, dummy_input.shape[1]]))]  # None for batch size
print("Initial types for ONNX:", input_type)
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(bst)
tflite_model = converter.convert()

# Save the TFLite model
with open(f'model/car_parts_xgboost_model_{int(f1*100)}F1.tflite', 'wb') as f:
    f.write(tflite_model)