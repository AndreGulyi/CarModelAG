import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from torch.utils.data import Dataset
from config import MAJOR_PARTS_NO_DIRECTION, CATEGORY_MAPP, DATASET_PATH
from prepare_dataframe import process_via_dataset
from tensorflow.keras.mixed_precision import Policy, set_global_policy

# Set mixed-precision policy
policy = Policy('mixed_float16')
set_global_policy(policy)
# Process dataset
data = process_via_dataset(DATASET_PATH, is_poly=True, is_no_direction=True)


# Pad sequence to a consistent size
def pad_sequence(sequence, max_parts, height, width, pad_value=0):
    padding_length = max_parts - len(sequence)
    return np.pad(sequence,
                  ((0, padding_length), (0, 0), (0, 0)),
                  mode='constant', constant_values=pad_value)


# Convert polygons to mask
def polygon_to_mask(image_shape, polygons):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for poly in polygons:
        poly_points = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly_points], color=1)
    return mask


# Calculate centroid of a polygon
def calculate_centroid(polygon):
    polygon = np.array(polygon)
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return np.mean(x_coords), np.mean(y_coords)


# Pad positions to a consistent size
def pad_positions(positions, max_parts, pad_value=0):
    padding_length = max_parts - len(positions)
    return np.pad(positions, ((0, padding_length), (0, 0)), mode='constant', constant_values=pad_value)


# Pad labels to a consistent size
def pad_labels(labels, max_parts, num_part_types, pad_value=0):
    padding_length = max_parts - len(labels)
    return np.pad(labels, ((0, padding_length), (0, 0)), mode='constant', constant_values=pad_value)


# Dataset class
class CarPartsDataset(Dataset):
    def __init__(self, data, image_shape=(640, 640), max_parts=20):
        self.data = data
        self.image_shape = image_shape
        self.num_part_types = len(MAJOR_PARTS_NO_DIRECTION)
        self.max_parts = max_parts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, polygons, labels, category = self.data[idx]

        masks = [polygon_to_mask(self.image_shape, [polygon]) for polygon in polygons]
        masks = pad_sequence(masks, self.max_parts, self.image_shape[0], self.image_shape[1])

        positions = [calculate_centroid(polygon) for polygon in polygons]
        positions = pad_positions(np.array(positions), self.max_parts)

        one_hot_labels = [np.eye(self.num_part_types)[label] for label in labels]
        one_hot_labels = pad_labels(one_hot_labels, self.max_parts, self.num_part_types)

        return masks, positions, one_hot_labels, category


# Model creation
def create_position_based_model(num_parts, height, width, num_part_types, num_categories):
    part_mask_input = layers.Input(shape=(num_parts, height, width, 1), name="part_masks")
    resized_masks = layers.TimeDistributed(layers.Resizing(256, 256))(part_mask_input)

    cnn_branch = layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu'))(resized_masks)
    cnn_branch = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(cnn_branch)
    cnn_branch = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'))(cnn_branch)
    cnn_branch = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(cnn_branch)
    cnn_branch = layers.TimeDistributed(layers.Flatten())(cnn_branch)
    cnn_branch = layers.TimeDistributed(layers.Dense(128, activation='relu'))(cnn_branch)
    cnn_branch = layers.Flatten()(cnn_branch)
    cnn_branch = layers.Dense(128, activation='relu')(cnn_branch)

    part_position_input = layers.Input(shape=(num_parts, 2), name="part_positions")
    position_branch = layers.Flatten()(part_position_input)
    position_branch = layers.Dense(128, activation='relu')(position_branch)

    part_labels_input = layers.Input(shape=(num_parts, num_part_types), name="part_labels")
    label_branch = layers.Flatten()(part_labels_input)
    label_branch = layers.Dense(128, activation='relu')(label_branch)

    combined = layers.concatenate([cnn_branch, position_branch, label_branch], axis=-1)
    output = layers.Dense(num_categories, activation='softmax', name="category_output")(combined)

    model = models.Model(inputs=[part_mask_input, part_position_input, part_labels_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Dataset preparation
num_parts = 20
height, width = 640, 640
num_part_types = len(MAJOR_PARTS_NO_DIRECTION)
num_categories = len(CATEGORY_MAPP)

dataset = CarPartsDataset(data, image_shape=(640, 640))
train_masks, train_positions, train_labels, train_categories = zip(
    *[dataset[i] for i in range(len(dataset))]
)

train_masks = np.array(train_masks)
train_positions = np.array(train_positions)
train_labels = np.array(train_labels)
train_categories = to_categorical(train_categories, num_categories)

# Model initialization
model = create_position_based_model(num_parts, height, width, num_part_types, num_categories)
model.summary()

# Training
model.fit(
    [train_masks, train_positions, train_labels],
    train_categories,
    epochs=10,
    batch_size=4
)
model.save('model/car_parts_model.h5')
