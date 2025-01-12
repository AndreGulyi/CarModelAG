import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from torch.utils.data import Dataset
from config import MAJOR_PARTS_NO_DIRECTION, CATEGORY_MAPP, DATASET_PATH
from tensorflow.keras.mixed_precision import Policy, set_global_policy

# Set mixed-precision policy
policy = Policy('mixed_float16')
set_global_policy(policy)
# Process dataset

image_shape = (320, 320)
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
    if not isinstance(polygon, (list, np.ndarray)) or len(polygon) == 0:
        raise ValueError("Invalid polygon data.")
    polygon = np.array(polygon)
    if len(polygon.shape) != 2 or polygon.shape[1] != 2:
        raise ValueError("Polygon must be a 2D array of (x, y) coordinates.")
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return np.mean(x_coords), np.mean(y_coords)


# Pad positions to a consistent size
def pad_positions(positions, max_parts, pad_value=0):
    padding_length = max_parts - len(positions)
    return np.pad(positions, ((0, padding_length), (0, 0)), mode='constant', constant_values=pad_value)


# Pad labels to a consistent size
def pad_labels(labels, max_parts, num_part_types, pad_value=0):
    # Ensure labels are 2D
    if not isinstance(labels, list ):
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        padding_length = max_parts - labels.shape[0]
    else:
        padding_length = max_parts - len(labels)
    if padding_length > 0:
        return np.pad(labels, ((0, padding_length), (0, 0)), mode='constant', constant_values=pad_value)
    return labels[:max_parts]


# Dataset class
class CarPartsDataset(Dataset):
    def __init__(self, data, image_shape=image_shape, max_parts=20):
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
    data = process_via_dataset(DATASET_PATH, is_poly=True, is_no_direction=True)
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
height, width = image_shape
num_part_types = len(MAJOR_PARTS_NO_DIRECTION)
num_categories = len(CATEGORY_MAPP)
if __name__ == "__main__":
    v = 3
    from prepare_dataframe import process_via_dataset

    data = process_via_dataset(DATASET_PATH, is_poly=True, is_no_direction=True)

    dataset = CarPartsDataset(data, image_shape=(height, width))
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
    print(f"Training mask Shape: {train_masks.shape}, Position Shape:{train_positions.shape}, Labels Shape: {train_labels.shape}")
    # raise Exception("invalid ")
    # Training
    model.fit(
        [train_masks, train_positions, train_labels],
        train_categories,
        epochs=3,
        batch_size=4
    )
    model.save(f"model/car_parts_model_v{v}.keras")

    """
    v: -1
     Total params: 31,884,135 (121.63 MB)
     Trainable params: 31,884,135 (121.63 MB)
     Non-trainable params: 0 (0.00 B)
    Training mask Shape: (1319, 20, 640, 640), Position Shape:(1319, 20, 2), Labels Shape: (1319, 20, 19)
    330/330 ━━━━━━━━━━━━━━━━━━━━ 3000s 9s/step - accuracy: 0.6547 - loss: 1.1151
    
    
    -----------------
    v:0
    Total params: 31,884,135 (121.63 MB)
     Trainable params: 31,884,135 (121.63 MB)
     Non-trainable params: 0 (0.00 B)
    Training mask Shape: (1319, 20, 640, 640), Position Shape:(1319, 20, 2), Labels Shape: (1319, 20, 19)
    Epoch 1/3
    330/330 ━━━━━━━━━━━━━━━━━━━━ 2200s 7s/step - accuracy: 0.6002 - loss: 1.2484
    Epoch 2/3
    330/330 ━━━━━━━━━━━━━━━━━━━━ 1989s 6s/step - accuracy: 0.9099 - loss: 0.2950
    Epoch 3/3
    330/330 ━━━━━━━━━━━━━━━━━━━━ 2615s 8s/step - accuracy: 0.9666 - loss: 0.1529

Process finished with exit code 130 (interrupted by signal 2: SIGINT)
    
    """