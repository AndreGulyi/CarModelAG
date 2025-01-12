# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 26/12/24
import os
import json
import shutil
from collections import defaultdict

import cv2
import random
import numpy as np
import yaml
from ultralytics import YOLO

from config import DATASET_PATH, CAR_PARTS_NAMES, cart_parts_region_dataset_path, is_poly, \
    MAJOR_PARTS_NO_DIRECTION_IDX_MAP, MAJOR_PARTS_NO_DIRECTION, CATEGORY_MAPP, CLASSY_MODEL_CLASS_NAME
from dataset_utils import CATEGORY_PARTS_RULE_HANDLER
# yolo_clasify = YOLO("/Users/yarramsettinaresh/PycharmProjects/cameraApp/runs/classify/train2/weights/best.pt")

CLASS_MAPPING = dict(zip(CAR_PARTS_NAMES.values(), CAR_PARTS_NAMES.keys()))

def get_part_label(part_label, is_no_direction=None):
    if is_no_direction:
        return MAJOR_PARTS_NO_DIRECTION_IDX_MAP[part_label.replace("left", "").replace("right", "")]
    return CLASS_MAPPING[part_label]

def create_data_yaml(output_path, is_no_direction=None, is_classy=None):
    if is_classy:
        data = {
            "train": "train",
            "val": "val",
            "test": "test",
            "nc": len(CLASSY_MODEL_CLASS_NAME),
            "names": CLASSY_MODEL_CLASS_NAME
        }
    else:
        data = {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(MAJOR_PARTS_NO_DIRECTION) if is_no_direction else len(CAR_PARTS_NAMES),
            "names": MAJOR_PARTS_NO_DIRECTION if is_no_direction else CAR_PARTS_NAMES
        }

    yaml_output_path = os.path.join(output_path, "data.yaml")

    if os.path.exists(yaml_output_path):
        os.remove(yaml_output_path)
    with open(yaml_output_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


# Function to convert polygon points to bounding box
def polygon_to_bbox(all_points_x, all_points_y):
    x_min, x_max = min(all_points_x), max(all_points_x)
    y_min, y_max = min(all_points_y), max(all_points_y)
    return x_min, y_min, x_max, y_max


# Function to normalize bounding box coordinates
def normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)


# Function to check if one bounding box is inside another
def is_inside(inner_box, outer_box):
    x1_in, y1_in, x2_in, y2_in = inner_box
    x1_out, y1_out, x2_out, y2_out = outer_box
    return x1_in >= x1_out and y1_in >= y1_out and x2_in <= x2_out and y2_in <= y2_out

def crop_polygon_to_new_image(image, all_points_x, all_points_y):
    """
    Crops a polygonal region from an image and returns the cropped image.

    Args:
        image (numpy.ndarray): The input image.
        all_points_x (list): List of x-coordinates of the polygon.
        all_points_y (list): List of y-coordinates of the polygon.

    Returns:
        cropped_image (numpy.ndarray): Cropped image containing the polygon region.
    """
    # Create a blank mask the same size as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define the polygon points
    polygon = np.array([[x, y] for x, y in zip(all_points_x, all_points_y)], dtype=np.int32)

    # Fill the polygon area on the mask
    cv2.fillPoly(mask, [polygon], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Compute the bounding box of the polygon
    x_min, x_max = min(all_points_x), max(all_points_x)
    y_min, y_max = min(all_points_y), max(all_points_y)

    # Crop the image to the bounding box
    cropped_image = masked_image[y_min:y_max, x_min:x_max]

    return cropped_image
def assemble_car_from_parts_exact(image, parts_position, img_width, img_height, canvas_size=(640, 640)):
    """
    Assembles a car image from its parts in their original positions.

    Args:
        parts_position (list): List of tuples containing part identity and position (polygon or bbox).
        img_width (int): Width of the input image.
        img_height (int): Height of the input image.
        canvas_size (tuple): Size of the canvas (width, height) for the assembled car.

    Returns:
        assembled_car (numpy.ndarray): The assembled car image.
    """
    # Create a blank canvas for the car
    canvas_width, canvas_height = canvas_size
    assembled_car = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for identity, position in parts_position:
        if isinstance(position[0], tuple):  # Polygon points
            # Extract polygon and mask
            all_points_x = [int(x * img_width) for x, y in position]
            all_points_y = [int(y * img_height) for x, y in position]
            part_img = crop_polygon_to_new_image(image, all_points_x, all_points_y)

            # Place polygon in its original position on the canvas
            x_min, y_min = min(all_points_x), min(all_points_y)
        else:  # Bounding box (normalized)
            # Convert normalized bbox to pixel values
            x_center, y_center, width, height = position
            x_min = int((x_center - width / 2) * img_width)
            x_max = int((x_center + width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            y_max = int((y_center + height / 2) * img_height)
            part_img = image[y_min:y_max, x_min:x_max]

        # Place the extracted part onto the canvas
        part_height, part_width = part_img.shape[:2]
        assembled_car[y_min:y_min + part_height, x_min:x_min + part_width] = part_img

    return assembled_car

# Main function to process VIA JSON files and convert to YOLO format
def process_via_dataset(base_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, is_no_direction=None, is_classy=None):
    # Delete existing output directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Create output directories
    os.makedirs(output_path, exist_ok=True)


    # Automatically detect all subfolders (groups) in the base path
    # groups = ["610910057eb77b4a469ebb37"]
    # output_path += groups[0]
    groups = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    if not is_classy:
        images_output = os.path.join(output_path, "images")
        labels_output = os.path.join(output_path, "labels")

        os.makedirs(images_output, exist_ok=True)
        os.makedirs(labels_output, exist_ok=True)

    valid_images = 0
    invalid_images = 0
    missing_images = 0
    total_images = 0

    group_all_image_paths = defaultdict(list)
    # if not  is_classy:
    #     category_label_file = open(os.path.join(output_path, "category_label_file.txt"),mode='w')
    # Collect all image paths first
    for group in groups:

        group_path = os.path.join(base_path, group)
        json_path = os.path.join(group_path, "via_region_data.json")

        # Load VIA JSON file
        if not os.path.exists(json_path):
            print(f"No JSON found in {group_path}. Skipping.")
            continue

        with open(json_path, "r") as file:
            via_data = json.load(file)

        for file_key, data in via_data.items():
            filename = data["filename"]
            image_path = os.path.join(group_path, filename)
            regions = data["regions"]
            # Check if image exists
            parts = set()
            for region in regions:
                identity = region["region_attributes"].get("identity", None)

                if identity not in CLASS_MAPPING:
                    continue  # Skip unknown classes
                parts.add(identity)



            if os.path.exists(image_path):
                category_name = CATEGORY_PARTS_RULE_HANDLER.get_category({CLASS_MAPPING[p] for p in parts})
                if category_name:
                    if is_classy:
                        group_all_image_paths[CATEGORY_MAPP[category_name]].append((image_path, data))
                    else:
                        group_all_image_paths[group].append((image_path, data))
            else:
                missing_images += 1
    total_split = defaultdict(int)
    for group, all_image_paths in group_all_image_paths.items():
        # Shuffle all images before splitting into train, val, and test sets
        # if is_classy:
        #     images_output = os.path.join(output_path, group)
        #     os.makedirs(images_output, exist_ok=True)
        random.shuffle(all_image_paths)

        # Calculate the split sizes
        total_files = len(all_image_paths)
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)
        test_size = total_files - train_size - val_size

        # Split dataset into train, val, and test
        splits = {
            'train': all_image_paths[:train_size],
            'val': all_image_paths[train_size:train_size + val_size],
            'test': all_image_paths[train_size + val_size:]
        }

        # Dataset insights per split
        dataset_insights = {
            'train': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0},
            'val': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0},
            'test': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0}
        }

        # Iterate over each split and process images
        for split_name, split_data in splits.items():
            if is_classy:
                images_output = os.path.join(output_path, split_name, group)
                os.makedirs(images_output, exist_ok=True)
            else:
                split_images_output = os.path.join(images_output, split_name)
                split_labels_output = os.path.join(labels_output, split_name)
                os.makedirs(split_images_output, exist_ok=True)

                os.makedirs(split_labels_output, exist_ok=True)

            for image_path, data in split_data:
                filename = data["filename"]
                regions = data["regions"]
                img = cv2.imread(image_path)

                if img is None:
                    dataset_insights[split_name]['missing'] += 1
                    print(f"Failed to read {image_path}. Skipping.")
                    continue

                # Copy image to the appropriate output folder
                if not is_classy:
                    new_image_path = os.path.join(split_images_output, filename)
                    shutil.copy(image_path, new_image_path)
                img_height, img_width = img.shape[:2]


                # Process regions
                yolo_annotations = []
                parts = set()
                parts_position = []
                for region in regions:
                    shape = region["shape_attributes"]
                    identity = region["region_attributes"].get("identity", None)

                    if identity not in CLASS_MAPPING:
                        continue  # Skip unknown classes
                    parts.add(identity)

                    # Extract polygon and convert to bbox
                    all_points_x = shape["all_points_x"]
                    all_points_y = shape["all_points_y"]
                    if is_poly:
                        normalized_points = [
                            (x / img_width, y / img_height)
                            for x, y in zip(all_points_x, all_points_y)
                        ]
                        normalized_points_flat = " ".join(
                            [f"{x:.6f} {y:.6f}" for x, y in normalized_points]
                        )
                        parts_position.append((identity, normalized_points))
                        yolo_annotations.append(
                            f"{get_part_label(identity, is_no_direction)} {normalized_points_flat}"
                        )
                    else:
                        bbox = polygon_to_bbox(all_points_x, all_points_y)

                        # Normalize coordinates for YOLO format
                        normalized_bbox = normalize_bbox(*bbox, img_width, img_height)
                        parts_position.append((identity, normalized_bbox))
                        yolo_annotations.append(
                            f"{get_part_label(identity, is_no_direction)} {normalized_bbox[0]} {normalized_bbox[1]} {normalized_bbox[2]} {normalized_bbox[3]}"
                        )


                if is_classy:
                    category_name = CATEGORY_PARTS_RULE_HANDLER.get_category({CLASS_MAPPING[p] for p in parts})
                    if category_name:
                        new_img_path = os.path.join(images_output, filename)
                        cat_rule = CATEGORY_PARTS_RULE_HANDLER.get_cat_rule(CATEGORY_MAPP[category_name])
                        cat_parts = cat_rule.all_parts()
                        # parts_position = [(k,v) for k,v in parts_position if k in cat_parts]
                        new_img = assemble_car_from_parts_exact(img, parts_position, img_width, img_height, canvas_size=(img_width,img_height))
                        # rr = yolo_clasify(new_img)
                        # if not rr[0].boxes is None:
                        #     print("cccc")
                        cv2.imwrite(new_img_path, new_img)
                        # cv2.imwrite("new_image_with_parts.jpg", new_img)
                        # cv2.imshow(f"{CATEGORY_MAPP[category_name]}", new_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                else:
                    # Save YOLO annotations to a text file

                    label_filename = os.path.splitext(filename)[0] + ".txt"
                    label_path = os.path.join(split_labels_output, label_filename)
                    with open(label_path, "w") as label_file:
                        label_file.write("\n".join(yolo_annotations))
                    category_name = CATEGORY_PARTS_RULE_HANDLER.get_category({CLASS_MAPPING[p] for p in parts})
                    # category_label_file.write(f"{filename} {category_name}")
                dataset_insights[split_name]['total'] += 1
                total_images += 1
        for split, c in splits.items():
            total_split[split]+=len(c)

    # Print dataset analysis
    print("\n=== Dataset Analysis ===")
    print(f"\nTotal images processed: {total_images}")
    print(f"Total valid images: {total_images - invalid_images - missing_images}")
    print(f"Total invalid images: {invalid_images}")
    print(f"Missing images: {missing_images}")

    print("===== image split ")
    for split, v in total_split.items():
        print(f"{split}: {v}")
    create_data_yaml(output_path, is_no_direction, is_classy)

        # category_label_file.close()

if __name__ == "__main__":
    process_via_dataset(DATASET_PATH, cart_parts_region_dataset_path, is_no_direction=True, is_classy=None)