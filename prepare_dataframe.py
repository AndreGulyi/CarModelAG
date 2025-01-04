# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 26/12/24
# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 26/12/24
import os
import json
import shutil
import cv2
import random
from PIL import Image

import yaml
from config import DATASET_PATH, CAR_PARTS_NAMES, cart_parts_region_dataset_path, is_poly, CAR_PARTS_NAMES_IDX_MAPP, \
    CATEGORY_MAPP, MAJOR_PARTS_NO_DIRECTION, MAJOR_PARTS_NO_DIRECTION_IDX_MAP
from dataset_handler import create_pdf_with_images
from dataset_utils import CATEGORY_PARTS_RULE_HANDLER

CLASS_MAPPING = dict(zip(CAR_PARTS_NAMES.values(), CAR_PARTS_NAMES.keys()))

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


# Main function to process VIA JSON files and convert to YOLO format
def process_via_dataset(base_path, is_poly=None, is_debug=False, is_no_direction=None):
    df = []

    # Automatically detect all subfolders (groups) in the base path
    groups = ["610910057eb77b4a469ebb37"]
    # groups = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    valid_images = 0
    invalid_images = 0
    missing_images = 0
    total_images = 0

    all_image_paths = []
    # category_label_file = open(os.path.join(output_path, "category_label_file.txt"),mode='w')
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
            # Check if image exists
            if os.path.exists(image_path):
                all_image_paths.append((image_path, data))
            else:
                missing_images += 1





    # Dataset insights per split
    dataset_insights = {
        'train': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0},
        'val': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0},
        'test': {'total': 0, 'valid': 0, 'invalid': 0, 'missing': 0}
    }
    empty_category_images = []

    for image_path, data in all_image_paths:
        filename = data["filename"]
        regions = data["regions"]
        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to read {image_path}. Skipping.")
            continue

        img_height, img_width = img.shape[:2]

        # Process regions
        part_bboxes = []
        part_labels = []
        for region in regions:
            shape = region["shape_attributes"]
            identity = region["region_attributes"].get("identity", None)
            if is_no_direction and identity:
                identity = identity.replace("partial", "").replace("_", "")

            if identity not in CLASS_MAPPING:
                continue  # Skip unknown classes
            part_labels.append(CLASS_MAPPING[identity])

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

                # yolo_annotations.append(
                #     f"{CLASS_MAPPING[identity]} {normalized_points_flat}"
                # )
                part_bboxes.append(normalized_points)
            else:
                bbox = polygon_to_bbox(all_points_x, all_points_y)
                # Normalize coordinates for YOLO format
                normalized_bbox = normalize_bbox(*bbox, img_width, img_height)

                part_bboxes.append(normalized_bbox)
            # bboxes = [torch.randint(0, 224, (4,)).tolist() for _ in range(num_parts)]  # Random bounding boxes
            # labels = [;np.random.randint(0, 10) for _ in range(num_parts)]  # Random labels for parts
            # car_category = np.random.randint(0, 5)  # Random car category
        category_name = CATEGORY_PARTS_RULE_HANDLER.get_category(set(part_labels))
        if "QHVrZ0XIB1_1627644849225" in image_path:
            print("****", image_path)
            print({CAR_PARTS_NAMES[i] for i in part_labels})
            cat_rule = CATEGORY_PARTS_RULE_HANDLER.get_cat_rule("rear")
            print(cat_rule.validate_parts(part_labels, debug=True))
            print("****")
        if not category_name:
            if is_debug:
                print(image_path)
                print({CAR_PARTS_NAMES[i] for i in part_labels})

            # print(image_path)
            # Image.open(image_path).show()
            # print("----------")
            if "QHVrZ0XIB1_1627644849225" in image_path:
                print("****", image_path)
                print({CAR_PARTS_NAMES[i] for i in part_labels})
                cat_rule = CATEGORY_PARTS_RULE_HANDLER.get_cat_rule("rightFront")
                print(cat_rule.validate_parts(part_labels, debug=True))
                print("****")
            empty_category_images.append({"filename": image_path, "labels": {CAR_PARTS_NAMES[i] for i in part_labels},
                                          "category": CATEGORY_MAPP[category_name]})
            # if part_labels:
            invalid_images +=1
            # else:
            #     missing_images +=1
        else:
            if 'frontws' in part_labels and 'rearws' in part_labels:
                invalid_images+=1
                print("frontws and rearws shoul not be in same image:",part_labels, image_path)
            if is_no_direction:
                part_labels = [MAJOR_PARTS_NO_DIRECTION_IDX_MAP[CAR_PARTS_NAMES[part_id].replace("left", "").replace("right", "")] for part_id in part_labels]
            df.append((image_path, part_bboxes, part_labels, category_name))



    # Print dataset analysis
    print("\n=== Dataset Analysis ===")
    if is_debug:
        create_pdf_with_images(empty_category_images, "log/dataset_empty_category_images.pdf")

    print(f"\nTotal images processed: {len(df)}")
    print(f"Total valid images: {total_images - invalid_images - missing_images}")
    print(f"Total empty images: {invalid_images}")
    print(f"No parts images: {missing_images}")

    for split_name, insights in dataset_insights.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total images: {insights['total']}")
    return df

if __name__ == "__main__":
    df = process_via_dataset(DATASET_PATH, is_poly=True)
    print(df)