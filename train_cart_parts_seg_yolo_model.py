# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 19/12/24
from ultralytics import YOLO

from config import cart_parts_region_dataset_path, is_poly, DATASET_PATH, CAR_PARTS_SEG_MODEL_TRAINING_EPOCHS, \
    CAR_PARTS_SEG_MODEL_BATCH
from prepare_dataset import process_via_dataset


def train_yolo_mode(output_dataset_path, is_poly=False):
    # Load a pre-trained YOLO model (YOLOv8)
    if is_poly:
        model = YOLO("yolo11n-seg.pt")
    else:
        model = YOLO("yolov11n.pt")  # 'yolov8n.pt' is a smaller, faster variant

    # Train the model
    model.train(data=f'{output_dataset_path}/data.yaml', epochs=CAR_PARTS_SEG_MODEL_TRAINING_EPOCHS, imgsz=640,
                batch=CAR_PARTS_SEG_MODEL_BATCH,
                project='yolo_model',
                name=f'carparts_poly',)
    model.export(format="tflite")

if __name__ == "__main__":
    process_via_dataset(DATASET_PATH, cart_parts_region_dataset_path, is_no_direction=True, is_classy=None)
    train_yolo_mode(cart_parts_region_dataset_path)


