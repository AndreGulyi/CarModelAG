# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 12/01/25
# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 19/12/24
import time

from ultralytics import YOLO

from config import cart_parts_region_dataset_path, is_poly, DATASET_PATH, CAR_PARTS_SEG_MODEL_TRAINING_EPOCHS, \
    CAR_PARTS_SEG_MODEL_BATCH, cart_parts_classy_dataset_path, CLASSIFY_MODEL_EPOCHS, CLASSIFY_MODEL_BATCH
from prepare_dataset import process_via_dataset


def train_yolo_mode(output_dataset_path):
    model = YOLO("yolo11n-cls.pt")
    # Train the model
    model.train(data=f'{output_dataset_path}', epochs=CLASSIFY_MODEL_EPOCHS, imgsz=640,
                batch=CLASSIFY_MODEL_BATCH,
                project='yolo_model',
                name=f'CLASSIFY_MODEL',)
    model.export(format="tflite")

if __name__ == "__main__":
    process_via_dataset(DATASET_PATH, cart_parts_classy_dataset_path, is_classy=True)
    # time.sleep(3)
    import os

    print(os.path.isfile('/Users/yarramsettinaresh/PycharmProjects/CarModel/_car_parts_crop_dataset/data.yaml'))

    train_yolo_mode(cart_parts_classy_dataset_path)


