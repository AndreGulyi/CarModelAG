# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 23/12/24

#         **********      1. TESTING          **********
import datetime

import yaml
import os

from __logger_init import new_version


def load_yaml_config(config_file):
    with open(config_file, 'r', ) as f:
        config = yaml.safe_load(f)
    return config
import logging
logging.basicConfig(
    filename=f"log/run/v_{new_version}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

car_parts_vs_category_graph_config = load_yaml_config('carparts_vs_category_graph_config.yaml')
conf= load_yaml_config('conf.yaml')

CNN_IMG_SIZE = (256, 256)
#    *****      Testing
TEST_FOLDER = "/Users/yarramsettinaresh/Downloads/exercise_1/6125c98e7eb77b4a469ef416"  # model will read images from a test folder
OUTPUT_CSV_PATH = "/carparts_vs_category_graph_config.yaml"  # store prediction in a pandas dataframe.

#       *********       3. Evalute
MODEL_PATH = "model/car_overlbestay_model.pth"


CAR_PARTS_SEG_TF_LIGHT_MODEL_PATH = conf["CAR_PARTS_SEG_TF_LIGHT_MODEL_PATH"]
CLASSIFY_MODE_PATH = conf["CLASSIFY_MODE_PATH"]
CAR_PARTS_SEG_MODEL_CONFIDENCE_THRESHOLD = float(conf["CAR_PARTS_SEG_MODEL_CONFIDENCE_THRESHOLD"])
CLASSIFY_MODEL_CONFIDENCE_THRESHOLD = float(conf["CLASSIFY_MODEL_CONFIDENCE_THRESHOLD"])
CAR_PARTS_SEG_MODEL_IOU = float(conf["CAR_PARTS_SEG_MODEL_IOU"])

#          *********    2. Model Training     ************

DATASET_PATH = conf["DATASET_PATH"]
CAR_PARTS_SEG_MODEL_TRAINING_EPOCHS = int(conf["CAR_PARTS_SEG_MODEL_TRAINING_EPOCHS"])
CAR_PARTS_SEG_MODEL_BATCH = float(conf["CAR_PARTS_SEG_MODEL_BATCH"])
CLASSIFY_MODEL_EPOCHS = int(conf["CLASSIFY_MODEL_EPOCHS"])
CLASSIFY_MODEL_BATCH = int(conf["CLASSIFY_MODEL_BATCH"])

CAR_PARTS_NAMES = car_parts_vs_category_graph_config["major_parts"]
CAR_PARTS_NAMES_IDX_MAPP = dict([(v, k) for k, v in CAR_PARTS_NAMES.items()])
CATEGORY_RULES = car_parts_vs_category_graph_config["CATEGORY_RULES"]
CATEGORY_MAPP = car_parts_vs_category_graph_config["CATEGORY_MAPP"]
CATEGORY_MAPP_IDX_MAPP = dict([(v,k) for k,v in CATEGORY_MAPP.items()])
MAJOR_PARTS_NO_DIRECTION = car_parts_vs_category_graph_config["MAJOR_PARTS_NO_DIRECTION"]
MAJOR_PARTS_NO_DIRECTION_IDX_MAP = dict([(v, k) for k, v in MAJOR_PARTS_NO_DIRECTION.items()])
# all_parts = set()
# for k,v in config["major_parts"].items():
#     # for _,p in v.items():
#         # for pp in p.split(" "):
#     all_parts.add(v.replace("left","").replace("right",""))
# for i, p in zip(range(len(all_parts)), all_parts):
#     print(f"{i} : {p}")

############
is_poly = True
cart_parts_region_dataset_path = os.path.join(os.getcwdb().decode('utf-8'),
                                                f"_car_parts_poly_dataset_")
cart_parts_classy_dataset_path = os.path.join(os.getcwdb().decode('utf-8'),
                                                f"_car_parts_crop_dataset")

CLASSY_MODEL_CLASS_NAME = {0: 'front', 1: 'leftFront', 2: 'leftRear', 3: 'rear', 4: 'rightFront', 5: 'rightRear'}

