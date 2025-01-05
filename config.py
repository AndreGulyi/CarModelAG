# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 23/12/24
#         **********      1. TESTING          **********
import yaml
import os
def load_yaml_config(config_file):
    with open(config_file, 'r', ) as f:
        config = yaml.safe_load(f)
    return config

TEST_FOLDER = "/Users/yarramsettinaresh/Downloads/exercise_1/6125c98e7eb77b4a469ef416"  # model will read images from a test folder
OUTPUT_CSV_PATH = "/Users/yarramsettinaresh/PycharmProjects/CarModel/yolo_config.yaml"  # store prediction in a pandas dataframe.
CNN_IMG_SIZE = (256, 256)

#          *********    2. Model Training     ************

DATASET_PATH = "/Users/yarramsettinaresh/Downloads/exercise_1/"  # Replace with  dataset path
EPOCHS = 20  #
LR = 0.001  # Learning rate

#       *********       3. Evalute
MODEL_PATH = "model/car_overlbestay_model.pth"

config_file = '/Users/yarramsettinaresh/PycharmProjects/CarModel/yolo_config.yaml'  # Replace with your actual config file path
config = load_yaml_config(config_file,)

print(config.keys())
CAR_PARTS_NAMES = config["major_parts"]
CAR_PARTS_NAMES_IDX_MAPP = dict([(v, k) for k, v in CAR_PARTS_NAMES.items()])
CATEGORY_RULES = config["CATEGORY_RULES"]
CATEGORY_MAPP = config["CATEGORY_MAPP"]
CATEGORY_MAPP_IDX_MAPP = dict([(v,k) for k,v in CATEGORY_MAPP.items()])
MAJOR_PARTS_NO_DIRECTION = config["MAJOR_PARTS_NO_DIRECTION"]
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
                                                f"_car_parts_{'poly_'if is_poly else ''}crop_dataset/")
