# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 27/12/24
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from config import DATASET_PATH, CATEGORY_MAPP, CAR_PARTS_NAMES, CAR_PARTS_NAMES_IDX_MAPP, MAJOR_PARTS_NO_DIRECTION_IDX_MAP
from dataset_handler import create_pdf_with_images

from prepare_dataframe import process_via_dataset
import numpy as np
from train_rcnn import MultimodalCarClassifierWithPositions
device = "cuda" if torch.cuda.is_available() else "cpu"

# model = MultimodalCarClassifierWithPositions(num_categories=len(CATEGORY_MAPP), image_size=CNN_IMG_SIZE)
# model.load_state_dict(torch.load("model/cnn1/best_model.pth"))
# model.to(device)
# model.eval()
# yolo = YOLO("yolo_model/carparts5/weights/best.pt") #---- 2104/3974
yolo = YOLO("model/segmentation/car_parts_seg.onnx",
            task='detect'
            )
# yolo.export(format="tflite", int8=True, data='./datasets/data.yaml')
input_image_path = "/Users/yarramsettinaresh/Downloads/exercise_1/610910057eb77b4a469ebb37/dLVioqdsaI_1627983747074.jpg"

result = yolo(input_image_path)
# yolo = YOLO("model/segmentation/carparts_poly_best_saved_model")
#
# yolo.export(format="tflite", )
#yolo.names[32] = "UnknownClass"
# yolo_clasify = YOLO("/Users/yarramsettinaresh/PycharmProjects/cameraApp/runs/classify/train2/weights/best.pt")
yolo_clasify = YOLO("model/car_parts_poly_crop_dataset.tflite")
# Export the model to TFLite format
# yolo.export(format="tflite")  # creates 'yolo11n_float32.tflite'
# img_path = "/Users/yarramsettinaresh/Downloads/exercise_1/625e744db098d00d5a42a07f/scraped_Qb59yY_1649431165516.jpg"
# debug_images = ["scraped_C6TxDz_1601833408820", "scraped_EqPLAm_1615049264549.jpg"]
# parts_result = yolo(img_path)
# parts_result[0].show()
# parts_result[0].save(filename=f"log/temp/{img_path.split('/')[-1]}")
# create_pdf_with_images([], "log/yolo_seg_predict_report.pdf", summery="HCDSAcscfscv sfcvsdv\n dasdfasfsadfv")
df = process_via_dataset(DATASET_PATH)

missing_no_parts = defaultdict(int)
cat_out_put = []
count = 0
success = 0
pdf_data = []
is_yolo = True
error_list = []
error_pdf_list = []
is_yolo_clasify = True
for img_path, _,true_parts, category in df:
    if is_yolo:
        parts = []
        parts_result = yolo(img_path)

        # Initialize a blank image to combine all extracted objects
        combined_image = np.zeros_like(parts_result[0].orig_img)

        # Iterate over detection results
        for result in parts_result:
            # Access the original image
            img = np.copy(result.orig_img)
            img_name = Path(result.path).stem

            # Iterate over each detected object
            for obj in result:
                # Access the class label
                label = obj.names[obj.boxes.cls.tolist().pop()]

                # Create a binary mask for the current object
                b_mask = np.zeros(img.shape[:2], np.uint8)

                # Create contour mask
                contour = obj.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(b_mask, [contour], -1, (255), cv2.FILLED)

                # Isolate the current object using the mask
                isolated_object = cv2.bitwise_and(img, img, mask=b_mask)

                # Combine the isolated object with the combined image
                combined_image = cv2.add(combined_image, isolated_object)
        if is_yolo_clasify:
            rrr = yolo_clasify(combined_image)
            y_pred = yolo_clasify.names[rrr[0].probs.top1]
            if not y_pred == CATEGORY_MAPP[category]:

                print(y_pred, CATEGORY_MAPP[category])
                print("---", img_path)
                print(f"----fails {count}/{count + success}")
                error_list.append((y_pred, CATEGORY_MAPP[category]))

                count += 1
            else:
                success+=1
        else:
            classes = parts_result[0].boxes.cls  # Class labels
            confidences = parts_result[0].boxes.conf
            bboxes = parts_result[0].boxes.xyxy

            clss = []
            for cls, conf,bbox in zip(classes,confidences, bboxes):
                clssss = yolo.names[int(cls)]
                if clssss in CAR_PARTS_NAMES_IDX_MAPP:
                    clss.append((CAR_PARTS_NAMES_IDX_MAPP[clssss], clssss))
                elif clssss in MAJOR_PARTS_NO_DIRECTION_IDX_MAP:
                    clss.append((MAJOR_PARTS_NO_DIRECTION_IDX_MAP[clssss], clssss))
                else:
                    print(clssss)

            labels  = [l for l, _ in clss]
            parts = [l for _,l in clss]
            mising_parts = [CAR_PARTS_NAMES[p] for p in true_parts if CAR_PARTS_NAMES[p] not in parts]
            if mising_parts:
                no_dir_parts = [p.replace("left","").replace("right","") for p in parts]
                no_true_parts = [CAR_PARTS_NAMES[p] for p in true_parts]
                no_true_parts = [p.replace("left","").replace("right","") for p in no_true_parts]
                no_missing_trues_parts = [tp for tp in no_true_parts if tp not in no_dir_parts]
                if no_missing_trues_parts:
                    print(no_missing_trues_parts, img_path)
                    # print(parts_result[0].show())
                    file_path = f"log/temp/{img_path.split('/')[-1]}"
                    parts_result[0].save(filename=file_path)
                    pdf_data.append({"filename": file_path, "labels": no_missing_trues_parts,
                                                  "category": CATEGORY_MAPP[category]})
                    # for img in debug_images:
                    #     if img in img_path:
                    #         print(img_path)

                    print(f"----fails {count}/{count+success}")
                    for p in no_missing_trues_parts:
                        missing_no_parts[p]+=1
                    count +=1
                else:
                    success += 1

            else:
                success += 1
    else:

       category_pred = RcnnPosittionPatternPredict(img_path,category,error_pdf_list=error_pdf_list)
       if isinstance(category_pred, list):
            category_pred = category_pred[0]
       if not category_pred == category:
           error_list.append((CATEGORY_MAPP[category], CATEGORY_MAPP[category_pred]))

           print(f"{count}/{count+success}",f"pred: {category_pred}, true: {category}")
           count+=1
       else:
           success+=1




print(f"success: {success}, fails:{count}, total:{success+count}")
print("error list:")
error_list.sort()
for i in error_list:
    print(i)
if is_yolo:
    print(missing_no_parts)

    create_pdf_with_images(pdf_data, "log/yolo_seg_predict_report.pdf", summery=f"success: {success}, fails:{count}, total:{success+count} \n {missing_no_parts}")
if error_pdf_list:
    create_pdf_with_images(error_pdf_list, "log/xbooot_model/debug/yolo_seg_predict_report.pdf")
    print(error_pdf_list)