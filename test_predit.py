# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 27/12/24
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from config import DATASET_PATH, CATEGORY_MAPP, CNN_IMG_SIZE, CAR_PARTS_NAMES, CAR_PARTS_NAMES_IDX_MAPP, \
    MAJOR_PARTS_NO_DIRECTION, MAJOR_PARTS_NO_DIRECTION_IDX_MAP
from dataset_handler import create_pdf_with_images
from goraka_agents.RcnnPosittionPattern import predict as RcnnPosittionPatternPredict
from positional_pattern_model import calculate_centroid, pad_sequence, pad_positions, pad_labels
from prepare_dataframe import process_via_dataset
from train_rcnn import MultimodalCarClassifierWithPositions
device = "cuda" if torch.cuda.is_available() else "cpu"

# model = MultimodalCarClassifierWithPositions(num_categories=len(CATEGORY_MAPP), image_size=CNN_IMG_SIZE)
# model.load_state_dict(torch.load("model/cnn1/best_model.pth"))
# model.to(device)
# model.eval()
# yolo = YOLO("yolo_model/carparts5/weights/best.pt") #---- 2104/3974
yolo = YOLO("yolo_model/carparts_poly3/weights/best.pt")


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
is_yolo = False
for img_path, _,true_parts, category in df:
    if is_yolo:
        parts = []
        parts_result = yolo(img_path)
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
                for img in debug_images:
                    if img in img_path:
                        print(img_path)

                print(f"----fails {count}/{count+success}")
                for p in no_missing_trues_parts:
                    missing_no_parts[p]+=1
                count +=1
            else:
                success += 1

        else:
            success += 1
    else:

       category_pred = RcnnPosittionPatternPredict(img_path)
       if isinstance(category_pred, list):
            category_pred = category_pred[0]
       if not category_pred == category:
           print(f"{count}/{count+success}",f"pred: {category_pred}, true: {category}")
           count+=1
       else:
           success+=1




print(f"success: {success}, fails:{count}, total:{success+count}")
if is_yolo:
    print(missing_no_parts)

    create_pdf_with_images(pdf_data, "log/yolo_seg_predict_report.pdf", summery=f"success: {success}, fails:{count}, total:{success+count} \n {missing_no_parts}")
#     predicted_label, logits = predict(model, img_path,bboxes, labels, device)
#     if category==0 or not category == predicted_label:
#         print(f"True label: {CATEGORY_MAPP[category]}, predicted: {CATEGORY_MAPP[predicted_label]}")
#         Image.open(img_path).show()
#     cat_out_put.append(f"{CATEGORY_MAPP[category]} == {CATEGORY_MAPP[predicted_label]}")
#     print(f"True label: {CATEGORY_MAPP[category]}, predicted: {CATEGORY_MAPP[predicted_label]}")

#image_path = '/Users/yarramsettinaresh/Downloads/exercise_1/61111d667eb77b4a469ec270/4cQQqtUeUW_1628511496035.jpg'


