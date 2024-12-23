# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 23/12/24
import datetime
import sys

from sklearn.metrics import f1_score, classification_report
from carmodel import CarModel, encode_labels, all_categories, decode_labels
from config import DATASET_PATH
from dataset_handler import prepare_dataset, create_pdf_with_images
import logging
logging.basicConfig(
    filename=f"log/eval/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.StreamHandler(sys.stdout)
logging.debug("Test message")
log_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

dataset_path = DATASET_PATH
df = prepare_dataset(dataset_path)
car_model = CarModel()
file_names = []
all_preds, all_true_labels = [], []
count = 0
empty = 0
for file_name,labels in zip(df["filename"], df["labels"]):
    r = car_model.predict(file_name, is_decoded=False)

    if any(r):
        # print(r)
        count +=1
        all_true_labels.append(encode_labels(labels))
        all_preds.append(list(r))
        file_names.append(file_name)
        pass
    else:
        # print("Empty")
        empty += 1
print(f"predicted: {count}/{count+empty}, empty: {empty}")
y_true_binary =all_true_labels
y_pred_binary = all_preds
f1 = f1_score(y_true_binary, y_pred_binary, average='micro')
print(f"\tF1 Score (Micro): {f1:.4f}")
print("\tClassification Report:")
print("\t", classification_report(y_true_binary, y_pred_binary, target_names=all_categories))
report =sorted(classification_report(y_true_binary, y_pred_binary, target_names=all_categories, output_dict=True).items(),
               key=lambda x: x[1]['f1-score'], reverse=True)
report = [r for r in report if r[0] in all_categories]

for r, info in report[:2]:
    low_score_images = []
    for image_path, true_label, pred_lable in zip(file_names, all_true_labels, all_preds):
        true_label = decode_labels(true_label)
        pred_lable = decode_labels(pred_lable)
        if not set(true_label) == set(pred_lable):
            low_score_images.append({"filename": image_path, "labels": true_label,
                                      "category": pred_lable})
    if low_score_images:
        pdf_name = f"reports/DebugLowF1Score_{r}__{info['f1-score']:.2f}.pdf"
        create_pdf_with_images(low_score_images, pdf_name, title=f"{r} low F1 score({info['f1-score']:.2f}) debug ")
        print(f"PDF report creadted :{pdf_name}")

