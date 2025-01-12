# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 11/01/25
from config import DATASET_PATH, CATEGORY_MAPP
from multi_model_inference import MultiModelTflightInference
from prepare_dataframe import process_via_dataset
from sklearn.metrics import f1_score

multi_model = MultiModelTflightInference()
df = process_via_dataset(DATASET_PATH, is_poly=True, is_no_direction=True)

failed_count = 0
error = []
y_true = []
y_pred = []

for img_path, poly, true_parts, true_category in df:
    category = None
    parts_img, parts = multi_model._pred_carts_seg_model(img_path)
    if parts is not None:
        dif = set(true_parts).difference(set(parts))
        if dif:
            print(dif, img_path)
        true_category = CATEGORY_MAPP[true_category]
        category = multi_model._pred_classy_mode(parts_img)
    else:
        error.append(img_path)
        # Skip this iteration if category prediction fails
        continue

    # Append predictions and true labels for F1 score calculation
    if category is not None:
        y_true.append(true_category)
        y_pred.append(category)
    else:
        # Handle failed predictions (optional: add a placeholder like "Unknown")
        y_true.append(true_category)
        y_pred.append("Unknown")  # You can remove this if you want to exclude it completely

    if category != true_category:
        print(category, true_category)
        failed_count += 1

# Calculate F1 score (micro, macro, or weighted as required)
try:
    f1 = f1_score(y_true, y_pred, average='weighted')  # Change average to 'macro' or 'micro' if needed
except ValueError as e:
    print(f"Error calculating F1 score: {e}")
    f1 = None

print(f"error: {len(error)}, {error}")
print(f"success: {len(df)-failed_count}, Failed: {failed_count}, tot: {len(df)}")
if f1 is not None:
    print(f"F1 Score: {f1}")
