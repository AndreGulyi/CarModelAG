from config import DATASET_PATH, CATEGORY_MAPP, CLASSY_MODEL_CLASS_NAME
from multi_model_inference import MultiModelTflightInference
from prepare_dataframe import process_via_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tabulate import tabulate

def main():
    # Initialize models and process dataset
    multi_model = MultiModelTflightInference()
    df = process_via_dataset(DATASET_PATH, is_poly=True, is_no_direction=True)

    failed_count = 0
    error = []
    y_true = []
    y_pred = []

    for img_path, poly, true_parts, true_category in df:
        try:
            parts_img, parts = multi_model._pred_carts_seg_model(img_path)
            if parts is None:
                error.append(img_path)
                continue

            missing_parts = set(true_parts).difference(set(parts))
            if missing_parts:
                print(f"Missing parts: {missing_parts}, Image: {img_path}")

            true_category = CATEGORY_MAPP[true_category]
            predicted_category = multi_model._pred_classy_mode(parts_img)

            if predicted_category is not None:
                y_true.append(true_category)
                y_pred.append(predicted_category)
                if predicted_category != true_category:
                    print(f"Mismatch: Predicted: {predicted_category}, True: {true_category}")
                    failed_count += 1
            else:
                y_true.append(true_category)
                y_pred.append("Unknown")
                failed_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            error.append(img_path)

    # Metrics calculation
    calculate_metrics(y_true, y_pred, failed_count, len(df), error)

def calculate_metrics(y_true, y_pred, failed_count, total_count, error):
    try:
        weighted_f1_score = f1_score(y_true, y_pred, average='weighted')
    except ValueError as e:
        print(f"Error calculating F1 score: {e}")
        weighted_f1_score = None

    print(f"\nError count: {len(error)}, Errors: {error}")
    print(f"Success: {total_count - failed_count}, Failed: {failed_count}, Total: {total_count}")

    if weighted_f1_score is not None:
        print(f"Weighted F1 Score: {weighted_f1_score:.2f}")

    labels = list(CLASSY_MODEL_CLASS_NAME.values())
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    index_to_char = {v: k for k, v in CLASSY_MODEL_CLASS_NAME.items()}
    classwise_metrics = {
        index_to_char[label]: {
            "Precision": precision[idx],
            "Recall": recall[idx],
            "F1 Score": f1_scores[idx],
        }
        for idx, label in enumerate(labels)
    }

    display_metrics(classwise_metrics, total_count, failed_count)

def display_metrics(classwise_metrics, total_count, failed_count):
    sorted_metrics = sorted(
        classwise_metrics.items(), key=lambda x: x[1]['F1 Score'], reverse=True
    )

    table_data = [
        [CLASSY_MODEL_CLASS_NAME[char], metrics['F1 Score'], metrics['Precision'], metrics['Recall']]
        for char, metrics in sorted_metrics
    ]

    headers = ["Character", "F1 Score", "Precision", "Recall"]

    accuracy = ((total_count - failed_count) / total_count) * 100 if total_count > 0 else 0

    print("\nMetrics sorted by F1 Score:")
    print(tabulate(table_data, headers=headers, floatfmt=".2f"))
    print(f"\nSuccess: {total_count - failed_count}, Failed: {failed_count}, Total testing: {total_count}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
