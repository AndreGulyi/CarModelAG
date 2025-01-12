# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 23/12/24
from carmodel import CarModel
import pandas as pd
import os

from config import TEST_FOLDER, OUTPUT_CSV_PATH
from multi_model_inference import MultiModelTflightInference

model = MultiModelTflightInference

def predict_folder(test_folder, output_path):
    """Run predictions on if images in test folder using the trained  model."""
    print(f"prediction started of {test_folder}")
    model = MultiModelTflightInference()

    results = []
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_folder, filename)
            try:
                predicted_classes = model.predict(image_path)  # Assuming model.predict() accepts PIL Image
                results.append((filename, predicted_classes))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    df = pd.DataFrame(results, columns=['filename', 'predictions'])
    df.to_csv(output_path, index=False)
    print(f"prediction save at {output_path}")



if __name__ == "__main__":
    predict_folder(TEST_FOLDER, OUTPUT_CSV_PATH)