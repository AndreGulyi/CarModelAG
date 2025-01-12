# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 09/01/25
import tensorflow as tf
from PIL import Image

from config import CLASSIFY_MODE_PATH, CLASSY_MODEL_CLASS_NAME

input_size = (640, 640)
import numpy as np


class ClassyModelTflightInference:
    def __init__(self, tflite_model_path):
        self.image_size = (640, 640)
        self.model_path = tflite_model_path
        self.interpreter = tf.lite.Interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self.output_details = self.interpreter.get_output_details()
        print("model output details : ", self.output_details)
        self.class_labels = CLASSY_MODEL_CLASS_NAME

    def load_image(self, image_path):
        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = Image.open(image_path)
            img = img.resize((640, 640))  # Resize the image to the expected size
            img = np.array(img)

        img = np.float32(img)  # Convert to float32
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def run_inference(self, input_image):
        interpreter = self.interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def predict(self, image_path, confidence=0.9):
        input_image = self.load_image(image_path)
        output = self.run_inference(input_image)

        predicted_class = np.argmax(output)
        if output[0][predicted_class] > confidence:
            # print(f"Predicted class: {self.class_labels[int(predicted_class)]}")
            # print(f"Confidence: {output[0][predicted_class]:.2f}")
            return self.class_labels[int(predicted_class)]
        else:
            print(f"Predicted class: {self.class_labels[int(predicted_class)]}")
            print(f"Confidence: {output[0][predicted_class]:.2f}")
        return None


if __name__ == '__main__':
    model_path = CLASSIFY_MODE_PATH
    # image_path = "tttt.png"
    # image_path = "/Users/yarramsettinaresh/Downloads/exercise_1/610910057eb77b4a469ebb37/dLVioqdsaI_1627983747074.jpg"
    image_path = "scraped_mzGgy7_1654878683127.jpg"
    cmtfli = ClassyModelTflightInference(model_path)
    r = cmtfli.predict(image_path)
    print(r)
