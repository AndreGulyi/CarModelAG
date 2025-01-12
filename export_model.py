# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 27/12/24
import torch.onnx
import tensorflow as tf


# Save the model in ONNX format
def save_onnx_model(model,IMG_SIZE, device, path="model.onnx"):
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor (use the shape of your input data)
    dummy_input = torch.randn(1, 3, *IMG_SIZE).to()  # (batch_size, channels, height, width)

    # Export the model
    torch.onnx.export(model, dummy_input, path, verbose=True, input_names=["image"], output_names=["output"])


# Convert TensorFlow model to TFLite
def convert_to_tflite(model_dir="model/model_tf", tflite_model_path="model/model.tflite"):
    # Load the saved TensorFlow model
    model = tf.saved_model.load(model_dir)

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)


# Convert the model to TFLite format
convert_to_tflite()
