# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 09/01/25
from ultralytics import YOLO

# yolo = YOLO("model/segmentation/car_parts_seg.pt")
# yolo = YOLO("model/segmentation/car_parts_seg.tflite", task="segment")
yolo = YOLO("/Users/yarramsettinaresh/PycharmProjects/CarModel/yolo_model/carparts_poly3/weights/best_saved_model/best_float16.tflite",
            task="segment")
input_image_path = "/Users/yarramsettinaresh/Downloads/exercise_1/610910057eb77b4a469ebb37/dLVioqdsaI_1627983747074.jpg"
input_image_path = '/Users/yarramsettinaresh/Downloads/exercise_1/625e744db098d00d5a42a07f/scraped_vGLssf_1649431048991.jpg'
result = yolo(input_image_path)
result[0].show()
# yolo.names.update([(k,str(k)) for k in range(20,100)])
# yolo.export(format="tflite", int8=True, data='/Users/yarramsettinaresh/PycharmProjects/CarModel/_car_parts_poly_dataset_/data.yaml')

#
#
# print("nkjn")
print("done")

"""
TensorFlow SavedModel: export success ✅ 2039.5s, saved as 'model/segmentation/car_parts_seg_saved_model' (39.0 MB)

TensorFlow Lite: starting export with tensorflow 2.16.2...
TensorFlow Lite: export success ✅ 0.0s, saved as 'model/segmentation/car_parts_seg_saved_model/car_parts_seg_int8.tflite' (3.1 MB)

Export complete (2047.5s)
Results saved to /Users/yarramsettinaresh/PycharmProjects/CarModel/model/segmentation
Predict:         yolo predict task=segment model=model/segmentation/car_parts_seg_saved_model/car_parts_seg_int8.tflite imgsz=640 int8 
Validate:        yolo val task=segment model=model/segmentation/car_parts_seg_saved_model/car_parts_seg_int8.tflite imgsz=640 data=/content/datasets/_car_parts_poly_dataset_/data.yaml int8 
Visualize:       https://netron.app
done
"""