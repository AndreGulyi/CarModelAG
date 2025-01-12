# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 11/01/25
from cart_parts_segment_inference import CarPartsSegTflightModel
from config import CAR_PARTS_SEG_TF_LIGHT_MODEL_PATH, CLASSIFY_MODE_PATH, CAR_PARTS_SEG_MODEL_CONFIDENCE_THRESHOLD, \
    CLASSIFY_MODEL_CONFIDENCE_THRESHOLD, CAR_PARTS_SEG_MODEL_IOU
from tflight_classy_model_inference import ClassyModelTflightInference


class MultiModelTflightInference:
    def __init__(self):
        self.carts_seg_model = CarPartsSegTflightModel(tflite_model_path=CAR_PARTS_SEG_TF_LIGHT_MODEL_PATH)
        self.classy_model = ClassyModelTflightInference(tflite_model_path=CLASSIFY_MODE_PATH)

    def _pred_carts_seg_model(self, image_path):

        return self.carts_seg_model.predict(image_path, combined_parts=True,
                                            confidence_threshold=CAR_PARTS_SEG_MODEL_CONFIDENCE_THRESHOLD,
                                            iou=CAR_PARTS_SEG_MODEL_IOU)

    def _pred_classy_mode(self, parts_image):
        return self.classy_model.predict(parts_image,
                                         confidence=CLASSIFY_MODEL_CONFIDENCE_THRESHOLD)

    def predict(self, image_path):
        parts_image, part_ids = self._pred_carts_seg_model(image_path)
        if len(part_ids) <= 3:
            return None
        return self._pred_classy_mode(parts_image)



if __name__ == "__main__":
    input_image_path = "/Users/yarramsettinaresh/Downloads/exercise_1/610910057eb77b4a469ebb37/dLVioqdsaI_1627983747074.jpg"

    r = MultiModelTflightInference().predict(input_image_path)
    print(r)