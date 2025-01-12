# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 11/01/25
import tensorflow as tf
import time

import torch
from PIL import Image
import numpy as np
import cv2

from config import MAJOR_PARTS_NO_DIRECTION
from geometry_utils import xywh2xyxy, process_mask

class_labels = MAJOR_PARTS_NO_DIRECTION


class CarPartsSegTflightModel:

    def __init__(self, tflite_model_path="model/segmentation/car_parts_seg.tflite"):
        # Specify model and image paths
        self.model_path = tflite_model_path
        self.image_size = (640, 640)

        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def predict(self, input_image_path, combined_parts=False, draw_labels=False,
                confidence_threshold=0.8, iou=0.7):
        # Load and preprocess the input image
        input_image = self.load_image(input_image_path)
        original_image = Image.open(input_image_path)

        # Run inference
        output1, output2 = self.run_inference(input_image)
        pred = self.non_max_suppression(
            torch.from_numpy(output1),
            conf_thres=confidence_threshold,
            iou_thres=iou,
            # agnostic=self.args.agnostic_nms,
            max_det=300,
            nc=len(MAJOR_PARTS_NO_DIRECTION),
            # classes=self.args.classes,
        )

        output21 = np.squeeze(output2, axis=0)  # Remove the batch dimension [1, 160, 160, 32] -> [160, 160, 32]

        output21 = np.transpose(output21, (2, 0, 1))
        output21 = torch.from_numpy(output21)
        proto = output21[-1] if isinstance(output21, tuple) else output21  # tuple if PyTorch model or array if exported
        pred = pred[0]
        boxes = pred[:, :4]
        class_ids = pred[:, 5:6]
        if class_ids.numel() == 0:
            if combined_parts:
                return None, None
            return None,None, None, None
        masks = process_mask(proto, pred[:, 6:], pred[:, :4], (640, 640), upsample=True)  # HWC
        confidences = pred[:, 4:5]
        class_ids = pred[:, 5:6]
        class_ids, confidences, boxes = class_ids.numpy(), confidences.numpy(), boxes.numpy()

        if combined_parts:
            img = self.draw_results_combined_mask(original_image, boxes, class_ids, confidences, masks, draw_labels)
        class_ids = class_ids.reshape(-1).astype(int).tolist()
        confidences = confidences.reshape(-1).astype(float).tolist()
        if combined_parts:
            return img, class_ids
        return class_ids,confidences, boxes, masks

    # Function to load and preprocess the input image
    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((640, 640))  # Resize the image to the expected size
        img = np.array(img)
        img = np.float32(img)  # Convert to float32
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    # Function to run inference using the TFLite interpreter
    def run_inference(self, input_image):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], input_image)
        self.interpreter.invoke()

        # Retrieve output tensors
        output1 = self.interpreter.get_tensor(output_details[0]['index'])  # Object detection
        output2 = self.interpreter.get_tensor(output_details[1]['index'])  # Segmentation masks
        return output1, output2

    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            nc=0,  # number of classes (optional)
                            max_time_img=0.05,
                            max_nms=30000,
                            max_wh=7680,
                            in_place=True,
                            # rotated=False,
                            ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Args:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels.
            in_place (bool): If True, the input prediction tensor will be modified in place.
            rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(prediction,
                      (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        if classes is not None:
            classes = torch.tensor(classes, device=prediction.device)

        if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
            output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
            if classes is not None:
                output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
            return output

        bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4  # number of masks
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 2.0 + max_time_img * bs  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

        t = time.time()
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == classes).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores

            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            # # Experimental
            # merge = False  # use merge-NMS
            # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            #     from .metrics import box_iou
            #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
            #     weights = iou * scores[None]  # box weights
            #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            #     redundant = True  # require redundant detections
            #     if redundant:
            #         i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
                break  # time limit exceeded

        return output

    def draw_results_combined_mask(self, image, boxes, class_ids, confidences, masks, draw_labels=False):
        original_width, original_height = (640, 640)
        image = image.resize((640, 640))
        image = np.array(image)

        # Initialize the combined output image (same shape as input image)
        combined_image = np.zeros_like(image, dtype=np.uint8)

        for box, class_id, confidence, mask in zip(boxes, class_ids, confidences, masks):
            x_min, y_min, x_max, y_max = map(float, box)
            x_min = int(x_min * original_width)
            y_min = int(y_min * original_height)
            x_max = int(x_max * original_width)
            y_max = int(y_max * original_height)
            label = f"{class_labels[int(class_id)]}, {float(confidence):.2f}"
            # print(label)

            # Skip invalid boxes
            if x_min >= x_max or y_min >= y_max:
                print("Invalid coordinates:", label, box)
                continue

            # Ensure mask is a NumPy array
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.float32)  # Convert to float32 if needed

                # Zero out regions outside the bounding box in the binary mask
            mask[:y_min, :] = 0  # Top region outside the bounding box
            mask[y_max:, :] = 0  # Bottom region outside the bounding box
            mask[:, :x_min] = 0  # Left region outside the bounding box
            mask[:, x_max:] = 0  # Right region outside the bounding box

            # Convert the mask to binary
            binary_mask = (mask > 0.5).astype(np.uint8) * 255

            isolated_object = cv2.bitwise_and(image, image, mask=binary_mask)

            # Combine the isolated object with the combined image
            combined_image = cv2.add(combined_image, isolated_object)

            # Optionally draw the bounding box and label
            if draw_labels:
                cv2.rectangle(combined_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(combined_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return combined_image

if __name__ == "__main__":
    cpstfl = CarPartsSegTflightModel("model/segmentation/car_parts_seg.tflite")
    # input_image_path = "/Users/yarramsettinaresh/Downloads/exercise_1/610910057eb77b4a469ebb37/dLVioqdsaI_1627983747074.jpg"
    input_image_path = '/Users/yarramsettinaresh/Downloads/exercise_1/625e744db098d00d5a42a07f/scraped_vGLssf_1649431048991.jpg'
    result_img, class_ids = cpstfl.predict(input_image_path, combined_parts=True, confidence_threshold=0.2)
    if class_ids is not None:
        Image.fromarray(result_img).show()

