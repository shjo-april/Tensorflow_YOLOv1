
import cv2
import numpy as np

from Define import *
from Utils import *
from DataAugmentation import *

class YOLOv1_Utils:
    def __init__(self, ):
        pass
        
    def Encode(self, xml_paths, augment = False):
        np_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), np.float32)
        np_label_data = np.zeros((BATCH_SIZE, S, S, B, 5 + CLASSES), np.float32)
        
        image_paths, gt_bboxes_list, gt_classes_list = [], [], []

        for i, xml_path in enumerate(xml_paths):
            image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = False)
            
            image = cv2.imread(image_path)
            
            if augment:
                image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)
            
            h, w, c = image.shape
            gt_bboxes = gt_bboxes.astype(np.float32)
            gt_bboxes /= [w, h, w, h]
            
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
            
            image_data = image.astype(np.float32)
            label_data = np.zeros([S, S, B, 5 + CLASSES], dtype = np.float32)

            for bbox, class_index in zip(gt_bboxes, gt_classes):
                ccwh_bbox = xyxy_to_ccwh(bbox)
                cx, cy, w, h = ccwh_bbox

                # get grid cell
                grid_x = int(cx * S)
                grid_y = int(cy * S)
                
                # get offset x, y
                grid_x_offset = (cx * S) - grid_x
                grid_y_offset = (cy * S) - grid_y

                for bbox_index in range(B):
                    if label_data[grid_y, grid_x, bbox_index, 4] == 0:
                        label_data[grid_y, grid_x, bbox_index, :4] = [grid_x_offset, grid_y_offset, w, h]
                        label_data[grid_y, grid_x, bbox_index, 4] = 1.0
                        label_data[grid_y, grid_x, bbox_index, 5:] = one_hot(class_index)
                        break
            
            np_image_data[i] = image_data
            np_label_data[i] = label_data
            
        return np_image_data, np_label_data

    def Decode(self, encode_data, detect_threshold = 0.4, size = (IMAGE_WIDTH, IMAGE_HEIGHT), detect_class_names = CLASS_NAMES, nms = False):
        pred_bboxes = []
        pred_classes = []

        img_w, img_h = size
        
        for y in range(S):
            for x in range(S):
                for bbox_index in range(B):
                    data = encode_data[y, x, bbox_index, : ]
                    if data[4] >= detect_threshold:
                        offset_x, offset_y, w, h = data[:4]

                        cx = (x + offset_x) / S * img_w
                        cy = (y + offset_y) / S * img_h
                        width = w * img_w
                        height = h * img_h
                        
                        bbox = ccwh_to_xyxy([cx, cy, width, height]).astype(np.float32)
                        class_index = np.argmax(data[5:])
                        
                        if CLASS_NAMES[class_index] in detect_class_names:
                            pred_bboxes.append(np.append(bbox, data[4]))
                            pred_classes.append(class_index)

        pred_bboxes = np.asarray(pred_bboxes, dtype = np.float32)
        pred_classes = np.asarray(pred_classes, dtype = np.int32)
        
        if nms:
            pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)

        return pred_bboxes, pred_classes

if __name__ == '__main__':
    xml_paths = glob.glob("D:/DB/VOC2007/train/xml/*")
    yolov2_utils = YOLOv2_Utils()

    for i in range(BATCH_SIZE):
        batch_xml_paths = xml_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        np_image_data, np_label_data = yolov2_utils.Encode(batch_xml_paths, True)

        for image, encode_data in zip(np_image_data, np_label_data):
            image = image.astype(np.uint8)
            bboxes, classes = yolov2_utils.Decode(encode_data)

            print(bboxes)
            print(classes)

            for bbox, class_index in zip(bboxes, classes):
                xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
                conf = bbox[4]
                class_name = CLASS_NAMES[class_index]

                string = "{} : {:.2f}%".format(class_name, conf * 100)
                cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            cv2.imshow('show', image)
            cv2.waitKey(0)
