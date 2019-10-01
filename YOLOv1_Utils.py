
import cv2
import numpy as np

from Define import *
from Utils import *
from DataAugmentation import *

class YOLOv1_Utils:
    def __init__(self, ):
        pass
        
    def Encode(self, gt_bboxes, gt_classes):
        label_data = np.zeros((S, S, B, 5 + CLASSES), np.float32)

        for bbox, class_index in zip(gt_bboxes, gt_classes):
            cx, cy, w, h = xyxy_to_ccwh(bbox)

            grid_x = int(cx / IMAGE_WIDTH * S)
            grid_y = int(cy / IMAGE_HEIGHT * S)
            
            label_data[grid_y, grid_x, :, :4] = bbox
            label_data[grid_y, grid_x, :, 4] = 1.0
            label_data[grid_y, grid_x, :, 5:] = one_hot(class_index)

        return label_data

    def Decode(self, encode_data, detect_threshold = 0.4, size = [IMAGE_WIDTH, IMAGE_HEIGHT], use_nms = False):
        encode_data = encode_data.reshape((-1, 5 + CLASSES))
        
        pos_mask = encode_data[:, 4] >= detect_threshold

        pred_confs = encode_data[pos_mask, 4][..., np.newaxis]
        pred_bboxes = convert_bboxes(encode_data[pos_mask, :4], size)
        pred_classes = np.argmax(encode_data[pos_mask, 5:], axis = 1)

        pred_bboxes = np.concatenate([pred_bboxes, pred_confs], axis = -1)
        
        if use_nms:
            pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)
        
        return pred_bboxes, pred_classes

if __name__ == '__main__':
    total_data_list = np.load('./dataset/train.npy', allow_pickle = True)
    yolov1_utils = YOLOv1_Utils()
    
    for data in total_data_list:
        image_name, gt_bboxes, gt_classes = data
        
        image_path = ROOT_DIR + image_name
        gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
        gt_classes = np.asarray(gt_classes, dtype = np.int32)
        
        image = cv2.imread(image_path)
        image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)

        image_h, image_w, image_c = image.shape
        gt_bboxes /= [image_w, image_h, image_w, image_h]
        gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
        
        label_data = yolov1_utils.Encode(gt_bboxes, gt_classes)
        pred_bboxes, pred_classes = yolov1_utils.Decode(label_data, size = [image_w, image_h], use_nms = True)

        print(len(gt_bboxes), len(pred_bboxes), len(gt_bboxes) - len(pred_bboxes))

        for bbox, class_index in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
            conf = bbox[4]
            class_name = CLASS_NAMES[class_index]

            string = "{} : {:.2f}%".format(class_name, conf * 100)
            cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('show', image)
        cv2.waitKey(0)
