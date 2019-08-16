
import cv2
import numpy as np

from DataAugmentation import *
from Define import *
from Utils import *

def Encode(xml_paths, bDataAugmentation = False):

    assert len(xml_paths) == BATCH_SIZE, "[!] Encode xml count : {}".format(len(xml_paths))
    
    np_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), np.float32)
    np_label_data = np.zeros((BATCH_SIZE, S, S, B, 5 + C), np.float32)
    
    image_paths, gt_bboxes_list, gt_classes_list = [], [], []
    for xml_path in xml_paths:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = not bDataAugmentation)
        
        image_paths.append(image_path)
        gt_bboxes_list.append(gt_bboxes)
        gt_classes_list.append(gt_classes)

    for index, image_path, gt_bboxes, gt_classes in zip(range(BATCH_SIZE), image_paths, gt_bboxes_list, gt_classes_list):
        
        image = cv2.imread(image_path)
        assert not image is None, "[!] cv2.imread : {}".format(image_path)

        if bDataAugmentation:
            gt_bboxes = np.asarray(gt_bboxes).astype(np.int32)
            gt_classes = np.asarray(gt_classes).astype(np.int32)

            image, gt_bboxes = random_flip(image, gt_bboxes)
            image, gt_bboxes = random_scale(image, gt_bboxes)
            image = random_blur(image)
            image = random_brightness(image)
            image = random_hue(image)
            image = random_saturation(image)
            image = random_gray(image)
            image, gt_bboxes, gt_classes = random_shift(image, gt_bboxes, gt_classes)
            image, gt_bboxes, gt_classes = random_crop(image, gt_bboxes, gt_classes)
            image, gt_bboxes, gt_classes = random_translate(image, gt_bboxes, gt_classes)

            h, w, c = image.shape
            gt_bboxes = gt_bboxes.astype(np.float32)
            gt_bboxes /= [w, h, w, h]
            
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_LINEAR)
        
        image_data = image.astype(np.float32)
        label_data = np.zeros([S, S, B, 5 + C], dtype = np.float32)

        for bbox, class_index in zip(gt_bboxes, gt_classes):
            ccwh_bbox = xyxy_to_ccwh(bbox)

            cx, cy, w, h = ccwh_bbox

            # get grid cell
            grid_x = int(cx * S)
            grid_y = int(cy * S)
            
            # get offset x, y
            grid_x_offset = (cx * S) - grid_x
            grid_y_offset = (cy * S) - grid_y

            # update
            for bbox_index in range(B):
                if label_data[grid_y, grid_x, bbox_index, 4] == 0:
                    label_data[grid_y, grid_x, bbox_index, 0:4] = [grid_x_offset, grid_y_offset, w, h]
                    label_data[grid_y, grid_x, bbox_index, 4] = 1.0
                    label_data[grid_y, grid_x, bbox_index, 5:] = one_hot(class_index)
                    break
        
        np_image_data[index] = image_data
        np_label_data[index] = label_data
            
    return np_image_data, np_label_data

def Decode(encode_data, detect_threshold = 0.1, size = (IMAGE_WIDTH, IMAGE_HEIGHT)):

    bboxes = []
    classes = []

    img_w, img_h = size

    for y in range(S):
        for x in range(S):
            for bbox_index in range(B):
                data = encode_data[y, x, bbox_index, : ]

                # confidence
                if data[4] >= detect_threshold:
                    offset_x, offset_y, w, h = data[:4]

                    cx = (x + offset_x) / S * img_w
                    cy = (y + offset_y) / S * img_h
                    width = w * img_w
                    height = h * img_h
                    
                    bbox = ccwh_to_xyxy([cx, cy, width, height]).astype(np.float32)

                    class_prob = data[5:]
                    max_class_prob = np.max(class_prob)
                    class_index = np.argmax(class_prob)
                    
                    bboxes.append(np.append(bbox, data[4]))
                    classes.append(class_index)

    return bboxes, classes

if __name__ == '__main__':
    # 1. YOLO Encode & Decode Test
    xml_paths = glob.glob("D:/DB/VOC2007/train/xml/" + "*")[:BATCH_SIZE]

    while True:
        np_image_data, np_label_data = Encode(xml_paths, True)

        for image, encode_data in zip(np_image_data, np_label_data):
            
            image = image.astype(np.uint8)
            bboxes, classes = Decode(encode_data)

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
