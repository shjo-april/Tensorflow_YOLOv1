import glob
import numpy as np
import xml.etree.ElementTree as ET

from Define import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def xml_read(xml_path, find_labels = CLASS_NAMES, normalize = False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = xml_path[:-3] + '*'
    image_path = image_path.replace('/xml', '/image')
    image_path = glob.glob(image_path)[0]

    if normalize:
        size = root.find('size')
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text

        if label in find_labels:
            classes.append(CLASS_DIC[label])
        else:
            continue

        bbox = obj.find('bndbox')
        
        bbox_xmin = int(bbox.find('xmin').text.split('.')[0])
        bbox_ymin = int(bbox.find('ymin').text.split('.')[0])
        bbox_xmax = int(bbox.find('xmax').text.split('.')[0])
        bbox_ymax = int(bbox.find('ymax').text.split('.')[0])

        if normalize:
            bbox_xmin /= image_width
            bbox_ymin /= image_height
            bbox_xmax /= image_width
            bbox_ymax /= image_height
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])

    return image_path, np.asarray(bboxes, dtype = np.float32), classes

def class_nms(bboxes, classes, threshold = 0.5, mode = 'Union'):
    data_dic = {}
    nms_bboxes = []
    nms_classes = []

    for bbox, class_index in zip(bboxes, classes):
        try:
            data_dic[class_index].append(bbox)
        except KeyError:
            data_dic[class_index] = []
            data_dic[class_index].append(bbox)

    for key in data_dic.keys():
        data_dic[key] = np.asarray(data_dic[key], dtype = np.float32)

        #print(data_dic[key].shape)
        keep_indexs = py_nms(data_dic[key], threshold)

        for bbox in data_dic[key][keep_indexs]:
            nms_bboxes.append(bbox)
            nms_classes.append(key)

    return nms_bboxes, nms_classes

def py_nms(dets, thresh, mode="Union"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def ccwh_to_xyxy(bbox):
    cx, cy, w, h = bbox

    xmin = max(cx - w / 2, 0)
    ymin = max(cy - h / 2, 0)
    xmax = cx + w / 2
    ymax = cy + h / 2 
    
    return np.asarray([xmin, ymin, xmax, ymax])

def xyxy_to_ccwh(bbox):
    xmin, ymin, xmax, ymax = bbox

    cx = float((xmax + xmin) / 2)
    cy = float((ymax + ymin) / 2)
    width = xmax - xmin
    height = ymax - ymin

    return np.asarray([cx, cy, width, height])

def one_hot(label, classes = C):
    vector = np.zeros(classes, dtype = np.float32)
    vector[label] = 1.
    return vector

def IOU(box1, box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h

    if (box1_area + box2_area - inter) <= 0:
        return 0.0

    ovr = inter * 1.0 / (box1_area + box2_area - inter)
    return ovr

def Precision_Recall(gt_boxes, gt_classes, pred_boxes, pred_classes, threshold_iou = 0.5):
    recall = 0.0
    precision = 0.0

    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return 1.0, 1.0
        else:
            return 0.0, 0.0

    if len(pred_boxes) != 0:
        gt_boxes_cnt = len(gt_boxes)
        pred_boxes_cnt = len(pred_boxes)

        recall_vector = np.zeros(gt_boxes_cnt)
        precision_vector = np.zeros(pred_boxes_cnt)

        for gt_index in range(gt_boxes_cnt):
            for pred_index in range(pred_boxes_cnt):
                if IOU(pred_boxes[pred_index], gt_boxes[gt_index]) >= threshold_iou:
                    recall_vector[gt_index] = True
                    if gt_classes[gt_index] == pred_classes[pred_index]:
                        precision_vector[pred_index] = True

        recall = np.sum(recall_vector) / gt_boxes_cnt
        precision = np.sum(precision_vector) / pred_boxes_cnt

    return precision, recall
