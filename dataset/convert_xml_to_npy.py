# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import sys
import glob

import numpy as np
import xml.etree.ElementTree as ET

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}

def xml_read(xml_path, find_labels = CLASS_NAMES):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = xml_path[:-3] + '*'
    image_path = image_path.replace('/xml', '/image')
    image_path = glob.glob(image_path)[0]

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        if not label in find_labels:
            continue
            
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(CLASS_DIC[label])

    return image_path, np.asarray(bboxes, dtype = np.float32), np.asarray(classes, dtype = np.int32)

ROOT_DIR = 'D:/DB/'
TRAIN_RATIO = 0.9

xml_paths = glob.glob(ROOT_DIR + 'VOC2007/train/xml/*.xml') + glob.glob(ROOT_DIR + 'VOC2012/train/xml/*.xml')
np.random.shuffle(xml_paths)

length = len(xml_paths)
train_length = int(length * TRAIN_RATIO)

train_xml_paths = xml_paths[:train_length]
valid_xml_paths = xml_paths[train_length:]

data_list = []
for i, xml_path in enumerate(train_xml_paths):
    image_path, gt_bboxes, gt_classes = xml_read(xml_path)
    
    data = [image_path.replace(ROOT_DIR, ''), gt_bboxes, gt_classes]
    data_list.append(data)

    sys.stdout.write('\r[{}/{}]'.format(i, length))
    sys.stdout.flush()
print()

data_list = np.asarray(data_list)
np.save('./dataset/train.npy', data_list)

data_list = []
for i, xml_path in enumerate(valid_xml_paths):
    image_path, gt_bboxes, gt_classes = xml_read(xml_path)

    data = [image_path.replace(ROOT_DIR, ''), gt_bboxes, gt_classes]
    data_list.append(data)

    sys.stdout.write('\r[{}/{}]'.format(i, length))
    sys.stdout.flush()
print()

data_list = np.asarray(data_list)
np.save('./dataset/validation.npy', data_list)
