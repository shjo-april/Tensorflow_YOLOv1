# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Define import *
from Utils import *

from YOLOv1 import *
from YOLOv1_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
test_xml_paths = glob.glob(ROOT_DIR + 'VOC2007/test/xml/*.xml')
test_xml_count = len(test_xml_paths)
print('[i] Test : {}'.format(len(test_xml_paths)))

# 2. build
yolov1_utils = YOLOv1_Utils()
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

pred_tensors = YOLOv1(input_var, False)

# 3. create Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 4. restore Model
saver = tf.train.Saver()
saver.restore(sess, './model/YOLOv1_{}.ckpt'.format(170000))

# 5. calculate AP@50
batch_image_id = []
batch_image_data = []
batch_image_wh = []

if not os.path.isdir('./YOLOv1/detection-results/'):
    os.makedirs('./YOLOv1/detection-results/')
        
for test_iter, xml_path in enumerate(test_xml_paths):
    image_path, gt_bboxes, gt_classes = xml_read(xml_path, CLASS_NAMES)
    image_name = os.path.basename(image_path)
    image_id = image_name.split('.')[0]

    ori_image = cv2.imread(image_path)
    image = cv2.resize(ori_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

    batch_image_id.append(image_id)
    batch_image_data.append(image.astype(np.float32))
    batch_image_wh.append(ori_image.shape[:-1][::-1])

    # calculate correct/confidence
    if len(batch_image_data) == BATCH_SIZE:
        pred_data = sess.run(pred_tensors, feed_dict = {input_var : batch_image_data})

        for i in range(BATCH_SIZE):
            pred_bboxes, pred_classes = yolov1_utils.Decode(pred_data[i], detect_threshold = 0.01, size = batch_image_wh[i], use_nms = True)

            f = open('./YOLOv1/detection-results/{}.txt'.format(batch_image_id[i]), 'w')
            for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
                xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
                f.write('{} {} {} {} {} {}\n'.format(CLASS_NAMES[pred_class], pred_bbox[4], xmin, ymin, xmax, ymax))
            f.close()

        batch_image_id = []
        batch_image_data = []
        batch_image_wh = []

    sys.stdout.write('\r# Test = {:.2f}%'.format(test_iter / test_xml_count * 100))
    sys.stdout.flush()
print()

if len(batch_image_data) != 0:
    pred_data = sess.run(pred_tensors, feed_dict = {input_var : batch_image_data})

    for i in range(len(batch_image_data)):
        pred_bboxes, pred_classes = yolov1_utils.Decode(pred_data[i], detect_threshold = 0.01, size = batch_image_wh[i], use_nms = True)

        f = open('./YOLOv1/detection-results/{}.txt'.format(batch_image_id[i]), 'w')
        for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
            f.write('{} {} {} {} {} {}\n'.format(CLASS_NAMES[pred_class], pred_bbox[4], xmin, ymin, xmax, ymax))
        f.close()

'''
./model/YOLOv1_140000.ckpt
66.07% = aeroplane AP
63.71% = bicycle AP
60.05% = bird AP
47.08% = boat AP
34.58% = bottle AP
62.01% = bus AP
67.37% = car AP
74.38% = cat AP
41.46% = chair AP
60.86% = cow AP
46.44% = diningtable AP
64.05% = dog AP
62.82% = horse AP
64.91% = motorbike AP
68.06% = person AP
37.84% = pottedplant AP
57.11% = sheep AP
48.44% = sofa AP
71.94% = train AP
65.64% = tvmonitor AP
mAP = 58.24%

./model/YOLOv1_170000.ckpt
70.65% = aeroplane AP
65.55% = bicycle AP
59.71% = bird AP
47.49% = boat AP
35.26% = bottle AP
64.01% = bus AP
67.48% = car AP
75.20% = cat AP
41.69% = chair AP
61.25% = cow AP
50.59% = diningtable AP
65.91% = dog AP
62.94% = horse AP
65.65% = motorbike AP
68.39% = person AP
38.22% = pottedplant AP
59.03% = sheep AP
49.89% = sofa AP
73.08% = train AP
65.50% = tvmonitor AP
mAP = 59.37%
'''
