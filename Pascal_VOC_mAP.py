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

prediction_op = YOLOv1(input_var, False)

# 3. create Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 4. restore Model
saver = tf.train.Saver()
saver.restore(sess, './model/YOLOv1_{}.ckpt'.format(155000))

# 5. calculate AP@50
ap_threshold = 0.5
nms_threshold = 0.5

# 6. class loop
correct_dic = {}
confidence_dic = {}
all_ground_truths_dic = {}

for class_name in CLASS_NAMES:
    correct_dic[class_name] = []
    confidence_dic[class_name] = []
    all_ground_truths_dic[class_name] = 0.

batch_image_data = []
batch_image_wh = []
batch_gt_bboxes_dic = []

test_time = time.time()

for test_iter, xml_path in enumerate(test_xml_paths):
    image_path, gt_bboxes_dic = class_xml_read(xml_path, CLASS_NAMES)

    ori_image = cv2.imread(image_path)
    image = cv2.resize(ori_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    
    batch_image_data.append(image.astype(np.float32))
    batch_image_wh.append(ori_image.shape[:-1][::-1])
    batch_gt_bboxes_dic.append(gt_bboxes_dic)

    # calculate correct/confidence
    if len(batch_image_data) == BATCH_SIZE:
        pred_encode_data = sess.run(prediction_op, feed_dict = {input_var : batch_image_data})

        for i in range(BATCH_SIZE):
            gt_bboxes_dic = batch_gt_bboxes_dic[i]
            for class_name in list(gt_bboxes_dic.keys()):
                gt_bboxes = np.asarray(gt_bboxes_dic[class_name], dtype = np.float32)

                gt_class = CLASS_DIC[class_name]
                all_ground_truths_dic[class_name] += gt_bboxes.shape[0]

                pred_bboxes, pred_classes = yolov1_utils.Decode(pred_encode_data[i], size = batch_image_wh[i], detect_threshold = 0.0, detect_class_names = [class_name], nms = True)

                if pred_bboxes.shape[0] == 0:
                    pred_bboxes = np.zeros((1, 5), dtype = np.float32)

                ious = compute_bboxes_IoU(pred_bboxes, gt_bboxes)
                
                # ious >= 0.50 (AP@50)
                correct = np.max(ious, axis = 1) > ap_threshold
                confidence = pred_bboxes[:, 4]

                correct_dic[class_name] += correct.tolist()
                confidence_dic[class_name] += confidence.tolist()

        batch_image_data = []
        batch_image_wh = []
        batch_gt_bboxes_dic = []

    sys.stdout.write('\r# Test = {:.2f}%'.format(test_iter / test_xml_count * 100))
    sys.stdout.flush()

if len(batch_image_data) != 0:
    pred_encode_data = sess.run(prediction_op, feed_dict = {input_var : batch_image_data})

    for i in range(len(batch_image_data)):
        gt_bboxes_dic = batch_gt_bboxes_dic[i]
        for class_name in list(gt_bboxes_dic.keys()):
            gt_bboxes = np.asarray(gt_bboxes_dic[class_name], dtype = np.float32)

            gt_class = CLASS_DIC[class_name]
            all_ground_truths_dic[class_name] += gt_bboxes.shape[0]
            
            pred_bboxes, pred_classes = yolov1_utils.Decode(pred_encode_data[i], size = batch_image_wh[i], detect_threshold = 0.0, detect_class_names = [class_name], nms = True)
            ious = compute_bboxes_IoU(pred_bboxes, gt_bboxes)

            # ious >= 0.50 (AP@50)
            correct = np.max(ious, axis = 1) >= ap_threshold
            confidence = pred_bboxes[:, 4]

            correct_dic[class_name] += correct.tolist()
            confidence_dic[class_name] += confidence.tolist()

print(all_ground_truths_dic)

test_time = int(time.time() - test_time)
print('\n[i] test time = {}sec'.format(test_time))

map_list = []

for class_name in CLASS_NAMES:
    if all_ground_truths_dic[class_name] == 0:
        continue
    
    correct_list = correct_dic[class_name]
    confidence_list = confidence_dic[class_name]
    all_ground_truths = np.sum(correct_list)

    # list -> numpy
    confidence_list = np.asarray(confidence_list, dtype = np.float32)
    correct_list = np.asarray(correct_list, dtype = np.bool)
    
    # Ascending (confidence)
    sort_indexs = confidence_list.argsort()[::-1]
    confidence_list = confidence_list[sort_indexs]
    correct_list = correct_list[sort_indexs]
    
    correct_detections = 0
    all_detections = 0
    
    # calculate precision/recall
    precision_list = []
    recall_list = []

    for confidence, correct in zip(confidence_list, correct_list):
        all_detections += 1
        if correct:
            correct_detections += 1    
        
        precision = correct_detections / all_detections
        recall = correct_detections / all_ground_truths
        
        precision_list.append(precision)
        recall_list.append(recall)

        # maximum correct detections
        if recall == 1.0:
            break

    precision_list = np.asarray(precision_list, dtype = np.float32)
    recall_list = np.asarray(recall_list, dtype = np.float32)
    
    # calculating the interpolation performed in 11 points (0.0 -> 1.0, +0.01)
    precision_interp_list = []
    interp_list = np.arange(0, 10 + 1) / 10

    for interp in interp_list:
        try:
            precision_interp = max(precision_list[recall_list >= interp])
        except:
            precision_interp = 0.0
        
        precision_interp_list.append(precision_interp)

    ap = np.mean(precision_interp_list) * 100
    map_list.append(ap)
    
    # matplotlib (precision&recall curve + interpolation)
    plt.clf()
    plt.plot(recall_list, precision_list, 'green')
    plt.plot(interp_list, precision_interp_list, 'ro')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('# Precision-recall curve ({} - {:.2f}%)'.format(class_name, ap))
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(['precision/recall', 'interpolation'], loc='lower left')
    # plt.show()
    plt.savefig('./results/{}.jpg'.format(class_name))

'''
# AP@50 aeroplane = 68.38%
# AP@50 bicycle = 70.28%
# AP@50 bird = 63.91%
# AP@50 boat = 49.64%
# AP@50 bottle = 57.35%
# AP@50 bus = 81.63%
# AP@50 car = 71.31%
# AP@50 cat = 78.95%
# AP@50 chair = 56.38%
# AP@50 cow = 72.50%
# AP@50 diningtable = 66.00%
# AP@50 dog = 76.45%
# AP@50 horse = 70.27%
# AP@50 motorbike = 73.30%
# AP@50 person = 70.37%
# AP@50 pottedplant = 56.02%
# AP@50 sheep = 75.43%
# AP@50 sofa = 67.10%
# AP@50 train = 76.29%
# AP@50 tvmonitor = 73.99%
# mAP@50 = 68.78%
'''
print()
for ap, class_name in zip(map_list, CLASS_NAMES):
    print('# AP@50 {} = {:.2f}%'.format(class_name, ap))
print('# mAP@50 = {:.2f}%'.format(np.mean(map_list)))
