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

from mAP_Calculator import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
test_xml_paths = glob.glob(ROOT_DIR + 'VOC2007/test/xml/*.xml')[:100]
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
nms_threshold = 0.5
mAP_calc = mAP_Calculator(classes = CLASSES)

test_time = time.time()

batch_image_data = []
batch_image_wh = []

batch_gt_bboxes = []
batch_gt_classes = []

for test_iter, xml_path in enumerate(test_xml_paths):
    image_path, gt_bboxes, gt_classes = xml_read(xml_path, CLASS_NAMES)

    ori_image = cv2.imread(image_path)
    image = cv2.resize(ori_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    
    batch_image_data.append(image.astype(np.float32))
    batch_image_wh.append(ori_image.shape[:-1][::-1])

    batch_gt_bboxes.append(gt_bboxes)
    batch_gt_classes.append(gt_classes)
    
    # calculate correct/confidence
    if len(batch_image_data) == BATCH_SIZE:
        pred_encode_data = sess.run(prediction_op, feed_dict = {input_var : batch_image_data})

        for i in range(BATCH_SIZE):
            gt_bboxes, gt_classes = batch_gt_bboxes[i], batch_gt_classes[i]
            pred_bboxes, pred_classes = yolov1_utils.Decode(pred_encode_data[i], size = batch_image_wh[i], detect_threshold = 0.00, nms = True)

            if pred_bboxes.shape[0] == 0:
                pred_bboxes = np.zeros((0, 5), dtype = np.float32)
            
            mAP_calc.update(pred_bboxes, pred_classes, gt_bboxes, gt_classes)

        batch_image_data = []
        batch_image_wh = []

        batch_gt_bboxes = []
        batch_gt_classes = []
    
    sys.stdout.write('\r# Test = {:.2f}%'.format(test_iter / test_xml_count * 100))
    sys.stdout.flush()

# if len(batch_image_data) != 0:
#     pred_encode_data = sess.run(prediction_op, feed_dict = {input_var : batch_image_data})

#     for i in range(len(batch_image_data)):
#         gt_bboxes_dic = batch_gt_bboxes_dic[i]
#         for class_name in list(gt_bboxes_dic.keys()):
#             gt_bboxes = np.asarray(gt_bboxes_dic[class_name], dtype = np.float32)

#             gt_class = CLASS_DIC[class_name]
#             all_ground_truths_dic[class_name] += gt_bboxes.shape[0]
            
#             pred_bboxes, pred_classes = yolov1_utils.Decode(pred_encode_data[i], size = batch_image_wh[i], detect_threshold = 0.0, detect_class_names = [class_name], nms = True)
#             ious = compute_bboxes_IoU(pred_bboxes, gt_bboxes)

#             # ious >= 0.50 (AP@50)
#             correct = np.max(ious, axis = 1) >= ap_threshold
#             confidence = pred_bboxes[:, 4]

#             correct_dic[class_name] += correct.tolist()
#             confidence_dic[class_name] += confidence.tolist()

test_time = int(time.time() - test_time)
print('\n[i] test time = {}sec'.format(test_time))

map_list = []

for i, class_name in enumerate(CLASS_NAMES):
    precisions, recalls = mAP_calc.get_precision_and_recall(i)
    ap = np.mean(precisions) * 100
    # ap = mAP_calc.compute_ap(precisions, recalls) * 100
    
    # matplotlib (precision&recall curve + interpolation)
    plt.clf()
    
    plt.fill_between(recalls, precisions, step = 'post', alpha = 0.2, color = 'green')
    plt.plot(recalls, precisions, 'green')
    plt.plot(np.arange(0, 10 + 1) / 10, precisions, 'ro')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('# Precision-recall curve ({} - {:.2f}%)'.format(class_name, ap))
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
    # plt.savefig('./results/{}.jpg'.format(class_name))

    map_list.append(ap)

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
