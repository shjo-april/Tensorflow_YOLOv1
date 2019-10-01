
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

from Define import *
from Utils import *
from DataAugmentation import *

from YOLOv1 import *
from YOLOv1_Loss import *
from YOLOv1_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. dataset
train_data_list = np.load('./dataset/train.npy', allow_pickle = True)
valid_data_list = np.load('./dataset/validation.npy', allow_pickle = True)
valid_count = len(valid_data_list)

open('log.txt', 'w')
log_print('[i] Train : {}'.format(len(train_data_list)))
log_print('[i] Valid : {}'.format(len(valid_data_list)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
label_var = tf.placeholder(tf.float32, [None, S, S, B, 5 + CLASSES])
is_training = tf.placeholder(tf.bool)

pred_tensors = YOLOv1(input_var, is_training)
log_print('[i] pred_tensors : {}'.format(pred_tensors))

loss_op, giou_loss_op, pos_conf_loss_op, neg_conf_loss_op, class_loss_op = YOLOv1_Loss(pred_tensors, label_var)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op = loss_op + l2_reg_loss_op

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)
    train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/GIoU_Loss' : giou_loss_op,
    'Loss/Positive_Confidence_Loss' : pos_conf_loss_op,
    'Loss/Negative_Confidence_Loss' : neg_conf_loss_op,
    'Loss/Class_Loss' : class_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

log_image_var = tf.placeholder(tf.float32, [None, SAMPLE_IMAGE_HEIGHT, SAMPLE_IMAGE_WIDTH, IMAGE_CHANNEL])
log_image_op = tf.summary.image('Image/Train', log_image_var, SAMPLES)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# '''
pretrained_vars = []
for var in vars:
    if 'resnet_v1_50' in var.name:
        pretrained_vars.append(var)

pretrained_saver = tf.train.Saver(var_list = pretrained_vars)
pretrained_saver.restore(sess, './resnet_v1_model/resnet_v1_50.ckpt')
# '''

saver = tf.train.Saver(max_to_keep = 30)
# saver.restore(sess, './model/YOLOv1_{}.ckpt'.format(115000))

best_valid_mAP = 0.0
learning_rate = INIT_LEARNING_RATE

train_iteration = len(train_data_list) // BATCH_SIZE

max_iteration = train_iteration * MAX_EPOCH
decay_iteration = np.asarray([0.5 * max_iteration, 0.75 * max_iteration], dtype = np.int32)

log_print('[i] max_iteration : {}'.format(max_iteration))
log_print('[i] decay_iteration : {}'.format(decay_iteration))

loss_list = []
giou_loss_list = []
pos_conf_loss_list = []
neg_conf_loss_list = []
class_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train')

sample_data_list = train_data_list[:SAMPLES]
train_ops = [train_op, loss_op, giou_loss_op, pos_conf_loss_op, neg_conf_loss_op, class_loss_op, l2_reg_loss_op, train_summary_op]

for iter in range(1, max_iteration + 1):
    if iter in decay_iteration:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))
    
    batch_xml_paths = random.sample(train_xml_paths, BATCH_SIZE)
    batch_image_data, batch_label_data = yolov1_utils.Encode(batch_xml_paths, augment = True)

    log = sess.run(train_ops, feed_dict = {input_var : batch_image_data, label_var : batch_label_data, is_training : True, learning_rate_var : learning_rate})

    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    xy_loss_list.append(log[2])
    wh_loss_list.append(log[3])
    obj_loss_list.append(log[4])
    noobj_loss_list.append(log[5])
    class_loss_list.append(log[6])
    l2_reg_loss_list.append(log[7])
    train_writer.add_summary(log[8], iter)

    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        xy_loss = np.mean(xy_loss_list)
        wh_loss = np.mean(wh_loss_list)
        obj_loss = np.mean(obj_loss_list)
        noobj_loss = np.mean(noobj_loss_list)
        class_loss = np.mean(class_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, xy_loss : {:.4f}, wh_loss : {:.4f}, obj_loss : {:.4f}, noobj_loss : {:.4f}, class_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss, l2_reg_loss, train_time))

        loss_list = []
        xy_loss_list = []
        wh_loss_list = []
        obj_loss_list = []
        noobj_loss_list = []
        class_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % VALID_ITERATION == 0:
        ap_threshold = 0.5
        nms_threshold = 0.6

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

        valid_time = time.time()

        for valid_iter, xml_path in enumerate(valid_xml_paths):
            image_path, gt_bboxes_dic = class_xml_read(xml_path, CLASS_NAMES)

            ori_image = cv2.imread(image_path)
            image = cv2.resize(ori_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
            
            batch_image_data.append(image.astype(np.float32))
            batch_image_wh.append(ori_image.shape[:-1][::-1])
            batch_gt_bboxes_dic.append(gt_bboxes_dic)

            # calculate correct/confidence
            if len(batch_image_data) == BATCH_SIZE:
                pred_encode_data = sess.run(prediction_op, feed_dict = {input_var : batch_image_data, is_training : False})

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
                        correct = np.max(ious, axis = 1) >= ap_threshold
                        confidence = pred_bboxes[:, 4]

                        correct_dic[class_name] += correct.tolist()
                        confidence_dic[class_name] += confidence.tolist()

                batch_image_data = []
                batch_image_wh = []
                batch_gt_bboxes_dic = []

            sys.stdout.write('\r# Validation = {:.2f}%'.format(valid_iter / valid_xml_count * 100))
            sys.stdout.flush()

        if len(batch_image_data) != 0:
            pred_encode_data = sess.run(prediction_op, feed_dict = {input_var : batch_image_data, is_training : False})

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

        valid_time = int(time.time() - valid_time)
        print('\n[i] valid time = {}sec'.format(valid_time))

        valid_mAP_list = []
        for class_name in CLASS_NAMES:
            if all_ground_truths_dic[class_name] == 0:
                continue

            correct_list = correct_dic[class_name]
            confidence_list = confidence_dic[class_name]
            correct_list = correct_dic[class_name]

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
            valid_mAP_list.append(ap)

        valid_mAP = np.mean(valid_mAP_list)
        if best_valid_mAP < valid_mAP:
            best_valid_mAP = valid_mAP
            saver.save(sess, './model/YOLOv1_{}.ckpt'.format(iter))
            
        log_print('[i] valid mAP : {:.6f}, best valid mAP : {:.6f}'.format(valid_mAP, best_valid_mAP))

saver.save(sess, './model/YOLOv1.ckpt')
