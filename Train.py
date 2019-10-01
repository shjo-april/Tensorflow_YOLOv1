
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
from Teacher import *

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

loss_op, giou_loss_op, conf_loss_op, class_loss_op = YOLOv1_Loss(pred_tensors, label_var)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op = loss_op + l2_reg_loss_op

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)
    # train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/GIoU_Loss' : giou_loss_op,
    'Loss/Confidence_Loss' : conf_loss_op,
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
log_image_op = tf.summary.image('Image/Train', log_image_var[..., ::-1], SAMPLES)

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
conf_loss_list = []
class_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

sample_data_list = train_data_list[:SAMPLES]

train_writer = tf.summary.FileWriter('./logs/train')
train_ops = [train_op, loss_op, giou_loss_op, conf_loss_op, class_loss_op, l2_reg_loss_op, train_summary_op]

train_threads = []
for i in range(NUM_THREADS):
    train_thread = Teacher('./dataset/train.npy', debug = False)
    train_thread.start()
    train_threads.append(train_thread)

for iter in range(1, max_iteration + 1):
    if iter in decay_iteration:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    # Thread
    find = False
    while not find:
        for train_thread in train_threads:
            if train_thread.ready:
                find = True
                batch_image_data, batch_label_data = train_thread.get_batch_data()        
                break
    
    log = sess.run(train_ops, feed_dict = {input_var : batch_image_data, label_var : batch_label_data, is_training : True, learning_rate_var : learning_rate})

    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    giou_loss_list.append(log[2])
    conf_loss_list.append(log[3])
    class_loss_list.append(log[4])
    l2_reg_loss_list.append(log[5])
    train_writer.add_summary(log[6], iter)

    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        giou_loss = np.mean(giou_loss_list)
        conf_loss = np.mean(conf_loss_list)
        class_loss = np.mean(class_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, giou_loss : {:.4f}, conf_loss : {:.4f}, class_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, giou_loss, conf_loss, class_loss, l2_reg_loss, train_time))

        loss_list = []
        giou_loss_list = []
        conf_loss_list = []
        class_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    if iter % SAMPLE_ITERATION == 0:
        sample_images = []
        batch_image_data = np.zeros((SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)

        for i, data in enumerate(sample_data_list):
            image_name, gt_bboxes, gt_classes = data

            image = cv2.imread(ROOT_DIR + image_name)
            tf_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

            batch_image_data[i] = tf_image.copy()
        
        total_pred_data = sess.run(pred_tensors, feed_dict = {input_var : batch_image_data, is_training : False})
        
        for i in range(BATCH_SIZE):
            image = batch_image_data[i]
            pred_bboxes, pred_classes = yolov1_utils.Decode(total_pred_data[i], detect_threshold = 0.20, size = [IMAGE_WIDTH, IMAGE_HEIGHT])
            
            for bbox, class_index in zip(pred_bboxes, pred_classes):
                xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
                conf = bbox[4]
                class_name = CLASS_NAMES[class_index]

                string = "{} : {:.2f}%".format(class_name, conf * 100)
                cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            image = cv2.resize(image, (SAMPLE_IMAGE_WIDTH, SAMPLE_IMAGE_HEIGHT))
            sample_images.append(image.copy())
        
        image_summary = sess.run(log_image_op, feed_dict = {log_image_var : sample_images})
        train_writer.add_summary(image_summary, iter)

    if iter % SAVE_ITERATION == 0:
        saver.save(sess, './model/YOLOv1_{}.ckpt'.format(iter))
            
saver.save(sess, './model/YOLOv1.ckpt')
