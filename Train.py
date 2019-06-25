
import os
import cv2
import sys
import glob
import time

import numpy as np
import tensorflow as tf

from Define import *
from YOLOv1 import *
from YOLO_Loss import *
from YOLO_Utils import *

from Utils import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# 1. dataset
TRAIN_XML_DIRS = ["D:/_ImageDataset/VOC2007/train/xml/", "D:/_ImageDataset/VOC2012/xml/"]
TEST_XML_DIRS = ["D:/_ImageDataset/VOC2007/test/xml/"]

train_xml_paths = []
test_xml_paths = []

for train_xml_dir in TRAIN_XML_DIRS:
    train_xml_paths += glob.glob(train_xml_dir + "*")

for test_xml_dir in TEST_XML_DIRS:
    test_xml_paths += glob.glob(test_xml_dir + "*")

np.random.shuffle(train_xml_paths)
train_xml_paths = np.asarray(train_xml_paths)

valid_xml_paths = train_xml_paths[:int(len(train_xml_paths) * 0.1)]
train_xml_paths = train_xml_paths[int(len(train_xml_paths) * 0.1):]

print('train : {}'.format(len(train_xml_paths)))
print('valid : {}'.format(len(valid_xml_paths)))
print('test : {}'.format(len(test_xml_paths)))

# 2. build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
label_var = tf.placeholder(tf.float32, [None, S, S, B, 5 + C])

is_training = tf.placeholder(tf.bool)
lr_var = tf.placeholder(tf.float32, name = 'lr')

pred_tensor = YOLOv1(input_var, is_training)
loss_op, xy_loss_op, wh_loss_op, obj_loss_op, noobj_loss_op, class_loss_op = YOLO_Loss(pred_tensor, label_var)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY

loss_op = loss_op + l2_reg_loss_op

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = tf.train.MomentumOptimizer(lr_var, MOMENTUM).minimize(loss_op)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

vgg_vars = []
for var in vars:
    if 'vgg' in var.name:
        vgg_vars.append(var)

vgg_saver = tf.train.Saver(var_list = vgg_vars)
vgg_saver.restore(sess, './vgg_16/vgg_16.ckpt')

saver = tf.train.Saver()

decay_epoch = np.asarray([0.5, 0.75])
decay_epoch *= MAX_EPOCHS
decay_epoch = decay_epoch.astype(np.int32)

lr = INIT_LR
max_iterations = len(train_xml_paths) // BATCH_SIZE

best_valid_mAP = 0.0

for epoch in range(1, MAX_EPOCHS):

    if epoch in decay_epoch:
        lr /= 10
        print('learning rate decay')

    loss_list = []
    
    xy_loss_list = []
    wh_loss_list = []
    obj_loss_list = []
    noobj_loss_list = []
    class_loss_list = []
    l2_reg_loss_list = []

    np.random.shuffle(train_xml_paths)
    for iter in range(len(train_xml_paths) // BATCH_SIZE):
        xml_paths = train_xml_paths[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]

        np_image_data, np_label_data = Encode(xml_paths, True)
        _feed_dict = {input_var : np_image_data, label_var : np_label_data, is_training : True, lr_var : lr}
        _, loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss, l2_reg_loss = sess.run([train_op, loss_op, xy_loss_op, wh_loss_op, obj_loss_op, noobj_loss_op, class_loss_op, l2_reg_loss_op], feed_dict = _feed_dict)
        
        # debug
        #print(loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss, l2_reg_loss)
        assert not np.isnan(loss), "Loss = Nan !"
        
        loss_list.append(loss)
        
        xy_loss_list.append(xy_loss)
        wh_loss_list.append(wh_loss)
        obj_loss_list.append(obj_loss)
        noobj_loss_list.append(noobj_loss)
        class_loss_list.append(class_loss)
        l2_reg_loss_list.append(l2_reg_loss)

        sys.stdout.write('\r[{}/{}]'.format(iter, max_iterations))
        sys.stdout.flush()

    loss = np.mean(loss_list)
    xy_loss = np.mean(xy_loss_list)
    wh_loss = np.mean(wh_loss_list)
    obj_loss = np.mean(obj_loss_list)
    noobj_loss = np.mean(noobj_loss_list)
    class_loss = np.mean(class_loss_list)
    l2_reg_loss = np.mean(l2_reg_loss_list)
    
    print(' epoch : {}, loss : {:.4f}, xy_loss : {:.4f}, wh_loss : {:.4f}, obj_loss : {:.4f}, noobj_loss : {:.4f}, class_loss : {:.4f}, l2_reg_loss : {:.4f}'.format(epoch, loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss, l2_reg_loss))

    #saver.save(sess, './model/YOLOv1_{}.ckpt'.format(epoch))

    # validation mAP
    precision_list = []
    recall_list = []

    # single batch (batch_norm check)
    for xml_path in valid_xml_paths:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path)

        image = cv2.imread(image_path)
        h, w, c = image.shape
        
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
        pred_encode_data = sess.run(pred_tensor, feed_dict = {input_var : [image], is_training : False})

        pred_bboxes, pred_classes = Decode(pred_encode_data[0], size = (w, h))
        pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes, threshold = 0.5)

        precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes)
    
        precision_list.append(precision)
        recall_list.append(recall)

    precision = np.mean(precision_list) * 100
    recall = np.mean(recall_list) * 100
    mAP = (precision + recall) / 2

    print('valid mAP : {:.2f}, best valid mAP : {:.2f}%'.format(mAP, best_valid_mAP))

    if best_valid_mAP < mAP:
        best_valid_mAP = mAP
        saver.save(sess, './model/YOLOv1_{}.ckpt'.format(epoch))

