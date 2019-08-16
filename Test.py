import os
import cv2
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Define import *
from Utils import *
from YOLOv1 import *
from YOLO_Utils import *

model_path = './model/YOLOv1_{}.ckpt'.format(PRETRAINED_MODEL_NAME)

if PRETRAINED_MODEL_NAME == 'VGG16':
    detect_threshold = 0.31
elif PRETRAINED_MODEL_NAME == 'InceptionResNetv2':
    detect_threshold = 0.34

# build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
pred_tensor = YOLOv1(input_var, False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, model_path)

# test
xml_paths = glob.glob("D:/DB/VOC2007/test/xml/*")
xml_count = len(xml_paths)

for index, xml_path in enumerate(xml_paths[:20]):

    image_path, gt_bboxes, gt_classes = xml_read(xml_path)
    
    image = cv2.imread(image_path)
    h, w, c = image.shape
    
    tf_image = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)]
    pred_encode_data = sess.run(pred_tensor, feed_dict = {input_var : tf_image})

    pred_bboxes, pred_classes = Decode(pred_encode_data[0], size = (w, h), detect_threshold = detect_threshold)
    pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes, threshold = 0.5)
    
    for bbox, class_index in zip(gt_bboxes, gt_classes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int32)

        # Text
        string = "{}".format(CLASS_NAMES[class_index])
        text_size = cv2.getTextSize(string, 0, 0.5, thickness = 1)[0]
        
        cv2.rectangle(image, (xmin, ymin), (xmin + text_size[0], ymin - text_size[1] - 5), (0, 0, 255), -1)
        cv2.putText(image, string, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType = cv2.LINE_AA)

        # Rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    for bbox, class_index in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
        conf = bbox[4]
        class_name = CLASS_NAMES[class_index]

        # Text
        string = "{} : {:.2f}%".format(class_name, conf * 100)
        text_size = cv2.getTextSize(string, 0, 0.5, thickness = 1)[0]

        cv2.rectangle(image, (xmin, ymin), (xmin + text_size[0], ymin - text_size[1] - 5), (0, 255, 0), -1)
        cv2.putText(image, string, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType = cv2.LINE_AA)

        # Rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # cv2.imshow(PRETRAINED_MODEL_NAME, image)
    # cv2.waitKey(0)

    image_name = os.path.basename(image_path)
    cv2.imwrite('./results/{}_Test_Samples/'.format(PRETRAINED_MODEL_NAME) + image_name, image)
