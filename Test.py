import cv2
import glob

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from YOLOv1 import *
from YOLO_Utils import *

# build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
pred_tensor = YOLOv1(input_var, False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/YOLOv1_Test_mAP(72.29%)_DataAugmentation(O).ckpt')

# test
xml_paths = glob.glob("D:/DB/VOC2007/train/xml/" + "*")
for xml_path in xml_paths:

    image_path, gt_bboxes, gt_classes = xml_read(xml_path)
    print(image_path)
    
    image = cv2.imread(image_path)
    h, w, c = image.shape

    tf_image = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)]
    pred_encode_data = sess.run(pred_tensor, feed_dict = {input_var : tf_image})

    pred_bboxes, pred_classes = Decode(pred_encode_data[0], size = (w, h))
    pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes, threshold = 0.5)
    
    for bbox in gt_bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    for bbox, class_index in zip(pred_bboxes, pred_classes):
        xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
        conf = bbox[4]
        class_name = CLASS_NAMES[class_index]

        string = "{} : {:.2f}%".format(class_name, conf * 100)
        cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes)
    print(precision, recall)

    cv2.imshow('show', image)
    cv2.waitKey(0)
    