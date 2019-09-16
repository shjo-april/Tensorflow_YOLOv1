import cv2
import glob

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from YOLOv2 import *
from YOLOv2_Utils import *

yolov2_utils = YOLOv2_Utils()

# build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
prediction_op = YOLOv2(input_var, False, yolov2_utils.anchors)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/YOLOv2_135000.ckpt')

# test
xml_paths = glob.glob("D:/_DeepLearning_DB/VOC2012/test/xml/" + "*")

for xml_path in xml_paths:
    image_path, gt_bboxes, gt_classes = xml_read(xml_path)
    print(image_path)
    
    image = cv2.imread(image_path)
    h, w, c = image.shape

    tf_image = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_LINEAR).astype(np.float32)]
    pred_encode_data = sess.run(prediction_op, feed_dict = {input_var : tf_image})

    pred_bboxes, pred_classes = yolov2_utils.Decode(pred_encode_data[0], size = (w, h), detect_threshold = 0.40, nms = True)

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

    cv2.imshow('show', image)
    cv2.waitKey(0)
    