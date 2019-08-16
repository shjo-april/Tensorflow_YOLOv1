import sys
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

# build
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
pred_tensor = YOLOv1(input_var, False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, model_path)

# test
precision_list = np.zeros((101), dtype = np.float32)
recall_list = np.zeros((101), dtype = np.float32)

xml_paths = glob.glob("D:/DB/VOC2007/test/xml/*")
xml_count = len(xml_paths)

for index, xml_path in enumerate(xml_paths):

    image_path, gt_bboxes, gt_classes = xml_read(xml_path)
    
    image = cv2.imread(image_path)
    h, w, c = image.shape
    
    tf_image = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)]
    pred_encode_data = sess.run(pred_tensor, feed_dict = {input_var : tf_image})

    for threshold in range(0, 101):
        pred_bboxes, pred_classes = Decode(pred_encode_data[0], size = (w, h), detect_threshold = threshold / 100)
        pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes, threshold = 0.5)

        precision, recall = Precision_Recall(gt_bboxes, gt_classes, pred_bboxes, pred_classes)
        
        precision_list[threshold] += precision
        recall_list[threshold] += recall

    sys.stdout.write('\r[{}/{}]'.format(index, xml_count))
    sys.stdout.flush()

thresholds = [threshold / 100 for threshold in range(0, 101)]
precision_list /= xml_count
recall_list /= xml_count
mAP_list = (precision_list + recall_list) / 2

plt.plot(thresholds, precision_list, color = 'red', label = 'Precision')
plt.plot(thresholds, recall_list, color = 'green', label = 'Recall')
plt.plot(thresholds, mAP_list, color = 'orange', label = 'mAP')

plt.xlim([0.0, 1.0])
plt.xlim([0.0, 1.05])

plt.title('Precision_Recall_Curve')
plt.legend(loc = "lower right")
# plt.show()
plt.savefig('./results/Precision_Recall_Curve_{}.jpg'.format(PRETRAINED_MODEL_NAME))

best_threshold = np.argmax(mAP_list)

print()
print('# {}, best_threshold = {:.2f}'.format(model_path, best_threshold / 100))
print('Precision : {:.2f}%'.format(precision_list[best_threshold] * 100))
print('Recall : {:.2f}%'.format(recall_list[best_threshold] * 100))
print('mAP : {:.2f}%'.format(mAP_list[best_threshold] * 100))

'''
# ./model/YOLOv1_InceptionResNetv2.ckpt, best_threshold = 0.34
Precision : 84.67%
Recall : 72.27%
mAP : 78.47%

# ./model/YOLOv1_VGG16.ckpt, best_threshold = 0.31
Precision : 76.49%
Recall : 69.10%
mAP : 72.79%
'''

