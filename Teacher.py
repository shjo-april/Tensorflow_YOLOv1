# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import time
import random
import threading

import numpy as np

from Define import *
from Utils import *
from DataAugmentation import *

from YOLOv1_Utils import *

class Teacher(threading.Thread):
    ready = False
    min_data_size = 0
    max_data_size = 5

    total_indexs = []
    total_data_list = []
    
    batch_data_list = []
    batch_data_length = 0

    debug = False
    name = ''
    retina_utils = None
    
    def __init__(self, npy_path, min_data_size = 1, max_data_size = 5, name = 'Thread', debug = False):
        self.name = name
        self.debug = debug
        
        self.yolov1_utils = YOLOv1_Utils()

        self.min_data_size = min_data_size
        self.max_data_size = max_data_size

        self.total_data_list = np.load(npy_path, allow_pickle = True)
        self.total_indexs = np.arange(len(self.total_data_list)).tolist()

        threading.Thread.__init__(self)
        
    def get_batch_data(self):
        batch_image_data, batch_label_data = self.batch_data_list[0]
        
        del self.batch_data_list[0]
        self.batch_data_length -= 1

        if self.batch_data_length < self.min_data_size:
            self.ready = False
        
        return batch_image_data, batch_label_data
    
    def run(self):
        while True:
            while self.batch_data_length >= self.max_data_size:
                continue
            
            batch_image_data = []
            batch_label_data = []
            batch_indexs = random.sample(self.total_indexs, BATCH_SIZE * 2)

            for data in self.total_data_list[batch_indexs]:
                image_name, gt_bboxes, gt_classes = data
                
                image_path = ROOT_DIR + image_name
                gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
                gt_classes = np.asarray(gt_classes, dtype = np.int32)
                
                image = cv2.imread(image_path)
                image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)

                image_h, image_w, image_c = image.shape
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

                gt_bboxes = gt_bboxes.astype(np.float32)
                gt_classes = np.asarray(gt_classes, dtype = np.int32)
                
                gt_bboxes /= [image_w, image_h, image_w, image_h]
                gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
                
                label_data = self.yolov1_utils.Encode(gt_bboxes, gt_classes)

                batch_image_data.append(image.astype(np.float32))
                batch_label_data.append(label_data)
                if len(batch_image_data) == BATCH_SIZE:
                    break
            
            batch_image_data = np.asarray(batch_image_data, dtype = np.float32) 
            batch_label_data = np.asarray(batch_label_data, dtype = np.float32)
            
            self.batch_data_list.append([batch_image_data, batch_label_data])
            self.batch_data_length += 1

            if self.debug:
                print('[D] stack = [{}/{}]'.format(self.batch_data_length, self.max_data_size))

            if self.batch_data_length >= self.min_data_size:
                self.ready = True
            else:
                self.ready = False