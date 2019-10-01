# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_DeepLearning_DB/'

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
IMAGE_HEIGHT = 448
IMAGE_WIDTH = 448
IMAGE_CHANNEL = 3

# ResNet (Normalize), OpenCV BGR -> RGB
R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
MEAN = [R_MEAN, G_MEAN, B_MEAN]

DIVIDE = 2 ** 6

S = IMAGE_WIDTH // DIVIDE
B = 2

# use thread (Dataset)
NUM_THREADS = 10

SAMPLES = 5

# loss parameters
COORD = 5.0
NOOBJ = 0.5

WEIGHT_DECAY = 0.0001

# train
BATCH_SIZE = 16
INIT_LEARNING_RATE = 1e-3

MAX_EPOCH = 200

LOG_ITERATION = 50
SAMPLE_ITERATION = 1000
SAVE_ITERATION = 5000
