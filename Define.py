
IMAGE_HEIGHT = 448
IMAGE_WIDTH = 448
IMAGE_CHANNEL = 3

# output feature shape : S x S x (B * 5 + C)
S = 7
B = 2
C = 20

COORD = 5
NOOBJ = 0.5

VGG_MEAN = [103.94, 116.78, 123.68]

BATCH_SIZE = 16
MOMENTUM = 0.9

DROP_OUT = 0.5
WEIGHT_DECAY = 0.0005

INIT_LR = 1e-3
MAX_EPOCHS = 135
# 1e-3 -> 1e-2 (75 epochs) -> 1e-3 (30 epochs) -> 1e-4 (30 epochs)

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}

assert C == len(CLASS_NAMES), '[!] class length != C'
