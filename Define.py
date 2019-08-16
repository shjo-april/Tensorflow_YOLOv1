
PRETRAINED_MODEL_NAME = 'VGG16'
# PRETRAINED_MODEL_NAME = 'InceptionResNetv2'

if PRETRAINED_MODEL_NAME == 'VGG16':
    IMAGE_HEIGHT = 448
    IMAGE_WIDTH = 448
    IMAGE_CHANNEL = 3

    S = 7
    B = 2
    C = 20
    
    BATCH_SIZE = 16
    VGG_MEAN = [103.94, 116.78, 123.68]

elif PRETRAINED_MODEL_NAME == 'InceptionResNetv2':
    IMAGE_HEIGHT = 416
    IMAGE_WIDTH = 416
    IMAGE_CHANNEL = 3

    S = 11
    B = 2
    C = 20

    BATCH_SIZE = 8

COORD = 5
NOOBJ = 0.5

WEIGHT_DECAY = 0.0005

INIT_LR = 1e-3
MAX_EPOCHS = 135

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}

