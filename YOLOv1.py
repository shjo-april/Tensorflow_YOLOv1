
import numpy as np
import tensorflow as tf
import resnet_v1.resnet_v1 as resnet_v1

from Define import *

kernel_initializer = tf.contrib.layers.xavier_initializer()

def conv_bn_relu(x, filters, kernel_size, strides, padding, is_training, scope, bn = True, activation = True, use_bias = True, upscaling = False):
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'upconv2d')
        
        if bn:
            x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn')

        if activation:
            x = tf.nn.relu(x, name = 'relu')
    return x

def Decode_Layer(pred_tensors):
    pred_tensors = tf.nn.sigmoid(pred_tensors)
    _, h, w, b, c = pred_tensors.shape.as_list()

    anchor_xy = np.zeros((h, w, b, 2), dtype = np.float32)
    for y in range(h):
        for x in range(w):
            for i in range(b):
                anchor_xy[y, x, i, :] = [x, y]
    
    pred_cxcy = (anchor_xy + pred_tensors[..., :2]) / [w, h] * [IMAGE_WIDTH, IMAGE_HEIGHT]
    pred_wh = pred_tensors[..., 2:4] * [IMAGE_WIDTH, IMAGE_HEIGHT]

    pred_min_xy = pred_cxcy - pred_wh / 2
    pred_max_xy = pred_cxcy + pred_wh / 2
    pred_conf = pred_tensors[..., 4][..., tf.newaxis]
    pred_classes = pred_tensors[..., 5:]

    pred_tensors = tf.concat([pred_min_xy, pred_max_xy, pred_conf, pred_classes], axis = -1)
    return pred_tensors

def YOLOv1_ResNetv1_50(input_var, is_training, reuse = False):

    # resnet
    x = input_var[..., ::-1] - MEAN
    with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(x, is_training = is_training, reuse = reuse)
    
    # YOLOv1
    with tf.variable_scope('YOLOv1', reuse = reuse):
        x = end_points['resnet_v1_50/block4']

        x = conv_bn_relu(x, 512, [3, 3], 1, 'same', is_training, 'conv1')
        x = conv_bn_relu(x, B * (5 + CLASSES), [1, 1], 1, 'valid', is_training, 'conv2', bn = False, activation = False)
        
        pred_tensors = tf.reshape(x, (-1, S, S, B, 5 + CLASSES), name = 'pred_tensors')
        
    pred_tensors = Decode_Layer(pred_tensors)
    return pred_tensors

YOLOv1 = YOLOv1_ResNetv1_50

if __name__ == '__main__':
    from YOLOv1_Utils import *

    input_var = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

    pred_tensors = YOLOv1(input_var, False)
    print(pred_tensors)
