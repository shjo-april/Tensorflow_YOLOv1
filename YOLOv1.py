
import vgg_16.VGG16 as vgg
import inception_resnet_v2.inception as inception

import tensorflow as tf
from Define import *

init_fn = tf.contrib.layers.xavier_initializer()

def YOLOv1_InceptionResNetv2(x, is_training):
    x = x / 127.5 - 1

    with tf.contrib.slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        x = inception.inception_resnet_v2(x, 1000, is_training = is_training)

    x = tf.layers.conv2d(x, kernel_size = [1, 1], filters = B * (5 + C), strides = 1, kernel_initializer = init_fn, padding = 'same')
    x = tf.layers.batch_normalization(x, training = is_training)
    
    x = tf.reshape(x, (-1, S, S, B, 5 + C))
    x = tf.nn.sigmoid(x, name = 'yolo_outputs')
    return x

def YOLOv1_VGG(x, is_training):
    x -= VGG_MEAN
    
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        x = vgg.vgg_16(x, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5)
    
    for i in range(2):
        x = tf.layers.conv2d(x, kernel_size = [1, 1], filters = 512, strides = 1, kernel_initializer = init_fn, padding='same')
        x = tf.layers.batch_normalization(x, training = is_training)
        x = tf.nn.relu(x)
        
        x = tf.layers.conv2d(x, kernel_size = [3, 3], filters = 1024, strides = 1, kernel_initializer = init_fn, padding='same')
        x = tf.layers.batch_normalization(x, training = is_training)
        x = tf.nn.relu(x)

    x = tf.layers.max_pooling2d(x, pool_size = [2, 2], strides = 2)
    
    x = tf.layers.conv2d(x, kernel_size = [1, 1], filters = B * (5 + C), strides = 1, kernel_initializer = init_fn, padding = 'same')
    x = tf.layers.batch_normalization(x, training = is_training)
    
    x = tf.reshape(x, (-1, S, S, B, 5 + C))
    x = tf.nn.sigmoid(x, name = 'yolo_outputs')
    return x

if PRETRAINED_MODEL_NAME == 'VGG16':
    YOLOv1 = YOLOv1_VGG
elif PRETRAINED_MODEL_NAME == 'InceptionResNetv2':
    YOLOv1 = YOLOv1_InceptionResNetv2

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name = 'images')
    pred_tensors = YOLOv1(input_var, False)
    print(pred_tensors)

