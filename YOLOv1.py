
import VGG16 as vgg
import tensorflow as tf

from Define import *

init_fn = tf.contrib.layers.xavier_initializer()

def YOLOv1_VGG(x, is_training):
    x -= VGG_MEAN
    feature_maps_list = []

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
    
    # YOLO
    x = tf.reshape(x, (-1, S, S, B, 5 + C))
    x = tf.nn.sigmoid(x, name = 'yolo_outputs')
    return x

YOLOv1 = YOLOv1_VGG

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 448, 448, 3], name = 'image')
    
    pred_tensor = YOLOv1(input_var, False)
    print(pred_tensor)

    assert True
