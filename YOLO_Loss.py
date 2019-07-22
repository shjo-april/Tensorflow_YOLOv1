import tensorflow as tf

from Define import *

def L2_Loss(tensor_1, tensor_2):
    return tf.pow(tensor_1 - tensor_2, 2)

def YOLO_Loss(pred_tensor, target_tensor):
    pos_mask = target_tensor[..., 4]
    neg_mask = 1 - pos_mask
    
    pos_pred_tensor = pos_mask[..., tf.newaxis] * pred_tensor
    pos_target_tensor = pos_mask[..., tf.newaxis] * target_tensor

    xy_loss = COORD * L2_Loss(pos_pred_tensor[..., :2], pos_target_tensor[..., :2])
    wh_loss = COORD * L2_Loss(tf.sqrt(pos_pred_tensor[..., 2:4] + 1e-10), tf.sqrt(pos_target_tensor[..., 2:4]))
    
    conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_tensor[..., 4], labels = target_tensor[..., 4])
    obj_loss = pos_mask * conf_loss
    noobj_loss = NOOBJ * neg_mask * conf_loss
    
    class_loss = pos_mask * tf.nn.softmax_cross_entropy_with_logits(logits = pred_tensor[..., 5:], labels = target_tensor[..., 5:])
    
    xy_loss = tf.reduce_sum(xy_loss) / BATCH_SIZE
    wh_loss = tf.reduce_sum(wh_loss) / BATCH_SIZE
    obj_loss = tf.reduce_sum(obj_loss) / BATCH_SIZE
    noobj_loss = tf.reduce_sum(noobj_loss) / BATCH_SIZE
    class_loss = tf.reduce_sum(class_loss) / BATCH_SIZE

    total_loss = xy_loss + wh_loss + obj_loss + noobj_loss + class_loss

    return total_loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [4, 13, 13, 5, 25])
    y = tf.placeholder(tf.float32, [4, 13, 13, 5, 25])

    YOLO_Loss(x, y)
