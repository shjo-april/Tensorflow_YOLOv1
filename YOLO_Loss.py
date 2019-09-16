import tensorflow as tf

from Define import *

def L2_Loss(tensor_1, tensor_2):
    return tf.pow(tensor_1 - tensor_2, 2)

def YOLO_Loss(pred_tensor, target_tensor):
    pos_mask = tf.expand_dims(target_tensor[..., 4], axis = -1)
    neg_mask = 1 - pos_mask

    pos_pred_tensor = pos_mask * pred_tensor
    neg_pred_tensor = neg_mask * pred_tensor

    pos_target_tensor = pos_mask * target_tensor
    neg_target_tensor = neg_mask * target_tensor

    xy_loss = L2_Loss(pos_pred_tensor[..., :2], pos_target_tensor[..., :2])
    wh_loss = L2_Loss(tf.sqrt(pos_pred_tensor[..., 2:4] + 1e-10), tf.sqrt(pos_target_tensor[..., 2:4]))
    
    obj_loss = L2_Loss(pos_pred_tensor[..., 4], pos_target_tensor[..., 4])
    noobj_loss = L2_Loss(neg_pred_tensor[..., 4], neg_target_tensor[..., 4])
    
    class_loss = L2_Loss(pos_pred_tensor[..., 5:], pos_target_tensor[..., 5:])
    
    xy_loss = COORD * tf.reduce_sum(xy_loss) / BATCH_SIZE
    wh_loss = COORD * tf.reduce_sum(wh_loss) / BATCH_SIZE
    obj_loss = tf.reduce_sum(obj_loss) / BATCH_SIZE
    noobj_loss = NOOBJ * tf.reduce_sum(noobj_loss) / BATCH_SIZE
    class_loss = tf.reduce_sum(class_loss) / BATCH_SIZE
    
    loss = xy_loss + wh_loss + obj_loss + noobj_loss + class_loss
    return loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss

if __name__ == '__main__':
    pred_tensors = tf.placeholder(tf.float32, [4, 13, 13, 5, 25])
    gt_tensors = tf.placeholder(tf.float32, [4, 13, 13, 5, 25])

    YOLO_Loss(pred_tensors, gt_tensors)
