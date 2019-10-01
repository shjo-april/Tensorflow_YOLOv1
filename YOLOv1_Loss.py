import tensorflow as tf

from Define import *

def L2_Loss(tensor_1, tensor_2):
    return tf.pow(tensor_1 - tensor_2, 2)

'''
pt = {
    p    , if y = 1
    1 − p, otherwise
}
FL(pt) = −(1 − pt)γ * log(pt)
'''
def Focal_Loss(pred_classes, gt_classes, alpha = 0.25, gamma = 2):
    with tf.variable_scope('Focal'):
        # focal_loss = [BATCH_SIZE, S, S, B, CLASSES]
        pt = gt_classes * pred_classes + (1 - gt_classes) * (1 - pred_classes) 
        focal_loss = -alpha * tf.pow(1. - pt, gamma) * tf.log(pt + 1e-10)

        # focal_loss = [BATCH_SIZE]
        focal_loss = tf.reduce_sum(tf.abs(focal_loss), axis = -1)

    return focal_loss

'''
GIoU = IoU - (C - (A U B))/C
Loss = 1 - GIoU
'''
def GIoU(bboxes_1, bboxes_2):
    with tf.variable_scope('GIoU'):
        # 1. calulate intersection over union
        area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])
        area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])
        
        intersection_wh = tf.minimum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.maximum(bboxes_1[..., :2], bboxes_2[..., :2])
        intersection_wh = tf.maximum(intersection_wh, 0)
        
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        union = (area_1 + area_2) - intersection
        
        ious = intersection / tf.maximum(union, 1e-10)

        # 2. (C - (A U B))/C
        C_wh = tf.maximum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.minimum(bboxes_1[..., :2], bboxes_2[..., :2])
        C_wh = tf.maximum(C_wh, 0.0)
        C = C_wh[..., 0] * C_wh[..., 1]
        
        giou = ious - (C - union) / tf.maximum(C, 1e-10)
    return giou

# Original YOLOv1 Loss
# def YOLOv1_Loss(pred_tensor, gt_tensor):
#     pos_mask = tf.expand_dims(gt_tensor[..., 4], axis = -1)
#     neg_mask = 1 - pos_mask

#     pos_pred_tensor = pos_mask * pred_tensor
#     neg_pred_tensor = neg_mask * pred_tensor

#     pos_gt_tensor = pos_mask * gt_tensor
#     neg_gt_tensor = neg_mask * gt_tensor

#     xy_loss = L2_Loss(pos_pred_tensor[..., :2], pos_gt_tensor[..., :2])
#     wh_loss = L2_Loss(tf.sqrt(pos_pred_tensor[..., 2:4] + 1e-10), tf.sqrt(pos_gt_tensor[..., 2:4]))
    
#     obj_loss = L2_Loss(pos_pred_tensor[..., 4], pos_gt_tensor[..., 4])
#     noobj_loss = L2_Loss(neg_pred_tensor[..., 4], neg_gt_tensor[..., 4])
    
#     class_loss = L2_Loss(pos_pred_tensor[..., 5:], pos_gt_tensor[..., 5:])

#     xy_loss = COORD * tf.reduce_sum(xy_loss) / BATCH_SIZE
#     wh_loss = COORD * tf.reduce_sum(wh_loss) / BATCH_SIZE
#     obj_loss = tf.reduce_sum(obj_loss) / BATCH_SIZE
#     noobj_loss = NOOBJ * tf.reduce_sum(noobj_loss) / BATCH_SIZE
#     class_loss = tf.reduce_sum(class_loss) / BATCH_SIZE
    
#     loss = xy_loss + wh_loss + obj_loss + noobj_loss + class_loss
#     return loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss

def YOLOv1_Loss(pred_tensor, gt_tensor):
    # get mask
    pos_mask = gt_tensor[..., 4]
    neg_mask = 1 - pos_mask

    # bboxes - xmin, ymin, xmax, ymax
    giou_loss_op = 1 - GIoU(pred_tensor[..., :4], gt_tensor[..., :4])
    
    # confidence - focal loss
    conf_loss_op = Focal_Loss(pred_tensor[..., 4], gt_tensor[..., 4])

    # classification - focal loss
    class_loss_op = Focal_Loss(pred_tensor[..., 5:], gt_tensor[..., 5:])
    
    # calculate total loss
    giou_loss_op = tf.reduce_sum(pos_mask * giou_loss_op) / BATCH_SIZE
    conf_loss_op = tf.reduce_sum(conf_loss_op) / BATCH_SIZE
    class_loss_op = tf.reduce_sum(pos_mask * class_loss_op) / BATCH_SIZE
    loss_op = giou_loss_op + conf_loss_op + class_loss_op

    return loss_op, giou_loss_op, conf_loss_op, class_loss_op

if __name__ == '__main__':
    pred_tensors = tf.placeholder(tf.float32, [BATCH_SIZE, S, S, B, 5 + CLASSES])
    gt_tensors = tf.placeholder(tf.float32, [BATCH_SIZE, S, S, B, 5 + CLASSES])

    YOLOv1_Loss(pred_tensors, gt_tensors)
