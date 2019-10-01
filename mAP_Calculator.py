
import math
import numpy as np

from Utils import *

class AP_Calculator:
    def __init__(self):
        self.fn = 0
        self.correct = []
        self.confidence = []

class mAP_Calculator:
    def __init__(self, classes, ap_threshold = 0.5):
        
        self.classes = classes
        self.ap_threshold = ap_threshold

        self.reset()

    def reset(self):
        self.ap_calc_list = []
        for i in range(self.classes):
            self.ap_calc_list.append(AP_Calculator())

    def update(self, total_pred_bboxes, total_pred_classes, total_gt_bboxes, total_gt_classes):
        
        # # calculate ious "shape = (pred_bboxes, gt_bboxes)"
        # gt_ious = compute_bboxes_IoU(gt_bboxes, pred_bboxes)

        # max_gt_ious = np.max(gt_ious, axis = -1, keepdims = True)
        # gt_mask = np.equal(gt_ious, max_gt_ious)

        # pred_ious = np.where(gt_mask, gt_ious, 0.0) >= self.ap_threshold
        # pred_mask = pred_ious.T.shape
        
        # max_gt_mask = np.max(gt_mask, axis = -1)
        
        for class_index, ap_calc in enumerate(self.ap_calc_list):
            # get masks
            gt_masks = total_gt_classes == class_index
            pred_masks = total_pred_classes == class_index

            # extract bboxes (with masks)
            gt_bboxes = total_gt_bboxes[gt_masks]
            pred_bboxes = total_pred_bboxes[pred_masks]

            gt_count = len(gt_bboxes)
            pred_count = len(pred_bboxes)

            # exception case 1. 
            if gt_count == 0:
                # add false positives
                if pred_count != 0:
                    ap_calc.correct += list(np.zeros(pred_count, dtype = np.bool))
                    ap_calc.confidence += list(pred_bboxes[:, 4])

                continue

            # exception case 2.
            if pred_count == 0:
                ap_calc.fn += gt_count
                continue
            
            # calculate intersection over union.
            gt_ious = compute_bboxes_IoU(gt_bboxes, pred_bboxes)

            # calculate masks
            gt_ap_masks = gt_ious >= self.ap_threshold
            gt_iou_masks = np.equal(gt_ious, np.max(gt_ious, axis = -1, keepdims = True))

            gt_masks = np.logical_and(gt_ap_masks, gt_iou_masks)

            # update fn
            ap_calc.fn += (gt_count - sum(np.max(gt_masks, axis = 1)))

            # update tp
            ap_calc.correct += list(np.max(gt_masks.T, axis = 1))
            ap_calc.confidence += list(pred_bboxes[:, 4])

            # print('fn : {}'.format(gt_count -  sum(np.max(gt_masks, axis = 1))))
            # print('correct : {}'.format(list(np.max(gt_masks.T, axis = 1))))
            # print('confidence : {}'.format(list(pred_bboxes[:, 4])))
            # input()

    def compute_precision_recall(self, class_index):
        # get correct, confidence, all_ground_truths
        correct_list = self.ap_calc_list[class_index].correct
        confidence_list = self.ap_calc_list[class_index].confidence
        all_ground_truths = np.sum(correct_list) + self.ap_calc_list[class_index].fn

        # list -> numpy
        confidence_list = np.asarray(confidence_list, dtype = np.float32)
        correct_list = np.asarray(correct_list, dtype = np.bool)
        
        # Ascending (confidence)
        sort_indexs = confidence_list.argsort()[::-1]
        confidence_list = confidence_list[sort_indexs]
        correct_list = correct_list[sort_indexs]
        
        correct_detections = 0
        all_detections = 0
        
        # calculate precision/recall
        precision_list = []
        recall_list = []

        for confidence, correct in zip(confidence_list, correct_list):
            all_detections += 1
            if correct: correct_detections += 1    
            
            precision = correct_detections / all_detections
            recall = correct_detections / all_ground_truths
            
            precision_list.append(precision)
            recall_list.append(recall)

            # maximum correct detections
            if recall == 1.0:
                break

        precision_list = np.asarray(precision_list, dtype = np.float32)
        recall_list = np.asarray(recall_list, dtype = np.float32)
        
        # calculating the interpolation performed in 11 points (0.0 -> 1.0, +0.01)
        precision_interp_list = []
        interp_list = np.arange(0, 10 + 1) / 10

        for interp in interp_list:
            try:
                precision_interp = max(precision_list[recall_list >= interp])
            except:
                precision_interp = 0.0
            
            precision_interp_list.append(precision_interp)

        ap = np.mean(precision_interp_list) * 100
        return ap, precision_list, recall_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mAP_calc = mAP_Calculator(classes = 4)
    
    # case 1
    pred_bboxes = np.array([[0.880688, 0.44609185, 0.95696718, 0.6476958, 0.95],
                     [0.84020283, 0.45787981, 0.99351478, 0.64294884, 0.75],
                     [0.78723741, 0.61799151, 0.9083041, 0.75623035, 0.4],
                     [0.22078986, 0.30151826, 0.36679274, 0.40551913, 0.3],
                     [0.0041579, 0.48359361, 0.06867643, 0.60145104, 1.0],
                     [0.4731401, 0.33888632, 0.75164948, 0.80546954, 1.0],
                     [0.75489414, 0.75228018, 0.87922037, 0.88110524, 0.75],
                     [0.21953127, 0.77934921, 0.34853417, 0.90626764, 0.5],
                     [0.81, 0.11, 0.91, 0.21, 0.5]])
    pred_classes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3], dtype = np.int32)

    gt_bboxes = np.array([[0.86132812, 0.48242188, 0.97460938, 0.6171875],
                    [0.18554688, 0.234375, 0.36132812, 0.41601562],
                    [0., 0.47265625, 0.0703125, 0.62109375],
                    [0.47070312, 0.3125, 0.77929688, 0.78125],
                    [0.8, 0.1, 0.9, 0.2]])
    gt_classes = np.array([0, 0, 1, 2, 2], dtype = np.int32)

    pred_bboxes[:, :4] = pred_bboxes[:, :4] * [100, 100, 100, 100]
    gt_bboxes = gt_bboxes * [100, 100, 100, 100]

    mAP_calc.update(pred_bboxes, pred_classes, gt_bboxes, gt_classes)

    ap, precisions, recalls = mAP_calc.compute_precision_recall(0)
    print(ap, precisions, recalls)
    
    # matplotlib (precision&recall curve + interpolation)
    plt.clf()
    plt.plot(recalls, precisions, 'green')
    plt.fill_between(recalls, precisions, step = 'post', alpha = 0.2, color = 'green')
    # plt.plot(interp_list, precision_interp_list, 'ro')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('# Precision-recall curve ({} - {:.2f}%)'.format('0', ap))
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
    # plt.savefig('./results/{}.jpg'.format(class_name))