import cv2
import random

import numpy as np

# prob = 50%
def random_flip(image, gt_bboxes, condition = [False, True]):
    if random.choice(condition):
        h, w, c = image.shape

        image = np.fliplr(image).copy()

        xmin = w - gt_bboxes[:, 2]
        xmax = w - gt_bboxes[:, 0]

        gt_bboxes[:, 0] = xmin
        gt_bboxes[:, 2] = xmax

    return image, gt_bboxes

# prob = 50%
def random_scale(image, gt_bboxes, condition = [False, True]):
    if random.choice(condition):
        h, w, c = image.shape

        w_scale = random.uniform(0.8, 1.2)
        h_scale = random.uniform(0.8, 1.2)

        image = cv2.resize(image, (int(w * w_scale), int(h * h_scale)))
        gt_bboxes = gt_bboxes * [w_scale, h_scale, w_scale, h_scale]

    return image, gt_bboxes

# prob = 20%
def random_blur(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        image = cv2.blur(image, (5, 5))
    return image

# prob = 20%
def random_brightness(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        adjust = random.uniform(0.5, 1.5)
        image = np.clip(image.astype(np.float32) * adjust, 0, 255).astype(np.uint8)

    return image

# prob = 20%
def random_hue(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        h = h.astype(np.float32)
        adjust = random.uniform(0.5, 1.5)

        h = np.clip(h * adjust, 0, 255).astype(np.uint8)

        hsv_image = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

# prob = 20%
def random_saturation(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        s = s.astype(np.float32)
        adjust = random.uniform(0.5, 1.5)

        s = np.clip(s * adjust, 0, 255).astype(np.uint8)

        hsv_image = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image

# prob = 20%
def random_gray(image, condition = [False, False, False, False, True]):
    if random.choice(condition):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.merge([image, image, image])
    return image

# prob = 25%
def random_shift(image, gt_bboxes, gt_classes, condition = [False, False, False, True]):
    if random.choice(condition):
        h, w, c = image.shape
        shift_image = np.zeros((h, w, c), dtype = np.uint8)

        shift_x = int(random.uniform(-w * 0.2, +w * 0.2))
        shift_y = int(random.uniform(-h * 0.2, +h * 0.2))

        if shift_x >= 0 and shift_y >= 0:
            shift_image[shift_y:, shift_x:, :] = image[:h - shift_y, :w - shift_x, :]
        elif shift_x >= 0 and shift_y < 0:
            shift_image[:h + shift_y, shift_x:, :] = image[-shift_y:, :w - shift_x, :]
        elif shift_x < 0 and shift_y >= 0:
            shift_image[shift_y:, :w + shift_x, :] = image[:h - shift_y, -shift_x:, :]
        elif shift_x < 0 and shift_y < 0:
            shift_image[:h + shift_y, :w + shift_x, :] = image[-shift_y: , -shift_x:, :]
        else:
            assert False, '[!] random shift image_shape = ({}, {}, {}), shift = ({}, {})'.format(h, w, c, shift_x, shift_y)

        centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        centers += [shift_x, shift_y]

        bboxes_in = []
        class_in = []

        for center, gt_bbox, gt_class in zip(centers, gt_bboxes, gt_classes):
            if center[0] > 0 and center[0] < w and center[1] > 0 and center[1] < h:
                bboxes_in.append(gt_bbox)
                class_in.append(gt_class)

        if len(bboxes_in) == 0:
            return image, gt_bboxes, gt_classes

        image = shift_image
        gt_bboxes = np.asarray(bboxes_in, dtype = np.int32) + [shift_x, shift_y, shift_x, shift_y]
        gt_classes = class_in

        #gt_bboxes = gt_bboxes.clip(min = 0, max = w)

    return image, gt_bboxes, gt_classes

# prob = 25%
def random_crop(image, gt_bboxes, gt_classes, condition = [False, False, False, True]):
    if random.choice(condition):
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(gt_bboxes[:, 0:2], axis=0), np.max(gt_bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

        gt_bboxes[:, [0, 2]] = gt_bboxes[:, [0, 2]] - crop_xmin
        gt_bboxes[:, [1, 3]] = gt_bboxes[:, [1, 3]] - crop_ymin

    return image, gt_bboxes, gt_classes

# prob = 25%
def random_translate(image, gt_bboxes, gt_classes, condition = [False, False, False, True]):
    if random.choice(condition):
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(gt_bboxes[:, 0:2], axis=0), np.max(gt_bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        gt_bboxes[:, [0, 2]] = gt_bboxes[:, [0, 2]] + tx
        gt_bboxes[:, [1, 3]] = gt_bboxes[:, [1, 3]] + ty

    return image, gt_bboxes, gt_classes

if __name__ == '__main__':
    import glob
    from Utils import *

    xml_paths = []
    xml_paths += glob.glob("D:/DB/VOC2007/train/xml/" + "*")
    xml_paths += glob.glob("D:/DB/VOC2012/xml/" + "*")

    for xml_path in xml_paths:
        image_path, gt_bboxes, gt_classes = xml_read(xml_path, normalize = False)
        print(image_path)

        image = cv2.imread(image_path)

        gt_bboxes = np.asarray(gt_bboxes).astype(np.int32)
        gt_classes = np.asarray(gt_classes).astype(np.int32)

        print(gt_bboxes)
        print(gt_classes)

        for bbox, class_index in zip(gt_bboxes, gt_classes):
            xmin, ymin, xmax, ymax = bbox
            if (xmax - xmin) * (ymax - ymin) > 0.0:
                pass
            else:
                print('original')
                print(bbox)
                print(xml_path)
                input()

        # image, gt_bboxes = random_flip(image, gt_bboxes, [True])
        # image, gt_bboxes = random_scale(image, gt_bboxes, [True])
        # image = random_blur(image, [True])
        # image = random_brightness(image, [True])
        # image = random_hue(image, [True])
        # image = random_saturation(image, [True])
        # image = random_gray(image, [True])
        # image, gt_bboxes, gt_classes = random_shift(image, gt_bboxes, gt_classes, [True])
        image, gt_bboxes, gt_classes = random_crop(image, gt_bboxes, gt_classes, [True])
        image, gt_bboxes, gt_classes = random_translate(image, gt_bboxes, gt_classes, [True])

        h, w, c = image.shape

        for bbox, _class in zip(gt_bboxes.astype(np.int32), gt_classes):
            xmin, ymin, xmax, ymax = bbox

            if (xmax - xmin) * (ymax - ymin) > 0.0:
                pass
            else:
                print('crop')
                print(bbox)
                print(xml_path)
                input()

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('show', image)
        cv2.waitKey(0)
        