
import glob
import numpy as np

from Define import *

TRAIN_RATIO = 0.9

train_xml_paths = glob.glob(ROOT_DIR + 'VOC2007/train/xml/*.xml') + glob.glob(ROOT_DIR + 'VOC2012/train/xml/*.xml')
train_xml_count = len(train_xml_paths)

np.random.shuffle(train_xml_paths)
valid_xml_paths = train_xml_paths[int(train_xml_count * TRAIN_RATIO):]
train_xml_paths = train_xml_paths[:int(train_xml_count * TRAIN_RATIO)]

with open('./dataset/train.txt', 'w') as f:
    for xml_path in train_xml_paths:
        f.write(xml_path.replace(ROOT_DIR, '') + '\n')
    f.close()

with open('./dataset/valid.txt', 'w') as f:
    for xml_path in valid_xml_paths:
        f.write(xml_path.replace(ROOT_DIR, '') + '\n')
    f.close()
