# Tensorflow_YOLOv1+

## Why YOLOv1+?
1. Fixed Loss to be simpler.
2. Applied the YOLOv2 concept.

## Results
### Pascal VOC 2007 Test
- (paper) YOLOv1-VGG16 : 66.4%
- (self) YOLOv1-VGG16 : 72.79%

![result](./results/Precision_Recall_Curve_VGG16.jpg)

- (self) YOLOv1-InceptionResNetv2 : 78.47%

![result](./results/Precision_Recall_Curve_InceptionResNetv2.jpg)

## Samples
### YOLOv1-InceptionResNetv2
![result](./results/InceptionResNetv2_Test_Samples/000014.jpg)
![result](./results/InceptionResNetv2_Test_Samples/000015.jpg)
![result](./results/InceptionResNetv2_Test_Samples/000029.jpg)

### YOLOv1-VGG16
![result](./results/VGG16_Test_Samples/000014.jpg)
![result](./results/VGG16_Test_Samples/000015.jpg)
![result](./results/VGG16_Test_Samples/000029.jpg)

## Requirements
- Tensorflow 1.13.1
- OpenCV 4.0.0
- Numpy 1.16.4

## Pretrained models 
- https://github.com/tensorflow/models/tree/master/research/slim