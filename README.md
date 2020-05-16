# detntorch
Convolutional Neural Networks for Detection tasks

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

1 - Clone this repository:
```
git clone https://github.com/dswanderley/detntorch.git
```

2 - Install the Python pakages:
```
conda install --yes --file /python/requirements.txt
```
This project was developed in Python 3.7 (and 3.6) using a virtual environment from Anaconda 3.

### Training

The network can be trained using the training script of the desired network. Below are some examples:

```
python train_rcnn.py --batch_size 8
python train_retina.py --batch_size 8
python train_yolo.py --model_name yolov3 --batch_size 6
python train_yolo.py --model_name yolov3-tiny --batch_size 8
```

### Detection

The visualization of each network detections can be done by running the correspondent network detect script. Below are some examples:
```
python detect_rcnn.py --weights_path ../weights/NAME_OF_SAVED_WEIGHTS.pth.tar
python detect_retina.py --weights_path ../weights/NAME_OF_SAVED_WEIGHTS.pth.tar
python detect_yolo.py --model_name yolov3 --weights_path ../weights/NAME_OF_SAVED_WEIGHTS.pth.tar
python detect_yolo.py --model_name yolov3-tiny --weights_path ../weights/NAME_OF_SAVED_WEIGHTS.pth.tar
```


## Networks

### YOLO v3

[<b>YOLOv3: An Incremental Improvement</b>](https://arxiv.org/abs/1804.02767)

<i>Joseph Redmon, Ali Farhadi</i>

<b>Abstract</b> - 
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

### Faster R-CNN

[<b>Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks</b>](https://arxiv.org/abs/1506.01497)

<i>Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun</i>

<b>Abstract</b> - 
State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.


### RetinaNet

[<b>Focal Loss for Dense Object Detection</b>](https://arxiv.org/abs/1708.02002)

<i>Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár</i>

<b>Abstract</b> - 
The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors.


## Evaluation Metrics

### Intersection over Union (IoU)

The IoU measures the overlap between two boundaries.
<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?IoU&space;=&space;\frac{\text{area&space;of&space;overlap}}{\text{area&space;of&space;union}}" title="IoU = \frac{\text{area of overlap}}{\text{area of union}}" />
</p> 

### Basic Concepts
- <b>True Positive (TP)</b>:  Number of correctly predictions of a class (detection with IoU ≥ threshold).
- <b>True Negative (TN)</b>:  Number of correctly predictions of the negative class (not applied in these tasks).
- <b>False Positive (FP)</b>: Number of incorrectly predictions of a class (detection with IOU < threshold).
- <b>False Negative (FN)</b>: Number of incorrectly predictions of the negative class (ground truth not detected).


### Precision
Precision is the fraction of predicted objects that are relevant to the image:
<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?\small&space;\text{precision}&space;=\frac{TP}{TP&space;&plus;&space;FP}" title="\small \text{precision} =\frac{TP}{TP + FP}" />
</p> 

### Recall
Recall is the fraction of the predicted documents that are successfully detected:
<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?\small&space;\text{recall}&space;=\frac{TP}{TP&space;&plus;&space;FN}" title="\small \text{recall} =\frac{TP}{TP + FN}" />
</p> 

### F<sub>1</sub> score
The F-measure (F1 score) is the harmonic mean of precision and recall:
<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?\small&space;F_1&space;=&space;2&space;\frac{\text{precision}&space;\cdot&space;\text{recall}}{\text{precision}&plus;&space;\text{recall}}" title="\small F_1 = 2 \frac{\text{precision} \cdot \text{recall}}{\text{precision}+ \text{recall}}" />
</p> 

### Mean Average Precision
Mean average precision (mAP) for a set of predictions is the mean of the average precision (AP) scores for each prediction.
<p align="center"> 
 <img src="https://latex.codecogs.com/gif.latex?mAP&space;=&space;\frac{1}{N}\sum_{i=1}^{N}AP_i" title="mAP = \frac{1}{N}\sum_{i=1}^{N}AP_i" />
</p> 
The AP is computed given the area under the curve (AUC) of the precision x recall curve.


## Acknowledgements

- This implemantation o Faster R-CNN on the [torchvision](https://github.com/pytorch/vision).
- This implementation of RetinaNet was based on the [Pytorch implementation of RetinaNet object detection](https://github.com/yhenon/pytorch-retinanet).
- This implemantation of YOLO v3 was based on the [PyTorch-YOLOv3
](https://github.com/eriklindernoren/PyTorch-YOLOv3).
- The original [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- Some relevant functions used for the calculation of evaluation metrics are from [Most popular metrics used to evaluate object detection algorithms](https://github.com/rafaelpadilla/Object-Detection-Metrics) and [Faster R-CNN (Python implementation)](https://github.com/rbgirshick/py-faster-rcnn).
