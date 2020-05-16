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

### YOLO

### Faster R-CNN

### RetinaNet


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
