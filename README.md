# detntorch
Convolutional Neural Networks for Detection tasks

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

### Training

Explain how to train

```
Give an example
```

### Testing

Explain what these test and why

```
Give an example
```


## Networks

### YOLO

### Faster R-CNN

### RetinaNet


## Evaluation Metrics

#### Precision
Precision is the fraction of predicted objects that are relevant to the image:

<img src="https://latex.codecogs.com/gif.latex?\small&space;\text{precision}&space;=\frac{TP}{TP&space;&plus;&space;FP}" title="\small \text{precision} =\frac{TP}{TP + FP}" />

### Recall
Recall is the fraction of the predicted documents that are successfully detected:

<img src="https://latex.codecogs.com/gif.latex?\small&space;\text{recall}&space;=\frac{TP}{TP&space;&plus;&space;FN}" title="\small \text{recall} =\frac{TP}{TP + FN}" />

### F_1 score
The F-measure (F1 score) is the harmonic mean of precision and recall:

<img src="https://latex.codecogs.com/gif.latex?\small&space;F_1&space;=&space;2&space;\frac{\text{precision}&space;\cdot&space;\text{recall}}{\text{precision}&plus;&space;\text{recall}}" title="\small F_1 = 2 \frac{\text{precision} \cdot \text{recall}}{\text{precision}+ \text{recall}}" />

### mAP
Mean average precision for a set of queries is the mean of the average precision scores for each query.



### Accuracy

<img src="https://latex.codecogs.com/gif.latex?\small&space;\text{accuracy}&space;=\frac{TP&space;&plus;&space;TN}{TP&space;&plus;&space;TN&space;&plus;&space;FP&space;&plus;&space;FN}" title="\small \text{accuracy} =\frac{TP + TN}{TP + TN + FP + FN}" />

## Acknowledgements

- This implementation of RetinaNet was based on the [Pytorch implementation of RetinaNet object detection](https://github.com/yhenon/pytorch-retinanet).
- The original [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- Some relevant functions used for the calculation of evaluation metrics are from [Most popular metrics used to evaluate object detection algorithms](https://github.com/rafaelpadilla/Object-Detection-Metrics) and [Faster R-CNN (Python implementation)](https://github.com/rbgirshick/py-faster-rcnn).
