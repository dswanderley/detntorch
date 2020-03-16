# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:54:15 2019
@author: Diego Wanderley
@python: 3.6
@description: Methods based on R-CNN for Object Detection.
"""

import torch
import torch.nn as nn

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(nn.Module):
    '''
    Faster R-CNN Class
    '''
    def __init__(self, num_channels=3, num_classes=91, pretrained=False, min_size=512):
        super(FasterRCNN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.min_size = min_size

        # Input conv is applied to convert the input to 3 ch depth
        self.inconv = None
        if num_channels != 3:
            self.inconv = nn.Sequential()
            self.inconv.add_module("conv_0", nn.Conv2d(num_channels, 3, 1, stride=1, padding=0))
            self.inconv.add_module("bnorm_0", nn.BatchNorm2d(3))
            self.inconv.add_module("relu_0", nn.ReLU(inplace=True))

        # Pre-trained model needs to be an identical network
        if pretrained:
            self.body = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=91, min_size=min_size)
            # Reset output
            if num_classes != 91:
                in_features = self.body.roi_heads.box_predictor.cls_score.in_features
                self.body.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        else:
            self.body = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=num_classes, min_size=min_size)


    def forward(self, x, tgts=None):
        # input layer to convert to 3 channels
        if self.inconv != None:
            x = self.inconv(x)
        # Verify if is traning (this situation requires targets)
        if self.body.training:
            x = list(im for im in x) # convert to list (as required)
            x_out = self.body(x,tgts)
        else:
            x_out = self.body(x)

        return x_out


if __name__ == "__main__":

    import os
    import numpy as np
    from skimage import io
    from skimage.color import rgb2gray

    #im_path = os.path.join('../datasets/cars', 'cars.jpg')
    #image = io.imread(im_path) / 255.
    #torch_img = torch.from_numpy(image)
    #torch_img = torch_img.permute(2,0,1)

    #grayscale = rgb2gray(image)
    #torch_img = torch.from_numpy(grayscale)
    #torch_img.unsqueeze_(0)

    #torch_img.unsqueeze_(0)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together
    # https://github.com/pytorch/vision/blob/master/references/detection/train.py

    # Images
    images = torch.randn(1, 1, 512, 512)
    images = [images, images]
    #images = [torch_img.float(), torch_img.float()]
    # Targets
    bbox = torch.FloatTensor([[120, 130, 300, 350], [200, 200, 250, 250]]) # [y1, x1, y2, x2] format
    lbls = torch.LongTensor([1, 2]) # 0 represents background

    # targets per image
    tgts = {
        'boxes':  bbox.to(device),
        'labels': lbls.to(device)
    }
    # targets to list (by batch)
    targets = [tgts, tgts]

    # Model
    model = FasterRCNN(num_channels=1, num_classes=2, pretrained=True).to(device)
    #model = FasterRCNN(num_channels=3, pretrained=True).to(device)
    model.eval()
    #model.train()

    # output
    images = torch.FloatTensor(images)
    loss_dict = model(images, targets)
    #loss_dict = model(images#.to(device))

    for i in range(len(loss_dict)):
        img = image
        bboxes = loss_dict[i]['boxes']
        color = np.array([0, 255, 0])/255.

        for bb in bboxes:
            bounding_box = bb.cpu().detach().numpy().round().astype(np.int)
            img[bounding_box[1], bounding_box[0]:bounding_box[2]] = color
            img[bounding_box[1]:bounding_box[3], bounding_box[0]] = color

            img[bounding_box[3], bounding_box[0]:bounding_box[2]] = color
            img[bounding_box[1]:bounding_box[3], bounding_box[2]] = color

        io.imsave("./debug.png", img)


    print('')
