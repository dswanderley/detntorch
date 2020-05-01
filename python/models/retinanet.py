# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:03:14 2020
@author: Diego Wanderley
@python: 3.6
@description: RetinaNet implementation
"""

import torch
import torch.nn as nn

try:
    from models.fpn import PyramidFeatures
except:
    from fpn import PyramidFeatures


class ClassificationModel(nn.Module):
    '''
    Classification network: a subnetwork responsible for performing
        object classification using the backbone’s output.
    '''
    def __init__(self, in_features, num_features=256, num_classes=2, num_anchors=9):
        super(ClassificationModel, self).__init__()
        # Parameters
        self.num_classes = num_classes
        self.in_features = in_features
        self.num_features = num_features
        self.num_anchors = num_anchors
        # Define model blocks
        self.conv1 = nn.Conv2d(in_features,  num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # Classification subnet
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        # out is B x C x W x H, with C = n_classes + n_anchors
        out = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out.shape

        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out = out.contiguous().view(x.shape[0], -1, self.num_classes)

        return out


class RegressionModel(nn.Module):
    '''
    Regression network: a subnetwork responsible for performing
        bounding box regression using the backbone’s output.
    '''
    def __init__(self, in_features, num_features=256, num_anchors=9):
        super(RegressionModel, self).__init__()
        # Parameters
        self.in_features = in_features
        self.num_features = num_features
        self.num_anchors = num_anchors
        # Define model blocks
        self.conv1 = nn.Conv2d(in_features,  num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        # Box subnet
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        # Output conv
        out = self.conv5(out)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, in_channels=1, num_classes=2, num_features=256):
        super(RetinaNet, self).__init__()

        self.fpn = PyramidFeatures(in_channels=in_channels, num_features=num_features)

    def forward(self, x):
        y = self.fpn(x)
        return y



if __name__ == "__main__":
    from torch.autograd import Variable

    net = RetinaNet(in_channels=1)
    preds = net( Variable( torch.randn(2,1,512,512) ) )

    for p in preds:
        print(p.shape)

    print('')
