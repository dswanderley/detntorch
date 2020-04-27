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
