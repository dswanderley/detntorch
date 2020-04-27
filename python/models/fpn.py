# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:55:10 2020
@author: Diego Wanderley
@python: 3.6
@description: Feature Pyramid Network class and auxiliary classes.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name, pretrained=True):
    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones

    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output


class ResNetBackbone(nn.Module):
    '''
    ResNet backbone for Feature Pyramid Net.
    '''
    def __init__(self, num_channels=3, backbone='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        # set parameters
        self.num_channels = num_channels
        # load backbone
        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone, pretrained=pretrained)
        # input conv
        self.conv1 = self.backbone.conv1
        if num_channels != 3:
            self.conv1.in_channels = num_channels
        self.conv1.load_state_dict(self.backbone.conv1.state_dict())
        self.bn1 = self.backbone.bn1
        self.bn1.load_state_dict(self.backbone.bn1.state_dict())
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        # Sequence 1
        self.layer1 = self.backbone.layer1
        self.layer1.load_state_dict(self.backbone.layer1.state_dict())
        # Sequence 2
        self.layer2 = self.backbone.layer2
        self.layer2.load_state_dict(self.backbone.layer2.state_dict())
        # Sequence 3
        self.layer3 = self.backbone.layer3
        self.layer3.load_state_dict(self.backbone.layer3.state_dict())
        # Sequence 4
        self.layer4 = self.backbone.layer4
        self.layer4.load_state_dict(self.backbone.layer4.state_dict())
        # Output features
        self.fpn_sizes = [self.layer2[- 1].conv3.out_channels,
                          self.layer3[- 1].conv3.out_channels,
                          self.layer4[- 1].conv3.out_channels]
        #print(self.layer4._get_name())

    def forward(self, x):
        # input conv
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        # Bottleneck sequence
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # output features
        return x2, x3, x4


class UpsampleLike(nn.Module):
    '''
    Class for upsample and add (Feature Pyramid lateral connection)
    '''
    def __init__(self, num_channels, num_features=256):
        super(UpsampleLike, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x_l, x_r=None):
        # input from downstream
        x = self.conv1(x_l)
        # connect with upstream if exist
        if x_r is not None:
            x = x + x_r
        # upsample and outpu
        x_up = self.upsample(x)
        x_out = self.conv2(x_up)

        return x_out


class PyramidFeatures(nn.Module):
    '''
    Features Pyramid Network
    '''
    def __init__(self, num_channels=3, num_features=256, backbone='resnet50', pretrained=True):
        super(PyramidFeatures, self).__init__()
        # parametes
        self.num_channels = num_channels
        self.num_features = num_features
        self.backbone_name = backbone
        # Bottom-up pathway
        self.backbone = ResNetBackbone(num_channels, backbone='resnet50', pretrained=True)
        # Top-down pathway
        self.uplayer1 = UpsampleLike(self.backbone.fpn_sizes[2], num_features)
        self.uplayer2 = UpsampleLike(self.backbone.fpn_sizes[1], num_features)
        self.uplayer3 = UpsampleLike(self.backbone.fpn_sizes[0], num_features)
        # High level feature maps
        self.conv6 = nn.Conv2d(self.backbone.fpn_sizes[2], num_features, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d( num_features, num_features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Bottom-up pathway
        c3, c4, c5 = self.backbone(x)
        # Top-down pathway
        p5 = self.uplayer1(c5)
        p4 = self.uplayer2(c4, p5)
        p3 = self.uplayer3(c3, p4)
        # High level output
        p6 = self.conv6(c5)
        p7 = self.relu(p6)
        p7 = self.conv7(p7)
        # Output from lower to higher level (larger to smaller spatial size)
        return p3, p4, p5, p6, p7


if __name__ == "__main__":
    from torch.autograd import Variable

    net = PyramidFeatures()
    preds = net( Variable( torch.randn(2,3,512,512) ) )

    for p in preds:
        print(p.shape)

    print('')
