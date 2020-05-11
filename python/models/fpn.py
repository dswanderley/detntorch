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
    def __init__(self, in_channels=3, backbone_model='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        # set parameters
        self.in_channels = in_channels
        # adjust channels
        if in_channels != 3:
            self.conv0 = nn.Conv2d(in_channels, 3, 1, stride=1, padding=0, bias=False)
            self.bn0 = nn.BatchNorm2d(3)
            self.conv0.weight.data.fill_(1/in_channels)
        self.relu = nn.ReLU(inplace=True)
        # load backbone
        backbone, shortcut_features, bb_out_name = get_backbone(backbone_model, pretrained=pretrained)
        # backbone input conv
        self.conv1 = backbone.conv1
        self.conv1.load_state_dict(backbone.conv1.state_dict())
        self.bn1 = backbone.bn1
        self.bn1.load_state_dict(backbone.bn1.state_dict())
        #self.relu = self.backbone.relu
        self.maxpool = backbone.maxpool
        # Sequence 1
        self.layer1 = backbone.layer1
        self.layer1.load_state_dict(backbone.layer1.state_dict())
        # Sequence 2
        self.layer2 = backbone.layer2
        self.layer2.load_state_dict(backbone.layer2.state_dict())
        # Sequence 3
        self.layer3 = backbone.layer3
        self.layer3.load_state_dict(backbone.layer3.state_dict())
        # Sequence 4
        self.layer4 = backbone.layer4
        self.layer4.load_state_dict(backbone.layer4.state_dict())
        # Output features
        self.fpn_sizes = [self.layer2[- 1].conv3.out_channels,
                          self.layer3[- 1].conv3.out_channels,
                          self.layer4[- 1].conv3.out_channels]
        #print(self.layer4._get_name())

    def forward(self, x):
        # adjust input channels
        if self.in_channels != 3:
            x = self.conv0(x)
            x = self.bn0(x)
            x = self.relu(x)
        # backbone input conv
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


class PyramidFeatures(nn.Module):
    '''
    Features Pyramid Network
    '''
    def __init__(self, in_channels=3, num_features=256, backbone_name='resnet50', pretrained=False):
        super(PyramidFeatures, self).__init__()
        # parametes
        self.in_channels = in_channels
        self.num_features = num_features
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        # Bottom-up pathway
        self.backbone = ResNetBackbone(in_channels=in_channels, backbone_model=backbone_name, pretrained=pretrained)
        # Lateral convolution pathway
        self.latlayer1 = nn.Conv2d(self.backbone.fpn_sizes[2], num_features, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.backbone.fpn_sizes[1], num_features, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(self.backbone.fpn_sizes[0], num_features, kernel_size=1, stride=1, padding=0)
         # High level feature maps
        self.conv6 = nn.Conv2d(self.backbone.fpn_sizes[2], num_features, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d( num_features, num_features, kernel_size=3, stride=2, padding=1)
        # Top-down pathway
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # Top layer convs
        self.toplayer1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
       
    def forward(self, x):
        # Bottom-up pathway
        c3, c4, c5 = self.backbone(x)
        # High level output
        p6 = self.conv6(c5)
        p7 = self.relu(p6)
        p7 = self.conv7(p7)
        # Top-down pathway
        p5 = self.latlayer1(c5)
        p4 = self.latlayer2(c4)
        p3 = self.latlayer3(c3)
        p4 = self.upsample1(p5) + p4
        p3 = self.upsample2(p4) + p3
        # Top layers output
        p5 = self.toplayer1(p5)
        p4 = self.toplayer2(p4)
        p3 = self.toplayer3(p3)

        # Output from lower to higher level (larger to smaller spatial size)
        return p3, p4, p5, p6, p7


if __name__ == "__main__":
    from torch.autograd import Variable

    net = PyramidFeatures(in_channels=1)
    preds = net( Variable( torch.randn(2,1,512,512) ) )

    for p in preds:
        print(p.shape)

    print('')
