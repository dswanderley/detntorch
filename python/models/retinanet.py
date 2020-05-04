# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:03:14 2020
@author: Diego Wanderley
@python: 3.6
@description: RetinaNet implementation
"""

import math
import torch
import torch.nn as nn

try:
    #import models.retina_utils.losses as losses
    from models.fpn import PyramidFeatures
    from models.retina_utils.utils import DataEncoder
    from models.retina_utils.losses2 import FocalLoss
    from models.retina_utils.anchors import Anchors
    from models.retina_utils.utils2 import BBoxTransform, ClipBoxes
except:
    #import retina_utils.losses as losses
    from fpn import PyramidFeatures
    from retina_utils.utils import DataEncoder
    from retina_utils.losses2 import FocalLoss
    from retina_utils.anchors import Anchors
    from retina_utils.utils2 import BBoxTransform, ClipBoxes


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
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(num_features, num_anchors * num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # Classification subnet
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        # Output conv
        out = self.conv5(out) # out is B x C x W x H, with C = n_classes * n_anchors
        out = self.sigmoid(out)
        # Permute to put W and H in the middle
        out = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out.shape
        # Split anchors and classes from channels
        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)
        # Strip all elements in a single dimension, excepet batch anc classes (one-hot-encoded)
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
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(num_features, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        # Box subnet
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        # Output conv
        out = self.conv5(out) # out is B x C x W x H, with C = 4*num_anchors
        # Permute to put W and H in the middle
        out = out.permute(0, 2, 3, 1)
        # Strip H, W and anchors in a single dimension
        out = out.contiguous().view(out.shape[0], -1, 4)

        return out


class RetinaNet(nn.Module):
    '''
        RetinaNet class.
    '''
    def __init__(self, in_channels=1, num_classes=2, num_features=256, num_anchors=9, pretrained=False):
        super(RetinaNet, self).__init__()
        # Parameters
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_anchors = num_anchors
        self.pretrained = pretrained
        # Define model blocks
        self.fpn = PyramidFeatures(in_channels=in_channels, num_features=num_features, pretrained=pretrained)
        self.classification = ClassificationModel(in_features=num_features,
                                                  num_features=num_features,
                                                  num_anchors=num_anchors,
                                                  num_classes=num_classes)
        self.regression = RegressionModel(in_features=num_features,
                                            num_features=num_features,
                                            num_anchors=num_anchors)
        # Loss
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()
        # Encoder
        self.encoder = DataEncoder()
        # Adjust output weights
        prior = 0.01
        self.classification.conv5.weight.data.fill_(0)
        self.classification.conv5.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regression.conv5.weight.data.fill_(0)
        self.regression.conv5.bias.data.fill_(0)
        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x, tgts=None):
        cls_preds = []
        box_preds = []

        # Get Pyramid features
        features = self.fpn(x)

        # Run subnets for each feature
        for feat in features:
            cpred = self.classification(feat)
            cls_preds.append(cpred)
            bpred = self.regression(feat)
            box_preds.append(bpred)
        # Convert to tensor
        lbl_preds = torch.cat(cls_preds, dim=1)
        box_preds = torch.cat(box_preds, dim=1)

        # Computes anchors to the current batch
        anchors = self.anchors(x)

        # Outputs - return loss if targets are give training
        if tgts is not None:
            loss =  self.focalLoss(lbl_preds, box_preds, anchors, tgts)
            return { 'box_loss':loss[1], 'cls_los':loss[0] }
        else:
            # output in tensorbord faster rcnn style
            detections = []
            for bbxs, lbls in zip(box_preds, lbl_preds):
                boxes, labels = encoder.decode(bbxs.data.squeeze(), lbls.data.squeeze(), (w,h))
                detections.append({
                        'boxes': boxes,
                        'labels': labels
                    })
            return detections


if __name__ == "__main__":
    from torch.autograd import Variable
    from retina_utils.utils import DataEncoder

    encoder = DataEncoder()

    in_channels = 1
    bs = 2
    w = h = 512
    training = True

    # Create inputs
    imgs = Variable( torch.randn(bs,in_channels,w,h) )
    boxes = [   torch.FloatTensor( [ [120, 130, 300, 350], [200, 200, 250, 250] ] ),
                torch.FloatTensor( [ [100, 150, 150, 200] ] ) ]
    labels = [ torch.LongTensor([1, 0]), torch.LongTensor([1]) ]
    # Encode targets
    """
    box_targets = []
    cls_targets = []
    for i in range(bs):
        loc_target, cls_target = encoder.encode(boxes[i], labels[i], input_size=(w,h))
        box_targets.append(loc_target)
        cls_targets.append(cls_target)
    
    # Set in a dict.
    tgts = {
        'boxes': boxes, #torch.stack(box_targets),
        'labels': labels #torch.stack(cls_targets)
    }
    """

    tgts = [{ 'boxes':  box,'labels': lbl } #.to(device)
                        for box, lbl in zip(boxes, labels)]

    # Load network
    net = RetinaNet(in_channels=in_channels)
    if training:
        net.train()
        out = net( imgs, tgts )
    else:
        net.eval()
        detections = net( imgs )

    print('end')
