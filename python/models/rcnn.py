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


class FasterRCNN(nn.Module):
    '''
    Faster R-CNN Class
    '''
    def __init__(self, n_channels=3, n_classes=91, pretrained=False):
        super(FasterRCNN, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Input conv is applied to convert the input to 3 ch depth
        self.inconv = None
        if n_channels != 3:
            self.inconv = nn.Sequential()
            self.inconv.add_module("conv_0", nn.Conv2d(n_channels, 3, 1, stride=1, padding=0))
            self.inconv.add_module("bnorm_0", nn.BatchNorm2d(3))
            self.inconv.add_module("relu_0", nn.ReLU(inplace=True))

        # Pre-trained model needs to be an identical network
        if pretrained:
            self.body = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=91, min_size=512)
            # Reset output
            if n_classes != 91:
                self.body.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
                self.body.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4*n_classes, bias=True)

        else:
            self.body = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=n_classes, min_size=512)


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

    import numpy as np
    from skimage import io
    from skimage.color import rgb2gray

    image = io.imread('/Users/Diego/Downloads/cars.jpg') / 255.
    #torch_img = torch.from_numpy(image)
    #torch_img = torch_img.permute(2,0,1)

    grayscale = rgb2gray(image)
    torch_img = torch.from_numpy(grayscale)
    torch_img.unsqueeze_(0)

    torch_img.unsqueeze_(0)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together
    # https://github.com/pytorch/vision/blob/master/references/detection/train.py

    # Images
    #images = torch.randn(2, 1, 512, 512)
    images = torch_img.float()
    # Targets
    bbox = torch.FloatTensor([[120, 130, 300, 350], [200, 200, 250, 250]]) # [y1, x1, y2, x2] format
    lbls = torch.LongTensor([1, 2]) # 0 represents background

    # targets per image
    tgts = {
        'boxes':  bbox.to(device),
        'labels': lbls.to(device)
    }
    # targets to list (by batch)
    targets = [tgts]#, tgts]

    # Model
    model = FasterRCNN(n_channels=1, pretrained=True).to(device)
    #model = FasterRCNN(n_channels=3, pretrained=True).to(device)
    model.eval()
    #model.train()

    # output
    #loss_dict = model(images.to(device), targets)
    loss_dict = model(images.to(device))

    for i in range(len(loss_dict)):
        img = image
        bboxes = loss_dict[i]['boxes']
        color = np.array([0, 255, 0])/255.

        for bb in bboxes:
            bounding_box = bb.detach().numpy().round().astype(np.int)
            img[bounding_box[1], bounding_box[0]:bounding_box[2]] = color
            img[bounding_box[1]:bounding_box[3], bounding_box[0]] = color

            img[bounding_box[3], bounding_box[0]:bounding_box[2]] = color
            img[bounding_box[1]:bounding_box[3], bounding_box[2]] = color

        io.imsave("./debug.png", img)


    print('')
