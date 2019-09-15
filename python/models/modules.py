# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:29:35 2019
@author: Diego Wanderley
@python: 3.6
@description: CNN modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DarknetConvBlock(nn.Module):
    '''
    Darknet Convolutional Block
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, rep=1):
        ''' Constructor '''
        super(DarknetConvBlock, self).__init__()
        # Parameters
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.batch_norm = batch_norm
        self.rep = rep

        # Conv 0 - with downsampling
        self.conv_down = nn.Sequential()
        index = 0
        self.conv_down.add_module("conv_{0}".format(index),
                            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
        if batch_norm:
            self.conv_down.add_module("bnorm_{0}".format(index),
                                nn.BatchNorm2d(out_ch))
        self.conv_down.add_module("leaky_{0}".format(index),
                             nn.LeakyReLU(inplace=True))

        # Convolution cycle - 1x1 - 3x3
        self.conv_loop = nn.Sequential()
        for index in range(1,2*rep+1):
            # Change input and output according the pair of convs (1x1 and 3x3)
            if index % 2 == 1:
                final_vol = round(out_ch / 2)
                init_vol  = out_ch
            else:
                final_vol = out_ch
                init_vol  = round(out_ch / 2)
            # Conv layer
            self.conv_loop.add_module("conv_{0}".format(index),
                                nn.Conv2d(init_vol, final_vol, 3, stride=1, padding=1))
            # Batch Normalization layer
            if batch_norm:
                self.conv_loop.add_module("bnorm_{0}".format(index),
                                nn.BatchNorm2d(final_vol))
            # Activation (Leaky) layer
            self.conv_loop.add_module("leaky_{0}".format(index),
                                nn.LeakyReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x0 = self.conv_down(x)
        x1 = self.conv_loop(x0)
        x = x0 + x1
        return x




# Main calls
if __name__ == '__main__':

    # Images
    images = torch.randn(2, 32, 320, 320)
    darknet = DarknetConvBlock(32,64, rep=5)

    detn = darknet(images)

    print('')