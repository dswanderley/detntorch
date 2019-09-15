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
    def __init__(self, in_ch, out_ch, batch_norm=True, rep=1, index=0):
        ''' Constructor '''
        super(DarknetConvBlock, self).__init__()

        # Parameters
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.batch_norm = batch_norm
        self.repetitions = rep

        # Conv 0 - with downsampling
        self.conv_down = nn.Sequential()
        self.conv_down.add_module("conv_{0}".format(index),
                            nn.Conv2d(in_ch, out_ch, 3,
                                    stride=2,
                                    padding=1,
                                    bias=not batch_norm))
        if batch_norm:
            self.conv_down.add_module("bnorm_{0}".format(index),
                                nn.BatchNorm2d(out_ch))
        self.conv_down.add_module("leaky_{0}".format(index),
                             nn.LeakyReLU(0.1, inplace=True))

        # Convolution cycle - 1x1 - 3x3
        self.conv_loop = nn.Sequential()
        for idx in range(1,2*rep+1):
            index += 1
            # Change input and output according the pair of convs (1x1 and 3x3)
            if idx % 2 == 1:
                final_vol = round(out_ch / 2)
                init_vol  = out_ch
                ks = 1
                pad = 0
            else:
                final_vol = out_ch
                init_vol  = round(out_ch / 2)
                ks = 3
                pad = 1
            # Conv layer
            self.conv_loop.add_module("conv_{0}".format(index),
                                nn.Conv2d(init_vol, final_vol, ks,
                                            stride=1,
                                            padding=pad,
                                            bias=not batch_norm))
            # Batch Normalization layer
            if batch_norm:
                self.conv_loop.add_module("bnorm_{0}".format(index),
                                nn.BatchNorm2d(final_vol))
            # Activation (Leaky) layer
            self.conv_loop.add_module("leaky_{0}".format(index),
                                nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x0 = self.conv_down(x)
        x1 = self.conv_loop(x0)
        x_out = x0 + x1
        return x_out



# Main calls
if __name__ == '__main__':

    # Images
    images = torch.randn(2, 32, 160, 160)
    darknet = DarknetConvBlock(32,64, rep=5, index=5)
    detn = darknet(images)


    print('')