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


class DarknetUpConvBlock(nn.Module):
    '''
    Darknet Upsampling and Convolutional Block.
    '''
    def __init__(self, in_ch, out_ch, res_ch=0,
                scale_factor=2, mode="nearest",
                batch_norm=True, rep=1, index=0):
        ''' Constructor '''
        super(DarknetUpConvBlock, self).__init__()

        # Parameters
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.res_channels = res_ch
        self.repetitions = rep
        self.batch_norm = False
        self.scale_factor = scale_factor
        self.mode = mode

        # Set Conv in block
        self.conv_in = nn.Sequential()
        self.conv_in.add_module("conv_{0}".format(index),
                            nn.Conv2d(in_ch, out_ch, 1,
                                    stride=1,
                                    padding=0,
                                    bias=not batch_norm))
        if batch_norm:
            self.conv_in.add_module("bnorm_{0}".format(index),
                                nn.BatchNorm2d(out_ch))
        self.conv_in.add_module("leaky_{0}".format(index),
                             nn.LeakyReLU(inplace=True))

        # Upsampling block
        if self.mode in ["linear", "bilinear", "bicubic", "trilinear"]:
            up_layer = nn.Upsample(scale_factor = self.scale_factor,
                                mode = self.mode,
                                align_corners=True)
        else:
            up_layer = nn.Upsample(scale_factor = self.scale_factor,
                                mode = self.mode)
        self.upsample = nn.Sequential()
        self.upsample.add_module("upsample_{0}".format(index), up_layer)

        # Set output conv block
        self.conv_loop = nn.Sequential()
        for idx in range(1,2*rep+1):
            index += 1
            # Change input and output according the pair of convs (1x1 and 3x3)
            if idx == 1:
                final_vol = round(out_ch / 2)
                init_vol  = out_ch + res_ch
                filter_sz = 1
                pad = 0
            elif idx % 2 == 1:
                final_vol = round(out_ch / 2)
                init_vol  = out_ch
                filter_sz = 1
                pad = 0
            else:
                final_vol = out_ch
                init_vol  = round(out_ch / 2)
                filter_sz = 3
                pad = 1
            # Conv layer
            self.conv_loop.add_module("conv_{0}".format(index),
                                nn.Conv2d(init_vol, final_vol, filter_sz,
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

    def forward(self, x, x_res=None):
        ''' Foward method '''
        # Input conv
        x_in = self.conv_in(x)
        # Upsampling
        x_up = self.upsample(x_in)
        # Concatenates
        if x_res is None:
            x_cat = x_up
        else:
            x_cat = torch.cat((x_up, x_res), 1)
        # Output conv loop
        x_out = self.conv_loop(x_cat)
        return x_out



# Main calls
if __name__ == '__main__':

    # Images
    x0 = torch.randn(2, 32, 160, 160)

    darknet_conv = DarknetConvBlock(32, 64, rep=5)
    darknet_up = DarknetUpConvBlock(64, 32, res_ch=32, rep=2)

    x1 = darknet_conv(x0)
    x2 = darknet_up(x1, x_res=x0)


    print('')