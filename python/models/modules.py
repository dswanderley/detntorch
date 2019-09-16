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
    Basic convolutional block
    '''
    def __init__(self, in_ch, out_ch,
                kernel_size=3, stride=1, padding=1,
                batch_norm=True, index=0):
        ''' Constructor '''
        super(DarknetConvBlock, self).__init__()

        # Parameters
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.index = index

        # Sequential moduler
        self.conv = nn.Sequential()
        self.conv.add_module("conv_{0}".format(index),
                            nn.Conv2d(in_ch, out_ch, kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=not batch_norm))
        if batch_norm:
            self.conv.add_module("bnorm_{0}".format(index),
                                nn.BatchNorm2d(out_ch))
        self.conv.add_module("leaky_{0}".format(index),
                             nn.LeakyReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class DarknetBottleneck(nn.Module):
    '''
    Darknet Bottlenec Convolutional Block
    '''
    def __init__(self, ch_in, batch_norm=True, index=1):
        super(DarknetBottleneck, self).__init__()
        '''Constructor'''
        self.in_channels = ch_in
        self.mid_channels = round(ch_in / 2)
        self.out_channels = ch_in
        self.batch_norm = batch_norm
        self.index_in = index

        # First Conv 1x1
        self.conv1 = DarknetConvBlock(self.in_channels, self.mid_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    batch_norm=batch_norm,
                                    index=index)
        # Second Conv 3x3
        index += 1
        self.conv2 = DarknetConvBlock(self.mid_channels, self.out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    batch_norm=batch_norm,
                                    index=index)
        self.index_out = index

    def forward(self, x):
        ''' Foward method '''
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x_out = x + x2
        return x_out


class DarknetBlock(nn.Module):
    '''
    Darknet Convolutional Block
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, rep=1, index=0):
        ''' Constructor '''
        super(DarknetBlock, self).__init__()

        # Parameters
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.batch_norm = batch_norm
        self.repetitions = rep
        self.index_in = index
        conv_modules = nn.ModuleList()
        # Conv 0 - with downsampling
        conv_down = DarknetConvBlock(in_ch, out_ch,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    batch_norm=True,
                                    index=index)
        conv_modules.append(conv_down)
        # Convolution cycle - 1x1 - 3x3
        for idx in range(1,rep+1):
            index += 1
            bneck = DarknetBottleneck(out_ch,
                                    batch_norm=batch_norm,
                                    index=index)
            conv_modules.append(bneck)
            index += 1
        self.index_out = index
        # Create sequential conv
        self.sequential = nn.Sequential(*conv_modules)

    def forward(self, x):
        ''' Foward method '''
        x = self.sequential(x)
        return x


class DarknetUpsampling(nn.Module):
    '''
    Darknet Upsampling and Convolutional Block.
    '''
    def __init__(self, in_ch, out_ch, res_ch=0,
                scale_factor=2, mode="nearest",
                batch_norm=True, rep=1, index=0):
        ''' Constructor '''
        super(DarknetUpsampling, self).__init__()

        # Parameters
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.res_channels = res_ch
        self.repetitions = rep
        self.batch_norm = False
        self.scale_factor = scale_factor
        self.mode = mode
        self.index_in = index

        # Set Conv in block
        self.conv_in = DarknetConvBlock(self.in_channels, self.out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    batch_norm=batch_norm,
                                    index=index)

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
        for idx in range(1,rep+1):
            index += 1
            # Change input and output according the pair of convs (1x1 and 3x3)
            if idx == 1:
                init_vol  = self.out_channels + self.res_channels
                final_vol = self.in_channels
                filter_sz = 1
                pad = 0
            elif idx % 2 == 1:
                init_vol  = self.out_channels
                final_vol = self.in_channels
                filter_sz = 1
                pad = 0
            else:
                init_vol  = self.in_channels
                final_vol = self.out_channels
                filter_sz = 3
                pad = 1
            # Conv layer
            conv_module = DarknetConvBlock(init_vol, final_vol,
                                        kernel_size=filter_sz,
                                        stride=1,
                                        padding=pad,
                                        batch_norm=batch_norm,
                                        index=index)
            self.conv_loop.add_module("conv_block_{0}".format(idx), conv_module)
        self.index_out = index

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


class YoloDetector(nn.Module):
    '''
    Output Dimensions of each detection kernel is 1 x 1 x (B x (4 + 1 + C))
        - B: Number of bounding boxes a cell on the feature map can predic;
        - 4: Bounding box attributes;
        - 1: Object confidence;
        - C: Number of classes.
        Kernel depth arrangement:
            [t_x ,t_y, t_w, t_h], [p_o], [p_1, p+2, ..., p_C] x B
    '''
    def __init__(self, in_ch, anchors, num_classes,
                        bb_by_cell=3, img_dim=512,
                        batch_norm=True, index=0):
        ''' Constructor '''
        super(YoloDetector, self).__init__()

        kernel_depth = bb_by_cell * (4 + 1 + num_classes)

        self.in_channels = in_ch
        self.mid_channels = round(in_ch / 2)
        self.out_channels = kernel_depth
        self.batch_norm = batch_norm
        self.bb_by_cell = bb_by_cell
        self.anchors = anchors
        self.index_in = index

        # Reduce depth with 3x3
        self.conv_in = DarknetConvBlock(self.in_channels, self.mid_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm=self.batch_norm,
                                        index=index)
        index += 1
        # Reduce to output dimension
        self.conv_out = nn.Sequential()
        self.conv_out.add_module("conv_{0}".format(index),
                            nn.Conv2d(self.mid_channels, self.out_channels, 1,
                                    stride=1,
                                    padding=0))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv_in(x)
        x = self.conv_out(x)
        return x


# Main calls
if __name__ == '__main__':

    # Images
    x0 = torch.randn(2, 32, 160, 160)

    darknet_conv = DarknetBlock(32, 64, rep=5)
    darknet_up = DarknetUpsampling(64, 32, res_ch=32, rep=5)
    yolo_dtn = YoloDetector(64, 3, 3)

    x1 = darknet_conv(x0)
    x2 = darknet_up(x1, x_res=x0)
    x3 = yolo_dtn(x2)

    print('')