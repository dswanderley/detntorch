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

try:
    import models.losses as losses
    from models.utils import *
except:
    import losses
    from utils import *


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
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_out = x + x_2
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
        for _ in range(1,rep+1):
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
        x_out = self.sequential(x)
        return x_out


class DarknetLoopBlock(nn.Module):
    '''
    Darknet Loop Convolutional Block
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, rep=1, index=0):
        ''' Constructor '''
        super(DarknetLoopBlock, self).__init__()

        # Parameters
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.repetitions = rep
        self.batch_norm = batch_norm
        self.index_in = index

        # Convolution cycle - 1x1 - 3x3
        self.conv_loop = nn.Sequential()
        for idx in range(1,rep+1):
            index += 1
            # Change input and output according the pair of convs (1x1 and 3x3)
            if idx % 2 == 1:
                init_vol  = self.in_channels
                final_vol = self.out_channels
                filter_sz = 1
                pad = 0
            else:
                init_vol  = self.out_channels
                final_vol = self.in_channels
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

    def forward(self, x):
        ''' Foward method '''
        x = self.conv_loop(x)
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
        self.batch_norm = batch_norm
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
                final_vol = self.out_channels
                filter_sz = 1
                pad = 0
            elif idx % 2 == 1:
                init_vol  = self.in_channels
                final_vol = self.out_channels
                filter_sz = 1
                pad = 0
            else:
                init_vol  = self.out_channels
                final_vol = self.in_channels
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
    def __init__(self, in_ch, num_classes=3, num_anchors=3,
                        batch_norm=True, index=0):
        ''' Constructor '''
        super(YoloDetector, self).__init__()

        kernel_depth = num_anchors * (4 + 1 + num_classes)

        self.in_channels = in_ch
        self.mid_channels = round(in_ch / 2)
        self.out_channels = kernel_depth
        self.batch_norm = batch_norm
        self.num_anchors = num_anchors
        self.num_classes = num_classes
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

        # Reshape prediction for an easier way to understand
        batch_size = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # [batch, anchor, grid_height, grid_width, classes_and_bbox]

        # Apply sigmoit to outputs (expept height and width)
        prediction[..., 0] = torch.sigmoid(prediction[..., 0])      # Center x
        prediction[..., 1] = torch.sigmoid(prediction[..., 1])      # Center y
        prediction[..., 4] = torch.sigmoid(prediction[..., 4])      # Confidence
        prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])    # Classes

        return prediction


class Darknet(nn.Module):
    '''
    Darknet body class
    '''
    def __init__(self, in_ch, filters=32,
                batch_norm=True):
        ''' Constructor '''
        super(Darknet, self).__init__()

        # Parameters
        self.in_channels = in_ch
        self.batch_norm = batch_norm
        self.filters = filters
        self.stride_out_1 = 32
        self.stride_out_2 = 16
        self.stride_out_3 = 8
        index = 0
        self.first_index = 0
        # Initial Conv
        self.conv0 = DarknetConvBlock(self.in_channels, self.filters,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    batch_norm=self.batch_norm,
                                    index=index)
        index += 1
        # First block
        self.conv1 = DarknetBlock(self.filters, self.filters * 2,
                                batch_norm=self.batch_norm,
                                rep=1,
                                index=index)
        index = self.conv1.index_out + 1
        # Second block
        self.conv2 = DarknetBlock(self.filters * 2, self.filters * 4,
                                batch_norm=self.batch_norm,
                                rep=2,
                                index=index)
        index = self.conv2.index_out + 1
        # Third block
        self.conv3 = DarknetBlock(self.filters * 4, self.filters * 8,
                                batch_norm=self.batch_norm,
                                rep=8,
                                index=index)
        index = self.conv3.index_out + 1
        # Fourth block
        self.conv4 = DarknetBlock(self.filters * 8, self.filters * 16,
                                batch_norm=self.batch_norm,
                                rep=8,
                                index=index)
        index = self.conv4.index_out + 1
        # Fifth block
        self.conv5 = DarknetBlock(self.filters * 16, self.filters * 32,
                                batch_norm=self.batch_norm,
                                rep=4,
                                index=index)
        index = self.conv5.index_out + 1

        # Output 1
        self.conv_low = DarknetLoopBlock(self.filters * 32, self.filters * 16,
                                    batch_norm=self.batch_norm,
                                    rep=5,
                                    index=index)
        index = self.conv_low.index_out + 1
        # Output 2
        self.conv_up1 = DarknetUpsampling(self.filters * 16, self.filters * 8,
                                    res_ch=self.filters * 16,
                                    batch_norm=self.batch_norm,
                                    rep=5,
                                    index=index)
        index = self.conv_up1.index_out + 1
        # Output 3
        self.conv_up2 = DarknetUpsampling(self.filters * 8, self.filters * 4,
                                    res_ch=self.filters * 8,
                                    batch_norm=self.batch_norm,
                                    rep=5,
                                    index=index)
        index = self.conv_up2.index_out + 1
        self.last_index = index

    def forward(self, x):
        ''' Foward method '''
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x_out1 = self.conv_low(x5)
        x_out2 = self.conv_up1(x_out1, x_res=x4)
        x_out3 = self.conv_up2(x_out2, x_res=x3)
        return x_out1, x_out2, x_out3


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=512):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = prepare_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thresh=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Yolo_v3(nn.Module):
    '''
    Yolo network
    '''
    def __init__(self, in_ch, num_classes=2, anchors=[(116,90),(156,198),(737,326)]):
        ''' Constructor '''
        super(Yolo_v3, self).__init__()

        # Parameters
        self.in_ch = in_ch
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes

        # Body
        self.darknet = Darknet(in_ch)

        # Scaled anchors
        std1 = self.darknet.stride_out_1
        self.scaled_anchors_1 = torch.FloatTensor([(a_w / std1, a_h / std1) for a_w, a_h in self.anchors])
        std2 = self.darknet.stride_out_2
        self.scaled_anchors_2 = torch.FloatTensor([(a_w / std2, a_h / std2) for a_w, a_h in self.anchors])
        std3 = self.darknet.stride_out_3
        self.scaled_anchors_3 = torch.FloatTensor([(a_w / std3, a_h / std3) for a_w, a_h in self.anchors])

        # Detectors
        self.yolo1 = YoloDetector(512,
                        num_classes=self.num_classes,
                        num_anchors=self.num_anchors)
        self.yolo2 = YoloDetector(256,
                        num_classes=self.num_classes,
                        num_anchors=self.num_anchors)
        self.yolo3 = YoloDetector(128,
                        num_classes=self.num_classes,
                        num_anchors=self.num_anchors)

        # Losses
        self.loss1 = losses.YoloLoss(self.scaled_anchors_1)
        self.loss2 = losses.YoloLoss(self.scaled_anchors_2)
        self.loss3 = losses.YoloLoss(self.scaled_anchors_3)

    def get_output_shape(self, pred_block, num_samples):
        # Convert pred to output shape
        output_shape = torch.cat(
            (
                pred_block[6].view(num_samples, -1, 4) * self.darknet.stride_out_1,
                pred_block[4].view(num_samples, -1, 1),
                pred_block[5].view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        return output_shape

    def forward(self, x, targets=None):
        ''' Foward method '''
        num_samples = x.size(0)
        self.image_size = x.shape[-1]
        # Run body
        x_1, x_2, x_3 = self.darknet(x)
        # Get ratio / stride
        self.net_stride_1 = self.image_size / x_1.shape[-1]
        self.net_stride_2 = self.image_size / x_2.shape[-1]
        self.net_stride_3 = self.image_size / x_3.shape[-1]
        # Prediction
        pred_1 = self.yolo1(x_1)
        pred_2 = self.yolo2(x_2)
        pred_3 = self.yolo3(x_3)

        pred_block_1 = get_pred_boxes(pred_1, self.scaled_anchors_1)
        pred_block_2 = get_pred_boxes(pred_2, self.scaled_anchors_1)
        pred_block_3 = get_pred_boxes(pred_3, self.scaled_anchors_1)

        # Convert predictions to output shape
        output_1 = self.get_output_shape(pred_block_1, num_samples)
        output_2 = self.get_output_shape(pred_block_2, num_samples)
        output_3 = self.get_output_shape(pred_block_3, num_samples)
        # Concat outputs
        outputs = [output_1, output_2, output_3]
        outputs = torch.cat(outputs, 1)

        if targets is None:
            return outputs

        else:
            # Compute losses
            l1 = self.loss1(pred_block_1, targets)
            l2 = self.loss2(pred_block_2, targets)
            l3 = self.loss3(pred_block_3, targets)

            # Total Loss
            loss = l1 + l2 + l3

            return outputs, loss


class Yolo_net(nn.Module):
    '''
    Yolo network
    '''
    def __init__(self, in_ch, num_classes=2,
                    anchors=[(116,90),(156,198),(737,326)],
                    img_dim=512, batch_norm=True):
        ''' Constructor '''
        super(Yolo_net, self).__init__()

        # Parameters
        self.in_ch = in_ch
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.batch_norm = batch_norm

        # Body
        self.darknet = Darknet(in_ch)
        index = self.darknet.first_index + 1
        # Outputs # Reduce depth with 3x3
        self.out_net1 = DarknetConvBlock(512, self.num_anchors * (self.num_classes + 5),
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm=self.batch_norm,
                                        index=index)
        index += 1
        self.out_net2 = DarknetConvBlock(256, self.num_anchors * (self.num_classes + 5),
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm=self.batch_norm,
                                        index=index)
        index += 1
        self.out_net3 = DarknetConvBlock(128, self.num_anchors * (self.num_classes + 5),
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm=self.batch_norm,
                                        index=index)
        index += 1
        # Detectors
        self.yolo1 = YOLOLayer(self.anchors, self.num_classes,img_dim=self.img_dim)
        self.yolo2 = YOLOLayer(self.anchors, self.num_classes,img_dim=self.img_dim)
        self.yolo3 = YOLOLayer(self.anchors, self.num_classes,img_dim=self.img_dim)


    def forward(self, x, targets=None):
        ''' Foward method '''
        num_samples = x.size(0)
        img_size = x.shape[-1]
        # Run body
        x_1, x_2, x_3 = self.darknet(x)
        x_1 = self.out_net1(x_1)
        x_2 = self.out_net2(x_2)
        x_3 = self.out_net3(x_3)
        # Yolo layer
        x_1, loss_1 = self.yolo1(x_1, targets, img_dim=img_size)
        x_2, loss_2 = self.yolo2(x_2, targets, img_dim=img_size)
        x_3, loss_3 = self.yolo3(x_3, targets, img_dim=img_size)

        yolo_outputs = [x_1, x_2, x_3]
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        loss = loss_1 + loss_2 + loss_3

        return yolo_outputs if targets is None else (yolo_outputs, loss)



# Main calls
if __name__ == '__main__':

    import torch.optim as optim

    # Images
    x = torch.randn(2, 1, 512, 512)

    '''
    x0 = torch.randn(2, 32, 160, 160)

    darknet_conv = DarknetBlock(32, 64, rep=5)
    darknet_up = DarknetUpsampling(64, 32, res_ch=32, rep=5)
    yolo_dtn = YoloDetector(64, 3, 3)

    x1 = darknet_conv(x0)
    x2 = darknet_up(x1, x_res=x0)
    x3 = yolo_dtn(x2)

    darknet = Darknet(1)
    x_out1, x_out2, x_out3 = darknet(x)
    '''
    # [2, 512, 16, 16]
    # [2, 256, 32, 32]
    # [2, 128, 64, 64]

    yolo = Yolo_net(1)
    optimizer = optim.Adam(yolo.parameters(), lr=0.001)
    #x_out1, x_out2, x_out3 = yolo(x)

    targets = [
        [0.0000, 1.0000, 0.6611, 0.4229, 0.3262, 0.4395],
        [0.0000, 1.0000, 0.4707, 0.3711, 0.1094, 0.1133]
    ]
    targets = torch.Tensor(targets)
    yolo.train()
    for _ in range(0,2):
        output, loss = yolo(x, targets=targets)
        loss.backward()

        optimizer.zero_grad()
        optimizer.step()

    print('')