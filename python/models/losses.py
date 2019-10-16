# -*- coding: utf-8 -*-
"""
Created on Mon Oct 07 21:58:12 2019

@author: Diego Wanderley
@python: 3.6
@description: Script with losses functions.

"""

import math
import torch

import numpy as np
import torch.nn as nn

from models.utils import prepare_targets


class YoloLoss(nn.Module):
    '''
    YOLO Loss

    Forward arguments:
        @param prediction: tensor with prediction volume
        @param groundtruth: list with a tensor of bouding boxes and classes
    '''

    def __init__(self, scaled_anchors, ignore_thresh=0.5):
        super(YoloLoss, self).__init__()

        self.anchors = scaled_anchors
        self.ignore_thresh = ignore_thresh

        self.obj_scale = 1
        self.noobj_scale = 100
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # Offset by grids
        self.num_anchors = len(self.anchors)
        self.anchor_w = self.anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.anchors[:, 1:2].view((1, self.num_anchors, 1, 1))


    def forward(self, pred, targets):

        # Get outputs
        pred_x = pred[..., 0]       # Center x
        pred_y = pred[..., 1]       # Center y
        pred_w = pred[..., 2]       # Width
        pred_h = pred[..., 3]       # Height
        pred_conf = pred[..., 4]    # Conf
        pred_cls =  pred[..., 5:]   # Cls pred.
        gs = pred.size(-2)     # grid size

        # Reference/index grids
        grid_x = torch.arange(gs).repeat(gs, 1).view([1,1,gs,gs]).type(torch.FloatTensor)
        grid_y = torch.arange(gs).repeat(gs, 1).t().view([1,1,gs,gs]).type(torch.FloatTensor)

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(pred[..., :4].shape)
        pred_boxes[..., 0] = pred_x.data + grid_x
        pred_boxes[..., 1] = pred_y.data + grid_y
        pred_boxes[..., 2] = torch.exp(pred_w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(pred_h.data) * self.anchor_h

        # Bouding boxes scores
        iou_scores, class_pred, obj_detn, noobj_detn, bb_x, bb_y, bb_w, bb_h, tcls, tconf = prepare_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.anchors,
                ignore_thresh=self.ignore_thresh,
            )

        # Bouding Box losses based on MSE
        loss_x = self.mse_loss(pred_x[obj_detn], bb_x[obj_detn])
        loss_y = self.mse_loss(pred_y[obj_detn], bb_y[obj_detn])
        loss_w = self.mse_loss(pred_w[obj_detn], bb_w[obj_detn])
        loss_h = self.mse_loss(pred_h[obj_detn], bb_h[obj_detn])
        # Object confidence error based on BCE
        loss_conf_obj = self.bce_loss(pred_conf[obj_detn], tconf[obj_detn])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_detn], tconf[noobj_detn])
        # Scal and sum the two BCE losses
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        # Classes error
        loss_cls = self.bce_loss(pred_cls[obj_detn], tcls[obj_detn])
        # Total loss
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics
        # Average all correct (1) or wrong (o) predictions and x 100%
        cls_acc = 100 * class_pred[obj_detn].mean()
        # Average all object confidence
        conf_obj = pred_conf[obj_detn].mean()
        # Average all no object confidence
        conf_noobj = pred_conf[noobj_detn].mean()
        # Find binary all pred_conf over 0.5
        conf50 = (pred_conf > 0.5).float()
        # Find binary all IoU over 0.50
        iou50 = (iou_scores > 0.5).float()
        # Find binary all IoU over 0.75
        iou75 = (iou_scores > 0.75).float()
        # Find all predicted right class bouding boces over 0.5 confidence
        obj_mask = conf50 * class_pred * tconf
        # Divide (sum of detected mask that is highter than 0.5 IOU) with (sum of con50)
        precision = torch.sum(iou50 * obj_mask) / (conf50.sum() + 1e-16)
        # Divide (sum of detected mask that is highter than 0.5 IOU) with (sum of obj_mask)
        recall50 = torch.sum(iou50 * obj_mask) / (obj_mask.sum() + 1e-16)
        # Divide (sum of detected mask that is highter than 0.75 IOU) with (sum of obj_mask)
        recall75 = torch.sum(iou75 * obj_mask) / (obj_mask.sum() + 1e-16)

        metrics = {
            "loss": total_loss.item(),
            "x": loss_x.item(),
            "y": loss_y.item(),
            "w": loss_w.item(),
            "h": loss_h.item(),
            "conf": loss_conf.item(),
            "cls": loss_cls.item(),
            "cls_acc": cls_acc.item(),
            "recall50": recall50.item(),
            "recall75": recall75.item(),
            "precision": precision.item(),
            "conf_obj": conf_obj.item(),
            "conf_noobj": conf_noobj.item(),
            "grid_size": gs,
        }

        return metrics
