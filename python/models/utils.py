# -*- coding: utf-8 -*-
"""
Created on Wed Oct 02 21:34:01 2019
@author: Diego Wanderley
@python: 3.6
@description: YOLO modules auxiliary functions.
"""


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    batch_size  = pred_boxes.size(0)
    num_anchors = pred_boxes.size(1)
    grid_size   = pred_boxes.size(2)
    num_classes = pred_cls.size(-1)

	# Object location by patch
    obj_detn = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.uint8)
    not_detn = torch.ones((batch_size, num_anchors, grid_size, grid_size), dtype=torch.uint8)
    # Bouding boxes
    iou_scores = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_x = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_y = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_h = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_w = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    # To understand
    tcls = torch.zeros((batch_size, num_anchors, grid_size, grid_size, num_classes), dtype=torch.float32)
    # Class prediction by batch
    class_pred = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)

    tconf = obj_detn.float()
    return iou_scores, class_pred, obj_detn, not_detn, bb_x, bb_y, bb_w, bb_h, tcls, tconf
