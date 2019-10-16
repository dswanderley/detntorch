# -*- coding: utf-8 -*-
"""
Created on Wed Oct 02 21:34:01 2019
@author: Diego Wanderley
@python: 3.6
@description: YOLO modules auxiliary functions.
"""

import torch


def relative_bbox_iou(w1, h1, w2, h2):
    """
    Compute the IOU given the width and the height of two bouding boxes
    """
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def prepare_targets(pred_boxes, pred_cls, target, anchors, ignore_thresh):
    """
        Adjust targets to the network output format and compare with anchor boxes.

        INPUTS
            pred_boxes: predicted boxes [bs, n_anchors, grid_height, grid_width, 4 (bbox_dim)]
            pred_cls:   predicted classes [bs, n_anchors, grid_height, grid_width, n_classes]
            targets:    bouding boxes from dataloader
            scaled_anchors: values of the pre-defined anchores [n_anchors , 2 (w & h)]
            ignore_thresh: threshold value to ignore iou between targets and anchors

        OUTPUTS
            iou_scores:
            class_pred: Class prediction by batch
            obj:   Object location
            no_obj:   Not detected positions to zero for larger iou
            bb_x:       Box center y by patch
            bb_y:       Box center x by patch
            bb_w:       Box width by patch
            bb_h:       Box height by patch
            tcls:       One-hot encoding per classes
            tconf:      obj.float()
    """
    batch_size  = pred_boxes.size(0)
    num_anchors = pred_boxes.size(1)
    grid_size   = pred_boxes.size(2)
    num_classes = pred_cls.size(-1)

	# Object location by patch
    obj = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.uint8)
    no_obj = torch.ones((batch_size, num_anchors, grid_size, grid_size), dtype=torch.uint8)
    # Bouding boxes
    iou_scores = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_x = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_y = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_h = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    bb_w = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)
    # One-hot encoding per classes
    tcls = torch.zeros((batch_size, num_anchors, grid_size, grid_size, num_classes), dtype=torch.float32)
    # Class prediction by batch
    class_pred = torch.zeros((batch_size, num_anchors, grid_size, grid_size), dtype=torch.float32)

    # Relative position on patch/box
    target_boxes = target[:, 2:6] * grid_size
    patch_xy = target_boxes[:, :2]
    patch_wh = target_boxes[:, 2:]

    # Compute the anchors with the best iou with the targets
    ious_list = []
    for anchor in anchors:
        w1, h1 = anchor[0], anchor[1]
        wh2 = patch_wh.t()
        w2, h2 = wh2[0], wh2[1]
        r_iou = relative_bbox_iou(w1, h1, w2, h2)
        ious_list.append(r_iou)
    ious = torch.stack(ious_list)
    # Get max iou
    best_ious, best_n = ious.max(0)

    # Separate target values
    batch_idx, target_labels = target[:, :2].long().t()
    patch_x, patch_y = patch_xy.t()
    patch_w, patch_h = patch_wh.t()
    patch_i, patch_j = patch_xy.long().t()

    # Set object (patch) positions
    obj[batch_idx, best_n, patch_j, patch_i] = 1
    no_obj[batch_idx, best_n, patch_j, patch_i] = 0

    # Set the not detected positions to zero for larger iou
    for i, anchor_ious in enumerate(ious.t()):
        anchor_thresh = anchor_ious > ignore_thresh
        no_obj[batch_idx[i], anchor_thresh, patch_j[i], patch_i[i]] = 0

    # Bouding boxes center
    bb_x[batch_idx, best_n, patch_j, patch_i] = patch_x - patch_x.floor()
    bb_y[batch_idx, best_n, patch_j, patch_i] = patch_y - patch_y.floor()
    # Bouding boxes width and height
    bb_w[batch_idx, best_n, patch_j, patch_i] = torch.log(patch_w / anchors[best_n][:, 0] + 1e-16)
    bb_h[batch_idx, best_n, patch_j, patch_i] = torch.log(patch_h / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of classes
    tcls[batch_idx, best_n, patch_j, patch_i, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_pred[batch_idx, best_n, patch_j, patch_i] = (
        pred_cls[batch_idx, best_n, patch_j, patch_i].argmax(-1) == target_labels
                                            ).float()
    iou_scores[batch_idx, best_n, patch_j, patch_i] = bbox_iou(
                                                pred_boxes[batch_idx, best_n, patch_j, patch_i],
                                                target_boxes
                                                )
    tconf = obj.float()
    return iou_scores, class_pred, obj, no_obj, bb_x, bb_y, bb_w, bb_h, tcls, tconf
