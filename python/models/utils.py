# -*- coding: utf-8 -*-
"""
Created on Wed Oct 02 21:34:01 2019
@author: Diego Wanderley
@python: 3.6
@description: YOLO modules auxiliary functions.
"""

import torch
import warnings

import numpy as np

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ResourceWarning)


def central_to_corners_coord(x):
    '''
        From (center x, center y, width, height) to (x1, y1, x2, y2)
    '''
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def get_pred_boxes(pred, anchors):
    '''
        Get outputs predictions and pred_boxes.
    '''
    # Anchors parameters
    num_anchors = len(anchors)
    anchor_w = anchors[:, 0:1].view((1, num_anchors, 1, 1))
    anchor_h = anchors[:, 1:2].view((1, num_anchors, 1, 1))

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
    pred_boxes[..., 2] = torch.exp(pred_w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(pred_h.data) * anchor_h

    return pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls, pred_boxes


def relative_bbox_iou(w1, h1, w2, h2):
    """
    Compute the IOU given the width and the height of two bouding boxes
    """
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
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
                                                target_boxes,
                                                x1y1x2y2=False
                                                )
    tconf = obj.float()
    return iou_scores, class_pred, obj, no_obj, bb_x, bb_y, bb_w, bb_h, tcls, tconf


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = central_to_corners_coord(prediction[..., :4])

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

