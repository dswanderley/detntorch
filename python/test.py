# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:23:12 2019
@author: Diego Wanderley
@python: 3.6
@description: Test script with evaluate function.
"""

import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from models.modules import Yolo_net
from models.yolo import Darknet
from models.utils import *
from utils.datasets import OvaryDataset
from utils.logger import Logger


def evaluate(model, data_loader, iou_thres, conf_thres, nms_thres, batch_size, device):
    """
        Evaluate model
    """
    # To evaluate on validation set
    model.eval()
    model = model.to(device)

    sample_metrics = [] # List of (TP, confs, pred)
    labels = []         # to recieve targets

    # Batch iteration - Validation dataset
    for batch_idx, (names, imgs, targets) in enumerate(data_loader):
        # Images
        imgs = Variable(imgs.to(device), requires_grad=False)
        img_size = imgs.shape[-1]
        batch_size = imgs.shape[0]
        # Labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = central_to_corners_coord(targets[:, 2:])
        targets[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs,
                                        conf_thres=conf_thres,
                                        nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs,
                                                targets,
                                                iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]

    return evaluation_metrics