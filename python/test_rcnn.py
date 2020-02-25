# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:23:12 2019
@author: Diego Wanderley
@python: 3.6
@description: Test script with evaluate function.
"""

import torch
import argparse
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image

from models.rcnn import FasterRCNN
from models.yolo_utils.utils import *
from utils.datasets import OvaryDataset, printBoudingBoxes
from utils.logger import Logger


def evaluate(model, data_loader, batch_size, device, save_bb=False):
    """
        Evaluate model
    """
    iou_threshold = 0.5
    # To evaluate on validation set
    model.eval()
    model = model.to(device)

    sample_metrics = [] # List of (TP, confs, pred)
    labels = []         # to recieve targets

    # Batch iteration - Validation dataset
    for batch_idx, samples in enumerate(data_loader):

        # Get data size
        bs = len(samples)
        if len(samples[0]['image'].shape) < 3:
            h, w = samples[0]['image'].shape
            ch = 1
        else:
            ch, h, w = samples[0]['image'].shape

        # Get images and targets
        images = torch.zeros(bs, ch, h, w)
        targets = []
        names = []
        for i in range(bs):
            images[i] = samples[i]['image'].to(device)
            targets.append(
                {
                    'boxes':  samples[i]['targets']['boxes'].to(device),
                    'labels': samples[i]['targets']['labels'].to(device)
                }
            )
            names.append(samples[i]['im_name'])

        # Run prediction
        with torch.no_grad():
            outputs = model(images.to(device))

        # [true_positives, pred_scores, pred_labels]
        batch_metrics = []
        for j in range(bs):
            out_pred = outputs[j]
            target_boxes = targets[i]['boxes']
            target_labels = targets[i]['labels']

            labels += target_labels.tolist()

            pred_boxes = out_pred['boxes']
            pred_labels = out_pred['labels']
            pred_scores = out_pred['scores']

            true_positives = np.zeros(pred_boxes.shape[0])
            detected_boxes = []

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(target_boxes):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

            batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])

        # Save images if needed
        if save_bb:
            im_name = names[j]
            im = images[j]
            # Get RGB image with BB
            im_np = printBoudingBoxes(im, pred_boxes, lbl=pred_labels, score=pred_scores)
            # Save image
            Image.fromarray((255*im_np).astype(np.uint8)).save('../predictions/faster_rcnn/' + im_name)

    sample_metrics += batch_metrics

    # Protect in case of no object detected
    if len(sample_metrics) == 0:
        precision = np.array([0])
        recall = np.array([0])
        AP = np.array([0])
        f1 = np.array([0])
        ap_class = np.array([0], dtype=np.int)
    else:
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Group metrics
    evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]

    return evaluation_metrics, ap_class


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    gt_path = "data/ovarian/gt/"
    im_path = "data/ovarian/im/"
    classes=2
    class_names = ['background','follicle']

    # Dataset
    dataset = OvaryDataset(im_dir='../datasets/ovarian/im/test/',
                           gt_dir='../datasets/ovarian/gt/test/',
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)
    data_loader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            collate_fn=dataset.collate_fn_list)

    # Initiate model
    model = FasterRCNN(n_channels=1, pretrained=True).to(device)

    # Eval
    evaluation_metrics, ap_class = evaluate(model,
                                data_loader,
                                4,
                                device=device,
                                save_bb=True)

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print("+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print("mAP: {AP.mean()}")