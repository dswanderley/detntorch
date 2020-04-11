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


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, score, class_pred)
    """
    output = [{'boxes':None, 'labels':None, 'scores':None} for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        boxes = pred['boxes']
        labels = pred['labels'].unsqueeze(1)
        scores = pred['scores'].unsqueeze(1)
        image_pred = torch.cat((boxes, scores, labels.float()), 1)
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        #class_confs, class_preds = image_pred[:, 4:].max(1, keepdim=True)
        detections = image_pred
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
            output[image_i]['boxes'] = torch.stack(keep_boxes)[:,:4]
            output[image_i]['labels'] = torch.stack(keep_boxes)[:,-1]
            output[image_i]['scores'] = torch.stack(keep_boxes)[:,4:-1]

    return output



def batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        if outputs[sample_i]['boxes'] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output['boxes'] 
        pred_scores = output['scores'] if len(output['scores'].shape) == 1 else output['scores'][:,0]
        pred_labels = output['labels'] if len(output['labels'].shape) == 1 else output['labels'][:,0]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[sample_i]
        target_labels = annotations['labels']
        target_boxes = annotations['boxes']

        if len(target_boxes):
            detected_boxes = []
            
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(pred_boxes):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
    return batch_metrics


def evaluate(model, data_loader, batch_size, device, save_bb=False):
    """
        Evaluate model
    """
    iou_thres = 0.5
    # To evaluate on validation set
    model.eval()
    model = model.to(device)

    sample_metrics = [] # List of (TP, confs, pred)
    labels = []         # to recieve targets

    # Batch iteration - Validation dataset
    for batch_idx, (names, imgs, targets) in enumerate(data_loader):

        # Get images and targets
        images = torch.stack(imgs).to(device)
        # Labels
        for tgt in targets:
            labels += tgt['labels'].tolist()
        # Targets
        targets = [{ 'boxes':  tgt['boxes'].to(device),'labels': tgt['labels'].to(device) } 
                    for tgt in targets]

        # Run prediction
        with torch.no_grad():
            outputs = model(images)
            outputs = non_max_suppression(outputs) # Removes detections with lower score 

        sample_metrics += batch_statistics(outputs,
                                targets,
                                iou_threshold=iou_thres) # [true_positives, pred_scores, pred_labels]

        # Save images if needed
        if save_bb:
            for i in range(len(names)):
                im_name = names[i]
                im = imgs[i]
                out_bb = outputs[i]
                # Get RGB image with BB
                im_np = printBoudingBoxes(im, out_bb['boxes'], 
                                            lbl=out_bb['labels'], 
                                            score=out_bb['scores'])
                # Save image
                Image.fromarray((255*im_np).astype(np.uint8)).save('../predictions/faster_rcnn/' + im_name)

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

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    
    from terminaltables import AsciiTable

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    n_classes = 2
    class_names = ['background','follicle']
    weights_path  = "../weights/20200324_1957_faster_rcnn_weights.pth.tar"#None

    # Dataset
    dataset = OvaryDataset(im_dir='../datasets/ovarian/im/test/',
                           gt_dir='../datasets/ovarian/gt/test/',
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)
    data_loader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            collate_fn=dataset.collate_fn_rcnn)
                            
    # Initiate model
    model = FasterRCNN(num_channels=1, num_classes=n_classes, pretrained=True).to(device)
    if weights_path is not None:
        # Load state dictionary
        state = torch.load(weights_path)
        model.load_state_dict(state['state_dict'])

    # Eval
    precision, recall, AP, f1, ap_class  = evaluate(model,
                                data_loader,
                                4,
                                device=device,
                                save_bb=True)

    # Group metrics
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]

     # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print("mAP: "+ str(AP.mean()))
    print('\n')