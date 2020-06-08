# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:23:12 2019
@author: Diego Wanderley
@python: 3.6
@description: Test script with evaluate function.
"""

import tqdm
import argparse
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

from PIL import Image

from models.modules import Yolo_net
from models.yolo import Darknet
from models.yolo_utils.utils import *

from utils.datasets import OvaryDataset, printBoudingBoxes


def evaluate(model, data_loader, iou_thres, conf_thres, nms_thres, device, save_bb=False):
    """
        Evaluate model
    """
    # To evaluate on validation set
    model.eval()
    model = model.to(device)

    sample_metrics = [] # List of (TP, confs, pred)
    labels = []         # to recieve targets

    # Batch iteration - Validation dataset
    for batch_idx, (names, imgs, targets) in enumerate(tqdm.tqdm(data_loader, desc="Validation")):
        # Images
        imgs = Variable(imgs.to(device), requires_grad=False)
        img_size = imgs.shape[-1]
        batch_size = imgs.shape[0]
        # Labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs,
                                        conf_thres=conf_thres,
                                        nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs,
                                                targets,
                                                iou_threshold=iou_thres)
        # [true_positives, pred_scores, pred_labels]

        # Save images if needed
        if save_bb:
            for i in range(batch_size):
                im_name = names[i]
                im = imgs[i]
                out_bb = outputs[i]
                # Get RGB image with BB
                im_np = printBoudingBoxes(im, out_bb)
                # Save image
                Image.fromarray((255*im_np).astype(np.uint8)).save('../predictions/yolo/' + im_name)

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


def parse_yolo_name(backbone_name, num_anchors, num_classes):
    """
    Get the .cfg filename given the Yolo v3 hyperparameters.
    """
    model_name = 'yolov3'

    if 'tiny' in backbone_name:
        model_name += '-tiny'
    elif 'spp' in backbone_name:
        model_name += '-spp'
    model_name += '_a' + str(num_anchors)
    model_name += '_c' + str(num_classes)

    return model_name


if __name__ == "__main__":

    from terminaltables import AsciiTable

    parser = argparse.ArgumentParser()
    # Network parameters
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--weights_path", type=str, default="../weights/20200512_2122_yolov3_weights.pth.tar", help="path to weights file")    
    parser.add_argument("--model_name", type=str, default="yolov3", help="name of the model definition (used to load the .cfg file)")
    parser.add_argument("--num_anchors", type=int, default=6, help="number of anchors")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    # Evaluation parameters
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--save_img", type=bool, default=False, help="save images with bouding box")

    opt = parser.parse_args()
    print(opt)

    # Classes names
    class_names = ['background','follicle','ovary']

    # Input parameters
    n_classes = opt.num_classes
    has_ovary = True if n_classes > 2 else False
    batch_size = opt.batch_size
    weights_path = opt.weights_path
    network_name = parse_yolo_name(opt.model_name, opt.num_anchors, n_classes)
    mode_config_path = 'config/'+ network_name

    # Get data configuration
    gt_path = "data/ovarian/gt/"
    im_path = "data/ovarian/im/"

    # Dataset
    dataset = OvaryDataset(im_dir='../datasets/ovarian/im/test/',
                           gt_dir='../datasets/ovarian/gt/test/',
                           clahe=False, transform=False,
                           ovary_inst=has_ovary)
    data_loader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            collate_fn=dataset.collate_fn_yolo)

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(mode_config_path).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    elif opt.weights_path.endswith(".pth"):
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    else:
        # Load state dictionary
        state = torch.load(opt.weights_path)
        model.load_state_dict(state['state_dict'])

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(model,
                                                data_loader,
                                                opt.iou_thres,
                                                opt.conf_thres,
                                                opt.nms_thres,
                                                device,
                                                save_bb=opt.save_img)
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