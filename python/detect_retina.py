# -*- coding: utf-8 -*-
"""
Created on Sun Mar 01 18:32:30 2020
@author: Diego Wanderley
@python: 3.6
@description: Script to detect objects from faster rcnn.
"""

import os
import sys
import csv
import time
import datetime
import argparse
import torch

import torch.optim as optim
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.retinanet import RetinaNet
from models.yolo_utils.utils import *
from utils.datasets import *
from test_retina import non_max_suppression, batch_statistics

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Network parameters
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--num_channels", type=int, default=1, help="number of channels in the input images")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes (including background)")
    parser.add_argument("--weights_path", type=str, default="../weights/20200511_1817_retinanet_weights.pth.tar", help="path to weights file")
    # Evaluation parameters
    parser.add_argument("--score_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--save_img", type=bool, default=False, help="save images with bouding box")

    opt = parser.parse_args()
    print(opt)

    # Classes names
    class_names = ['background','follicle','ovary']

    # Get data configuration
    n_classes = opt.num_classes
    weights_path  = opt.weights_path
    im_path = '../datasets/ovarian/im/test/'
    gt_path = '../datasets/ovarian/gt/test/'
    # Get network name
    networkname = weights_path.split('/')[-1]
    networkname = networkname.split('.')[0]
    if ('_weights' in networkname):
        networkname = networkname.replace('_weights', '')

    # Dataset
    dataset = OvaryDataset(im_dir=im_path,
                           gt_dir=gt_path,
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)

    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            collate_fn=dataset.collate_fn_rcnn)

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = RetinaNet(in_channels=opt.num_channels, num_classes=n_classes, pretrained=True).to(device)
    if weights_path is not None:
        # Load state dictionary
        state = torch.load(weights_path)
        model.load_state_dict(state['state_dict'])
    
    model.eval()

    img_names = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    table = [] # Table of content
    table.append(['fname', 'img_idx', 'bb_idx', 'labels', 'scores', 'x1', 'y1', 'x2', 'y2'])

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (names, imgs, targets) in enumerate(dataloader):
                
        # Get images and targets
        images = torch.stack(imgs).to(device)

        targets = [{ 'boxes':  tgt['boxes'].to(device),'labels': tgt['labels'].to(device) } 
                    for tgt in targets]

        # Get detections
        with torch.no_grad():
            detections = model(images)
            detections = non_max_suppression(detections, opt.score_thres, opt.nms_thres) # Removes detections with lower score 

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        img_names.extend(names)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Save images stage
    print("\nSaving images:")
    outfolder = "../predictions/retina/" + networkname + '/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # Iterate through images and save plot of detections
    for img_i, (fname, detections) in enumerate(zip(img_names, img_detections)):

        full_path = os.path.join(im_path, fname)
        print("(%d) Image: '%s'" % (img_i, full_path))

        # Create plot
        img = np.array(Image.open(full_path).convert('LA'))
        if len(img.shape) == 3:
            img = img[:,:,0]
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections['boxes'] is not None:
            unique_labels = detections['labels'].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for idx, (x1, y1, x2, y2) in enumerate(detections['boxes'].cpu()):

                cls_conf = detections['scores'][idx].cpu()

                if cls_conf > 0.:
                    cls_pred = detections['labels'][idx].cpu()
                    # Add data to table
                    table.append([fname, str(img_i + 1), str(idx + 1), 
                                class_names[int(cls_pred.item())], str(cls_conf.item()), 
                                str(x1.item()), str(y1.item()), str(x2.item()), str(y2.item())])

                    print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))

                    # Prepare box to print
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=class_names[int(cls_pred)][0]+str(idx+1),
                        color=color,
                        verticalalignment="top",
                        bbox={"color":"white", "alpha":.1, "pad": 0}
                    )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(outfolder + fname, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    # Save results on a ;csv
    with open(outfolder + "results.csv", 'w', newline='') as fp:
        writer = csv.writer(fp, delimiter=';')
        writer.writerows(table)