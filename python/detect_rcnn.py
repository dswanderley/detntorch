# -*- coding: utf-8 -*-
"""
Created on Sun Mar 01 18:32:30 2020
@author: Diego Wanderley
@python: 3.6
@description: Script to detect objects from faster rcnn.
"""

import os
import sys
import time
import datetime
import argparse
import torch

import torch.optim as optim
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.rcnn import FasterRCNN
from models.yolo_utils.utils import *
from utils.datasets import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    n_classes = 2
    class_names = ['background','follicle']
    weights_path  = "../weights/20200308_1313_faster_rcnn_weights.pth.tar"#None
    im_path = '../datasets/ovarian/im/test/'
    gt_path = '../datasets/ovarian/gt/test/'

    # Dataset
    dataset = OvaryDataset(im_dir=im_path,
                           gt_dir=gt_path,
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=dataset.collate_fn_rcnn)
                            
    # Initiate model
    model = FasterRCNN(num_channels=1, num_classes=n_classes, pretrained=True).to(device)
    if weights_path is not None:
        # Load state dictionary
        state = torch.load(weights_path)
        model.load_state_dict(state['state_dict'])
    
    model.eval()

    img_names = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

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
            #detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

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

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (fname, detections) in enumerate(zip(img_names, img_detections)):

        full_path = os.path.join(im_path, fname)
        print("(%d) Image: '%s'" % (img_i, full_path))

        # Create plot
        img = np.array(Image.open(full_path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections['labels'].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for idx, (x1, y1, x2, y2) in enumerate(detections['boxes'].cpu()):

                cls_conf = detections['scores'][idx].cpu()

                if cls_conf > 0.:
                    cls_pred = detections['labels'][idx].cpu()

                    print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))

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
                        s=class_names[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(f"../predictions/faster_rcnn/{fname}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
