# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:02:15 2020
@author: Diego Wanderley
@python: 3.6
@description: Read images and their ground truths and save data as a csv.
"""

import os
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from torch.utils.data import DataLoader
from utils.datasets import OvaryDataset
from matplotlib.ticker import NullLocator

# Dataset data
dataset_path = '../datasets/ovarian'
dataset_names = ['train', 'validation', 'test']
dataset_folder = ['train', 'val', 'test']
class_names = ['background','follicle','ovary']

# output table
table_header = ['dataset', 'filename', 'class',
                'x1', 'y1', 'x2', 'y2',
                'xc', 'yc', 'w', 'h' ]
data_table = [table_header]

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colormap = [cmap(i) for i in np.linspace(0, 1, 20)]
# colors = random.sample(colormap, len(dataset_names))
colors = [ colormap[1], colormap[6], colormap[14] ]

# Read datasets
for dname, fname in zip(dataset_names, dataset_folder):
    # Set paths
    path_im = dataset_path + '/im/' + fname + '/'
    path_gt = path_im.replace('/im/', '/gt/')
    # pre-set dataset
    dataset = OvaryDataset(im_dir=path_im,gt_dir=path_gt,
                        ovary_inst=False, out_tuple=True)
    # define pytorch data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=dataset.collate_fn_rcnn)
    # Iterate dataset
    for batch_idx, (names, imgs, targets) in enumerate(data_loader):
        # Iterate inside batch
        for i in range(len(names)):
            filename = names[i]
            full_path = os.path.join(path_im, filename)
            # Create plot
            img = np.array(Image.open(full_path).convert('RGB')) # Convert to RGB to save the image with original grayscale colormap
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            # Iterate bouding boxes
            for idx, (box, lbl) in enumerate(zip(targets[i]['boxes'], targets[i]['labels'])):
                label = lbl.item()
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                h = y2 - y1
                w = x2 - x1
                # Add data to table
                data_table.append([ dname, filename,
                                  class_names[label],
                                  x1, y1, x2, y2,
                                  xc, yc, w, h
                                ])
                # Plot bouding box
                color = colors[label]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=str(idx+1),
                    color=color,
                    verticalalignment="top",
                    bbox={"color":"white", "alpha":.1, "pad": 0},
                )

        # Save generated image with detections
        path_bb = path_im.replace('/im/', '/bb/')
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(path_bb + filename, bbox_inches="tight", pad_inches=0.0)
        plt.close()

# Save data on a csv
with open( dataset_path + "/data.csv", 'w', newline='') as fp:
    writer = csv.writer(fp, delimiter=';')
    writer.writerows(data_table)