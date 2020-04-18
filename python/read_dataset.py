# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:02:15 2020
@author: Diego Wanderley
@python: 3.6
@description: Read images and their ground truths and save data as a csv.
"""

import os
import csv
from torch.utils.data import DataLoader
from utils.datasets import OvaryDataset


dataset_path = '../datasets/ovarian/'
dataset_names = ['train', 'validation', 'test']
dataset_folder = ['train', 'val', 'test']

table_header = ['dataset', 'filename', 'class', 
                'x1', 'y1', 'x2', 'y2',
                'xc', 'yc', 'w', 'h' ]
data_table = [table_header]

for dname, fname in zip(dataset_names, dataset_folder):
    # Set paths
    path_im = dataset_path + '/im/' + fname
    path_gt = path_im.replace('/im/', '/gt/')
    # pre-set dataset
    dataset = OvaryDataset(im_dir=path_im,gt_dir=path_gt,
                        ovary_inst=False, out_tuple=True)
    # define pytorch data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=dataset.collate_fn_rcnn)
    # Iterate dataset
    for batch_idx, (names, imgs, targets) in enumerate(data_loader):
        for i in range(len(names)):
            filename = names[i]
            for box, lbl in zip(targets[i]['boxes'], targets[i]['labels']):
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
                                  label,
                                  x1, y1, x2, y2,
                                  xc, yc, w, h
                                ])

# Save data on a csv
with open( dataset_path + "data.csv", 'w', newline='') as fp:
    writer = csv.writer(fp, delimiter=';')
    writer.writerows(data_table)