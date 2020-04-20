# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:31:10 2020
@author: Diego Wanderley
@python: 3.6
@description: Read CNN torch states dictionary and weights.
"""

import os
import csv
import torch
import argparse

#from models.rcnn import FasterRCNN
#from models.yolo import Darknet


def read_state_dict(state):

    header = [  'epoch',
                'batch_size',
                'optimizer',
                'best_loss_train',
                'best_ap_val',
                'val_precision',
                'val_recall',
                'val_mAP',
                'val_f1',
                'device'  ]
    row  = [ '', '', '', '', '','', '', '', '', '' ]

    # Read state dict
    for key, value in state.items():
        # Epoch of saving
        if key == 'epoch':
            row[0] = value
        if key == 'batch_size':
            row[1] = value
        # Optimizer used
        if key == 'optimizer':
            row[2] = value.split(" ")[0]
        # Train loss score these weights
        if key == 'best_loss_train':
            row[3] = value
        # Validation score for these weights
        if key == 'best_ap_val':
            row[4] = value
        if key == 'val_precision':
            row[5] = value
        # Validation score for these weights
        if key == 'val_recall':
            row[6] = value
        # Validation score for these weights
        if key == 'val_mAP':
            row[7] = value
        # Validation score for these weights
        if key == 'val_f1':
            row[8] = value
        # Device used during training
        if key == 'device':
            row[9] = value

    return header, row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_folder", type=bool, default=True, help="flag to read all weights files in a folder or a single file.")
    parser.add_argument("--folder", type=str, default="../weights", help="folder with weights file")
    #parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../weights/20200324_2047_Yolo_v3_weights.pth.tar", help="path to weights file")
    opt = parser.parse_args()

    if (opt.read_folder):
        flist = []
        fnames = []
        for r, d, f in os.walk(opt.folder):
            [ (flist.append(os.path.join(r, x)), fnames.append(x)) for x in f if '.tar' in x ]

        table = []
        for wpath, wname in zip(flist,fnames):
            # split weights name
            name_splited = wname.split("_weights.")[0].split("_")
            train_date = name_splited[0]
            train_time = name_splited[1]
            model_name = '_'.join(name_splited[2:])
            # Read weights state dict
            state = torch.load(wpath)
            header, row = read_state_dict(state)

            if len(table) == 0:
                table_header = [ "model", "date", "time" ] + header
                table.append(table_header)
                print( table_header )

            table_row = [ model_name, int(train_date), int(train_time) ] + row
            table.append(table_row)
            print( table_row )

        # Save data on a csv
        with open(opt.folder + "/state.csv", 'w', newline='') as fp:
            writer = csv.writer(fp, delimiter=';')
            writer.writerows(table)

    else:
        # Load state dictionary
        state = torch.load(opt.weights_path)
        #model.load_state_dict(state['state_dict'])
        header, row = read_state_dict(state)
        print(header)
        print(row)

    print('')