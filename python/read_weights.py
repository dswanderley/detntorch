# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:31:10 2020
@author: Diego Wanderley
@python: 3.6
@description: Read CNN torch states dictionary and weights.
"""

import os
import csv
import copy
import torch
import argparse
import collections

#from models.rcnn import FasterRCNN
#from models.yolo import Darknet


def read_state_dict(state, _dict):

    state_dict = copy.deepcopy(_dict)
    # Read state dict
    for key, value in state.items():
        # Check if exist and assigns
        if key in state_dict:
            if key == 'optimizer':
                value = value.split(" ")[0]
            state_dict[key] = value
       
    return state_dict


def get_dict(states):
    '''
    Create a dictionary with all tags from a list of states dictionary.
    '''
    key_list = ["model", "date", "time"]
    for state in states:
        # Read state dict
        for key, val in state.items():
            if (key not in key_list) and (type(val) is not dict and type(val) is not collections.OrderedDict): 
                key_list.append(key)
    # Convert list to dict
    res_dct = {item: None for item in key_list}

    return res_dct


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

        dict_table = []
        
        states = [ torch.load(wpath) for wpath in flist ]
        states_dict = get_dict(states)
        keys = states_dict.keys()

        for wname, state in zip(fnames, states):
            # split weights name
            name_splited = wname.split("_weights.")[0].split("_")
            train_date = name_splited[0]
            train_time = name_splited[1]
            model_name = '_'.join(name_splited[2:])

            # Read weights state dict
            row = read_state_dict(state, states_dict)
            row['model'] = model_name
            row['date'] = train_date
            row['time'] = train_time

            dict_table.append(row)

        # Save data on a csv
        with open(opt.folder + "/state.csv", 'w', newline='') as fp:
            dict_writer = csv.DictWriter(fp, keys, delimiter=';')
            dict_writer.writeheader()
            dict_writer.writerows(dict_table)
            #writer = csv.writer(fp, delimiter=';')
            #writer.writerows(table)

    else:
        # Load state dictionary
        state = torch.load(opt.weights_path)
        header, row = read_state_dict(state)
        print(header)
        print(row)

    print('')