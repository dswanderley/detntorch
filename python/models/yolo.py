from __future__ import division

import torch
import torch.nn as nn

import numpy as np


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def read_conv_block(block, prev_filters, index=0):
    '''

    '''
    conv_module = nn.Sequential()

    #Get the info about the layer
    activation = block["activation"]
    try:
        batch_normalize = int(block["batch_normalize"])
        bias = False
    except:
        batch_normalize = 0
        bias = True

    filters= int(block["filters"])
    padding = int(block["pad"])
    kernel_size = int(block["size"])
    stride = int(block["stride"])

    if padding:
        pad = (kernel_size - 1) // 2
    else:
        pad = 0

    #Add the convolutional layer
    conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
    conv_module.add_module("conv_{0}".format(index), conv)

    #Add the Batch Norm Layer
    if batch_normalize:
        bn = nn.BatchNorm2d(filters)
        conv_module.add_module("batch_norm_{0}".format(index), bn)

    #Check the activation.
    #It is either Linear or a Leaky ReLU for YOLO
    if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace = True)
        conv_module.add_module("leaky_{0}".format(index), activn)

    return conv_module


def create_modules(blocks):
    '''
    The create_modules function takes a list blocks returned by the parse_cfg function.
    '''
    net_info = blocks[0]     #Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []


    for index, x in enumerate(blocks[1:]):
        module = None

        if (x["type"] == "convolutional"):

            module = read_conv_block(x, prev_filters, index=index)


        if type(module) == nn.Sequential():
            module_list.append(module)

    print('')



# Main calls
if __name__ == '__main__':

    cfgfile = "./configs/yolov3.cfg"
    blocks = parse_cfg(cfgfile)
    create_modules(blocks)

    print('')