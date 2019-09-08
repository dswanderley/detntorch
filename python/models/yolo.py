from __future__ import division

import torch
import torch.nn as nn

import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


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


def get_conv_layer(block, prev_filters, index=0):
    '''
    Get a convolutional block from configuration file and returns a pytorch layer.
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

    filters = int(block["filters"])
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

    return conv_module, filters


def get_upsampling_layer(block, index=0):
    '''
    Get a upsampling block from configuration file and returns a pytorch layer.
    '''
    up_module = nn.Sequential()
    # Get data
    stride = int(block["stride"])
    # Create layer
    upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
    up_module.add_module("upsample_{}".format(index), upsample)

    return up_module


def get_route_layer(block, index=0):
    '''
    Get a route block from configuration file and returns a pytorch layer.
    '''
    route_module = nn.Sequential()

    block["layers"] = block["layers"].split(',')
    #Start  of a route
    start = int(block["layers"][0])
    #end, if there exists one.
    try:
        end = int(block["layers"][1])
    except:
        end = 0
    #Positive anotation
    if start > 0:
        start = start - index
    if end > 0:
        end = end - index
    # Create empty layer for route layer
    route = EmptyLayer()
    route_module.add_module("route_{0}".format(index), route)

    return route_module, start, end


def get_shortcut_layer(block, index=0):
    '''
    Get a Shortcut block from configuration file and returns a pytorch layer.
    '''
    module = nn.Sequential()
    shortcut = EmptyLayer()
    module.add_module("shortcut_{}".format(index), shortcut)

    return module


def get_yolo_layer(block, index=0):
    '''
    Get a YOLO block from configuration file and returns a pytorch layer.
    '''
    dtn_module = nn.Sequential()
    # Read mask data
    mask = block["mask"].split(",")
    mask = [int(x) for x in mask]
    # Read anchors data
    anchors = block["anchors"].split(",")
    anchors = [int(a) for a in anchors]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
    anchors = [anchors[i] for i in mask]
    # init detection layer class
    detection = DetectionLayer(anchors)
    dtn_module.add_module("Detection_{}".format(index), detection)

    return dtn_module


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

        if (x["type"] == "net"):
            continue

        # Conv layer
        if (x["type"] == "convolutional"):
            module, filters = get_conv_layer(x, prev_filters, index=index)
        # Upsampling layer
        elif (x["type"] == "upsample"):
            module = get_upsampling_layer(x, index=index)
        # Route layer
        elif (x["type"] == "route"):
            module, start, end = get_route_layer(x, index=index)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        # Shortcut or skip connection
        elif x["type"] == "shortcut":
            module = get_shortcut_layer(x, index=index)
        # Yolo / detection layer
        elif x["type"] == "yolo":
            module = get_yolo_layer(x, index=index)

        # Add to module list
        if type(module) != None:
            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.hyperparams, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer

        write = 0     #This is explained a bit later
        for i, (module_def, module) in enumerate(zip(modules, self.module_list)):
            # Get layer type
            module_type = (module_def["type"])

            # Simple forward
            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
                outputs[i] = x

            # Concatenations
            elif module_type == "route":
                layers = module_def["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

                outputs[i] = x

            # Skip connections
            elif  module_type == "shortcut":
                from_ = int(module_def["from"])
                x = outputs[i-1] + outputs[i+from_]

                outputs[i] = x





# Main calls
if __name__ == '__main__':

    cfgfile = "./configs/yolov3.cfg"
    darknet = Darknet(cfgfile)


    # Images
    images = torch.randn(2, 3, 416, 416)
    darknet(images)

    print('')