# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:03:15 2019
@author: Diego Wanderley
@python: 3.6
@description: Train script for fast r-cnn
"""

import math
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.transformations as tsfrm

from test import evaluate
from models.rcnn import FasterRCNN
from utils.datasets import OvaryDataset
from utils.logger import Logger
from utils.helper import reduce_dict


class Training:
    """
        Training classe
    """

    def __init__(self, model, device, train_set, valid_set, optim,
                 train_name='fast_rcnn', logger=None):
        '''
            Training class - Constructor
        '''
        self.model = model
        self.device = device
        self.train_set = train_set
        self.valid_set = valid_set
        self.optimizer = optim
        self.train_name = train_name
        self.logger = logger
        self.epoch = 0


    def _saveweights(self, state):
        '''
            Save network weights.
            Arguments:
            @state (dict): parameters of the network
        '''
        path = '../weights/'
        filename = path + self.train_name + '_weights.pth.tar'
        torch.save(state, filename)


    def _iterate_train(self, data_loader):

        # Init loss count
        loss_train_sum = 0
        data_train_len = len(self.train_set)

        # Active train
        self.model.train()
        self.model = self.model.to(self.device)

        # Batch iteration - Training dataset
        for batch_idx, samples in enumerate(tqdm(data_loader, desc="Training epoch")):
            batches_done = len(data_loader) * self.epoch + batch_idx

            # Get data size
            bs = len(samples)
            if len(samples[0]['image'].shape) < 3:
                h, w = samples[0]['image'].shape
                ch = 1
            else:
                ch, h, w = samples[0]['image'].shape

            # Get images and targets
            images = torch.zeros(bs, ch, h, w)
            targets = []
            for i in range(bs):
                images[i] = samples[i]['image'].to(self.device)
                targets.append(
                    {
                        'boxes':  samples[i]['targets']['boxes'].to(self.device),
                        'labels': samples[i]['targets']['labels'].to(self.device)
                    }
                )

            # Forward and loss
            loss_dict = self.model(images, targets)

            # Compute loss
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            # Test if valid to continue
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        return loss_value


    def train(self, epochs=100, batch_size=4):
        '''
        Train network function
        Arguments:
            @param net: network model
            @param epochs: number of training epochs (int)
            @param batch_size: batch size (int)
        '''

        # Load Dataset
        data_loader_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                        collate_fn=self.train_set.collate_fn_list)
        data_loader_val = DataLoader(self.valid_set, batch_size=1, shuffle=False,
                                        collate_fn=self.valid_set.collate_fn_list)

        # Define parameters
        best_precision = 0    # Init best loss with a too high value

        # Run epochs
        for e in range(epochs):
            self.epoch = e
            print('Starting epoch {}/{}.'.format(self.epoch + 1, epochs))

            # ========================= Training =============================== #
            avg_loss_train = self._iterate_train(data_loader_train)
            print('training loss:  {:f}'.format(avg_loss_train))



if __name__ == "__main__":

    from utils.aux import gettrainname

    # Input parameters
    n_epochs = 500
    batch_size = 4
    input_channels = 1
    network_name = 'fast_rcnn'
    train_name = gettrainname(network_name)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load network model
    model = FasterRCNN(n_channels=input_channels, pretrained=True).to(device)

    # Transformation parameters
    transform = tsfrm.Compose([tsfrm.RandomHorizontalFlip(p=0.5),
                           tsfrm.RandomVerticalFlip(p=0.5),
                           tsfrm.RandomAffine(90, translate=(0.15, 0.15),
                            scale=(0.75, 1.5), resample=3, fillcolor=0)
                           ])

    # Dataset definitions
    dataset_train = OvaryDataset(im_dir='../datasets/ovarian/im/train/',
                           gt_dir='../datasets/ovarian/gt/train/',
                           clahe=False, transform=transform,
                           ovary_inst=False,
                           out_tuple=True)
    dataset_val = OvaryDataset(im_dir='../datasets/ovarian/im/val/',
                           gt_dir='../datasets/ovarian/gt/val/',
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)

    # Optmization
    optimizer = optim.Adam(model.parameters(), lr=0.001)


     # Set logs folder
    logger = Logger('../logs/' + train_name + '/')

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optimizer, logger=logger, train_name=train_name)
    training.train(epochs=n_epochs, batch_size=batch_size)

    print('')