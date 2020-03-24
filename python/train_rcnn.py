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
from terminaltables import AsciiTable

import utils.transformations as tsfrm

from test_rcnn import evaluate
from models.rcnn import FasterRCNN
from utils.datasets import OvaryDataset
#from utils.logger import Logger
from utils.helper import reduce_dict


class Training:
    """
        Training classe
    """

    def __init__(self, model, device, train_set, valid_set, optim, class_names,
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
        self.class_names = class_names
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
        for batch_idx, (names, imgs, targets) in enumerate(tqdm(data_loader, desc="Training epoch")):
            batches_done = len(data_loader) * self.epoch + batch_idx

            # Get images and targets
            if self.model.num_channels == 3:
                images = [img.to(self.device) for img in imgs]
            else:
                images = torch.stack(imgs).to(self.device)

            targets = [{ 'boxes':  tgt['boxes'].to(self.device),'labels': tgt['labels'].to(self.device) } 
                        for tgt in targets]

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
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        return loss_value


    def _logging(self, epoch, avg_loss_train, val_evaluation):

        # 1. Log scalar values (scalar summary)
        info = val_evaluation
        info.append(('avg_loss_train', avg_loss_train))
        for tag, value in info:
            self.logger.scalar_summary(tag, value, epoch+1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
            if not value.grad is None:
                self.logger.histo_summary(tag +'/grad', value.grad.data.cpu().numpy(), epoch+1)


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
                                        collate_fn=self.train_set.collate_fn_rcnn)
        data_loader_val = DataLoader(self.valid_set, batch_size=1, shuffle=False,
                                        collate_fn=self.valid_set.collate_fn_rcnn)

        # Define parameters
        best_loss = 1000000    # Init best loss with a too high value
        best_ap = 0            # Init best average precision as zero

        # Run epochs
        for e in range(epochs):
            self.epoch = e
            print('Starting epoch {}/{}.'.format(self.epoch + 1, epochs))

            # ========================= Training =============================== #
            avg_loss_train = self._iterate_train(data_loader_train)
            print('Training loss:  {:f}'.format(avg_loss_train))

            # ========================= Validation ============================= #
            precision, recall, AP, f1, ap_class = evaluate(self.model,
                                            data_loader_val,
                                            1, #batch_size,
                                            device=self.device)

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
                ap_table += [[c, self.class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print("mAP: "+ str(AP.mean()))
            print('\n')

            # ======================== Save weights ============================ #
            if (avg_loss_train <= best_loss) and (AP.mean() >= best_ap):
                best_loss = avg_loss_train
                best_ap = AP.mean()
                # save
                self._saveweights({
                'epoch': self.epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss_train': best_loss,
                'best_ap_val': best_ap,
                'optimizer': str(self.optimizer),
                'optimizer_dict': self.optimizer.state_dict(),
                'device': str(self.device)
                })

            # ====================== Tensorboard Logging ======================= #
            if self.logger:
                self._logging(self.epoch, avg_loss_train, val_evaluation)



if __name__ == "__main__":

    from utils.helper import gettrainname

    # Input parameters
    n_classes = 2
    n_epochs = 150
    batch_size = 4
    input_channels = 1
    network_name = 'faster_rcnn'
    train_name = gettrainname(network_name)

    cls_names = ['background','follicle','ovary']

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load network model
    model = FasterRCNN(num_channels=input_channels, num_classes=n_classes, pretrained=True).to(device)

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
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

     # Set logs folder
    #logger = Logger('../logs/' + train_name + '/')

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optimizer, #logger=logger,
                        class_names=cls_names[:2],
                        train_name=train_name)
    training.train(epochs=n_epochs, batch_size=batch_size)

    print('')