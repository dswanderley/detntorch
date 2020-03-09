# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:04:11 2019
@author: Diego Wanderley
@python: 3.6
@description: Train script with training class
"""

import tqdm
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from terminaltables import AsciiTable

import utils.transformations as tsfrm

from test_yolo import evaluate
#from models.modules import Yolo_net
from models.yolo import Darknet
from models.yolo_utils.utils import *
from utils.datasets import OvaryDataset
#from utils.logger import Logger


class Training:
    """
        Training classe
    """

    def __init__(self, model, device, train_set, valid_set, optim,
                 class_names, train_name='yolov3', logger=None):
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
        self.gradient_accumulations = 2
        self.iou_thres = 0.5
        self.conf_thres = 0.5
        self.nms_thres = 0.5
        self.metrics  = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]
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
        lotal_loss = 0
        data_train_len = len(self.train_set)

        # Active train
        self.model.train()
        self.model = self.model.to(self.device)

        # Batch iteration - Training dataset
        for batch_idx, (names, imgs, targets) in enumerate(tqdm.tqdm(data_loader, desc="Training epoch")):
            batches_done = len(data_loader) * self.epoch + batch_idx
            
            targets = Variable(targets.to(self.device), requires_grad=False)
            imgs = Variable(imgs.to(self.device))
            bs = len(imgs)
            
            # Forward and loss
            loss, output = self.model(imgs, targets=targets)
            loss.backward()

            if batches_done % self.gradient_accumulations:
                # Accumulates gradient before each step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.model.seen += imgs.size(0)

            # Log metrics at each YOLO layer
            batch_factor = bs / data_train_len
            for i, metric in enumerate(self.metrics):
                out_metrics = [(yolo.metrics.get(metric, 0) * batch_factor) for yolo in self.model.yolo_layers]
                # Fill average
                for j in range(len(self.avg_metrics[metric])):
                    self.avg_metrics[metric][j] += out_metrics[j]

            lotal_loss += loss.item() * batch_factor

        return lotal_loss


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
                                        collate_fn=self.train_set.collate_fn_yolo)
        data_loader_val = DataLoader(self.valid_set, batch_size=1, shuffle=False,
                                        collate_fn=self.valid_set.collate_fn_yolo)

        # Define parameters
        best_precision = 0    # Init best loss with a too high value

        # Run epochs
        for e in range(epochs):
            self.epoch = e
            print('Starting epoch {}/{}.'.format(self.epoch + 1, epochs))
            log_str = ''
            metric_table = [["Metrics", *["YOLO Layer " + str(i) for i in range(len(model.yolo_layers))]]]
            self.avg_metrics = { i : [0]*len(self.model.yolo_layers) for i in self.metrics }

            # ========================= Training =============================== #
            loss_train = self._iterate_train(data_loader_train)
            
            # Log metrics at each YOLO layer
            for i, metric in enumerate(self.metrics):
                formats = {m: "%.6f" for m in self.metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = self.avg_metrics[metric]
                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table
            log_str += "\nTotal loss: %0.5f"%loss_train

            print(log_str)
            print('')

            # ========================= Validation ============================= #
            precision, recall, AP, f1, ap_class = evaluate(self.model,
                                        data_loader_val,
                                        self.iou_thres,
                                        self.conf_thres,
                                        self.nms_thres,
                                        1, # batch_size
                                        self.device)
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
            if best_precision < loss_train:
                best_precision = loss_train
                # save
                self._saveweights({
                'epoch': self.epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_precision': best_precision,
                'optimizer': str(self.optimizer),
                'optimizer_dict': self.optimizer.state_dict(),
                'device': str(self.device)
                })

            # ====================== Tensorboard Logging ======================= #
            #if self.logger:
            #    self._logging(self.epoch, avg_loss_train, val_evaluation)


if __name__ == "__main__":

    from utils.helper import gettrainname

    # Input parameters
    n_epochs = 150
    batch_size = 4
    input_channels = 1
    network_name = 'Yolo_v3_tiny'
    train_name = gettrainname(network_name)

    cls_names = ['background','follicle','ovary']
    mode_config_path = 'config/yolov3-tiny.cfg'

    # Load network model
    model = Darknet(mode_config_path)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=0.005,
    #                            momentum=0.9, weight_decay=0.0005)
    
     # Set logs folder
    #logger = Logger('../logs/' + train_name + '/')

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optimizer, 
                        #logger=logger,
                        class_names=cls_names[:2],
                        train_name=train_name)
    training.train(epochs=n_epochs, batch_size=batch_size)

    print('')