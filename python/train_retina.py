# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 13:10:11 2019
@author: Diego Wanderley
@python: 3.6
@description: Train script for RetinaNet.
"""

import sys
import math
import argparse
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from terminaltables import AsciiTable

import utils.transformations as tsfrm
import models.retinanet as retinanet

from test_retina import evaluate
from utils.datasets import OvaryDataset
from utils.helper import gettrainname


class Training:
    """
        Training class
    """

    def __init__(self, model, device, train_set, valid_set, optim, class_names,
                 train_name='retinanet', logger=None,
                 iou_thres=0.5, score_thres=0.05, nms_thres=0.5):
        '''
            Training class - Constructor
        '''
        self.model = model
        self.device = device
        self.train_set = train_set
        self.valid_set = valid_set
        self.optimizer = optim
        self.train_name = train_name
        self.model_name = "_".join(train_name.split('_')[2:])
        self.logger = logger
        self.class_names = class_names
        self.iou_thres = iou_thres
        self.score_thres = score_thres
        self.nms_thres = nms_thres
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
        loss_cls_sum = 0
        loss_box_sum = 0
        data_train_len = len(self.train_set)

        # Active train
        self.model.train()
        self.model.freeze_bn()

        # Batch iteration - Training dataset
        for batch_idx, data in enumerate(tqdm(data_loader, desc="Training epoch")):
            batches_done = len(data_loader) * self.epoch + batch_idx

            # Get images and targets
            images = data['img'].float().to(self.device)
            
            # Set targets
            targets = data['annot'].to(self.device)

            # Forward and loss
            self.optimizer.zero_grad()
            classification_loss, regression_loss = self.model( [ images, targets ] )

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            # Compute loss
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            # Test if valid to continue
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            # Sum ponderated batch loss 
            loss_train_sum += loss.item() * batch_size / data_train_len
            loss_cls_sum   += float(classification_loss.item()) * batch_size / data_train_len
            loss_box_sum   += float(classification_loss.item()) * batch_size / data_train_len

        return loss_train_sum, loss_cls_sum, loss_box_sum


    def _logging(self, epoch, focal_loss_train, cls_loss_train, box_loss_train, val_evaluation):

        # 1. Log scalar values (scalar summary)
        info = val_evaluation
        info.append(('train_focal_loss', focal_loss_train))
        info.append(('train_cls_loss', cls_loss_train))
        info.append(('train_box_loss', box_loss_train))
        for tag, value in info:
            self.logger.add_scalar(tag, value, epoch+1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for rcnn_tag, value in self.model.named_parameters():
            # Define tag name
            tag_parts = rcnn_tag.split('.')
            if tag_parts[0] == 'inconv':
                tag = self.model_name + '/backbone/' + '/'.join(tag_parts[1:])
            else:
                tag = self.model_name + '/' + '/'.join(tag_parts[1:])
            # Ignore bias from batch normalization
            if (not 'bn' in tag) or (not 'bias' in tag):
                # add data to histogram
                self.logger.add_histogram(tag, value.data.cpu().numpy(), epoch+1)
                # add gradient if exist
                if not value.grad is None:
                    self.logger.add_histogram(tag +'/grad', value.grad.data.cpu().numpy(), epoch+1)


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
                                        collate_fn=self.train_set.collate_fn_retina)
        data_loader_val = DataLoader(self.valid_set, batch_size=1, shuffle=False,
                                        collate_fn=self.valid_set.collate_fn_retina)

        # Define parameters
        best_loss = 1000000    # Init best loss with a too high value
        best_ap = 0            # Init best average precision as zero

        # Run epochs
        for e in range(epochs):
            self.epoch = e
            print('Starting epoch {}/{}.'.format(self.epoch + 1, epochs))

            # ========================= Training =============================== #
            avg_loss_train, loss_cls_train, loss_box_train = self._iterate_train(data_loader_train)
            print('Training loss:  {:f}'.format(avg_loss_train))
            print('')
            
            # ========================= Validation ============================= #
            precision, recall, AP, f1, ap_class = evaluate(self.model,
                                                    data_loader_val,
                                                    self.iou_thres,
                                                    self.score_thres,
                                                    self.nms_thres,
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
            best_loss = avg_loss_train if avg_loss_train <= best_loss else best_loss
            is_best = AP.mean() >= best_ap
            if is_best:
                best_ap = AP.mean()
                # save
                self._saveweights({
                    'epoch': self.epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_focal_loss': avg_loss_train,
                    'train_box_loss': loss_box_train,
                    'train_cls_loss': loss_cls_train,
                    'train_best_loss': best_loss,
                    'val_precision': precision.mean(),
                    'val_recall': recall.mean(),
                    'val_mAP': AP.mean(),
                    'val_f1': f1.mean(),
                    'batch_size': batch_size,
                    'optimizer': str(self.optimizer),
                    'optimizer_dict': self.optimizer.state_dict(),
                    'device': str(self.device),
                    'iou_thres': self.iou_thres,
                    'score_thres': self.score_thres,
                    'nms_thres': self.nms_thres
                })
                
                print('Model {:s} updated!'.format(self.train_name))
                print('\n')

            # ====================== Tensorboard Logging ======================= #
            if self.logger:
                self._logging(self.epoch, avg_loss_train, loss_cls_train, loss_box_train, evaluation_metrics)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--num_epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--num_channels", type=int, default=1, help="number of channels in the input images")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes (including background)")
    # Evaluation parameters
    parser.add_argument("--iou_thres", type=float, default=0.3, help="iou threshold required to qualify as detected")
    parser.add_argument("--score_thres", type=float, default=0.3, help="object score threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")

    opt = parser.parse_args()
    print(opt)

    # Classes names
    cls_names = ['background','follicle','ovary']

    # Input parameters
    n_classes = opt.num_classes
    n_epochs = opt.num_epochs
    batch_size = opt.batch_size
    input_channels = opt.num_channels
    network_name = 'retinanet'
    train_name = gettrainname(network_name)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load network model
    model = retinanet.resnet50(num_classes=n_classes, pretrained=True).to(device)
    #RetinaNet(in_channels=input_channels, num_classes=n_classes, pretrained=True).to(device)
    #model = torch.nn.DataParallel(model).to(device)

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
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=0.005,
    #                            momentum=0.9, weight_decay=0.0005)

    # Set logs folder
    log_dir = '../logs/' + train_name + '/'
    writer = SummaryWriter(log_dir=log_dir)

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optimizer,
                        logger=writer,
                        class_names=cls_names[:2],
                        train_name=train_name,
                        iou_thres=opt.iou_thres,
                        score_thres=opt.score_thres,
                        nms_thres=opt.nms_thres)
    training.train(epochs=n_epochs, batch_size=batch_size)

    print('')