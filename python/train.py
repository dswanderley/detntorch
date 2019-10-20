# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 13:04:11 2019
@author: Diego Wanderley
@python: 3.6
@description: Train script with training class
"""

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

import utils.transformations as tsfrm

from models.modules import Yolo_v3
from models.utils import non_max_suppression, central_to_corners_coord
from utils.datasets import OvaryDataset
from utils.logger import Logger

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ResourceWarning)


class Training:
    """
        Training classe
    """

    def __init__(self, model, device, train_set, valid_set, optim,
                 train_name='yolov3', logger=None):
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
        for batch_idx, (names, imgs, targets) in enumerate(data_loader):

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # Forward and loss
            loss, outputs = self.model(imgs, targets)
            loss.backward()
            # Optmize
            self.optimizer.zero_grad()
            self.optimizer.step()

            # Update epoch loss
            loss_train_sum += len(imgs) * loss.item()

        # Calculate average loss per epoch
        avg_loss_train = loss_train_sum / data_train_len

        return avg_loss_train


    def _iterate_val(self, data_loader):

        # Init loss count
        loss_val_sum = 0
        data_val_len = len(self.valid_set)

        # To evaluate on validation set
        self.model.eval()
        self.model = self.model.to(self.device)

        sample_metrics = []  # List of (TP, confs, pred)

        # Batch iteration - Validation dataset
        for batch_idx, (names, imgs, targets) in enumerate(data_loader):
            # Images
            imgs = Variable(imgs.to(self.device), requires_grad=False)
            img_size = imgs.shape(-1)
            batch_size = imgs.shape(0)
            # Labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = central_to_corners_coord(targets[:, 2:])
            targets[:, 2:] *= img_size

            with torch.no_grad():
                outputs = self.model(imgs)
                outputs = non_max_suppression(outputs,
                                            conf_thres=self.conf_thres,
                                                nms_thres=self.nms_thres)

            sample_metrics += get_batch_statistics(outputs,
                                                    targets,
                                                    iou_threshold=self.iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]

        return evaluation_metrics


    def _logging(self, epoch, avg_loss_train, val_evaluation):

        # 1. Log scalar values (scalar summary)
        info = val_evaluation
        info.append(('avg_loss_train', avg_loss_train))
        for tag, value in info.items():
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
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

            # ========================= Training =============================== #
            avg_loss_train = self._iterate_train(data_loader_train)
            print('training loss:  {:f}'.format(avg_loss_train))

            # ========================= Validation ============================= #
            val_evaluation = self._iterate_val(data_loader_val)
            val_precision = val_evaluation[0]
            print(val_precision[0] + ': {:f}'.format(val_precision[1]))
            print('')

            # ======================== Save weights ============================ #
            if best_precision < val_precision[0]:
                best_precision = val_precision[0]
                # save
                self._saveweights({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_precision': best_precision,
                'optimizer': str(self.optimizer),
                'optimizer_dict': self.optimizer.state_dict(),
                'device': str(self.device)
                })

            # ====================== Tensorboard Logging ======================= #
            if self.logger:
                self._logging(epoch, avg_loss_train, val_evaluation)


if __name__ == "__main__":

    from utils.aux import gettrainname

    # Input parameters
    n_epochs = 500
    batch_size = 4
    input_channels = 1
    network_name = 'Yolo_v3'
    train_name = gettrainname(network_name)

    # Load network model
    model = Yolo_v3(input_channels)

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
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)
    dataset_val = OvaryDataset(im_dir='../datasets/ovarian/im/val/',
                           gt_dir='../datasets/ovarian/gt/val/',
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)

    # Optmization
    optmizer = optim.Adam(model.parameters(), lr=0.001)


     # Set logs folder
    logger = Logger('../logs/' + train_name + '/')

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optmizer, logger=logger, train_name=train_name)
    training.train(epochs=n_epochs, batch_size=batch_size)

    print('')