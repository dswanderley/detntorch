# -*- coding: utf-8 -*-
"""
Created on Wed Fev 27 18:37:04 2019

@author: Diego Wanderley
@python: 3.6
@description: Dataset loaders (images + ground truth)

"""

import os
import torch
import random

import numpy as np

from PIL import Image
from skimage import exposure, filters
from torchvision import transforms
from torch.utils.data import Dataset

from skimage.draw import polygon_perimeter
from scipy import ndimage as ndi


class OvaryDataset(Dataset):
    """
    Dataset of ovarian structures from B-mode images.
    """

    def __init__(self, im_dir='im', gt_dir='gt',
            one_hot=True, clahe=False,
            ovary_inst=False, transform=None, out_tuple=False):
        """
        Args:
            im_dir (string): Directory with all the images.
            gt_dir (string): Directory with all the masks, with the same name of
                the original images.
            one_hot (bool): Optional output encoding one-hot-encoding or gray levels.
            ovary_inst(bool, optional): Define if ovary/stroma needs to be encoded
                as an instance.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            out_tuple (bool, optional): Return a Tuple with all data or an object
                with labes - default is False.
        """
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.clahe = clahe
        self.one_hot = one_hot
        self.ovary_instance = ovary_inst
        self.out_tuple = out_tuple

         # Output encod accepts two types: one-hot-encoding or gray scale levels
        # Variable encods contains a list of each data encoding:
        # 1) Full GT mask, 2) Ovary mask, 3) Follicle mask
        if type(self.one_hot) is list:      # When a list is provided
            self.encods = []
            for i in range(4):
                if i > len(one_hot) - 1: # If provided list is lower than expected
                    self.encods.append(True)
                else:
                    self.encods.append(self.one_hot[i])
        elif type(self.one_hot) is bool:    # When a single bool is provided
            self.encods = [self.one_hot, self.one_hot, self.one_hot, self.one_hot]
        else:
            self.encods = [True, True, True, True]

        if self.out_tuple is False:
            self.encods[3] = False

        ldir_im = set(x for x in os.listdir(self.im_dir))
        ldir_gt = set(x for x in os.listdir(self.gt_dir))
        self.images_name  = [ fname for fname in list(ldir_im.intersection(ldir_gt)) if fname[0] is not '.']


    def __len__(self):
        """
            Get dataset length.
        """
        return len(self.images_name)


    def __getitem__(self, idx):
        """
            Get batch of images and related data.

            Args:
                @idx (int): file index.
            Returns:
                @sample (dict): im_name, image, gt_mask, ovary_mask,
                    follicle_mask, follicle_instances, num_follicles.
        """

        '''
            Load images
        '''
        # Image names: equal for original image and ground truth image
        im_name = self.images_name[idx]
        # Load Original Image (B-Mode)
        im_path = os.path.join(self.im_dir, im_name)    # PIL image in [0,255], 1 channel
        image = Image.open(im_path)
        # Load Ground Truth Image Image
        gt_path = os.path.join(self.gt_dir, im_name)    # PIL image in [0,255], 1 channel
        gt_im = Image.open(gt_path)

        # Apply transformations
        if self.transform:
            image, gt_im = self.transform(image, gt_im)

        '''
            Input Image preparation
        '''
        # Image to array
        im_np = np.array(image).astype(np.float32) / 255.
        if len(im_np.shape) > 2:
            im_np = im_np[:,:,0]

        '''
            Main Ground Truth preparation - Gray scale GT and Multi-channels GT
        '''
        # Grouth truth to array
        gt_np = np.array(gt_im).astype(np.float32)
        if (len(gt_np.shape) > 2):
            gt_np = gt_np[:,:,0]

        # Multi mask - background (R = 255) / ovary (G = 255) / follicle (B = 255)
        t1 = 128./2.
        t2 = 255. - t1
        # Background mask
        mask_bkgound = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_bkgound[gt_np < t1] = 1.
        # Stroma mask
        mask_stroma = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_stroma[(gt_np >= t1) & (gt_np <= t2)] = 1.

        # Follicles mask
        mask_follicle = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_follicle[gt_np > t2] = 1.

        # Main mask output
        if self.encods[0]:
            # Multi mask - background (R = 1) / ovary (G = 1) / follicle (B = 1)
            multi_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 3))
            multi_mask[...,0] = mask_bkgound
            multi_mask[...,1] = mask_stroma
            multi_mask[...,2] = mask_follicle
            gt_mask = multi_mask.astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            gt_mask = gt_np.astype(np.float32)

        '''
            Ovary Ground Truth preparation
        '''
        # Ovary mask
        mask_ovary = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_ovary[gt_np >= t1] = 1.

        # Ovarian auxiliary mask output
        if self.encods[1]:
            # Multi mask - background (R = 1) / ovary (G = 1)
            ov_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            ov_mask[...,0] = mask_bkgound
            ov_mask[...,1] = mask_ovary
            ov_mask = ov_mask.astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            ov_mask = mask_ovary.astype(np.float32)

        '''
            Follicles edge Ground Truth preparation
        '''
        # 2*Dilate - 2*Erode
        f_erode = ndi.morphology.binary_erosion(mask_follicle)
        f_erode = ndi.morphology.binary_erosion(f_erode).astype(np.float32)
        f_dilate = ndi.morphology.binary_dilation(mask_follicle)
        f_dilate = ndi.morphology.binary_dilation(f_dilate).astype(np.float32)
        mask_edges = f_dilate - f_erode

        # Follicle auxiliary mask output
        if self.encods[2]:
            # Multi mask - background (R = 1) / follicle (G = 1)
            # follicle mask
            fol_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            fol_mask[...,1] = mask_follicle
            fol_mask[...,0] = 1. - fol_mask[...,1]
            fol_mask = (fol_mask).astype(np.float32)
            # final edge
            fol_edge = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            fol_edge[...,0] = 1. - mask_edges
            fol_edge[...,1] = mask_edges
            fol_edge = (fol_edge).astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            fol_mask = mask_follicle.astype(np.float32)
            fol_edge = mask_edges.astype(np.float32)

        '''
            Instance Follicles mask
        '''
        # Get mask labeling each follicle from 1 to N value.
        mask_inst, num_inst = ndi.label(mask_follicle)
        # If ovary is treated as an instance
        if self.ovary_instance:
            mask_inst  = mask_inst + mask_ovary
            num_inst += 1

        # Instance masks output
        masks = np.zeros((mask_inst.shape[0], mask_inst.shape[1], num_inst))
        for i in range(num_inst):
            aux = np.zeros((mask_inst.shape[0], mask_inst.shape[1]))
            aux[mask_inst == i+1] = 1
            masks[...,i] = aux
        #if encods[3]:
        #    inst_mask = masks
        #else:
        inst_mask = mask_inst.astype(np.float32)

        '''
            Instance Bouding Boxes
        '''
        boxes = []
        labels = []
        # Follicles boxes
        for i in range(0,num_inst):
            if (self.ovary_instance and i==0):
                labels.append(2)
            else:
                labels.append(1)
            slice_y, slice_x = ndi.find_objects(mask_inst==i+1)[0]
            box = [ float(slice_x.start),
                    float(slice_y.start),
                    float(slice_x.stop),
                    float(slice_y.stop)]
            boxes.append(box)

        '''
            Input data: Add CLAHE if necessary
        '''
        # Check has clahe
        if self.clahe:
            if len(im_np.shape) == 2:
                im_np = im_np.reshape(im_np.shape+(1,))
            imclahe = np.zeros((im_np.shape[0], im_np.shape[1], 1))
            imclahe[...,0] = exposure.equalize_adapthist(im_np[...,0], kernel_size=im_np.shape[0]/8,
                            clip_limit=0.02, nbins=256)
            im_np = np.concatenate((imclahe, im_np), axis=2).astype(np.float32)

        # Print data if necessary
        #Image.fromarray((255*gt_mask).astype(np.uint8)).save("gt_mask.png")

        # Convert to torch (to be used on DataLoader)
        torch_im = torch.from_numpy(im_np)
        if len(torch_im.shape) > 2:
            torch_im = torch_im.permute(2, 0, 1).contiguous()

        torch_gt = torch.from_numpy(gt_mask)
        if len(torch_gt.shape) > 2:
            torch_gt = torch_gt.permute(2, 0, 1).contiguous()

        torch_ov = torch.from_numpy(ov_mask)
        if len(torch_ov.shape) > 2:
            torch_ov = torch_ov.permute(2, 0, 1).contiguous()

        torch_fol = torch.from_numpy(fol_mask)
        if len(torch_fol.shape) > 2:
            torch_fol = torch_fol.permute(2, 0, 1).contiguous()

        torch_edge = torch.from_numpy(fol_edge)
        if len(torch_edge.shape) > 2:
            torch_edge = torch_edge.permute(2, 0, 1).contiguous()

        torch_is = torch.from_numpy(inst_mask)
        if len(torch_is.shape) > 2:
            torch_is = torch_is.permute(2, 0, 1).contiguous()

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        masks = torch.from_numpy(masks)

        # Return tuple
        if self.out_tuple:
            return  im_name, torch_im, \
                    torch_gt, torch_ov, torch_fol, torch_edge, \
                    torch_is, num_inst, boxes, labels, masks

        # Return tensors
        else:
            sample =  { 'im_name': im_name,
                        'image': torch_im,
                        'gt_mask': torch_gt,
                        'ovary_mask': torch_ov,
                        'follicle_mask': torch_fol,
                        'follicle_edge': torch_edge,
                        'follicle_instances': torch_is,
                        'num_follicles':  num_inst,
                        'targets': {
                                    'boxes': boxes,
                                    'labels': labels,
                                    'masks': masks
                                }
                        }
            return sample


    def collate_fn_list(self, batch):
        '''
            Merges a list of samples to form a mini-batch
            dictionary for OvaryDataset.
        '''
        el_list = []
        for b in batch:
            el_list.append(
                {
                    'im_name': b[0],
                    'image': b[1],
                    'gt_mask': b[2],
                    'ovary_mask': b[3],
                    'follicle_mask': b[4],
                    'follicle_edge': b[5],
                    'follicle_instances': b[6],
                    'num_follicles':  b[7],
                    'targets': {
                                'boxes':  b[8],
                                'labels': b[9],
                                'masks':  b[10]
                            }
                }
            )
        return el_list


    def collate_fn_rcnn(self, batch):
        '''
            Merges a list of samples to form a faster rcnn batch (coco based)

            Output: list of dict
                - im_name:    tuple with filenames
                - image:     tensor - size (batch_size, channels, height, width)
                - targets:  dict - {labels, boxes: [x1, y1, x2, y2]}

        '''
        batch_list = []
        for b in batch:
            batch_list.append(
                {
                    'im_name': b[0],
                    'image': b[1],
                    'targets': {
                                'boxes': b[8],
                                'labels': b[9]
                            }
                }
            )
        return batch_list


    def collate_fn_yolo(self, batch):
        '''
        Merges a list of samples to form a YOLO batch.

        Outputs
            - names:    tuple with filenames
            - imgs:     tensor - size (batch_size, channels, height, width)
            - targets:  tensor - [batch_index, class, xc, yc, h, w]
        '''
        batch_list = list(zip(*batch))
        names = batch_list[0]
        imgs  = batch_list[1]
        tgts = batch_list[8]
        lbls = batch_list[9]
        targets = []

        # Add sample index to targets
        for i, (lbl, tgt) in enumerate(zip(lbls, tgts)):

            im_height, im_width = imgs[i].shape[-2], imgs[i].shape[-1] # [channels, height, width]
            # Compute bouding boxes on YOLO style
            center_x = (tgt[:,0] + tgt[:,2]) / 2 / im_width
            center_y = (tgt[:,1] + tgt[:,3]) / 2 / im_height
            obj_width = (tgt[:,2] - tgt[:,0]) / im_width
            obj_height = (tgt[:,3] - tgt[:,1]) / im_height
            # Add target data to the same array
            boxes = torch.zeros((len(tgt), 6))
            boxes[:,0] = i             # index of image on batch
            boxes[:,1] = lbl           # classes
            boxes[:,2] = center_x      # bouding boxes
            boxes[:,3] = center_y      # bouding boxes
            boxes[:,4] = obj_width     # bouding boxes
            boxes[:,5] = obj_height    # bouding boxes
            targets.append(boxes)
        # Convert list to a single tensor
        targets = torch.cat(targets, 0) # [batch, class, cx, cy, w, h]

        # images to input shape
        imgs = torch.stack([img if (len(img.shape) > 2) else img.unsqueeze_(0) for img in imgs])

        return names, imgs, targets


def printBoudingBoxes(img, bboxes, score=None, lbl=None):
    '''
        Print bouding boxes over image and return a RGB numpy.
    '''

    # Check image type
    if type(img) == torch.Tensor or type(img) == torch.tensor:
        im_np = img.permute(1,2,0).data.numpy()
    else:
        im_np = img

    # Grayscale to RGB
    if im_np.shape[2] == 1:
        im_np = np.tile(im_np,(1,1,3))

    # Check if has no bouding box
    if len(bboxes) > 0:
        # Get Detected Bouding Boxes
        for i in range(len(bboxes)):
            bb = bboxes[i]
            x1 = round(bb[0].item())
            x2 = round(bb[2].item())
            y1 = round(bb[1].item())
            y2 = round(bb[3].item())
            # pred scores
            if score is None:
                dtn = bb[4]
            else:
                dtn = score[i]
            dtn = dtn.item()
            # Class
            if lbl is None:
                _, cla = torch.max(bb[5:],0)
            else:
                cla = lbl[i]
            cla = cla.item()
            # Get rectangle
            rr, cc = polygon_perimeter([y1, y1, y2, y2],
                                    [x1, x2, x2, x1],
                                    shape=im_np.shape, clip=True)
            # Write rectangle on
            im_np[rr, cc, cla] = dtn

    return im_np


# Main calls
if __name__ == '__main__':

    import os
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from torch.utils.data import DataLoader

    classes = ['background','follicle','ovary']
    colors = ['w', 'm', 'y']
    path_im = '../datasets/ovarian/im/test/'
    path_gt = path_im.replace('/im/', '/gt/')

    # pre-set
    dataset = OvaryDataset(im_dir=path_im,
                           gt_dir=path_gt,
                           clahe=False, transform=False,
                           ovary_inst=True,
                           out_tuple=True)
    # Loader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True,
                                    collate_fn=dataset.collate_fn_rcnn)
    # iterate
    for _, (fnames, imgs, targets) in enumerate(data_loader):
        # Load data
        '''
        images = list(s['image'] for s in sample)

        targets = [s['targets'] for s in sample]
        '''
        img_size = imgs.shape[-1]

        # read each image in batch
        for i in range(len(imgs)):

            filename = fnames[i]

            img = imgs[i]
            img_np = img.permute(1,2,0).numpy()

            # Create plot
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img_np[:,:,0])

            for box in targets:

                # Check batch index
                if i == int(box[0]):
                    cl = box[1]
                    x1 = (box[2]-box[4]/2).item()*img_size
                    y1 = (box[3]-box[5]/2).item()*img_size
                    box_w = box[4].item()*img_size
                    box_h = box[5].item()*img_size
                    bbox = patches.Rectangle((x1, y1), box_w, box_h,
                            linewidth=2, edgecolor=colors[int(cl)], facecolor="none")

                    # add box to plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cl)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": colors[int(cl)], "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())

            os.makedirs("output", exist_ok=True)
            plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()

    print('')

