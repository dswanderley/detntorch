from __future__ import division

from models.yolo import Darknet
from models.yolo_utils.utils import *
from utils.datasets import *

import os
import sys
import csv
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Network parameters
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--weights_path", type=str, default="../weights/20200419_1419_yolov3_weights.pth.tar", help="path to weights file")
    parser.add_argument("--model_name", type=str, default="yolov3", help="newtork model file")
    # Evaluation parameters
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")

    opt = parser.parse_args()
    print(opt)

    # Classes names
    class_names = ['background','follicle','ovary']

    # Input parameters
    batch_size = opt.batch_size
    weights_path = opt.weights_path
    mode_config_path = 'config/'+ opt.model_name + '.cfg'
    # Get network name
    networkname = weights_path.split('/')[-1]
    networkname = networkname.split('.')[0]
    if ('_weights' in networkname):
        networkname = networkname.replace('_weights', '')

    # Get data configuration
    path = '../datasets/ovarian/im/test/'
    path_gt = path.replace('/im/', '/gt/')
    
    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(mode_config_path).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    elif opt.weights_path.endswith(".pth"):
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    else:
        # Load state dictionary
        state = torch.load(opt.weights_path)
        model.load_state_dict(state['state_dict'])

    model.eval()  # Set in evaluation mode

    dataset = OvaryDataset(im_dir=path,
                           gt_dir=path_gt,
                           clahe=False, transform=False,
                           ovary_inst=False,
                           out_tuple=True)

    dataloader = DataLoader(dataset, 
                        batch_size=opt.batch_size, 
                        shuffle=False,
                        collate_fn=dataset.collate_fn_yolo)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    table = [] # Table of content
    table.append(['fname', 'img_idx', 'bb_idx', 'cls_pred', 'conf', 'cls_conf', 'x1', 'y1', 'x2', 'y2'])

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs, targets) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Save images stage
    print("\nSaving images:")
    outfolder = "../predictions/yolo/" + networkname + '/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # Iterate through images and save plot of detections
    for img_i, (fname, detections) in enumerate(zip(imgs, img_detections)):

        full_path = os.path.join(path, fname)
        print("(%d) Image: '%s'" % (img_i, full_path))

        # Create plot
        img = np.array(Image.open(full_path).convert('LA'))
        if len(img.shape) == 3:
            img = img[:,:,0]
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img.shape[-1], img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for bb_idx, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                # Add data to table
                table.append([fname, str(img_i + 1), str(bb_idx + 1), 
                                class_names[int(cls_pred.item())], str(cls_conf.item()), 
                                str(x1.item()), str(y1.item()), str(x2.item()), str(y2.item())])

                print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))

                # Prepare box to print
                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=class_names[int(cls_pred)][0] + str(bb_idx + 1),
                    color=color,
                    verticalalignment="top",
                    bbox={"color":"white", "alpha":.1, "pad": 0}
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(outfolder + fname, bbox_inches="tight", pad_inches=0.0)
        plt.close()

    # Save results on a ;csv
    with open(outfolder + "results.csv", 'w', newline='') as fp:
        writer = csv.writer(fp, delimiter=';')
        writer.writerows(table)
