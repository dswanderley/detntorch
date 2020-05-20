# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:51:01 2019
@author: Diego Wanderley
@python: 3.7
@description: Compare all models predictions with groud truth dataset.
"""

import os
import csv
import copy
import torch
import numpy as np
from models.yolo_utils.utils import bbox_iou, ap_per_class

# Classes names
class_names = ['background','follicle','ovary']

# Files paths
dataset_file = '../datasets/ovarian/data.csv'
predict_path = '../predictions'
result_file = 'results.csv'

prediction_files = [ os.path.join(r,result_file)  for r, d, f in os.walk(predict_path) if result_file in f ]


def point_in_box(p, box):
    px, py = p
    bx1, by1, bx2, by2 = box

    if px < bx1 or px > bx2:
        return False
    elif py < by1 or py > by2:
        return False
    else:
        return True

# dataset	filename	class	x1	y1	x2	y2	xc	yc	w	h

# Faster RCNN
# fname	img_idx	bb_idx	labels	scores	x1	y1	x2	y2

# Retinanet
# fname	img_idx	bb_idx	labels	scores	x1	y1	x2	y2

# Yolo
# fname	img_idx	bb_idx	cls_pred	cls_conf	conf	x1	y1	x2	y2

# Load dataset
with open(dataset_file) as f:
    dataset = [ { k: v for k, v in row.items() }
        for row in csv.DictReader(f, delimiter=';') ]
# Creates a list of filenames in test dataset
testset_names = [ gt_data['filename'] for gt_data in dataset if gt_data['dataset'] == 'test' ]
testset_names = list(set(testset_names))
# Creates a dictionary with the filenames
testset = { fname: [] for fname in testset_names }
groundtruth = copy.deepcopy(testset)
[ groundtruth[t_data['filename']].append(t_data) for t_data in dataset if t_data['filename'] in testset_names ]

models_eval = []

# Read Detections
for pfile in prediction_files:
    model_name = pfile.split('\\')[-2]
    predictions = copy.deepcopy(testset) # Prediction dictionary / detection by filename
    
    # Load detections
    with open(pfile) as f:
        detections = [ { k: v for k, v in row.items() }
            for row in csv.DictReader(f, delimiter=';') ]
    lbl_key = 'cls_pred' if 'cls_pred' in detections[0] else 'labels'

    # Convert list of detection to a list of dictionary by filename
    [ predictions[p_data['fname']].append(p_data) for p_data in detections if p_data['fname'] in testset_names ]
    comparison_items = []
    comparison_ovral = []

    # Set of data for all images evaluates with this model
    model_true_positives = []
    model_pred_scores = []
    model_pred_labels = []
    model_tgt_labels = []
    model_num_tgt = []
    model_num_dtn = []
    model_num_fn = []
    model_num_duplicate = []
    
    # Iterate filenames
    for fname in testset_names:
        # Get list of detections and ground truth from the image
        detect = predictions[fname]
        gtruth = groundtruth[fname]
        # Temp storages
        iou_list = []
        gt_id_list = []
        pred_tp_list = []
        pred_scores_list = []
        pred_lbls_list = []
        gt_lbls_list = [ class_names.index(gt['class']) for gt in gtruth ]

        # Iterate image detections 
        for d in range(len(detect)):
            dtn = detect[d]
            score = dtn['scores'] if 'scores' in dtn else dtn['cls_conf']
            box_dt = torch.tensor([ float(dtn['x1']), float(dtn['y1']), float(dtn['x2']), float(dtn['y2']) ])
            box_center = ( (box_dt[0] + box_dt[2]) / 2, (box_dt[1] + box_dt[3]) / 2)
            best_iou = 0
            best_gt_id = -1

            # Iterate image GTs
            for g in range(len(gtruth)):
                gt = gtruth[g]
                # Check class to compute IoU if possible
                if dtn[lbl_key] == gt['class']:                    
                    box_gt = torch.tensor([ float(gt['x1']),  float(gt['y1']),  float(gt['x2']),  float(gt['y2']) ])
                    # Calculate IoU
                    iou = bbox_iou(box_dt.unsqueeze(0), box_gt.unsqueeze(0))
                    # Store if the is the best
                    if iou.item() > best_iou:
                        best_iou = iou.item()
                        if point_in_box(box_center, box_gt):
                            best_gt_id = g
            
            # Add comparison to data
            dtn['iou'] = best_iou
            dtn['gt_idx'] = best_gt_id + 1
            if best_gt_id < 0:
                dtn['status'] = 'FP'
                pred_tp_list.append(0)
            else:
                dtn['status'] = 'TP'
                pred_tp_list.append(1)
            # Append to be compared
            gt_id_list.append(best_gt_id)
            iou_list.append(best_iou)
            pred_scores_list.append(float(score))
            pred_lbls_list.append(class_names.index(dtn[lbl_key]))
            
        
    
        # Check for duplicated (when NMS fail)
        duplicate = []
        for i in range(len(gt_id_list)):
            for j in range(len(gt_id_list)):
                if gt_id_list[i] == gt_id_list[j] and gt_id_list[i] >= 0 and i != j:
                    if iou_list[j] > iou_list[i]:
                        duplicate.append(i)
                        break

        for d in reversed(duplicate):
            detect[d]['status']='DP'
            del pred_tp_list[d]
            del gt_id_list[d]
            del pred_scores_list[d]
            del pred_lbls_list[d]

        # Image prediction evaluation
        precision, recall, AP, f1, ap_class = ap_per_class( np.array(pred_tp_list), 
                                                            np.array(pred_scores_list), 
                                                            np.array(pred_lbls_list), 
                                                            gt_lbls_list )
            
        # False positives, false negatives and true positives
        false_negative = [ gt_i for gt_i in range(len(gtruth)) if gt_i not in gt_id_list ]
        false_positive = [ dt_i for dt_i in range(len(gt_id_list)) if gt_id_list[dt_i] < 0 ]
        true_positive =  [ gt_i for gt_i in range(len(gtruth)) if gt_i in list(set(gt_id_list)) and gt_i ]
        # Append to count
        model_num_tgt.append(len(gtruth))
        model_num_dtn.append(len(detect))
        model_num_fn.append(len(false_negative))
        model_num_duplicate.append(len(duplicate))

        # Dictionary with general data
        overall_dict = {
            'filename': fname,
            'num_gt': len(gtruth),
            'num_detn': len(detect),
            'num_fn': len(false_negative),
            'num_fp': len(false_positive),
            'num_tp': len(true_positive),
            'num_duplicates': len(duplicate),
            'fn': false_negative,
            'fp': false_positive,
            'tp': true_positive,
            'duplicates': duplicate,
            'precision_mean':precision.mean(),
            'recall_mean':recall.mean(),
            'ap_mean':AP.mean(),
            'f1_mean':f1.mean()
        }
        for i in range(len(ap_class)):
            overall_dict['precision_' + str(i+1)] = precision[i]
            overall_dict['recall_' + str(i+1)] = recall[i]
            overall_dict['ap_' + str(i+1)] = AP[i]
            overall_dict['f1_' + str(i+1)] = f1[i]

        # Store on a list with all detections of all files
        comparison_items += detect
        comparison_ovral.append(overall_dict)
        # Add to mode evaluation list
        model_true_positives += pred_tp_list
        model_pred_scores    += pred_scores_list
        model_pred_labels    += pred_lbls_list
        model_tgt_labels     += gt_lbls_list

    # Evaluate model
    precision, recall, AP, f1, ap_class = ap_per_class( np.array(model_true_positives), 
                                                        np.array(model_pred_scores), 
                                                        np.array(model_pred_labels), 
                                                        model_tgt_labels )
    overall_dict = {
        'filename': 'model',
        'num_gt': sum(model_num_tgt),
        'num_detn': sum(model_num_dtn),
        'num_fn': sum(model_num_fn),
        'num_fp':  model_true_positives.count(0),
        'num_tp': model_true_positives.count(1),
        'num_duplicates': sum(model_num_duplicate),
        'precision_mean':precision.mean(),
        'recall_mean':recall.mean(),
        'ap_mean':AP.mean(),
        'f1_mean':f1.mean()
    }
    for i in range(len(ap_class)):
        overall_dict['precision_' + str(i+1)] = precision[i]
        overall_dict['recall_' + str(i+1)] = recall[i]
        overall_dict['ap_' + str(i+1)] = AP[i]
        overall_dict['f1_' + str(i+1)] = f1[i]
    comparison_ovral.append(overall_dict)

    # Save comparisons
    keys_i = comparison_items[0].keys()
    with open(pfile.replace('results', 'results_comparison'), 'w', newline='') as fp:
        dict_writer = csv.DictWriter(fp, keys_i, delimiter=';')
        dict_writer.writeheader()
        dict_writer.writerows(comparison_items)

    # Save genreal comparisons
    keys_g = comparison_ovral[0].keys()
    with open(pfile.replace('results', 'results_general'), 'w', newline='') as fp:
        dict_writer = csv.DictWriter(fp, keys_g, delimiter=';')
        dict_writer.writeheader()
        dict_writer.writerows(comparison_ovral)

    # Store 
    overall_dict['filename'] = model_name
    models_eval.append(overall_dict)

# Save models evaluation
keys_m = models_eval[0].keys()
with open(os.path.join(predict_path, 'models_results.csv'), 'w', newline='') as fp:
    dict_writer = csv.DictWriter(fp, keys_m, delimiter=';')
    dict_writer.writeheader()
    dict_writer.writerows(models_eval)

print('finish')
    