import os
import csv
import copy
import torch

from models.yolo_utils.utils import bbox_iou


dataset_file = '../datasets/ovarian/data.csv'
predict_path = '../predictions'
result_file = 'results.csv'

prediction_files = [ os.path.join(r,result_file)  for r, d, f in os.walk(predict_path) if result_file in f ]

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

# Read Detections
for pfile in prediction_files:
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
    
    # Iterate filenames
    for fname in testset_names:
        # Get list of detections and ground truth from the image
        detect = predictions[fname]
        gtruth = groundtruth[fname]
        # Temp storages
        iou_list = []
        gt_id_list = []
        
        # Iterate image detections 
        for d in range(len(detect)):
            dtn = detect[d]
            box_dt = torch.tensor([ float(dtn['x1']), float(dtn['y1']), float(dtn['x2']), float(dtn['y2']) ])
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
                        best_gt_id = g
            
            # Add comparison to data
            dtn['iou'] = best_iou
            dtn['gt_idx'] = best_gt_id
            if best_gt_id < 0:
                dtn['status'] = 'FP'
            else:
                dtn['status'] = 'TP'
            # Append to be compared
            gt_id_list.append(best_gt_id)
            iou_list.append(best_iou)

        # Check for duplicated (when NMS fail)
        duplicate = []
        for i in range(len(gt_id_list)-1):
            for j in range(i+1,len(gt_id_list)):
                if gt_id_list[i] == gt_id_list[j] and gt_id_list[i] >= 0:
                    if iou_list[i] < iou_list[j]:
                        duplicate.append(j)
                    else:
                        duplicate.append(i)
        # False positives, false negatives and true positives
        false_negative = [ gt_i for gt_i in range(len(gtruth)) if gt_i not in gt_id_list ]
        false_positive = [ dt_i for dt_i in range(len(detect)) if gt_id_list[dt_i] < 0 ]
        true_positive =  [ gt_i for gt_i in range(len(gtruth)) if gt_i in list(set(gt_id_list)) ]
                
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
            'duplicates': duplicate
        }

        # Store on a list with all detections of all files
        comparison_items += detect
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

print('finish')
    