# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
torch.use_deterministic_algorithms(True)

# In-package imports
from lib.utils import AverageMeter
from lib.data import imagenet_tester, INet_Evaluator, Salient_Evaluator
from lib.data import im_normalization as in_norm
from lib.data import im_denormalization as in_denorm
from lib.score_bboxes import (BoxCoords, binarize_mask,
                              compute_saliency_metric,
                              get_image_bboxes, get_loc_scores,
                              get_rectangular_mask)

from models import model_selection
from evaluation.pointingGame import energy_point_game, standard_point_game

# Package imports
import os
osp = os.path
osj = osp.join

#import pyutils.io as io
import argparse
import copy
import json
import numpy as np


# ========================================================================
def main():
    parser = argparse.ArgumentParser()
    # System Parameters
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Using GPU Acceleration?')
    # Data Initialization
    parser.add_argument('--root_data', default=osj('/data1', 'data',
                        'corpus', 'imagenet_2012_cr', 'validation', 'val'),
                        type=str, help='Path to data root')
    parser.add_argument('--path_bboxes', default=osj('data',
                        'val_bboxes.json'), type=str,
                        help=('Path to bounding box json file?'))
    parser.add_argument('--path_data', default=osj('data',
                        '50k_sorted.csv'), type=str,
                        help='Path to csv with data and annotations')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Images per batch')
    parser.add_argument('--path_saliency', default=osj('SaliencyMaps',
                        'ImageNet', 'ResNet50'), type=str,
                        help='Path to Saliency Maps')
    parser.add_argument('--method', default='gradcam', type=str,
                        help='Saliency Attribution to compute for')
    parser.add_argument('--store_dir', default='Evaluation', type=str,
                        help='Path to store diagnostics')
    parser.add_argument('--lab', default='groundtruth', type=str,
                        help='Dow we use gt class, pred or least probable')
    # Model Initialization
    parser.add_argument('--model', default='resnet50', type=str,
                        help='Model to Evaluate')
    # Parsing Arguments
    args = parser.parse_args()
    args.store_dir = osj(args.store_dir, args.model, args.lab)
    # ====================================================================
    # Checking parsed arguments
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)

    # Checking for devices
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    # ====================================================================
    # Dataloading
    with open(args.path_data, 'r') as dump:
        pure_data = dump.readlines()
    
    transform = imagenet_tester(256, 224)
    args.path_saliency = osj(args.path_saliency, args.method, args.lab)
    experimental_dataset = INet_Evaluator(args.root_data, pure_data,
                                          transform)
    experimental_dataset = Salient_Evaluator(experimental_dataset,
                                             args.path_saliency)
    loaded = DataLoader(experimental_dataset, batch_size=args.batch_size,
                        num_workers=0, shuffle=False)
    with open(args.path_bboxes, 'r') as data:
        gt_bboxes = json.load(data)
    # ====================================================================
    # Model Selection, Targetting and Wrapping.
    model = model_selection(args.model)
    model = model.to(args.device).eval()
    # ====================================================================
    # Preparations for iteration
    fnames = []
    om_list = []
    le_list = []
    ep_list = []
    sp_list = []
    # ====================================================================
    # Iteration through the dataset
    for bi, batch in enumerate(loaded):
        # Unpacking the batch
        images = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        smap = batch[2]
        names = batch[3]
        outputs = model(images)

        preds = torch.argmax(outputs, dim=-1)
        compared = (labels==preds)*1

        # Preparing for computation
        if smap.shape[-1] == 3:
            smap = smap.mean(dim=-1)
        elif smap.shape[-1] == 1:
            smap = smap[:, :, :, 0]
        smap = smap.unsqueeze(1).to(args.device)
        box_coord_ls = [BoxCoords(0, 0, 0, 0)]*len(pure_data)        
        for idx, im_id in enumerate(names):
            om_meter = AverageMeter()
            le_meter = AverageMeter()
            gt_boxes = get_image_bboxes(bboxes_dict=gt_bboxes, path=im_id)
            # ============================================================
            # First Pointing Games
            epg = float(energy_point_game(gt_boxes, smap[idx].cpu().numpy()))
            spg = standard_point_game(gt_boxes, smap[idx].cpu().numpy())*1
            ep_list.append(epg)
            sp_list.append(spg)
            # ============================================================
            # Now for OM and LE
            binarized = binarize_mask(smap[idx])
            rectangular = torch.empty_like(binarized).to(args.device)
            m = binarized.squeeze().cpu().numpy()
            rectangular, box_coord_ls[idx] = get_rectangular_mask(m)
            f1_img = []
            iou_img = []
            bboxes = []
            for gt_box in gt_boxes:
                f1_box, iou_box = get_loc_scores(gt_box,
                                smap[idx].detach().cpu().numpy(),
                                rectangular.squeeze().detach().cpu().numpy())
                iou_img.append(iou_box)
            le_meter.update(1-np.array(iou_img).max())
            om_meter.update(1-(np.array(iou_img)).max()*compared[idx])
            le_list.append(le_meter.avg)
            om_list.append(om_meter.avg)

        fnames.extend(names); 

    dict_pure = {}

    for x in range(len(fnames)):
        # Element wise diagnostics
        dict_pure[fnames[x]] = {'EPG': str(ep_list[x]),
                                'SPG': str(sp_list[x]),
                                'OM': str(om_list[x]),
                                'LE': str(le_list[x])}
                      
    # Summary diagnostics                     
    dict_summary = {'EPG': str(np.asarray(ep_list).mean()), 
                    'SPG': str(np.asarray(sp_list).mean()),
                    'LE': str(np.asarray(om_list).mean()),
                    'OM': str(np.asarray(le_list).mean())}

    path_element = osj(args.store_dir, 'loc_{}_per_element.json'.format(\
                       args.method))
    path_summary = osj(args.store_dir, 'loc_{}_summary.json'.format(args.method))

    with open(path_element, 'w') as dump:
        json.dump(dict_pure, dump, ensure_ascii=False, indent=4)

    with open(path_summary, 'w') as dump:
        json.dump(dict_summary, dump, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

