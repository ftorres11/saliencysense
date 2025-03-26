# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True

# In-package imports
from lib.data import imagenet_tester, INet_Evaluator

from models import model_selection

# Package imports
import os
osp = os.path
osj = osp.join

import argparse
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
    parser.add_argument('--path_data', default=osj('data',
                        'imagenet_val_2012_raw.csv'), type=str,
                        help='Path to csv file with data and annotations')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Images per batch')
    parser.add_argument('--model', default='resnet50', type=str,
                        help='Model to Evaluate')
    parser.add_argument('--store_dir', default='Evaluation', type=str,
                        help='Where to store the saliency maps')
    # Parsing Arguments
    args = parser.parse_args()
    # ====================================================================
    # Checking parsed arguments
    args.store_dir = osj(args.store_dir)
    # Empty Lists to store probabilities
    labs_gt = []; labs_pred = []; labs_least = []
    probs_gt = []; probs_pred = []; probs_least = []
    fnames = []
    
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)

    # Checking for devices
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    # ====================================================================
    # Dataloading
    with open(args.path_data, 'r') as fil:
        data = fil.readlines()
    transform = imagenet_tester(256, 224)
    experimental_dataset = INet_Evaluator(args.root_data, data, transform)
    # Model Selection, Targetting and Wrapping.
    model = model_selection(args.model)
    model = model.to(args.device).eval()

    # Loading data
    loaded = DataLoader(experimental_dataset, batch_size=args.batch_size,
                        num_workers=0, shuffle=False)
    softmax = nn.Softmax(dim=-1)
    for images, labels, names in loaded:
        # Retrieving the batch
        images = images.to(args.device)
        labels = labels.to(args.device)
        # Forward through the model to get logits
        outputs = model(images)
        probs = softmax(outputs)
        preds = torch.argmax(probs, dim=-1)
        worst = torch.argmin(probs, dim=-1)
        predicted_probs = probs[torch.arange(len(probs)), preds].detach()
        groundtruth_probs = probs[torch.arange(len(probs)), labels].detach()
        worst_probs = probs[torch.arange(len(probs)), worst].detach()
        # ================================================================
        # Appending to lists
        fnames.extend(names)
        # Groundtruth
        labs_gt.extend(labels.cpu().numpy().tolist())
        probs_gt.extend(groundtruth_probs.cpu().numpy().tolist())
        # Predictions
        labs_pred.extend(preds.cpu().numpy().tolist())
        probs_pred.extend(predicted_probs.cpu().numpy().tolist())
        # Least probable
        labs_least.extend(worst.cpu().numpy().tolist())
        probs_least.extend(worst_probs.cpu().numpy().tolist())

    # ====================================================================
    # Complete Diagnostic Storage
    with open(osj(args.store_dir, 'complete_info.json'), 'w') as data:
        data_dict = {}
        data_dict['Filenames'] = fnames
        # Groundtruth - info
        data_dict['GroundtruthLabels'] = labs_gt
        data_dict['GroundtruthProbs'] = probs_gt
        # Predicted - info
        data_dict['PredictedLabels'] = labs_pred
        data_dict['PredictedProbs'] = probs_pred
        # Least-likely  - info
        data_dict['WorstLabels'] = labs_least
        data_dict['WorstProbs'] = probs_least
        # Storing
        json.dump(data_dict, data, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    main()

