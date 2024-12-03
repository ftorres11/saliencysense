# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
#import torch.hub as models
from torch.utils.data import DataLoader

from torchvision import transforms
# Jacob-gil imports

# In-package importsa
from lib.data import imagenet_tester, INet_Evaluator
from models import model_selection

# Package imports
import os
osp = os.path
osj = osp.join

import argparse
import copy
import json
import numpy as np
import pdb


# ========================================================================
def main():
    parser = argparse.ArgumentParser()
    # System Parameters
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Using GPU Acceleration?')
    # Data Initialization
    parser.add_argument('--root_data', default=osj('imagenet_2012_cr',
                        'validation', 'val'),
                        type=str, help='Path to data root')
    parser.add_argument('--path_data', default=osj('data',
                        'imagenet_val_2012_raw.csv'), type=str,
                        help='Path to csv file with data and annotations')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Images per batch')
    parser.add_argument('--fraction', default=None, type=float,
                        help='Fraction of the dataset to generate?')
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed for splitting data')
    # Model Initialization
    parser.add_argument('--model', default='resnet50', type=str,
                        help='Model to Evaluate')
    parser.add_argument('--method', default='gradcam', type=str,
                        help='Saliency Approach')
    parser.add_argument('--store_dir', default='SaliencyMaps', type=str,
                        help='Where to store the saliency maps')
    parser.add_argument('--pred', action='store_true', default=False,
                        help='Do we compute saliency maps for gt or pred?')
    # Parsing Arguments
    args = parser.parse_args()
    # ====================================================================
    # Checking parsed arguments
    root = copy.deepcopy(args.store_dir)
    prediction = 'predicted' if args.pred else 'groundtruth'
    args.store_dir = osj(args.store_dir, args.method, prediction)
    missclassifications = []
    wrongprobs = []
    corrprobs = []
    wronglabs = []
    corrlabs = []
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)

    # Checking for devices
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    # Checking for seed
    if args.seed:
        np.random.seed(args.seed)

    # ====================================================================
    # Dataloading
    with open(args.path_data, 'r') as fil:
        data = fil.readlines()
    
    # Shuffling and random permutation if choosing a split
    if args.fraction:
        mod_size = int(args.fraction*len(data))
        perm = np.random.permutation(mod_size)
        data = [data[elem] for elem in perm]

    transform = imagenet_tester(224)
    experimental_dataset = INet_Evaluator(args.root_data, data, transform)
    # Model Selection, Targetting and Wrapping.
    model = model_selection(args.model)
    model = model.to(args.device)
    # Checking for reshaping
    if 'transformer' in str(type(model)).lower():
        reshape = reshape_transform
    else:
        reshape = None
    target = cam_targetter(model)
    cam_approach = dict_cam[args.method]

    if 'transformer' in args.method: # Transformer-based methods
        cam = cam_approach(model, args.device, target, max_iter=50,
                           reshape_transform=reshape,
                           learning_rate=0.1, name_f='logit_predict',
                           name_loss='plain', name_norm='max_min')

    else: # CNN
        cam = cam_approach(model=model, target_layers=target,
                           use_cuda=args.use_gpu,
                           reshape_transform=reshape)
    
    # Loading data
    loaded = DataLoader(experimental_dataset, batch_size=args.batch_size,
                        num_workers=0, shuffle=False)
    softmax = nn.Softmax(dim=-1)
    for images, labels, names in loaded:
        # Retrieving the batch
        images = images.to(torch.cuda.current_device())
        labels = labels.to(torch.cuda.current_device())
        # Forward through the model to get logits
        outputs = model(images)
        probs = softmax(outputs)
        preds = torch.argmax(probs, dim=-1)
        predicted_probs = probs[torch.arange(len(probs)), preds].detach()
        groundtruth_probs = probs[torch.arange(len(probs)), labels].detach()
        compare = (preds==labels)*1
        incorrect_idx = torch.where(compare==0)[0].cpu().numpy()
        # Slicing to get correct, incorrect labels and predictions, alongside 
        wrong_names = (np.array(names)[incorrect_idx]).tolist()
        wrong_probs = (predicted_probs[incorrect_idx]).tolist()
        wrong_labs = preds[incorrect_idx].cpu().numpy().tolist()
        corr_probs = (groundtruth_probs[incorrect_idx]).cpu().numpy().tolist()
        corr_labs = (labels[incorrect_idx].cpu().numpy()).tolist()
        # Appending to lists
        missclassifications.extend(wrong_names) # Misclassified names
        wronglabs.extend(wrong_labs) # Predicted Labs
        wrongprobs.extend(wrong_probs) # Predicted Label probs
        corrlabs.extend(corr_labs) # Correct Labels
        corrprobs.extend(corr_probs) # Correct Label probs

        # Bizarre Override. Adapted to use groundtruth labels as in repo.
        if args.pred:
            targets = None
        else:
            targets = [ClassifierOutputTarget(label) \
                       for label in labels.tolist()]

        salient = cam(images, targets)
        for idx in range(len(salient)):
            saliency = salient[idx] # Storing index
            name = names[idx].replace('.JPEG', '') # Name of the image
            np.save(osj(args.store_dir, '{}.npy').format(name), saliency)

    with open(osj(root, 'misclassified_info.json'), 'w') as data:
        data_dict = {}
        # Storing Dictionary with data
        data_dict['Filenames'] = missclassifications
        data_dict['PredictedLabels'] = wronglabs
        data_dict['PredictedProbs'] = wrongprobs
        data_dict['GroundtruthLabels'] = corrlabs
        data_dict['GrondturthProbs'] = corrprobs

        json.dump(data_dict, data, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

