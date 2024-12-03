# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True

# In-package imports
from lib.data import imagenet_tester, INet_Evaluator
from lib.RISE import RISEBatch

from models import model_selection

# Package imports
import os
osp = os.path
osj = osp.join

import sys
epsilon = sys.float_info.epsilon
import argparse
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
    parser.add_argument('--method', default='RISE', type=str,
                        help='Saliency Approach')
    parser.add_argument('--store_dir', default='SaliencyMaps', type=str,
                        help='Where to store the saliency maps')
    parser.add_argument('--lab', default='groundtruth', type=str,
                        help='Do we compute saliency maps for gt or pred?')
    # Parsing Arguments
    args = parser.parse_args()
    # ====================================================================
    # Checking parsed arguments
    prediction = args.lab
    args.store_dir = osj(args.store_dir, args.method, prediction)
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

    transform = imagenet_tester(256, 224)
    experimental_dataset = INet_Evaluator(args.root_data, data, transform)
    # Model Selection, Targetting and Wrapping.
    model = model_selection(args.model)
    model = model.to(args.device).eval()
    # Selecting attributions
    explainer = RISEBatch(model, (224, 224), args.batch_size)
    explainer.generate_masks(N=32, s=4, p1=0.1, savepath=args.store_dir)
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
        worst = torch.argmin(probs, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        # Using selected labels
        if args.lab == 'predicted':
            targets = preds
        elif args.lab == 'least':
            targets = worst
        else:
            targets = labels

        # Forwarding through the explainer
        salient = explainer(images)
        salient_min = salient.amin(dim=(2, 3), keepdim=True)
        salient_max = salient.amax(dim=(2, 3), keepdim=True)
        salient = (salient-salient_min)/(salient_max-salient_min+epsilon)
        for idx in range(len(salient)):
            saliency = salient[idx, targets[idx], :, :] # Storing index
            saliency = saliency.detach().cpu().numpy()
            name = names[idx].replace('.JPEG', '') # Name of the image
            np.save(osj(args.store_dir, '{}.npy').format(name), saliency)
        del salient, saliency
    # ====================================================================

if __name__ == '__main__':
    main()

