# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True


# In-package imports
from lib.data import imagenet_tester, INet_Evaluator
from lib.IBA import IBA

from models import model_selection
from models.utils import cam_targetter

# Package imports
import os
osp = os.path
osj = osp.join

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
    parser.add_argument('--method', default='IBA', type=str,
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
    target = cam_targetter(model)
    # Loading data
    loaded = DataLoader(experimental_dataset, batch_size=args.batch_size,
                        num_workers=0, shuffle=False)
    iba = IBA(target[0])
    iba.estimate(model, loaded, n_samples=len(experimental_dataset), progbar=True)
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

        if args.lab == 'groundtruth':
            targets = labels
        elif args.lab == 'predicted':
            targets = preds
        elif args.lab == 'least':
            targets = worst
        
        model_loss_closure = lambda x: -torch.log_softmax(model(x), dim=1)\
                             [:, targets].mean()
        salient = iba.analyze(images, model_loss_closure, beta=10)
        salient = (salient-salient.min())/(salient.max()-salient.min())
        salient = torch.from_numpy(salient).unsqueeze(dim=0)
        # Needs min-max normalization
        for idx in range(len(salient)):
            saliency = salient[idx] # Storing index
            name = names[idx].replace('.JPEG', '') # Name of the image
            np.save(osj(args.store_dir, '{}.npy').format(name), saliency)
    # ====================================================================

if __name__ == '__main__':
    main()

