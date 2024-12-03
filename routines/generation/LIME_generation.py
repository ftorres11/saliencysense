# -*- coding: utf -8 -*-
# Author: Felipe Torres Figueroa
# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True

# Jacob-gil imports

# Captum imports
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import (SkLearnLinearRegression,
                                               SkLearnLasso)
from captum.attr._core.lime import get_exp_kernel_similarity_function

# Nat LIME imports
from lime.wrappers.scikit_image import SegmentationAlgorithm

# In-package imports
from lib.data import imagenet_tester, INet_Evaluator

from models import model_selection

# Package imports
import os
osp = os.path
osj = osp.join

import argparse
import numpy as np
from sklearn.utils import check_random_state
import sys
epsilon = sys.float_info.epsilon


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
    parser.add_argument('--fraction', default=None, type=float,
                        help='Fraction of the dataset to generate?')
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed for splitting data')
    # Model Initialization
    parser.add_argument('--model', default='resnet50', type=str,
                        help='Model to Evaluate')
    parser.add_argument('--store_dir', default='SaliencyMaps', type=str,
                        help='Where to store the saliency maps')
    parser.add_argument('--lab', default='groundtruth', type=str,
                        help='Do we compute saliency maps for gt or pred?')
    # Parsing Arguments
    args = parser.parse_args()
    # ====================================================================
    # Checking parsed arguments
    prediction = args.lab
    args.method = 'LIME'
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

    # LIME defn and hparams
    exp_eucl_dist = get_exp_kernel_similarity_function('euclidean',
                        kernel_width=1000)
    LIME = Lime(model, interpretable_model=SkLearnLasso(alpha=0.08),
                similarity_func=exp_eucl_dist)
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
           max_dist=200, ratio=0.2,
          random_seed=check_random_state(args.seed).randint(0, high=1000))
    
    # Loading data
    loaded = DataLoader(experimental_dataset, batch_size=1,
                        num_workers=0, shuffle=False)
    softmax = nn.Softmax(dim=-1)
    for images, labels, names in loaded:
        # Retrieving the batch
        images = images.to(torch.cuda.current_device())
        bsz, _, wdth, hght = images.shape
        labels = labels.to(torch.cuda.current_device())
        # Forward through the model to get logits
        outputs = model(images)
        probs = softmax(outputs)
        worst = torch.argmin(probs, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        # Bizarre Override. Adapted to use groundtruth labels as in repo.
        if args.lab == 'predicted':
            targets = preds.cpu().numpy()
        elif args.lab == 'least':
            targets = worst.cpu().numpy()
        else:
            targets = labels.cpu().numpy()
        # LIME preparations
        img = images.cpu().numpy()
        img = np.transpose(img, (0, 2, 3, 1))
        segments = segmentation_fn(img[0])
        segments = torch.from_numpy(segments).unsqueeze(0).to(args.device)
        n_interpret_features = len(segments.unique())
        # LIME Forwarding
        saliency = LIME.attribute(images, target=int(targets[0]), 
                       feature_mask=segments, n_samples=40,
                       perturbations_per_eval=16, show_progress=True)
        # Deprocessing LIME saliency
        saliency = saliency.pow(2).sum(axis=1).sqrt().squeeze().detach()
        saliency = saliency.cpu().numpy()
        saliency = (saliency-saliency.min())/\
                   (saliency.max()-saliency.min()+epsilon)
        name = names[0].replace('.JPEG', '') # Name of the image
        np.save(osj(args.store_dir, '{}.npy').format(name), saliency)
    # ====================================================================

if __name__ == '__main__':
    main()

