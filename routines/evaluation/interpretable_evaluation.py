# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
torch.use_deterministic_algorithms(True)

# In-package imports
from evaluation.ins_del import auc, gkern, CausalMetric
from evaluation.object_recognition import gradcampp_recognition
from lib.data import imagenet_tester, CompleteEvaluator
from lib.data import im_normalization as in_norm
from lib.data import im_denormalization as in_denorm
from lib.utils import AverageMeter
from models import model_selection

# Package imports
import os
osp = os.path
osj = osp.join

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
    parser.add_argument('--root_data', default=osj('imagenet_2012_cr',
                        'validation', 'val'),
                        type=str, help='Path to data root')
    parser.add_argument('--path_data', default='Diagnostics',
                        help='Path where data dict is stored')
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
        pure_data = json.load(dump)

    transform = imagenet_tester(256, 224)
    args.path_saliency = osj(args.path_saliency, args.method, args.lab)
    experimental_dataset = CompleteEvaluator(pure_data, args.root_data,
                                             args.path_saliency, args.lab,
                                             transform)
    loaded = DataLoader(experimental_dataset, batch_size=args.batch_size,
                        num_workers=0, shuffle=False)

    # ====================================================================
    # Model Selection, Targetting and Wrapping.
    model = model_selection(args.model)
    model = model.to(args.device).eval()
    # ====================================================================
    # Preparations for iteration
    softmax = nn.Softmax(dim=-1)
    klen = 5; ksig = 5 # Ins-Del kernels
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern.to(args.device),
                                          padding=2)
    insert = CausalMetric(model, 'ins', 224*8, blur, 224, 1000)
    delete = CausalMetric(model, 'del', 224*8, torch.zeros_like, 224, 
                          1000)
    # Storing Variables
    fnames = []
    drops = []; gains = []; incrs = []
    ins = []; dels = []

    # ====================================================================
    # Iteration through the dataset
    for batch in loaded:
        # Unpacking the batch
        names = batch[0]
        images = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        base_probs = batch[3].to(args.device) # Precomputed logits
        smap = batch[4]
        # Preparing for computation
        if smap.shape[-1] == 3:
            smap = smap.mean(dim=-1)
        elif smap.shape[-1] == 1:
            smap = smap[:, :, :, 0]
        smap = smap.unsqueeze(1).to(args.device)

        # Preparing for computation
        # Denormalizing
        denorm = in_denorm(images)
        exp_map = in_norm(smap*denorm) # Normalized
        exp_map = exp_map.type(images.type()) # Ensuring typecasting
        # Fwd, get probs
        log_map = model(exp_map)
        map_probs = softmax(log_map)
        map_probs = map_probs[torch.arange(len(map_probs)), labels].detach()
        smap = smap.squeeze(1).detach().cpu().numpy()
        # Computing metrics.
        ad_i, ic_i, ag_i = gradcampp_recognition(base_probs,
                                                 map_probs)
        ins_i = []
        del_i = []
        ins_pr = insert.evaluate(images, smap, images.shape[0], labels)
        [ins_i.append(auc(ins_pr[:,x])) for x in range(ins_pr.shape[1])]
        ins_i = torch.tensor(ins_i).cpu().numpy().tolist()

        del_pr = delete.evaluate(images, smap, images.shape[0], labels)
        [del_i.append(auc(del_pr[:,x])) for x in range(del_pr.shape[1])]
        del_i= torch.tensor(del_i).cpu().numpy().tolist()
        # Appending to storing lists
        fnames.extend(names); drops.extend(ad_i); gains.extend(ag_i)
        incrs.extend(ic_i); ins.extend(ins_i); dels.extend(del_i)

    dict_pure = {}

    for x in range(len(fnames)):
        # Element wise diagnostics
        dict_pure[fnames[x]] = {'Drop': str(drops[x]),
                                'Gain': str(gains[x]),
                                'Increase': str(incrs[x]),
                                'Insertion': str(ins[x]),
                                'Deletion': str(dels[x])}
                      
    # Summary diagnostics                     
    dict_summary = {'Drop': str(np.asarray(drops).mean()), 
                    'Gain': str(np.asarray(gains).mean()),
                    'Increase': str(np.asarray(incrs).mean()),
                    'Insertion': str(np.asarray(ins).mean()),
                    'Deletion': str(np.asarray(dels).mean())}


    path_element = osj(args.store_dir, '{}_per_element.json'.format(\
                       args.method))
    path_summary = osj(args.store_dir, '{}_summary.json'.format(args.method))

    with open(path_element, 'w') as dump:
        json.dump(dict_pure, dump, ensure_ascii=False, indent=4)

    with open(path_summary, 'w') as dump:
        json.dump(dict_summary, dump, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

