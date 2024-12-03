# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
#torch.use_deterministic_algorithms(True)

# Quantus imports
import quantus

# In-package imports
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
                        'ImageNet', 'ResNet50') , type=str,
                        help='Path to Saliency Maps')
    parser.add_argument('--method', default='gradcam', type=str,
                        help='Saliency Attribution to compute for')
    parser.add_argument('--store_dir', default='Evaluation', type=str,
                        help='Path to store diagnostics')
    parser.add_argument('--lab', default='groundtruth', type=str,
                        help='Dow we use gt class, pred or least probable')
    parser.add_argument('--explain_funct', default='Saliency', type=str,
                        help='Type of the explanation function to use')
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
    # Completeness 
    completeness = quantus.Completeness(normalise=False,
                                        disable_warnings=True)
    # Complexity
    complexity = quantus.Complexity(normalise=False)
    eff_complexity = quantus.EffectiveComplexity(normalise=False)
    # Sparseness
    sparseness = quantus.Sparseness(normalise=False)
    # Monotonicity
    monotonicity = quantus.MonotonicityCorrelation(normalise=False,
                        nr_samples=10,
                        features_in_step=224*32,
                        perturb_baseline="uniform",
                        perturb_func=quantus.perturb_func.baseline_replacement_by_indices)
    # Untested - not working
    avg_sensitivity = quantus.AvgSensitivity(abs=True, normalise=False,
                          nr_samples=10,
                          lower_bound=0.2,
                          norm_numerator=quantus.norm_func.fro_norm,
                          norm_denominator=quantus.norm_func.fro_norm,
                          perturb_func=quantus.perturb_func.uniform_noise,
                          similarity_func=quantus.similarity_func.difference
                          )
    max_sensitivity = quantus.MaxSensitivity(abs=True, normalise=False,
                          nr_samples=10,
                          lower_bound=0.2,
                          norm_numerator=quantus.norm_func.fro_norm,
                          norm_denominator=quantus.norm_func.fro_norm,
                          perturb_func=quantus.perturb_func.uniform_noise,
                          similarity_func=quantus.similarity_func.difference
                          )

    # Storing Variables
    fnames = []
    completes = []
    complexes = []
    eff_complexes = []
    monotones = []
    sensitivity_avg = []
    sensitivity_max = []
    sparses = []

    # ====================================================================
    # Iteration through the dataset
    for batch in loaded:
        # Unpacking the batch
        names = batch[0]
        images = batch[1].numpy()
        labels = batch[2].numpy()
        smap = batch[4]
        # Preparing for computation
        if smap.shape[-1] == 3:
            smap = smap.mean(dim=-1)
        elif smap.shape[-1] == 1:
            smap = smap[:, :, :, 0]
        smap = smap.unsqueeze(1).numpy()
        # Computing Completeness
        comp_b = completeness(model=model, x_batch=images,
                     y_batch=labels, a_batch=smap,
                     device=args.device,
                     explain_func=quantus.explain,
                     explain_func_kwargs={"method": args.explain_funct})

        # Computing Complexity
        komp_b = complexity(model=model, x_batch=images, y_batch=labels,
                            a_batch=smap, device=args.device,
                            batch_size=args.batch_size)

        eff_comp_b = eff_complexity(model=model, x_batch=images,
                                    y_batch=labels, a_batch=smap,
                                    device=args.device,
                                    batch_size=args.batch_size)

        # Computing Sparseness
        sparse_b = sparseness(model=model, x_batch=images,
                              y_batch=labels, a_batch=smap,
                              device=args.device,
                              batch_size=args.batch_size)

        # Computing Monotonicity
        monoton_b = monotonicity(model=model, x_batch=images,
                                 y_batch=labels, a_batch=smap,
                                 device=args.device,
                                 batch_size=args.batch_size)

        # Computing Average Sensitivity
        maxsens_b = max_sensitivity(model=model, x_batch=images,
                        y_batch=labels, a_batch=smap,
                        device=args.device,
                        explain_func=quantus.explain,
                        explain_func_kwargs={"method": args.explain_funct})

        avgsens_b = avg_sensitivity(model=model, x_batch=images,
                        y_batch=labels, a_batch=smap,
                        device=args.device,
                        explain_func=quantus.explain,
                        explain_func_kwargs={"method": args.explain_funct})

        # Appending to storing lists
        fnames.extend(names); #completes.extend(comp_b)
        complexes.extend(komp_b); eff_complexes.extend(eff_comp_b)
        monotones.extend(monoton_b);
        sensitivity_max.extend(maxsens_b)
        sensitivity_avg.extend(avgsens_b); sparses.extend(sparse_b); 

    dict_pure = {}

    for x in range(len(fnames)):
        # Element wise diagnostics
        dict_pure[fnames[x]] = {'Completeness': str(completes[x]),
                                'Complexity': str(complexes[x]),
                                'EffectiveComplexity': str(eff_complexes[x]),
                                'MaxSensitivity': str(sensitivity_max[x]),
                                'AvgSensitivity': str(sensitivity_avg[x]),
                                'Monotonicity': str(monotones[x]),
                                'Sparseness': str(sparses[x])}
                      
    # Summary diagnostics                     
    dict_summary = {'Completeness': str(np.asarray(completes).sum()/len(experimental_dataset)),
                    'Complexity': str(np.asarray(complexes).mean()),                    
                    'EffectiveComplexity': str(np.asarray(eff_complexes).mean()),
                    'MaxSensitivity': str(np.asarray(sensitivity_max).mean()),
                    'AvgSensitivity': str(np.asarray(sensitivity_avg).mean()),
                    'Monotonicity': str(np.asarray(monotones).mean()),
                    'Sparseness': str(np.asarray(sparses).mean())}


    path_element = osj(args.store_dir, '{}_complex_per_element.json'.format(\
                       args.method))
    path_summary = osj(args.store_dir, '{}_complex_summary.json'.format(args.method))

    with open(path_element, 'w') as dump:
        json.dump(dict_pure, dump, ensure_ascii=False, indent=4)

    with open(path_summary, 'w') as dump:
        json.dump(dict_summary, dump, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

