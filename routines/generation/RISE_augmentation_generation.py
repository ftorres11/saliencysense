# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2, InterpolationMode
torch.backends.cudnn.deterministic = True

# In-package imports
from lib.data import imagenet_tester, INet_Evaluator, seed_worker
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
                        'rand1k_1perclass_val.csv'), type=str,
                        help='Path to csv file with data and annotations')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Images per batch')
    # Model Initialization
    parser.add_argument('--model', default='resnet50', type=str,
                        help='Model to Evaluate')
    parser.add_argument('--store_dir', default='Augmented', type=str,
                        help='Where to store the saliency maps')
    parser.add_argument('--lab', default='groundtruth', type=str,
                        help='Do we compute saliency maps for gt or pred?')
    parser.add_argument('--seed', default=44, type=int,
                        help='Seed to control deterministic behaviour')
    # Augmentations
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='Are we using mixup augmentation?')
    parser.add_argument('--cutmix', action='store_true', default=False,
                        help='Are we using cutmix augmentation?')
    parser.add_argument('--rotate', type=int, default=None,
                        help='Are we augmenting using rotations?')
    parser.add_argument('--resize', type=int, default=None,
                        help='Are we resizing the image during forward?')
    parser.add_argument('--crop', action='store_true', default=False,
                        help='Are we extracting random crops?')
    parser.add_argument('--second', action='store_true', default=False,
                        help='Are we generating smaps for mixup second '
                        'class?')
    # Parsing Arguments
    args = parser.parse_args()
    # ====================================================================
    # Checking parsed arguments
    prediction = args.lab
    args.method = 'RISE'
    args.store_dir = osj(args.store_dir, args.method, prediction)
    suffix = 'aug'
    suffix += '_rotate_{}'.format(args.rotate) if args.rotate else ''
    suffix += '_mixup_seed{}'.format(args.seed) if args.mixup else ''
    suffix += '_cutmix_seed{}'.format(args.seed) if args.cutmix else ''
    suffix += '_resize_{}'.format(args.resize) if args.resize else ''
    suffix += '_crop_seed{}'.format(args.seed) if args.crop else ''
    suffix += '_second' if args.second else ''
    args.store_dir = osj(args.store_dir, suffix)
    path_aug = osj(args.store_dir, 'images')
    path_smp = osj(args.store_dir, 'smaps')

    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)
        os.makedirs(path_aug)
        os.makedirs(path_smp)

    # Checking for devices
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    gt_ls = []; aug_ls = []; all_names = []
    # ====================================================================
    # Dataloading
    with open(args.path_data, 'r') as fil:
        data = fil.readlines()
    
    transform = imagenet_tester(256, 224)
    experimental_dataset = INet_Evaluator(args.root_data, data, transform)
    # Model Selection, Targetting and Wrapping.
    model = model_selection(args.model)
    model = model.to(args.device).eval()
    # Selecting attributions
    resz = 224 if not args.resize else args.resize
    explainer = RISEBatch(model, (resz, resz),
                          args.batch_size)
    explainer.generate_masks(N=32, s=4, p1=0.1, savepath=args.store_dir)
  
    # Loading data
    loaded = DataLoader(experimental_dataset, batch_size=args.batch_size,
                        num_workers=0, shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=generator)

    softmax = nn.Softmax(dim=-1)

    # Mixup
    if args.mixup:
        mixup = v2.MixUp(num_classes=1000)

    # CutMix
    if args.cutmix:
        cutmix = v2.CutMix(num_classes=1000)        

    # Resizing
    if args.resize:
        resize = v2.Resize(size=(args.resize, args.resize),
                           interpolation=InterpolationMode.BILINEAR)

    # Random Crops
    if args.crop:
        crop = v2.RandomResizedCrop(size=224,
                           interpolation=InterpolationMode.BILINEAR)

    for images, labels, names in loaded:
        # Retrieving the batch
        images = images.to(torch.cuda.current_device())
        labels = labels.to(torch.cuda.current_device())
        all_names.extend(names)
        if args.mixup:
            images, lbs = mixup(images, labels)
            _, tops = torch.topk(lbs, 2)
            gt_ls.extend(tops[:, 0].cpu().numpy())
            aug_ls.extend(tops[:, 1].cpu().numpy())
            if args.second:
                labels = tops[:,1]

        if args.cutmix:
            images, lbs = cutmix(images, labels)
            _, tops = torch.topk(lbs, 2)
            gt_ls.extend(tops[:, 0].cpu().numpy())
            aug_ls.extend(tops[:, 1].cpu().numpy())

        if args.rotate:
            images = v2.functional.rotate(images, args.rotate,
                        interpolation=InterpolationMode.BILINEAR)
        if args.resize:
            images = resize(images)

        if args.crop:
            images = crop(images)

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
            transformed = images[idx].cpu().numpy()
            transformed = transformed.transpose((1, 2, 0))
            name = names[idx].replace('.JPEG', '') # Name of the image
            np.save(osj(path_smp, '{}.npy').format(name), saliency)
            np.save(osj(path_aug, '{}.npy').format(name), transformed)
    # ====================================================================
    if args.cutmix or args.mixup:
        with open(osj(path_aug, 'image_annotations.csv'), 'w') as data:
            data.write('Filename,Label,Augmented\n')
            for idx in range(len(all_names)):
                name = all_names[idx]
                gt_lab = gt_ls[idx]
                aug_lab = aug_ls[idx]
                data.write('{},{},{}\n'.format(name, gt_lab, aug_lab))

if __name__ == '__main__':
    main()
