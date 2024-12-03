# -*- coding: utf -8 -*-

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2, InterpolationMode
torch.backends.cudnn.deterministic = True
# Jacob-gil imports
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# In-package importsa
from lib import dict_cam
from lib.data import imagenet_tester, INet_Evaluator, seed_worker

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
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

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
    parser.add_argument('--method', default='gradcam', type=str,
                        help='Saliency Approach')
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
                        help='Are we generating images for the second '
                        'augmentation image?')
    # Parsing Arguments
    args = parser.parse_args()
    # ====================================================================
    # Checking parsed arguments
    prediction = args.lab
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
    # Checking for reshaping
    if 'transformer' in str(type(model)).lower():
        reshape = reshape_transform
    else:
        reshape = None
    target = cam_targetter(model)
    cam_approach = dict_cam[args.method]

    # Spetial parameters for CAM in CNN or transformer
    if 'transformer' in args.method: # Transformer-based methods
        cam = cam_approach(model, args.device, target, max_iter=50,
                           reshape_transform=reshape,
                           learning_rate=0.1, name_f='logit_predict',
                           name_loss='plain', name_norm='max_min')
    elif 'fakecam' in args.method:
        pass
    else: # CNN
        cam = cam_approach(model=model, target_layers=target,
                           use_cuda=args.use_gpu,
                           reshape_transform=reshape)
    
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
                labels = tops[:,1 ]

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
        # Bizarre Override. Adapted to use groundtruth labels as in repo.
        if args.lab == 'predicted':
            targets = None
        elif args.lab == 'least':
            targets = [ClassifierOutputTarget(label) 
                       for label in worst.cpu().numpy().tolist()]
        else:
            targets = [ClassifierOutputTarget(label) \
                       for label in labels.tolist()]

        if 'fakecam' == args.method:
            salient = torch.ones((images.shape[0], images.shape[-1], images.shape[-1]))
            salient[:,0,0] = 0
        else:
            salient = cam(images, targets)
        for idx in range(len(salient)):
            saliency = salient[idx] # Storing index
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

