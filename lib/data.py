# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, distributed
import random

# In package imports

# Package imports
import os
import pdb
import pickle
import numpy as np
from PIL import Image

import sys
epsilon = sys.float_info.epsilon

osp = os.path
osj = osp.join


# Library for dataloading. Classes and functions for different attempts at
# dataloading
# ========================================================================
# Standard Variables
tensor_transform = transforms.Compose([transforms.ToTensor()])

# ImageNet Normalization
im_normalization = transforms.Normalize(mean=[.485, .456, .406],
                                         std=[.229, .224, .225])

# ImageNet Denormalization
im_denormalization = transforms.Normalize(\
                         mean=[-.485/.229, -.456/.224, -.406/.225],
                         std=[1/.229, 1/.224, 1/.225])

# ========================================================================
# For specific seeds for data shuffling
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ========================================================================
# Standard imagenet preprocessing for testing. Resizing -> Center Crop 
def imagenet_tester(resize, crop_size):
    inet_trans = transforms.Compose([transforms.Resize((resize, resize),
                      interpolation=transforms.InterpolationMode.BILINEAR),
                      transforms.CenterCrop((crop_size, crop_size)),
                      transforms.ToTensor(),
                      im_normalization])
    return  inet_trans

# ========================================================================
# Deprocessing for gradient images following Jacobgil approach. Mean value
# centered around 0.5
def outlier_deprocessor(grad_batch):
    grad_mean = grad_batch.mean(dim=(2, 3), keepdim=True)
    grad_std = grad_batch.std(dim=(2, 3), keepdim=True)
    grad_batch = (grad_batch-grad_mean)/grad_std
    grad_batch = grad_batch*0.1 + 0.5
    grad_batch = torch.clamp(grad_batch, min=0, max=1)
    return grad_batch

# ========================================================================
# Deprocessing for gradient images keeping mean values in 0.
def smooth_deprocessor(grad_batch):
    grad_batch, _ = torch.max(torch.abs(grad_batch), dim=1, keepdim=True)
    grad_min = grad_batch.amin(dim=(2, 3), keepdim=True)
    grad_max = grad_batch.amax(dim=(2, 3), keepdim=True)
    grad_batch = (grad_batch-grad_min)/(grad_max-grad_min+epsilon)
    return grad_batch
# ========================================================================
# Classes
# ========================================================================
# Standard class for loading imagenet validation set, requires a path for 
# the data and the ids in a list.
class INet_Evaluator(Dataset):
    def __init__(self, root_path, listed_data, transform=None):
        self.root = root_path
        self.data = listed_data
        self.ids = []
        for idx, _ in enumerate(listed_data):
            self.ids.append(idx)
        self.transform = transform

    def __getitem__(self, x):
        name, label = self.data[x].strip().split(',')
        image = Image.open(osj(self.root, name)).convert('RGB')
        name = name.split('/')[-1]
        if self.transform:
            image = self.transform(image)
        return image, int(label), name
    def __len__(self):
        return len(self.data)

# ========================================================================
# Loads data for salient evaluation. Requires an ImageNet loader class, 
# and the path for saliency maps.
class Salient_Evaluator(Dataset):
    def __init__(self, parent_data, root_saliency):
        self.path = root_saliency
        self.data = parent_data

    def __getitem__(self, x):
        image, label, name = self.data[x]
        loading_name = name.split('.')[0]
        loading_name = loading_name.split('/')[-1]
        try:
            saliency_map = np.load(osj(self.path,
                           '{}.npy'.format(loading_name)),
                           allow_pickle=True)
        except FileNotFoundError:
            assert 'fakecam' in self.path
            saliency_map = np.ones((224, 224))
            saliency_map[0,0] = 0

        return image, label, saliency_map, loading_name

    def __len__(self):
        return len(self.data)

# ========================================================================
# Loads the data for augmentation evaluation. Similar to previous class,
# requires an instance of ImageNet evaluator and the path to the saliency 
# maps.
class AugmentedEvaluator(Dataset):
    def __init__(self, parent_data, root_saliency):
        self.data = parent_data
        self.path_augmented = osj(root_saliency, 'images')
        self.path_smaps = osj(root_saliency, 'smaps')

    def __getitem__(self, idx):
        name, label = self.data[idx].strip().split(',')
        name = name.split('/')[-1].split('/')[0]
        name = name.replace('.JPEG', '.npy')
        img = np.load(osj(self.path_augmented, '{}'.format(name)),
                      allow_pickle=True)
        smap = np.load(osj(self.path_smaps, '{}'.format(name)),
                       allow_pickle=True)
        label = int(label)
        return name, tensor_transform(img), tensor_transform(smap), label

    def __len__(self):
        return len(self.data)
        
# ========================================================================
# Class to perform complete evaluation. Requires a dictionary generated by 
# the diagnostics script, the path to raw images and the path of saliency 
# maps. Runs the evaluation set for either groundtruth, predictions and 
# worst instances of prediction.
class CompleteEvaluator(Dataset):
    def __init__(self, data_dict, root_data, root_saliency,
                 target_set='groundtruth', transform=None):
        self.root = root_data
        assert target_set in ['groundtruth', 'predicted', 'least'], \
                "target set ought to be either 'groundtruth', 'predicted'"\
                " or 'least'"
        self.target_set = target_set
        self.files = data_dict['Filenames']
        self.path_smap = root_saliency
        if self.target_set == 'groundtruth':
            self.labels = data_dict['GroundtruthLabels']
            self.probs = data_dict['GroundtruthProbs']
        elif self.target_set == 'predicted':
            self.labels = data_dict['PredictedLabels']
            self.probs = data_dict['PredictedProbs']
        elif self.target_set == 'least':
            self.labels = data_dict['WorstLabels']
            self.probs = data_dict['WorstProbs']

        self.transform = transform

    def __getitem__(self, x):
        # Loading raw image
        name = self.files[x]
        fname = name.replace('.JPEG', '')
        image = Image.open(osj(self.root, name)).convert('RGB')
        # Saliency Maps
        try:
            smap = np.load(osj(self.path_smap,
                           '{}.npy'.format(fname)), allow_pickle=True)
        except FileNotFoundError:
            assert 'fakecam' in self.path_smap
            smap = np.ones((224, 224))
            smap[0,0] = 0

        label = int(self.labels[x])
        prob = float(self.probs[x])

        if self.transform:
            image = self.transform(image)

        return fname, image, label, prob, smap

    def __len__(self):
        return len(self.files)

# ========================================================================
# Wrapper for loading data in a training setup. Mostly auxiliar and not 
# used for these evaluations. Still can provide useful for distributed 
# data parallelism.
def dataset_wrapper(loader, size, rank, collate_fn, args):
    if size > 1:
        sampler = distributed.DistributedSampler(loader,
                      num_replicas=size, rank=rank, shuffle=True)
    else:
        sampler = None

    # Runs data loading/sampling with a fixed seed if desired
    if args.fixed:
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        data = DataLoader(loader, batch_size=args.batch_size,
                          num_workers=args.cpus_task,
                          sampler=sampler,
                          worker_init_fn=seed_worker,
                          generator=gen,
                          shuffle=False if sampler else True,
                          drop_last=True,
                          collate_fn=collate_fn)
    else:
        data = DataLoader(loader, batch_size=args.batch_size,
                num_workers=args.cpus_task,
                sampler=sampler,
                drop_last=True,
                shuffle=False if sampler else True,
                collate_fn=collate_fn)
        
    return data, sampler
                             
