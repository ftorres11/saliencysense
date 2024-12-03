# -*- coding: utf-8 -*- 

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# In-package imports

# Package imports
import numpy as np
import sys
epsilon = sys.float_info.epsilon


# ========================================================================
def energy_point_game(bbox, saliency_map):
    saliency_map = saliency_map.squeeze()
    w, h = saliency_map.shape

    empty = np.zeros((w, h))
    for box in bbox:
        empty[box.xslice,box.yslice]=1
    mask_bbox = saliency_map * empty

    energy_bbox =  mask_bbox.sum()
    energy_whole = saliency_map.sum()
    
    proportion = energy_bbox / (epsilon + energy_whole)

    return proportion

# ========================================================================
def standard_point_game(bbox, saliency_map):
    saliency_map = saliency_map.squeeze()
    Hit = False
    x_p, y_p = np.where(saliency_map == saliency_map.max())
    for box in bbox:
        for i in range(x_p.shape[0]):
            hit = True
            if box.xmin > x_p[i]:
                hit = False
            if x_p[i] > box.xmax:
                hit = False
            if y_p[i] < box.ymin:
                hit = False
            if y_p[i] > box.ymax:
                hit = False
            if hit == True:
                Hit = True
    return Hit
