# -*- coding: utf-8 -*-
# Timm Imports
import timm
# Torch Imports
import torch
import torch.hub as hub
from torchvision.models import get_model

# In-package imports

# Package Imports
import os
osp = os.path
osj = osp.join

# ========================================================================
# Basic Configuration
def model_selection(model_name):
    if 'deit' in model_name:
        model = timm.create_model(model_name.lower(), pretrained=True)
    elif 'vit' in model_name:
        model = timm.create_model(model_name.lower(), pretrained=True)
    else:
        model = hub.load('pytorch/vision', model_name,
                         weights='IMAGENET1K_V1')
    return model


