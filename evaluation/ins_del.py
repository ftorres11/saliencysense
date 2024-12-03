# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa 
# Code base from: https://github.com/eclique/RISE/blob/master/evaluation.py

# Torch Imports
import torch
from torch import nn

# In-package Imports

# Package Imports
import copy
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter


# ========================================================================
# Functions
def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

# ========================================================================
def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

# ========================================================================
# Causal Metrics
class CausalMetric():
    def __init__(self, model, mode, step, substrate_fn, HW, classes):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): model wrapped in MAE to be explained.
            mode (str): 'del' or 'ins' or 'del-mae'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.HW = HW**2
        self.n_classes = classes

    def evaluate(self, img_batch, exp_batch, batch_size, top):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step 
            for every image.
        """
        # Baseline parameters
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, self.n_classes)
        n_steps = (self.HW + self.step - 1) // self.step
        # Flatten -> min to max -> max to min.
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, self.HW),
                                axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)
        assert n_samples % batch_size == 0
        # Forwarding
        #predictions = self.model(img_batch)
        top = top.cpu()
        #top = torch.argmax(predictions.detach(), dim=-1).cpu()
        scores = torch.zeros((n_steps + 1, n_samples)).cpu()

        # Generating the starting image-reconstruction to forward
        substrate = torch.zeros_like(img_batch)
        for j in range(n_samples // batch_size):
            # Masks every slice of the minibatch using the substrate function
            substrate = self.substrate_fn(img_batch)
        if 'del' in self.mode:
            start = img_batch.clone()
            finish = substrate.flatten(start_dim=2)
        elif 'ins' in self.mode:
            start = substrate
            finish = img_batch.clone().flatten(start_dim=2)
        # While not all pixels are changed
        for i in range(n_steps+1):
            # Iterate over batches
            for j in range(n_samples // batch_size):
            # Iterates over minibatches to retrieve the predicted
            # probabilities for masking step i. Keep Scores on CPU.
                # Compute new scores
                preds = self.model(start[j*batch_size:(j+1)*batch_size])
                preds = nn.functional.softmax(preds, dim=-1)
                # Gets the predicted probabilities of baseline predictions
                preds = preds.cpu().detach()
                preds = preds[range(batch_size),
                              top[j*batch_size:(j+1)*batch_size]]
                # Appends probabilities to scores
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient px to substrate px
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start = start.flatten(start_dim=2)
            start[r, :, coords.copy()] = finish[r, :, coords.copy()]
            start = start.view(img_batch.shape)
        return scores
