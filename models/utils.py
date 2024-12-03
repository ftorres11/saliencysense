# -*- coding: utf-8 -*- 
# Author: Felipe Torres Figueroa 

# Torch imports
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# In-package imports


# Package imports


# ========================================================================
class GuidedBackPropReLU(Function):
    @staticmethod
    def forward(self, input_map):
        positive_mask = (input_map > 0).type_as(input_map)
        output = input_map * positive_mask
        self.save_for_backward(input_map, output)
        return output
    @staticmethod
    def backward(self, output_grad):
        input_map, _ = self.saved_tensors

        positive_mask_a = (input_map > 0).type_as(input_map)
        positive_mask_b = (output_grad > 0).type_as(input_map)
        grad_input = torch.zeros(input_map.size()).type_as(input_map)
        grad_input = (output_grad*positive_mask_a*positive_mask_b)
        return grad_input

# ========================================================================
class GBP_ReLU(nn.Module):
    def forward(self,x):
        return GuidedBackPropReLU.apply(x)

# ========================================================================
def cam_targetter(model):
    # Check, activations, saliency maps WILL change given the different
    # layer targeted, also prediction vs groundtruth
    name = str(type(model))
    if 'resnet' in name.lower():
        target = [model.layer4[-1]]
    elif 'vgg' in name.lower():
        target = [model.features[-1]] # To Revisit
    elif 'convnext' in name.lower():
        target = [model.features[-1][-1].block[5]] # To Revisit
    elif 'visiontransformer' in name.lower():
        target = [model.blocks[-1].norm1]
    else:
        target = [model.features[-1]] # To Revisit
    return target

# ========================================================================
class Activation_Wrapper:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))

    def save_activation(self, module, input, output):
        activation = output
        # Adding support for multiple outputs
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation)#.detach())

    def __call__(self, x):
        self.activations = []
        return self.activations, self.model(x)


    def release(self):
        for handle in self.handles:
            handle.remove()

