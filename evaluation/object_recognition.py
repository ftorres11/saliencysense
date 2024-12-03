# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
# In Package imports

# Package imports
import sys
epsilon = sys.float_info.epsilon
import numpy as np


# ========================================================================
# Supporting Functions
# ========================================================================
def gradcampp_recognition(original, explanation):
    original = original.cpu().detach().squeeze()
    explanation = explanation.cpu().detach().squeeze()

    # Average Drop
    ad = torch.clamp(original-explanation, min=0)
    ad = ad/(original+epsilon).numpy()
    # Average Gain
    ag = torch.clamp(explanation-original, min=0)
    ag = ag/(1-original+epsilon).numpy()
    # Increase in Confidence
    ic = torch.sign(explanation-original).clamp(min=0)
    
    # Vectorized
    ad = np.asarray(ad)
    ic = np.asarray(ic)
    ag = np.asarray(ag)

    return ad, ic, ag
