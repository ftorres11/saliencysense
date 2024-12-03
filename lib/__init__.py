# -*- coding: utf-8 -*-

# Torch imports

# Jacob-gil imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam import EigenGradCAM
from pytorch_grad_cam import GradCAMPlusPlus

# In-package imports
from lib.opticam import OptCAM_CNN, OptCAM_Transformer

# Package imports


# ========================================================================
dict_cam = {'gradcam': GradCAM,
            'scorecam': ScoreCAM,
            'xgradcam': XGradCAM,
            'layercam': LayerCAM,
            'ablationcam': AblationCAM,
            'gradcampp': GradCAMPlusPlus,
            'eigengradcam': EigenGradCAM,
            'opticam_cnn': OptCAM_CNN,
            'opticam_transformer': OptCAM_Transformer}

