# Saliency Maps Give a False Sense of Explanability to Images: Experimental Analysis to Saliency Methods and Evaluation Metrics
## Overview
This repository covers the experimental procedure for the 2024 ACML article titled
[Saliency Maps Give a False Sense of Explanability to Images: Experimental Analysis to Saliency Methods and Evaluation
Metrics](https://openreview.net/pdf?id=Hftgajppmz)
<p align="center"><img src="misc/instances.svg"><p>

## Requirements
 Check here the list of [requirements](misc/requirements.txt).
 CASME utils are provided inside the repository. Thanks to the [original](https://github.com/kondiz/casme) from kondiz.

## Usage 

**Please bear in mind that you can gain insight into how each script works and its required commands by typing help script_name.py**

**For ease of use, command line scripts are also provided in sh_scripts for each procedure here mentioned**

1. **Diagnostics** Move into main and run the script [diagnostics_generation.py](routines/diagnostics_generation.py). Generates a json file containing the prediction probabilities for most-least likely predictions for the set of images, as well as the labels in each case. 

2. **Generation** To generate the many different activations/representations, select from the scripts in
   [generation](routines/generation).
   Options include CAM variants, gradient (standard, guided, integrated), LIME, RISE and IBA. Each script allows for
   generation of groundtruth labels, predicted labels and worst instances.
   - It's important to note that gradient visualizations have no standard deprocessing for visualization, be it either
     the mean gradient zeroed out and +/- gradients between 0-1 (Smoothgrad deprocessing); or keeping the mean as 0.5,
     performing outlier pruning and scaling to 0-1 (Jacobgil deprocessing). A switch between these options is found in the gradient generation
     scripts and routines.
   - Check for support for CAM variants, as some more can be included by adding them into [lib initialization](lib/__init__.py) from jacobgil's [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) repository.

3. **Evaluation** On the main article *interpretable recognition* and *complexity analysis* were considered for
   evaluation of the generated saliency maps per attribution method. However, it is also possible to perform
   localization evaluation although its results were not considered for this assessment given their misalignment between
   classifier and human centric evaluation.

   - For interpretable evaluation (metrics from [Grad-CAM++](https://arxiv.org/abs/1710.11063) and [RISE](https://arxiv.org/abs/1806.07421)) refer to script [interpretable recognition](py_scripys/evaluation/intepretable_recon.py).

   - For metrics relating to Fidelity, refer to script [complex analysis](routines/evaluation/complex_analysis.py).

   - Lastly, for evaluation experiments and similar to interpretable recognition, refer to script [localization
  evaluation](routines/evaluation/localization_evaluation.py).

4. **Cases for augmented images** Similar to previous scripts for generation, an "augmented" version is used to induce
   image modifications. Additionally, the script [augmented evaluation](routines/evaluation/augmented_evaluation.py)
   runs the evaluation procedure once per augmentation type selected.

### Additional Notes
Additional attribution methods are actively being developed and added as support to this repository. Most importantly
methods derivating from *Layer-wise Relevance Propagation* are planned towards the future although support for different
architectures with skip connections and complex designs is still pending.

### Citation
If this work is relevant to you, you can cite it as:
```
@inproceedings{zhang2024saliency,
    title={Saliency Maps Give a False Sense of Explanability to Image Classifiers: An Empirical Evaluation across Methods and Metrics},
    author={Hanwei Zhang and Felipe Torres Figueroa and Holger Hermanns},
    booktitle={The 16th Asian Conference on Machine Learning (Conference Track)},
    year={2024},
    url={https://openreview.net/forum?id=Hftgajppmz}                                                
}
```

