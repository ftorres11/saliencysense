#!/bin/bash
# Evaluating Augmentations for R50  - Mixup
# Fake-CAM
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method fakecam

# IBA Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method IBA

# Int-Grad Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method int_grad_smooth

# RISE Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method RISE

# LIME Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method LIME

# Backprop Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method backprop_smooth


# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab least --method backprop_smooth

# Guided backprop Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method guidedbackprop_smooth


# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab least --method guidedbackprop_smooth

# Grad-CAM Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab predicted 

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab least 

# Grad-CAM++ Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method gradcampp

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab predicted  --method gradcampp

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab least --method gradcampp

# Score-CAM Mix Up
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --method scorecam

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab predicted  --method scorecam

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 44 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 55 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 93 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype mixup --seed 32 --lab least --method scorecam

echo "All Done"!
wait
