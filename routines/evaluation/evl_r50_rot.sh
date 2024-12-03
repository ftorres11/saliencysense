#!/bin/bash

# Evaluating Augmentations for R50 Rotations
# Fake-CAM Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method fakecam

# IBA Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method IBA

# Int-Grad Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method int_grad_smooth

# RISE Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method RISE

# LIME Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method LIME

# Backprop Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method backprop_smooth

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab least --method backprop_smooth

# Guided Backprop Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method guidedbackprop_smooth

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab least --method guidedbackprop_smooth

# Grad-CAM Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab predicted 

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab least 

# Grad-CAM++ Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method gradcampp

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab predicted  --method gradcampp

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab least --method gradcampp

# Score-CAM Rotation
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --method scorecam

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab predicted  --method scorecam

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 45 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 135 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 225 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype rotate --rotate 315 --lab least --method scorecam

echo "All Done"!
wait
