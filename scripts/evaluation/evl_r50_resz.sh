#!/bin/bash

# Evaluating Augmentations R50 - Random Resizing
# Fake-CAM Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 8 --augtype resize --resize 448 --method fakecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method fakecam

# IBA Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 8 --augtype resize --resize 448 --method IBA
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method IBA

# Int-Grad Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --method int_grad_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method int_grad_smooth

# RISE Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --method RISE
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method RISE

# LIME Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --method LIME
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method LIME

# Backprop Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method backprop_smooth

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab least --method backprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab least --method backprop_smooth

# Guided Backprop Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method guidedbackprop_smooth

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab least --method guidedbackprop_smooth
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab least --method guidedbackprop_smooth


# Grad-CAM Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab predicted 

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab least 

# Grad-CAM++ Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method gradcampp

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab predicted --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab predicted  --method gradcampp

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab least --method gradcampp
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab least --method gradcampp

# Score-CAM Resizing
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --method scorecam

# Predicted
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab predicted --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab predicted  --method scorecam

# Least
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 128 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 64 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 448 --lab least --method scorecam
#python augmented_evaluation.py --use_gpu --batch_size 16 --augtype resize --resize 32 --lab least --method scorecam

echo "All Done"!
wait
