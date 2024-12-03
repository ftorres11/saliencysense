#!/bin/bash

# Generating Augmentations for R50 - Resizing
# IntegratedGradient Resizing
# python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --norm smooth
# python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --norm smooth
# python integrated_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --norm smooth
# python integrated_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --norm smooth


# Least
# python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least --norm smooth
# python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least --norm smooth
# python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least --norm smooth
# python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least --norm smooth

# IBA Resizing
# python IBA_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 
# python IBA_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 
# python IBA_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 
# python IBA_augmentation_generation.py --batch_size 2 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 


# Least
# python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least 
# python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least 
# python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least 
# python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least 


# RISE Resizing
# python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 
# python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 
# python RISE_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 
# python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 


# Least
# python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least 
# python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least 
# python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least 
# python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least 

# LIME Resizing
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 


# Least
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least 
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least 
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least 
# python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least 

# Gradient Resizing
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --norm smooth


# Least
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least --norm smooth

# Guided Gradient Resizing
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --norm smooth

# Least
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least --norm smooth
# python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least --norm smooth

# Fake-CAM Resizing
 python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --resize 128
 python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --resize 64
 python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --resize 448
 python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --resize 32


# Grad-CAM Resizing
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 128
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 64
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 448
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 32

# Predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab predicted

# Least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least

# Grad-CAM++ Resizing
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 128
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 64
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 448
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 32

# Predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab predicted
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab predicted

# Least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least
# python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least

# Score-CAM Resizing
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 128
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 64
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 448
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 32

# Predicted
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab predicted
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab predicted
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab predicted
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab predicted

# Least
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 128 --lab least
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 64 --lab least
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 448 --lab least
# python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --resize 32 --lab least

echo "All Done"!
wait
