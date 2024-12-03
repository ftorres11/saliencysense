#!/bin/bash

# Generating Augmentations for R50 - Mixup
# Integrated Gradients - Smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --norm smooth


# Least
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --lab least --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --lab least --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --lab least --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --lab least --norm smooth

# IBA Mixup
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32

# Least
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --lab least
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --lab least
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --lab least
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --lab least

# RISE Mixup
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32

# Least
#python RISE_augmentation_generation.py --batch_size 50 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --lab least
#python RISE_augmentation_generation.py --batch_size 50 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --lab least
#python RISE_augmentation_generation.py --batch_size 50 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --lab least
#python RISE_augmentation_generation.py --batch_size 50 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --lab least

# LIME Mixup
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32

# Least
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --lab least
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --lab least
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --lab least
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --lab least

# Backprop - Smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --norm smooth


# Least
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --lab least --norm smooth


# Guided backprop - Smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --norm smooth


# Least
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --lab least --norm smooth

# Fake-CAM Mixup
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32

# Grad-CAM Mixup
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32

# Grad-CAM++ Mixup
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --second


# Score-CAM Mixup
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 44
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 55
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 93
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --mixup --seed 32 --second
echo "All Done"!
wait
