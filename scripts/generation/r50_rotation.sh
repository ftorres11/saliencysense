#!/bin/bash
# Generating Augmentations for R50 - Rotation

# Integrated Gradient Rotation
#python integrated_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --norm smooth
#python integrated_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --norm smooth
#python integrated_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --norm smooth
#python integrated_augmentation_generation.py --batch_size 1 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --norm smooth

# Least
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least --norm smooth
#python integrated_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least --norm smooth

# IBA Rotation
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 

# Least
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least 
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least 
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least 
#python IBA_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least 

# RISE Rotation
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 


# Least
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least 
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least 
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least 
#python RISE_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least 

# LIME Rotation
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 


# Least
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least 
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least 
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least 
#python LIME_augmentation_generation.py --batch_size 5 --model resnet50  --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least 

# Gradient Rotation
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --norm smooth


# Least
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method backprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least --norm smooth

# Guided Gradient Rotation
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --norm smooth

# Least
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least --norm smooth
#python gradient_augmentation_generation.py --batch_size 5 --model resnet50 --method guidedbackprop --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least --norm smooth

# Fake-CAM Rotation
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 45
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 135
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 225
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method fakecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 315

# Grad-CAM Rotation
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 45
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 135
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 225
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 315

# Predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab predicted

# Least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcam --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least

# Grad-CAM++ Rotation
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 45
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 135
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 225
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 315

# Predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab predicted

# Least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method gradcampp --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least

# Score-CAM Rotation
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 45
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 135
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 225
#python cam_augmentation_generation.py --batch_size 5 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 315

# Predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab predicted
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab predicted

# Least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 45 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 135 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 225 --lab least
#python cam_augmentation_generation.py --batch_size 50 --model resnet50 --method scorecam --store_dir Augmentations/ResNet50 --use_gpu --rotate 315 --lab least
echo "All Done"!
wait
