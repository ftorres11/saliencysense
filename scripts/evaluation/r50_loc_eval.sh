#!/bin/bash

# Localization Evaluation for R50

# IBA 
#python localization_evaluation.py --batch_size 10 --model resnet50 --method IBA --use_gpu  

# LIME 
#python localization_evaluation.py --batch_size 10 --model resnet50 --method LIME --use_gpu  

# RISE
#python localization_evaluation.py --batch_size 10 --model resnet50 --method RISE --use_gpu  

# Fake-CAM
#python localization_evaluation.py --batch_size 10 --model resnet50 --method fakecam --use_gpu  
#python localization_evaluation.py --batch_size 10 --model resnet50 --method fakecam --use_gpu  --lab predicted

# Grad-CAM
#python localization_evaluation.py --batch_size 10 --model resnet50 --method gradcam --use_gpu  
#python localization_evaluation.py --batch_size 10 --model resnet50 --method gradcam --use_gpu  --lab predicted


# Grad-CAM++
#python localization_evaluation.py --batch_size 10 --model resnet50 --method gradcampp --use_gpu  
#python localization_evaluation.py --batch_size 10 --model resnet50 --method gradcampp --use_gpu --lab predicted

# Score-CAM
#python localization_evaluation.py --batch_size 10 --model resnet50 --method scorecam --use_gpu  
#python localization_evaluation.py --batch_size 10 --model resnet50 --method scorecam --use_gpu   --lab predicted


# Backprop-Smooth
#python localization_evaluation.py --batch_size 10 --model resnet50 --method backprop_smooth --use_gpu   

# Guided Backprop-Smooth
#python localization_evaluation.py --batch_size 10 --model resnet50 --method guidedbackprop_smooth --use_gpu   

# Integrated Gradients-Smooth
#python localization_evaluation.py --batch_size 10 --model resnet50 --method int_grad_smooth --use_gpu  

echo "All Done"!
wait

