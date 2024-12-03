#!/bin/bash

# VGG-Complex Analysis

# IBA 
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method IBA --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# LIME 
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method LIME --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# RISE
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method RISE --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Fake-CAM
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method fakecam --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Grad-CAM
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method gradcam --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Grad-CAM++
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method gradcampp --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Score-CAM
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method scorecam --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Opti-CAM
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method opticam_cnn --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Backprop-Smooth
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method backprop_smooth --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN --explain_funct Gradient

# Guided Backprop-Smooth
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method guidedbackprop_smooth --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN --explain_funct Gradient

# Integrated Gradients-Smooth
#python complex_analysis.py --batch_size 16 --model vgg16_bn --method int_grad_smooth --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN --explain_funct Gradient

echo "All Done"!
wait
