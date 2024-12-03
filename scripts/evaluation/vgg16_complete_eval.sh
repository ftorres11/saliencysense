#!/bin/bash

# Evaluating for VGG16 with BatchNorm
# IBA 
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method IBA --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method IBA --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method IBA --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# LIME 
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method LIME --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method LIME --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method LIME --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# RISE
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method RISE --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method RISE --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json  --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method RISE --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Fake-CAM
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method fakecam --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method fakecam --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method fakecam --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Grad-CAM
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcam --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcam --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcam --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Grad-CAM++
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcampp --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcampp --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method gradcampp --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN


# Score-CAM
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method scorecam --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method scorecam --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method scorecam --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_smooth --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_smooth --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method backprop_smooth --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Guided Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_smooth --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_smooth --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method guidedbackprop_smooth --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

# Integrated Gradients-Smooth
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method int_grad_smooth --use_gpu  --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method int_grad_smooth --use_gpu --lab predicted --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN
#python interpretable_evaluation.py --batch_size 16 --model vgg16_bn --method int_grad_smooth --use_gpu  --lab least --path_data Evaluation/vgg16_bn/complete_info.json --path_saliency SaliencyMaps/ImageNet/VGG16BN

echo "All Done"!
wait
