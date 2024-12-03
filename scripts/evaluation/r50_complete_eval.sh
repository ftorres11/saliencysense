#!/bin/bash

# Evaluating for R50
# IBA 
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method IBA --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method IBA --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method IBA --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# LIME 
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method LIME --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method LIME --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method LIME --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# RISE
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method RISE --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method RISE --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method RISE --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Fake-CAM
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method fakecam --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method fakecam --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method fakecam --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Grad-CAM
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcam --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcam --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcam --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Grad-CAM++
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcampp --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcampp --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method gradcampp --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Score-CAM
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method scorecam --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method scorecam --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method scorecam --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Backprop-Outlier
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_outlier --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_outlier --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_outlier --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Guided Backprop-Outlier
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_outlier --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_outlier --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_outlier --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_smooth --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_smooth --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method backprop_smooth --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Guided Backprop-Smooth
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_smooth --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_smooth --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method guidedbackprop_smooth --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

# Integrated Gradients-Smooth
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method int_grad_smooth --use_gpu  --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method int_grad_smooth --use_gpu --lab predicted --path_data Evaluation/resnet50/complete_info.json
#python interpretable_evaluation.py --batch_size 16 --model resnet50 --method int_grad_smooth --use_gpu  --lab least --path_data Evaluation/resnet50/complete_info.json

echo "All Done"!
wait
