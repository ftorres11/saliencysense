#!/bin/bash
# Complexity Eval for R50

# IBA 
#python complex_analysis.py --batch_size 32 --model resnet50 --method IBA --use_gpu  --path_data Evaluation/resnet50/complete_info.json

# LIME 
#python complex_analysis.py --batch_size 32 --model resnet50 --method LIME --use_gpu  --path_data Evaluation/resnet50/complete_info.json

# RISE
#python complex_analysis.py --batch_size 32 --model resnet50 --method RISE --use_gpu  --path_data Evaluation/resnet50/complete_info.json

# Fake-CAM
#python complex_analysis.py --batch_size 32 --model resnet50 --method fakecam --use_gpu  --path_data Evaluation/resnet50/complete_info.json

# Grad-CAM
#python complex_analysis.py --batch_size 32 --model resnet50 --method gradcam --use_gpu  --path_data Evaluation/resnet50/complete_info.json

# Grad-CAM++
#python complex_analysis.py --batch_size 32 --model resnet50 --method gradcampp --use_gpu  --path_data Evaluation/resnet50/complete_info.json

# Score-CAM
#python complex_analysis.py --batch_size 32 --model resnet50 --method scorecam --use_gpu  --path_data Evaluation/resnet50/complete_info.json

# Backprop-Smooth
#python complex_analysis.py --batch_size 32 --model resnet50 --method backprop_smooth --use_gpu  --path_data Evaluation/resnet50/complete_info.json --explain_funct Gradient

# Guided Backprop-Smooth
#python complex_analysis.py --batch_size 32 --model resnet50 --method guidedbackprop_smooth --use_gpu  --path_data Evaluation/resnet50/complete_info.json --explain_funct Gradient

# Integrated Gradients-Smooth
#python complex_analysis.py --batch_size 32 --model resnet50 --method int_grad_smooth --use_gpu  --path_data Evaluation/resnet50/complete_info.json --explain_funct Gradient

echo "All Done"!
wait
