#!/bin/bash

# Generating for ResNet50
# IBA
#python IBA_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv --batch_size=1
#python IBA_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv --batch_size=1 --lab predicted
#python IBA_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv --batch_size=1 --lab least

# LIME
#python LIME_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv
#python LIME_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv --lab predicted
#python LIME_generation.py --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv --lab least


# RISE 
#srun python RISE_generation.py --batch_size 5 --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv
#srun python RISE_generation.py --batch_size 5 --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv --lab predicted
#srun python RISE_generation.py --batch_size 5 --model resnet50 --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv --lab least

# Grad-CAM
#srun python cam_generation.py --batch_size 10 --model resnet50 --method gradcam --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu  --path_data data/50k_sorted.csv
#srun python cam_generation.py --batch_size 10 --model resnet50 --method gradcam --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted --path_data data/50k_sorted.csv

#srun python cam_generation.py --batch_size 10 --model resnet50 --method gradcam --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least --path_data data/50k_sorted.csv


# Grad-CAM++
#srun python cam_generation.py --batch_size 1 --model resnet50 --method gradcampp --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu
#srun python cam_generation.py --batch_size 1 --model resnet50 --method gradcampp --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted
#srun python cam_generation.py --batch_size 1 --model resnet50 --method gradcampp --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least

# Score-CAM
#srun python cam_generation.py --batch_size 1 --model resnet50 --method scorecam --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu
#srun python cam_generation.py --batch_size 1 --model resnet50 --method scorecam --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted
#srun python cam_generation.py --batch_size 1 --model resnet50 --method scorecam --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least

# Backprop - Jacbogil
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu 
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least

# Guided Backprop - Jacobgil
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least

# Backprop - Smoothgrad denorm
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --norm smooth
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted --norm smooth
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method backprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least --norm smooth

# Guided Backprop - Smoothgrad denorm
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --norm smooth
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted --norm smooth
#srun python gradient_generation.py --batch_size 1 --model resnet50 --method guidedbackprop --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least --norm smooth

# Integrated Gradients-smooth
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --norm smooth
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted --norm smooth
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least --norm smooth

# Integrated Gradients-Jacobgil
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu 
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab predicted 
#python integrated_grad_generation.py --batch_size 1 --model resnet50 --method int_grad --store_dir SaliencyMaps/ImageNet/ResNet50/Res_4 --use_gpu --lab least 

echo "All Done"!
wait