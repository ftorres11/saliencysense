#!/bin/bash
python diagnostics_generation.py --batch_size 32 --model vgg16_bn --store_dir Evaluation/vgg16_bn/ --use_gpu

echo "All Done"!
wait
