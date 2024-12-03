#!/bin/bash
python diagnostics_generation.py --batch_size 32 --model resnet50 --store_dir Evaluation/resnet50/ --use_gpu
echo "All Done"!
wait
