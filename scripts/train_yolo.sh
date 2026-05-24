#!/bin/bash

python src/models/yolo_detector.py \
    --train \
    --dataset_root $1 \
    --save_path $2