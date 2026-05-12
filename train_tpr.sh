#!/bin/bash
# TPR (Task-Performance-based Routing for multi-teacher) training script
# Example: Pancreas dataset with two teacher checkpoints

# Optional: activate conda (uncomment and set path)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

export CUDA_VISIBLE_DEVICES=0

python train_tpr.py \
    --data-dir /path/to/your/dataset \
    --teacher-1-path /path/to/teacher1/best_model.pth \
    --teacher-2-path /path/to/teacher2/best_model.pth \
    --batch-size 2 \
    --num-workers 4 \
    --roi-size 96 96 96 \
    --student-feature-size 48 \
    --num-classes 3 \
    --hard-region-threshold 0.5 \
    --routing-temperature 0.7 \
    --epochs 1000 \
    --lr 3e-4 \
    --weight-decay 1e-5 \
    --warmup-epochs 20 \
    --num-samples 1 \
    --pos 9 \
    --neg 1 \
    --flip-prob 0.2 \
    --rotate-prob 0.2 \
    --lambda-seg 1.0 \
    --lambda-align 0.2 \
    --lambda-logits 0.2 \
    --lambda-balance 0.1 \
    --balance-temperature 0.4 \
    --use-logits-distillation \
    --gpu 0 \
    --save-dir ./checkpoints_tpr \
    --val-interval 10 \
    --seed 42
