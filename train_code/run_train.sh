#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

python swift/cli/sft.py \
  --model checkpoint/ \
  --train_type full \
  --dataset ./data/data_demo.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --target_modules all-linear \
  --gradient_accumulation_steps 1 \
  --save_steps 2000 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 20000 \
  --output_dir output \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --model_author swift \
  --model_name swift-robot \
  --use_compression true \
  --compression_times 2 \
  --dino_from_pretrained facebook/dinov3-vitb16-pretrain-lvd1689m \
  --dino_size vitb16 \
  --dino_freeze true \
