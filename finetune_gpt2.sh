#!/bin/bash

python finetune_gpt2.py \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --num_train_epochs=2 \
    --warmup_ratio="0.1" \
    --block_size=210 \
    --seed=21 \
    --logging_steps=200 \
    --learning_rate=5e-05 \
    --eval_steps=200 \
    --evaluation_strategy steps \
    --save_strategy="epoch" \
    --train_file ./new_multi_conan.csv \
    --validation_file ./multi_conan_validation.csv \
    --do_train \
    --do_eval \
    --output_dir checkpoints \
    --logging_dir='./logs' \
    --overwrite_output_dir
