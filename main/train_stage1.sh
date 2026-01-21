#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29505

DATA_PATH="../data/stage1/stage1_train_data.jsonl"       
ENCODER_NAME="path/to/encoder/model/"                   
GENERATOR_NAME="path/to/generator/model/"      
OUTPUT_DIR="../checkpoint/stage1"            
LOG_DIR="../log/stage1"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_stage1.py \
    --data_path "$DATA_PATH" \
    --encoder_name "$ENCODER_NAME" \
    --generator_name "$GENERATOR_NAME" \
    --model_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --log_name "train_stage1.log" \