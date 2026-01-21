#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29506

DATA_PATH="../data/stage2/stage2_data_top1_all.jsonl"       
ENCODER_NAME="path/to/encoder/model/"                   
GENERATOR_NAME="path/to/generator/model/"      
CHECKPOINT_DIR="../checkpoint/stage1/Qwen3embedding0.6B-Mistral7B"  
OUTPUT_DIR="../checkpoint/stage2"            
LOG_DIR="../log/stage2"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_stage2.py \
    --data_path "$DATA_PATH" \
    --encoder_name "$ENCODER_NAME" \
    --generator_name "$GENERATOR_NAME" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --log_name "train_stage2.log" \