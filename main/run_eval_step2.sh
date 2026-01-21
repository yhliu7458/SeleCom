#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RESULT_PATH="../results/results.jsonl"   
JUDGE_MODEL_NAME="/path/to/judge/model/"                  

python evaluate_step2_metric.py \
    --result_path "$RESULT_PATH" \
    --judge_model_name "$JUDGE_MODEL_NAME" \
    --num_gpus 8
