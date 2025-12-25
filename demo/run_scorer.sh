#!/bin/bash

# 配置参数
PROJECT_PATH="path/to/GEMGen"
DICT_PATH="$PROJECT_PATH/models/scorer"
MODEL_PATH="$PROJECT_PATH/models/scorer/gemgen_scorer_epoch3.safetensors"
DATA_PATH="$PROJECT_PATH/data/scorer_test_demo.tsv"
TOKENIZER_PATH="$PROJECT_PATH/models/scorer"
OUTPUT_PATH="$PROJECT_PATH/results/scorer_test_demo_with_score.tsv"
BATCH_SIZE=32
MAX_LEN=8192
NUM_WORKERS=4

# 执行命令
python $PROJECT_PATH/gemgen/scorer.py \
    --dict_path "$DICT_PATH" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    --num_workers "$NUM_WORKERS"