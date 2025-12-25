#!/bin/bash

PROJECT_PATH="/home/lizhen/jiangqun/druggen/GEMGen"
CKPT_PATH="$PROJECT_PATH/models/generator/checkpoint1"
INPUT_FILE="$PROJECT_PATH/data/generator_prompts.txt"
OUTPUT_FILE="$PROJECT_PATH/results/test_generator_output.json"

# vLLM parameters
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.8

# Sampling parameters
MAX_NEW_TOKENS=512
TEMPERATURE=1
TOP_P=0.95
SAMPLE_COUNT=100

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file does not exist $INPUT_FILE"
    exit 1
fi

# Check if model directory exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "Error: Model directory does not exist $CKPT_PATH"
    exit 1
fi

# Check if model files exist in the directory
if [ ! -f "$CKPT_PATH/model.safetensors" ]; then
    echo "Error: No model files found in $CKPT_PATH"
    echo "Please check if the model files (model.safetensors) exist"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

echo "Running GEMGen generator..."
python $PROJECT_PATH/gemgen/generator.py \
    --tokenizer_path "$CKPT_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --sample_count $SAMPLE_COUNT