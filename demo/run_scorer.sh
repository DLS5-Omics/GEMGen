#!/bin/bash

# Configuration parameters
PROJECT_PATH="path/to/GEMGen"
DICT_PATH="$PROJECT_PATH/models/scorer"
MODEL_PATH="$PROJECT_PATH/models/scorer/model.safetensors"
DATA_PATH="$PROJECT_PATH/data/scorer_test_demo.tsv"
TOKENIZER_PATH="$PROJECT_PATH/models/scorer"
OUTPUT_PATH="$PROJECT_PATH/results/scorer_test_demo_with_score.tsv"
BATCH_SIZE=32
MAX_LEN=8192
NUM_WORKERS=4

# Function to check if file/directory exists
check_path() {
    local path_type=$1
    local path=$2
    local description=$3
    
    if [ ! -e "$path" ]; then
        echo "ERROR: $description '$path' does not exist!"
        return 1
    else
        echo "INFO: $description '$path' found."
        return 0
    fi
}

# Function to create directory if it doesn't exist
create_directory() {
    local dir_path=$1
    local description=$2
    
    if [ ! -d "$dir_path" ]; then
        echo "INFO: Creating $description directory: $dir_path"
        mkdir -p "$dir_path"
        if [ $? -eq 0 ]; then
            echo "INFO: Directory created successfully."
        else
            echo "ERROR: Failed to create directory: $dir_path"
            return 1
        fi
    else
        echo "INFO: $description directory already exists: $dir_path"
    fi
}

# Input validation and checks
echo "=== Starting GEMGen Scorer ==="
echo "Timestamp: $(date)"

# Check project path
if [ "$PROJECT_PATH" = "path/to/GEMGen" ]; then
    echo "ERROR: Please update PROJECT_PATH to the actual project directory."
    exit 1
fi

# Check input paths
echo "--- Checking input paths ---"
check_path "directory" "$PROJECT_PATH" "Project root" || exit 1
check_path "directory" "$DICT_PATH" "Dictionary path" || exit 1
check_path "file" "$MODEL_PATH" "Model file" || exit 1
check_path "file" "$DATA_PATH" "Data file" || exit 1
check_path "directory" "$TOKENIZER_PATH" "Tokenizer path" || exit 1

# Create output directory if it doesn't exist
echo "--- Preparing output directory ---"
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
create_directory "$OUTPUT_DIR" "output" || exit 1

echo "INFO: Parameters validated:"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  MAX_LEN: $MAX_LEN"
echo "  NUM_WORKERS: $NUM_WORKERS"

# Check if Python script exists
SCORER_SCRIPT="$PROJECT_PATH/gemgen/scorer.py"
check_path "file" "$SCORER_SCRIPT" "Scorer script" || exit 1

# Execute the command
echo "--- Starting scoring process ---"
echo "INFO: Input data: $DATA_PATH"
echo "INFO: Output will be saved to: $OUTPUT_PATH"
echo "INFO: Starting execution..."

# Ensure the correct CUDA runtime is used (avoid system CUDA overriding pip wheels).
LD_LIBRARY_PATH= python "$SCORER_SCRIPT" \
    --dict_path "$DICT_PATH" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    --num_workers "$NUM_WORKERS" \
    --save_input_columns