#!/bin/bash

PROJECT_PATH="path/to/GEMGen"
INPUT_JSON="$PROJECT_PATH/results/test_generator_output.json"
OUTPUT_CSV="$PROJECT_PATH/results/test_evaluation_output.csv"
GT_CSV="$PROJECT_PATH/data/test_gt.csv"

# --- Evaluation Options ---
EVALUATE_PROPERTIES=true      # Set to 'true' or 'false'
EVALUATE_HIT=false            # Set to 'true' or 'false'

# --- Parallelism ---
N_JOBS=8                     # Set to empty or comment out to use default (all CPUs)


if [ "$PROJECT_PATH" = "path/to/GEMGen" ]; then
    echo "ERROR: Please update PROJECT_PATH in the script."
    exit 1
fi

# Check input JSON
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input JSON file not found: $INPUT_JSON"
    exit 1
fi

# Handle evaluate_hit dependency
if [ "$EVALUATE_HIT" = true ]; then
    if [ -z "$GT_CSV" ] || [ ! -f "$GT_CSV" ]; then
        echo "Error: --evaluate_hit is enabled, but GT_CSV is missing or invalid: $GT_CSV"
        exit 1
    fi
fi

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"

CMD_ARGS=(
    --input_json "$INPUT_JSON"
    --output_csv "$OUTPUT_CSV"
)

if [ "$EVALUATE_PROPERTIES" = true ]; then
    CMD_ARGS+=(--evaluate_properties)
fi

if [ "$EVALUATE_HIT" = true ]; then
    CMD_ARGS+=(--evaluate_hit --gt_csv "$GT_CSV")
fi

if [ -n "$N_JOBS" ]; then
    CMD_ARGS+=(--n_jobs "$N_JOBS")
fi


echo "Running GEMGen evaluator with options:"
echo "  evaluate_properties: $EVALUATE_PROPERTIES"
echo "  evaluate_hit:        $EVALUATE_HIT"
[ "$EVALUATE_HIT" = true ] && echo "  gt_csv:              $GT_CSV"
[ -n "$N_JOBS" ] && echo "  n_jobs:              $N_JOBS"
echo "----------------------------------------"
echo "Input:  $INPUT_JSON"
echo "Output: $OUTPUT_CSV"
echo "----------------------------------------"

python -u "$PROJECT_PATH/gemgen/evaluate.py" "${CMD_ARGS[@]}"
