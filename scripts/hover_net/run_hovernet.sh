#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SCRIPT_DIR}/pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

datasets=(
    INT1 INT2 INT3 INT4 INT5 INT6 INT7 INT8 INT9 INT10
    INT11 INT12 INT13 INT14 INT15 INT16 INT17 INT18 INT19 INT20
    INT21 INT22 INT23 INT24
    NCBI642 NCBI643
    NCBI681 NCBI682 NCBI683 NCBI684
)

for dataset in "${datasets[@]}"; do
    INPUT_DIR="/data/${dataset}/patches"
    OUTPUT_DIR="/data/${dataset}/segment"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "Skipping ${dataset}: patches directory not found"
        continue
    fi

    echo "Processing ${dataset}..."
    python3 "${SCRIPT_DIR}/run_infer.py" \
        --gpu='0' \
        --nr_types=0 \
        --batch_size=32 \
        --model_mode=fast \
        --model_path="$MODEL_PATH" \
        --nr_inference_workers=0 \
        --nr_post_proc_workers=4 \
        tile \
        --input_dir="$INPUT_DIR" \
        --output_dir="$OUTPUT_DIR" \
        --mem_usage=0.1 \
        --draw_dot \
        --save_qupath

    echo "Finished ${dataset}"
    echo "---"
done

echo "All datasets processed."
