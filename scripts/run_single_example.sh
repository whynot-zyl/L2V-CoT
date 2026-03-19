#!/bin/bash
# ============================================================================
# run_single_example.sh
# Run L2V-CoT on a single image+question for a quick demonstration.
#
# Usage:
#   bash scripts/run_single_example.sh --image /path/to/image.jpg \
#       --question "What is in this image?"
#
# Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
# ============================================================================

set -e

# Default values
IMAGE_PATH="${IMAGE:-}"
QUESTION="${QUESTION:-What do you see in this image? Please reason step by step.}"
VLM_MODEL="${VLM_MODEL:-llava-hf/llava-1.5-7b-hf}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
COT_REPR_PATH="${COT_REPR_PATH:-outputs/cot_repr/llama3_8b_layer31.pt}"
DEVICE="${DEVICE:-cuda}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image) IMAGE_PATH="$2"; shift 2;;
        --question) QUESTION="$2"; shift 2;;
        --vlm_model) VLM_MODEL="$2"; shift 2;;
        --llm_model) LLM_MODEL="$2"; shift 2;;
        --cot_repr) COT_REPR_PATH="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$IMAGE_PATH" ]; then
    echo "Error: --image is required"
    echo "Usage: bash scripts/run_single_example.sh --image /path/to/image.jpg"
    exit 1
fi

echo "=== L2V-CoT Single Image Inference ==="
echo "VLM: $VLM_MODEL"
echo "Image: $IMAGE_PATH"
echo "Question: $QUESTION"
echo ""

# Use pre-extracted representations if available, otherwise extract on-the-fly
if [ -f "$COT_REPR_PATH" ]; then
    echo "Using pre-extracted CoT representations: $COT_REPR_PATH"
    python run_l2v_cot.py \
        --vlm_model "$VLM_MODEL" \
        --cot_representations "$COT_REPR_PATH" \
        --image "$IMAGE_PATH" \
        --question "$QUESTION" \
        --device "$DEVICE"
else
    echo "Pre-extracted representations not found."
    echo "Extracting on-the-fly from LLM: $LLM_MODEL"
    python run_l2v_cot.py \
        --vlm_model "$VLM_MODEL" \
        --llm_model "$LLM_MODEL" \
        --image "$IMAGE_PATH" \
        --question "$QUESTION" \
        --device "$DEVICE"
fi
