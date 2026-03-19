#!/bin/bash
# ============================================================================
# extract_cot.sh
# Extract CoT direction vectors from an LLM for L2V-CoT.
#
# This script runs the CoT direction extraction step for the L2V-CoT method.
# The extracted representations are saved and used for VLM intervention.
#
# Usage:
#   bash scripts/extract_cot.sh
#
# Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
# ============================================================================

set -e

# ─── Configuration ────────────────────────────────────────────────────────────

# Source LLM for CoT extraction (must be compatible with the VLM's language backbone)
LLM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Output path for CoT representations
OUTPUT_PATH="outputs/cot_repr/llama3_8b_layer31.pt"

# Dataset to draw questions from
DATASET="scienceqa"

# Number of samples to use (more = better representations, but slower)
NUM_SAMPLES=500

# LLM layer to extract from (-1 = last layer, which is default)
LAYER_IDX=-1

# Low-pass filter cutoff ratio (0.1 = keep lowest 10% of frequencies)
CUTOFF_RATIO=0.1

# Scaling factor for CoT direction strength
ALPHA=1.5

# VLM hidden size for resampling (set to the VLM's language model hidden size)
# For LLaVA-1.5-7b (Vicuna-7b backbone): 4096
# For LLaVA-1.5-13b (Vicuna-13b backbone): 5120
# For LLaVA-Next-8b (LLaMA-3-8b backbone): 4096
# Leave empty to skip resampling (only valid if LLM and VLM have same hidden size)
TARGET_HIDDEN_SIZE=4096

# Device settings
DEVICE="cuda"
BATCH_SIZE=4

# ─── Run extraction ───────────────────────────────────────────────────────────

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=== L2V-CoT: Extracting CoT Representations ==="
echo "Source LLM: $LLM_MODEL"
echo "Output: $OUTPUT_PATH"
echo "Dataset: $DATASET (${NUM_SAMPLES} samples)"
echo "Layer: $LAYER_IDX, Cutoff: $CUTOFF_RATIO, Alpha: $ALPHA"
echo ""

python extract_cot_directions.py \
    --llm_model "$LLM_MODEL" \
    --output_path "$OUTPUT_PATH" \
    --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" \
    --layer_idx "$LAYER_IDX" \
    --cutoff_ratio "$CUTOFF_RATIO" \
    --target_hidden_size "$TARGET_HIDDEN_SIZE" \
    --alpha "$ALPHA" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "=== Extraction complete! ==="
echo "CoT representations saved to: $OUTPUT_PATH"
