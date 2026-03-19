#!/bin/bash
# ============================================================================
# evaluate.sh
# Evaluate VLMs with and without L2V-CoT intervention using VLMEvalKit.
#
# This script runs evaluation on standard multimodal reasoning benchmarks
# to reproduce the results from the L2V-CoT paper.
#
# Prerequisites:
#   1. Install VLMEvalKit:
#      git clone https://github.com/open-compass/VLMEvalKit.git
#      cd VLMEvalKit && pip install -e .
#
#   2. Extract CoT representations:
#      bash scripts/extract_cot.sh
#
# Usage:
#   bash scripts/evaluate.sh
#
# Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
# ============================================================================

set -e

# ─── Configuration ────────────────────────────────────────────────────────────

# Target VLM for evaluation
VLM_MODEL="llava-hf/llava-1.5-7b-hf"

# Path to pre-extracted CoT representations
COT_REPR_PATH="outputs/cot_repr/llama3_8b_layer31.pt"

# VLM transformer layers to apply intervention to
# (last third of the model by default, but can be specified explicitly)
# For LLaVA-1.5-7b (32 layers): layers 20-31
TARGET_LAYERS="20 21 22 23 24 25 26 27 28 29 30 31"

# Intervention hyperparameters
INTERVENTION_STRENGTH=1.5
CUTOFF_RATIO=0.1

# Output directory
OUTPUT_DIR="results/l2v_cot"

# Benchmarks to evaluate on (space-separated)
# Standard benchmarks from the L2V-CoT paper:
BENCHMARKS="ScienceQA_VAL MMBench_DEV_EN MMStar MathVista_MINI"

# Device
DEVICE="cuda"

# ─── Setup ───────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"

echo "=== L2V-CoT Evaluation ==="
echo "VLM Model: $VLM_MODEL"
echo "CoT Representations: $COT_REPR_PATH"
echo "Target Layers: $TARGET_LAYERS"
echo "Benchmarks: $BENCHMARKS"
echo ""

# ─── Baseline evaluation (no intervention) ────────────────────────────────────

echo "--- Baseline evaluation (no intervention) ---"
python run_l2v_cot.py \
    --vlm_model "$VLM_MODEL" \
    --cot_representations "$COT_REPR_PATH" \
    --baseline \
    --eval_mode \
    --benchmark "$BENCHMARKS" \
    --output_dir "${OUTPUT_DIR}/baseline" \
    --device "$DEVICE"

echo ""
echo "--- L2V-CoT evaluation (with intervention) ---"
python run_l2v_cot.py \
    --vlm_model "$VLM_MODEL" \
    --cot_representations "$COT_REPR_PATH" \
    --target_layers $TARGET_LAYERS \
    --intervention_strength "$INTERVENTION_STRENGTH" \
    --cutoff_ratio "$CUTOFF_RATIO" \
    --eval_mode \
    --benchmark "$BENCHMARKS" \
    --output_dir "${OUTPUT_DIR}/l2v_cot" \
    --device "$DEVICE"

echo ""
echo "=== Evaluation complete! ==="
echo "Results saved to: $OUTPUT_DIR"
