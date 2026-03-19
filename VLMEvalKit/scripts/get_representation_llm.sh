#!/bin/bash
CUDA_ID=7  # 设置你想使用的 GPU 编号，例如 0、1、2 等
# DOMAIN="math"
# DOMAIN="physic"
# DOMAIN="chemistry"
DOMAIN="biology"
# DOMAIN="code"
# DOMAIN="puzzle"
LAMBDA=0.1
SCRIPT_PATH="/home/zhanyuliang/Project/CoT/VLM_Merging/VLMEvalKit/get_representation_llm.py"

for LAYER in $(seq 27 32); do
  echo "🚀 Running layer $LAYER..."
  CUDA_VISIBLE_DEVICES=$CUDA_ID python "$SCRIPT_PATH" --domain "$DOMAIN" --layer "$LAYER" --lambda_inj "$LAMBDA"
done
