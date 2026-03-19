#!/bin/bash
CUDA_ID=6  # 设置你想使用的 GPU 编号，例如 0、1、2 等
# DOMAIN="math"
# DOMAIN="physic"
# DOMAIN="chemistry"
# DOMAIN="biology"
DOMAIN="code"
# DOMAIN="puzzle"
LAMBDA=0.1
SCRIPT_PATH="/home/zhanyuliang/Project/CoT/VLM_Merging/VLMEvalKit/get_representation.py"
# ModelName="llava_next_merge_7b"
ModelName="Qwen2-VL-7B-Instruct"
# ModelName="idefics2_8b"
# ModelName="InternVL2-8B"
for LAYER in $(seq 1 32); do
  echo "🚀 Running layer $LAYER..."
  CUDA_VISIBLE_DEVICES=$CUDA_ID python "$SCRIPT_PATH" --domain "$DOMAIN" --layer "$LAYER" --lambda_inj "$LAMBDA" --model_name "$ModelName"
done
