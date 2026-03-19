#!/bin/bash
# Configure cache directories and environment
unset http_proxy https_proxy
export TRANSFORMERS_CACHE='/mnt2/zhanyuliang/CoT/VLM_Merginf/model_hub/'
export HF_DATASETS_CACHE='/mnt2/zhanyuliang/CoT/VLM_Merginf/model_hub/'
export HF_HOME='/mnt2/zhanyuliang/CoT/VLM_Merginf/model_hub/'
export PYTORCH_KERNEL_CACHE_PATH='/mnt2/zhanyuliang/CoT/VLM_Merginf/cache/'
export CUDA_VISIBLE_DEVICES='4,5'  # Specify which GPUs to use
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY="sk-123456"
export OPENAI_API_BASE="http://0.0.0.0:23333/v1/chat/completions"
export LOCAL_LLM="internlm/internlm2-chat-1_8b"
# Optional: Set proxy and API keys if needed
# export https_proxy="your_proxy_here"
# export OPENAI_API_KEY='your_openai_key_here'

# Set CUDA environment if needed
# export CUDA_HOME="/path/to/cuda"
# export PATH="${CUDA_HOME}/bin:$PATH"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# Change to the project directory
cd /home/zhanyuliang/Project/CoT/VLM_Merging/VLMEvalKit

# Source conda to make conda activate available in this script
# eval "$(conda shell.bash hook)"
# conda activate vlmm

# Count available GPUs
export GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Detected ${GPU} GPUs for distributed training"

# Define evaluation settings
OUTPUT_DIR="/mnt2/zhanyuliang/CoT/VLM_Merginf/output/"
MERGED_MODELS_DIR="/mnt2/zhanyuliang/CoT/VLM_Merginf/merged_models/linear_merge"

# Define tasks and models as lists
tasks=(
    "MathVista_MINI"
    # "MathVerse_MINI_Vision_Only"
    # "MathVerse_MINI_Vision_Dominant"
    # "MathVerse_MINI_Vision_Intensive"
    # "MathVerse_MINI_Text_Lite"
    # "MathVerse_MINI_Text_Dominant"
    # "MathVerse_MINI_Text_Intensive"
    # "MathVision_MINI"
    # "MM-Math"
    # "DynaMath"
    # "MMStar"
)

models=(
    # "llava_next_merge_7b"
    # "Qwen2-VL-7B-Instruct"
    # "idefics2_8b"
    # "InternVL2-8B"    
    # "InternVL2-26B" 
    "InternVL2-40B" 
    # "InternVL2-76B"
)

# Define merged model weights to evaluate
merges=(
    "${MERGED_MODELS_DIR}/merged_model_0.5.pth"
    # "${MERGED_MODELS_DIR}/merged_model_0.7.pth"
    # "${MERGED_MODELS_DIR}/merged_model_0.9.pth"
)

echo "Starting evaluation of VLM models..."

# Run combinations of tasks and models
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        for lambda_inj in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            for inj_layer in $(seq 1 26); do
                # First evaluate base models without merging
                lambda_str=$(printf "%.1f" $lambda_inj)
                echo "Evaluating base model: ${model} on task: ${task}"
                torchrun --nproc-per-node=${GPU} --master-port=2224 run_txy_representation.py \
                    --data "$task" \
                    --model "$model" \
                    --verbose \
                    --work-dir "${OUTPUT_DIR}/base_models_${inj_layer}_${lambda_str}" \
                    --inj_layer "$inj_layer" \
                    --lambda_inj "$lambda_inj" \
                
                # Then evaluate with merged weights
                # for merge in "${merges[@]}"; do
                #     echo "Evaluating merged model: ${model} with weights: ${merge} on task: ${task}"
                #     torchrun --nproc-per-node=${GPU} --master-port=2345 run_txy.py \
                #         --data "$task" \
                #         --model "$model" \
                #         --verbose \
                #         --merge_model "$merge" \
                #         --work-dir "${OUTPUT_DIR}/merged_models"
                # done
            done
        done
    done
done

echo "All evaluation tasks completed successfully!" 