#!/bin/bash

# Configure cache directories
export HF_HOME='/path/to/model_hub'
export PYTORCH_KERNEL_CACHE_PATH='/path/to/cache'
export CUDA_VISIBLE_DEVICES='0'  # Specify which GPU to use

# Optional: Set Hugging Face token if needed
export HF_TOKEN="your_hf_token_here"

# Change to the project directory
cd /path/to/VLMmerging/

# Source conda to make conda activate available in this script
eval "$(conda shell.bash hook)"
conda activate your_environment

# Output directory for merged models
OUTPUT_DIR='/path/to/merged_models'

############# LLaVA #############
# basemodel_path "meta-llama/Meta-Llama-3-8B"

# Dart-Uniform
echo "Merging models with linear interpolation..."
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path hkust-nlp/dart-math-llama3-8b-uniform \
    --output_dir "${OUTPUT_DIR}/llava_dart_uniform" --alpha 0.9 --mode 'base'

# Dart-Prop
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path hkust-nlp/dart-math-llama3-8b-prop2diff \
    --output_dir "${OUTPUT_DIR}/llava_dart_prop" --alpha 0.9 --mode 'base'

# Mammoth-1
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path EtashGuha/llama3-mammoth-dcft \
    --output_dir "${OUTPUT_DIR}/llava_mammoth_1" --alpha 0.9 --mode 'base'

# Mammoth-2
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path TIGER-Lab/MAmmoTH2-8B \
    --output_dir "${OUTPUT_DIR}/llava_mammoth_2" --alpha 0.9 --mode 'base'

# Magpie
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path Magpie-Align/Llama-3-8B-Magpie-Align-SFT-v0.3 \
    --output_dir "${OUTPUT_DIR}/llava_magpie" --alpha 0.9 --mode 'base'

############# Idefics #############
# basemodel_path "mistralai/Mistral-7B-v0.1"

# Meta-Math
python merge.py --model1_path HuggingFaceM4/idefics2-8b --model2_path meta-math/MetaMath-Mistral-7B \
    --output_dir "${OUTPUT_DIR}/idefics_meta_math" --alpha 0.9 --mode 'base'

# Dart-Uniform
python merge.py --model1_path HuggingFaceM4/idefics2-8b --model2_path hkust-nlp/dart-math-mistral-7b-uniform \
    --output_dir "${OUTPUT_DIR}/idefics_dart_uniform" --alpha 0.9 --mode 'base'

# Dart-Prop
python merge.py --model1_path HuggingFaceM4/idefics2-8b --model2_path hkust-nlp/dart-math-mistral-7b-prop2diff \
    --output_dir "${OUTPUT_DIR}/idefics_dart_prop" --alpha 0.9 --mode 'base'

# Mammoth-1
python merge.py --model1_path HuggingFaceM4/idefics2-8b --model2_path TIGER-Lab/MAmmoTH-7B-Mistral \
    --output_dir "${OUTPUT_DIR}/idefics_mammoth_1" --alpha 0.9 --mode 'base'

# Mammoth-2
python merge.py --model1_path HuggingFaceM4/idefics2-8b --model2_path TIGER-Lab/MAmmoTH2-7B \
    --output_dir "${OUTPUT_DIR}/idefics_mammoth_2" --alpha 0.9 --mode 'base'

############# InternVL #############
# basemodel_path "OpenGVLab/InternVL2-Llama3-76B"

# Dart-Uniform
python merge.py --model1_path OpenGVLab/InternVL2-Llama3-76B --model2_path hkust-nlp/dart-math-llama3-70b-uniform \
    --output_dir "${OUTPUT_DIR}/internvl_dart_uniform" --alpha 0.9 --mode 'base'

############# other merging methods #############

# TIES merging
# Results of the following merging: MathVision: 14.47	MathVerse-TextDominant: 31.35	MathVista: 37.1
echo "Merging models with TIES strategy..."
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path hkust-nlp/dart-math-llama3-8b-prop2diff \
    --basemodel_path meta-llama/Meta-Llama-3-8B \
    --output_dir "${OUTPUT_DIR}/ties_merge" --alpha 1.6 --mode 'ties' --density 0.2 --alpha2 0.2

# Layer swapping
# Results of the following merging: MathVision: 15.13	MathVerse-TextDominant: 29.57	MathVista: 37.9
echo "Merging models with layer swapping strategy..."
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path hkust-nlp/dart-math-llama3-8b-prop2diff \
    --basemodel_path meta-llama/Meta-Llama-3-8B \
    --output_dir "${OUTPUT_DIR}/layerswap_merge" --alpha 0.9 --mode 'layerswap' --base_layer_num 5

echo "All merging operations completed successfully!"