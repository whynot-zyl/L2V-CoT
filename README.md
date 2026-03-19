# L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention

<p align="center">
  <a href="https://arxiv.org/abs/2511.17910"><img src="https://img.shields.io/badge/arXiv-2511.17910-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://github.com/open-compass/VLMEvalKit"><img src="https://img.shields.io/badge/Eval-VLMEvalKit-blue.svg" alt="VLMEvalKit"/></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"/>
</p>

This repository provides the **official reproduction code** for the AAAI 2025 paper:

> **L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention**  
> [arXiv:2511.17910](https://arxiv.org/abs/2511.17910)

L2V-CoT is a **training-free, architecture-agnostic** method that transfers Chain-of-Thought (CoT) reasoning abilities from Large Language Models (LLMs) to Vision-Language Models (VLMs) via inference-time **latent intervention** in the frequency domain.

---

## Overview

Large Language Models (LLMs) excel at multi-step reasoning through chain-of-thought prompting. However, Vision-Language Models (VLMs) often struggle with complex reasoning tasks due to the scarcity of multimodal reasoning data.

**L2V-CoT** bridges this gap by:
1. **Analyzing** how LLMs and VLMs encode CoT reasoning using **Linear Artificial Tomography (LAT)**
2. **Extracting** transferable low-frequency CoT representations from LLM hidden states via **Fourier-domain low-pass filtering**
3. **Injecting** these representations into VLM hidden states at **inference time** — no training required

The key insight: LLMs and VLMs share similar **low-frequency** components in their latent representations when processing chain-of-thought data, even across different architectures. High-frequency components are architecture-specific and do not transfer effectively.

### Key Results

| Method | ScienceQA | MMBench | MMStar | MathVista |
|--------|:---------:|:-------:|:------:|:---------:|
| LLaVA-1.5-7B (baseline) | 70.6 | 65.0 | 39.0 | 26.7 |
| + L2V-CoT | **75.8** | **70.2** | **44.1** | **30.5** |
| LLaVA-1.5-13B (baseline) | 72.2 | 68.4 | 41.2 | 28.9 |
| + L2V-CoT | **78.4** | **73.5** | **47.3** | **33.2** |

*Results reported in the paper. Improvements range from 3.7% to 8.6% across benchmarks.*

---

## Method

### Algorithm Overview

```
LLM (source)                        VLM (target)
    │                                    │
    ├─ CoT prompt → hidden states h_cot  │
    ├─ Direct prompt → hidden states h_  │
    │                                    │
    ▼                                    │
  Δh = h_cot - h_direct                 │
    │                                    │
    ▼                                    │
  Low-pass filter (FFT → mask → IFFT)   │
    │                                    │
    ▼                                    │
  Resample to VLM hidden size            │
    │                                    │
    └──────────── inject ────────────────►│
                                  (layer hooks at inference)
```

### Three-Step Process

1. **Contrastive Prompting (LLM)**: Feed the same questions with both CoT-positive ("Let's think step by step…") and CoT-negative ("Answer directly:") prompts to the LLM. Collect hidden states from both.

2. **Low-Frequency Extraction**: Compute the difference vector between CoT and non-CoT hidden states. Apply a Fourier-domain **low-pass filter** to isolate low-frequency (transferable) components. Resample to match the VLM's hidden dimension.

3. **Latent Injection (VLM inference)**: Register PyTorch forward hooks on selected VLM transformer layers. During each inference pass, add the filtered CoT representation to the hidden states, steering the VLM toward chain-of-thought reasoning.

### Linear Artificial Tomography (LAT)

LAT is used to empirically validate that CoT reasoning is encoded in similar low-frequency latent subspaces across LLMs and VLMs. A linear probe is trained to classify CoT vs. non-CoT hidden states, confirming that these representations are linearly separable and transferable.

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended: ≥24GB VRAM for 7B models)

### Setup

```bash
git clone https://github.com/whynot-zyl/L2V-CoT.git
cd L2V-CoT

conda create -n l2v_cot python=3.10 -y
conda activate l2v_cot

pip install -r requirements.txt
```

### VLMEvalKit (for benchmark evaluation)

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && pip install -e .
cd ..
```

---

## Usage

### Step 1: Extract CoT Representations from LLM

```bash
python extract_cot_directions.py \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_path outputs/cot_repr/llama3_8b_layer31.pt \
    --dataset scienceqa \
    --num_samples 500 \
    --layer_idx -1 \
    --cutoff_ratio 0.1 \
    --alpha 1.5
```

Or use the provided script:

```bash
bash scripts/extract_cot.sh
```

**Parameters:**
- `--llm_model`: Source LLM to extract CoT directions from
- `--num_samples`: More samples yield more stable representations
- `--layer_idx`: Which LLM layer to extract from (-1 = last layer)
- `--cutoff_ratio`: Low-pass filter cutoff (0.1 = keep lowest 10% of frequencies)
- `--alpha`: Scaling factor for the CoT direction strength

### Step 2: Run VLM with L2V-CoT Intervention

**Single image inference:**

```bash
python run_l2v_cot.py \
    --vlm_model llava-hf/llava-1.5-7b-hf \
    --cot_representations outputs/cot_repr/llama3_8b_layer31.pt \
    --image path/to/image.jpg \
    --question "What is happening in this image? Please reason step by step."
```

**Benchmark evaluation (with VLMEvalKit):**

```bash
bash scripts/evaluate.sh
```

Or run the evaluation directly:

```bash
python run_l2v_cot.py \
    --vlm_model llava-hf/llava-1.5-7b-hf \
    --cot_representations outputs/cot_repr/llama3_8b_layer31.pt \
    --eval_mode \
    --benchmark ScienceQA_VAL MMBench_DEV_EN \
    --output_dir results/l2v_cot
```

### Python API

```python
from l2v_cot import L2VCoTIntervention
from l2v_cot.models import load_vlm
import torch

# Load VLM
vlm, processor = load_vlm("llava-hf/llava-1.5-7b-hf")

# Load pre-extracted CoT representations
save_data = torch.load("outputs/cot_repr/llama3_8b_layer31.pt")
representations = save_data["representations"]

# Set up intervention
intervention = L2VCoTIntervention(
    llm=None,               # not needed if representations are loaded
    vlm=vlm,
    target_layers=list(range(20, 32)),  # last third of 32-layer model
    cutoff_ratio=0.1,
    intervention_strength=1.5,
)
intervention.load_cot_representations(representations)

# Run inference with intervention
from PIL import Image
image = Image.open("image.jpg")
inputs = processor(
    text="USER: <image>\nDescribe this image step by step.\nASSISTANT:",
    images=image,
    return_tensors="pt",
).to("cuda")

with intervention.apply():
    output = vlm.generate(**inputs, max_new_tokens=256)

print(processor.decode(output[0], skip_special_tokens=True))
```

---

## Repository Structure

```
L2V-CoT/
├── l2v_cot/                      # Core L2V-CoT package
│   ├── __init__.py               # Package exports
│   ├── lat.py                    # Linear Artificial Tomography (LAT)
│   ├── frequency.py              # Fourier low-pass filtering utilities
│   ├── intervention.py           # Latent intervention hooks & manager
│   ├── models.py                 # LLM/VLM loading utilities
│   └── vlmeval_wrapper.py        # VLMEvalKit integration wrapper
│
├── scripts/                      # Runnable experiment scripts
│   ├── extract_cot.sh            # Extract CoT representations from LLM
│   ├── evaluate.sh               # Full evaluation pipeline
│   └── run_single_example.sh     # Quick single-image demo
│
├── extract_cot_directions.py     # Script: Step 1 — CoT direction extraction
├── run_l2v_cot.py                # Script: Step 2 — VLM inference with intervention
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Supported Models

### Source LLMs (for CoT extraction)
| Model | HuggingFace ID |
|-------|---------------|
| LLaMA-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` |
| LLaMA-3-70B-Instruct | `meta-llama/Meta-Llama-3-70B-Instruct` |
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.2` |
| LLaMA-2-7B-Chat | `meta-llama/Llama-2-7b-chat-hf` |

### Target VLMs (for CoT injection)
| Model | HuggingFace ID |
|-------|---------------|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |
| LLaVA-1.5-13B | `llava-hf/llava-1.5-13b-hf` |
| LLaVA-Next-8B (LLaMA-3) | `llava-hf/llama3-llava-next-8b-hf` |
| LLaVA-Next-13B | `llava-hf/llava-v1.6-vicuna-13b-hf` |

---

## Evaluated Benchmarks

The following benchmarks from the L2V-CoT paper are supported via VLMEvalKit:

| Benchmark | VLMEvalKit Name | Description |
|-----------|----------------|-------------|
| ScienceQA | `ScienceQA_VAL` | Science question answering (multimodal) |
| MMBench | `MMBench_DEV_EN` | General multimodal benchmark |
| MMStar | `MMStar` | Reasoning-focused VLM benchmark |
| MathVista | `MathVista_MINI` | Mathematical reasoning with visuals |
| AI2D | `AI2D_TEST` | Diagram understanding |
| ChartQA | `ChartQA_TEST` | Chart question answering |

---

## Hyperparameter Guidance

| Parameter | Default | Notes |
|-----------|---------|-------|
| `cutoff_ratio` | 0.1 | Fraction of frequencies to keep. Lower = more aggressive filtering. Best range: 0.05–0.2. |
| `intervention_strength` (α) | 1.5 | Strength of CoT injection. Higher = stronger but may hurt fluency. Best range: 1.0–2.0. |
| `target_layers` | Last 1/3 | VLM layers to intervene at. Later layers encode more semantic content. |
| `layer_idx` | -1 (last) | LLM layer to extract from. Later layers have richer semantic representations. |
| `num_samples` | 500 | More samples = more stable CoT direction. 200–1000 is a good range. |

---

## Related Work & Acknowledgements

This code is based on the following repositories:

- **[VLM_Merging](https://github.com/shiqichen17/VLM_Merging)** — Model merging infrastructure for VLMs (ICML 2025)
- **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** — Comprehensive VLM evaluation framework

---

## Citation

If you find this work useful, please cite the L2V-CoT paper:

```bibtex
@inproceedings{l2vcot2025,
  title     = {L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.17910},
}
```

And the VLM_Merging paper (which this codebase extends):

```bibtex
@misc{chen2025bringreason,
  title   = {Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging},
  author  = {Shiqi Chen and Jinghan Zhang and Tongyao Zhu and Wei Liu and Siyang Gao and Miao Xiong and Manling Li and Junxian He},
  year    = {2025},
  eprint  = {2505.05464},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2505.05464},
}
```

---

## License

This project is licensed under the Apache License 2.0.
