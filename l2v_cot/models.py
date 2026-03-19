"""
Model loading utilities for L2V-CoT.

Provides convenience functions for loading LLMs and VLMs
supported in the L2V-CoT experiments.

Supported LLMs (source models for CoT extraction):
  - LLaMA-2/3 (meta-llama/*)
  - Mistral (mistralai/*)
  - Phi-3 (microsoft/Phi-3*)
  - Any AutoModelForCausalLM-compatible HuggingFace model

Supported VLMs (target models for CoT injection):
  - LLaVA-1.5/1.6 (llava-hf/*)
  - Qwen-VL (Qwen/*)
  - InstructBLIP (Salesforce/*)
  - Any HuggingFace VLM with transformer layers

Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
"""

import torch
from typing import Optional, Tuple, Union

# These are imported at function call time to avoid requiring all dependencies
# when only a subset of models are needed.


def load_llm(
    model_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple:
    """
    Load a Large Language Model (LLM) for CoT representation extraction.

    Args:
        model_path: HuggingFace model path or local directory.
                    Examples:
                      - "meta-llama/Llama-2-7b-hf"
                      - "meta-llama/Meta-Llama-3-8B-Instruct"
                      - "mistralai/Mistral-7B-Instruct-v0.2"
        device: Target device ('cuda', 'cpu', or 'auto').
        dtype: Model dtype. If None, uses bfloat16 on CUDA, float32 on CPU.
        load_in_8bit: Load in 8-bit quantization (requires bitsandbytes).
        load_in_4bit: Load in 4-bit quantization (requires bitsandbytes).

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading LLM: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    elif device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if device not in ("auto",) and not load_in_8bit and not load_in_4bit:
        model = model.to(device)

    model.eval()
    print(f"LLM loaded: {model_path} on {device}")
    return model, tokenizer


def load_vlm(
    model_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple:
    """
    Load a Vision-Language Model (VLM) for CoT intervention.

    The function auto-detects the VLM type and loads the appropriate
    model class and processor.

    Args:
        model_path: HuggingFace model path or local directory.
                    Examples:
                      - "llava-hf/llava-1.5-7b-hf"
                      - "llava-hf/llama3-llava-next-8b-hf"
                      - "Qwen/Qwen-VL-Chat"
                      - "Salesforce/instructblip-vicuna-7b"
        device: Target device ('cuda', 'cpu', or 'auto').
        dtype: Model dtype. If None, uses bfloat16 on CUDA, float32 on CPU.
        load_in_8bit: Load in 8-bit quantization (requires bitsandbytes).
        load_in_4bit: Load in 4-bit quantization (requires bitsandbytes).

    Returns:
        Tuple of (model, processor).
        The processor handles both text and image inputs.
    """
    from transformers import AutoConfig

    if dtype is None:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading VLM: {model_path}")

    # Detect model type from config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])
    arch_str = " ".join(architectures).lower()

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    elif device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    if "llava" in model_type or "llava" in arch_str:
        model, processor = _load_llava(model_path, model_kwargs)
    elif "qwen" in model_type:
        model, processor = _load_qwen_vl(model_path, model_kwargs)
    elif "instructblip" in model_type or "instructblip" in arch_str:
        model, processor = _load_instructblip(model_path, model_kwargs)
    else:
        # Generic loading for other VLMs
        model, processor = _load_generic_vlm(model_path, model_kwargs)

    if device not in ("auto",) and not load_in_8bit and not load_in_4bit:
        model = model.to(device)

    model.eval()
    print(f"VLM loaded: {model_path} on {device}")
    return model, processor


def _load_llava(model_path: str, model_kwargs: dict) -> Tuple:
    """Load LLaVA model (1.5 or 1.6/Next)."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, **model_kwargs
    )
    return model, processor


def _load_qwen_vl(model_path: str, model_kwargs: dict) -> Tuple:
    """Load Qwen-VL model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    return model, tokenizer


def _load_instructblip(model_path: str, model_kwargs: dict) -> Tuple:
    """Load InstructBLIP model."""
    from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

    processor = InstructBlipProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_path, **model_kwargs
    )
    return model, processor


def _load_generic_vlm(model_path: str, model_kwargs: dict) -> Tuple:
    """Generic VLM loading via AutoModel."""
    from transformers import AutoModel, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    except Exception:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    return model, processor


def get_model_info(model_path: str) -> dict:
    """
    Get basic information about a model without loading its weights.

    Args:
        model_path: HuggingFace model path.

    Returns:
        Dictionary with model type, hidden size, num layers, etc.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    info = {
        "model_type": getattr(config, "model_type", "unknown"),
        "architectures": getattr(config, "architectures", []),
    }

    # Try to get hidden size and num layers
    for attr in ["hidden_size", "d_model", "n_embd"]:
        if hasattr(config, attr):
            info["hidden_size"] = getattr(config, attr)
            break

    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(config, attr):
            info["num_layers"] = getattr(config, attr)
            break

    return info
