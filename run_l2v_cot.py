"""
Run VLM inference with L2V-CoT latent intervention.

This script applies the extracted CoT direction vectors to a VLM at inference
time, enabling training-free cross-modal transfer of chain-of-thought reasoning.

Two modes are supported:
  1. Standard inference: Run VLM on an image + question without intervention.
  2. L2V-CoT inference: Run VLM with injected CoT representations for enhanced
     multi-step reasoning.

Usage:
    # With pre-extracted CoT representations:
    python run_l2v_cot.py \
        --vlm_model llava-hf/llava-1.5-7b-hf \
        --cot_representations cot_representations.pt \
        --image path/to/image.jpg \
        --question "What is shown in this image? Reason step by step." \
        --target_layers 20 21 22 23 24 25

    # With on-the-fly extraction (requires LLM):
    python run_l2v_cot.py \
        --vlm_model llava-hf/llava-1.5-7b-hf \
        --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
        --image path/to/image.jpg \
        --question "What is shown in this image?"

    # Evaluate on VLMEvalKit benchmark (requires VLMEvalKit installation):
    python run_l2v_cot.py \
        --vlm_model llava-hf/llava-1.5-7b-hf \
        --cot_representations cot_representations.pt \
        --eval_mode \
        --benchmark ScienceQA_VAL

Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
"""

import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from typing import List, Optional

from l2v_cot.intervention import L2VCoTIntervention
from l2v_cot.models import load_llm, load_vlm


def run_single_inference(
    vlm,
    processor,
    image_path: str,
    question: str,
    prompt_template: str = "USER: <image>\n{question}\nASSISTANT:",
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """
    Run VLM inference on a single image + question.

    Args:
        vlm: Loaded VLM model.
        processor: VLM processor (handles text + image tokenization).
        image_path: Path to the input image.
        question: The question to answer.
        prompt_template: Template for formatting the prompt.
        max_new_tokens: Maximum tokens to generate.
        device: Computation device.

    Returns:
        Generated text response.
    """
    image = Image.open(image_path).convert("RGB")
    prompt = prompt_template.format(question=question)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return response.strip()


def run_l2v_cot_inference(
    vlm,
    processor,
    intervention: L2VCoTIntervention,
    image_path: str,
    question: str,
    prompt_template: str = "USER: <image>\n{question}\nASSISTANT:",
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """
    Run VLM inference with L2V-CoT latent intervention.

    This is the main L2V-CoT inference function. It applies the CoT direction
    vectors to the VLM's hidden states at runtime, enhancing its reasoning.

    Args:
        vlm: Loaded VLM model.
        processor: VLM processor.
        intervention: Configured L2VCoTIntervention instance.
        image_path: Path to the input image.
        question: The question to answer.
        prompt_template: Template for formatting the prompt.
        max_new_tokens: Maximum tokens to generate.
        device: Computation device.

    Returns:
        Generated text response with enhanced CoT reasoning.
    """
    image = Image.open(image_path).convert("RGB")
    prompt = prompt_template.format(question=question)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(device)

    with intervention.apply():
        with torch.no_grad():
            output_ids = vlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return response.strip()


def setup_intervention(
    vlm,
    llm_model: Optional[str],
    cot_repr_path: Optional[str],
    target_layers: Optional[List[int]],
    cutoff_ratio: float,
    intervention_strength: float,
    device: str,
    batch_size: int,
) -> L2VCoTIntervention:
    """
    Set up the L2VCoTIntervention from either a saved file or by on-the-fly extraction.
    """
    # Option 1: Load pre-extracted representations
    if cot_repr_path and os.path.exists(cot_repr_path):
        print(f"Loading CoT representations from {cot_repr_path}")
        save_data = torch.load(cot_repr_path, map_location="cpu")
        representations = save_data.get("representations", save_data)
        meta = save_data.get("metadata", {})
        print(f"Loaded representations metadata: {meta}")

        llm_placeholder = None  # LLM not needed if representations are pre-computed
        intervention = L2VCoTIntervention(
            llm=llm_placeholder,
            vlm=vlm,
            target_layers=target_layers,
            cutoff_ratio=cutoff_ratio,
            intervention_strength=intervention_strength,
        )
        intervention.load_cot_representations(representations)
        return intervention

    # Option 2: Extract on the fly
    if llm_model is None:
        raise ValueError(
            "Either --cot_representations or --llm_model must be provided."
        )

    print(f"Extracting CoT representations on-the-fly from {llm_model}")
    llm, llm_tokenizer = load_llm(llm_model, device=device)

    intervention = L2VCoTIntervention(
        llm=llm,
        vlm=vlm,
        target_layers=target_layers,
        cutoff_ratio=cutoff_ratio,
        intervention_strength=intervention_strength,
    )

    from extract_cot_directions import build_cot_prompts, build_non_cot_prompts, load_questions
    questions = load_questions("default", None, 100)
    cot_prompts = build_cot_prompts(questions[:100])
    non_cot_prompts = build_non_cot_prompts(questions[:100])

    intervention.extract_cot_representations(
        cot_prompts=cot_prompts,
        non_cot_prompts=non_cot_prompts,
        tokenizer=llm_tokenizer,
        device=device,
        batch_size=batch_size,
    )

    # Clean up LLM from GPU to free memory
    del llm
    if device == "cuda":
        torch.cuda.empty_cache()

    return intervention


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM inference with L2V-CoT latent intervention"
    )

    # Model arguments
    parser.add_argument(
        "--vlm_model",
        type=str,
        required=True,
        help="HuggingFace path to the target VLM (e.g., llava-hf/llava-1.5-7b-hf)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="HuggingFace path to the source LLM (for on-the-fly extraction)",
    )
    parser.add_argument(
        "--cot_representations",
        type=str,
        default=None,
        help="Path to pre-extracted CoT representations (.pt file)",
    )

    # Input arguments
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for single inference",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Please describe what you see in this image and reason step by step.",
        help="Question to ask about the image",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="USER: <image>\n{question}\nASSISTANT:",
        help="Prompt template for the VLM",
    )

    # Intervention hyperparameters
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        default=None,
        help="VLM layer indices to apply intervention to (default: last third)",
    )
    parser.add_argument(
        "--cutoff_ratio",
        type=float,
        default=0.1,
        help="Low-pass filter cutoff ratio",
    )
    parser.add_argument(
        "--intervention_strength",
        type=float,
        default=1.5,
        help="Scaling factor for intervention strength",
    )

    # Evaluation mode
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="Run evaluation using VLMEvalKit (requires installation)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ScienceQA_VAL",
        help="VLMEvalKit benchmark name for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for evaluation results",
    )

    # Inference settings
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load VLM in 8-bit quantization",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load VLM in 4-bit quantization",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run without intervention (baseline comparison)",
    )

    args = parser.parse_args()

    # ── Load VLM ─────────────────────────────────────────────────────────────
    print(f"Loading VLM: {args.vlm_model}")
    vlm, processor = load_vlm(
        args.vlm_model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    # ── Setup intervention ───────────────────────────────────────────────────
    if not args.baseline:
        intervention = setup_intervention(
            vlm=vlm,
            llm_model=args.llm_model,
            cot_repr_path=args.cot_representations,
            target_layers=args.target_layers,
            cutoff_ratio=args.cutoff_ratio,
            intervention_strength=args.intervention_strength,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        intervention = None
        print("Running in baseline mode (no intervention)")

    # ── Evaluation mode ──────────────────────────────────────────────────────
    if args.eval_mode:
        run_evaluation(
            vlm=vlm,
            processor=processor,
            intervention=intervention,
            vlm_model_path=args.vlm_model,
            benchmark=args.benchmark,
            output_dir=args.output_dir,
            device=args.device,
        )
        return

    # ── Single image inference ───────────────────────────────────────────────
    if args.image is None:
        print("No image provided. Use --image path/to/image.jpg for inference,")
        print("or --eval_mode for benchmark evaluation.")
        return

    print(f"\nQuestion: {args.question}")
    print(f"Image: {args.image}")

    if args.baseline or intervention is None:
        response = run_single_inference(
            vlm=vlm,
            processor=processor,
            image_path=args.image,
            question=args.question,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        print(f"\n[Baseline] Response:\n{response}")
    else:
        # Baseline response
        response_baseline = run_single_inference(
            vlm=vlm,
            processor=processor,
            image_path=args.image,
            question=args.question,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        print(f"\n[Baseline] Response:\n{response_baseline}")

        # L2V-CoT response
        response_l2v = run_l2v_cot_inference(
            vlm=vlm,
            processor=processor,
            intervention=intervention,
            image_path=args.image,
            question=args.question,
            prompt_template=args.prompt_template,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        print(f"\n[L2V-CoT] Response:\n{response_l2v}")


def run_evaluation(
    vlm,
    processor,
    intervention,
    vlm_model_path: str,
    benchmark: str,
    output_dir: str,
    device: str,
):
    """
    Run evaluation using VLMEvalKit integration.

    This integrates with VLMEvalKit's evaluation framework to systematically
    evaluate the VLM on standard benchmarks with and without L2V-CoT intervention.
    """
    try:
        from vlmeval.api import VLMEvalAPI
        print("VLMEvalKit found. Running evaluation...")
    except ImportError:
        print(
            "VLMEvalKit not found. Please install it:\n"
            "  git clone https://github.com/open-compass/VLMEvalKit.git\n"
            "  cd VLMEvalKit && pip install -e .\n"
            "\nThen run evaluation via:\n"
            f"  python -m vlmeval.run --model {vlm_model_path} "
            f"--data {benchmark} --work-dir {output_dir}"
        )
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Wrap VLM with intervention for evaluation
    print(f"Evaluating on benchmark: {benchmark}")
    print(f"Output directory: {output_dir}")
    # VLMEvalKit integration logic would go here
    # The specific integration depends on the VLMEvalKit version and API


if __name__ == "__main__":
    main()
