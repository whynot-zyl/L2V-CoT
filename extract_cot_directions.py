"""
Extract CoT direction vectors from a Large Language Model (LLM).

This script implements Step 1 of L2V-CoT: using contrastive prompting to
generate CoT and non-CoT hidden states from an LLM, then extracting the
low-frequency CoT direction vectors for later injection into a VLM.

Usage:
    python extract_cot_directions.py \
        --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
        --output_path cot_representations.pt \
        --dataset scienceqa \
        --num_samples 500 \
        --layer_idx -1 \
        --cutoff_ratio 0.1

Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
"""

import argparse
import json
import os
import random
import torch
from pathlib import Path
from typing import List, Optional

from l2v_cot.lat import LinearArtificialTomography, collect_hidden_states
from l2v_cot.frequency import extract_low_frequency_cot_representation
from l2v_cot.models import load_llm, get_model_info


# ─── CoT prompt templates ────────────────────────────────────────────────────

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant that reasons step by step."
)

COT_PROMPT_TEMPLATE = (
    "Let's think step by step to answer the following question:\n\n"
    "Question: {question}\n\n"
    "Let's think through this carefully:"
)

NON_COT_PROMPT_TEMPLATE = (
    "Answer the following question directly and concisely:\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def load_questions(dataset: str, data_path: Optional[str], num_samples: int) -> List[str]:
    """
    Load questions from a dataset for CoT direction extraction.

    Args:
        dataset: Dataset name ('scienceqa', 'mmb', 'custom', or 'default').
        data_path: Optional path to custom data file (JSONL format with 'question' field).
        num_samples: Number of questions to use.

    Returns:
        List of question strings.
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading questions from {data_path}")
        questions = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                q = item.get("question") or item.get("query") or item.get("text", "")
                if q:
                    questions.append(q)
        if num_samples > 0:
            questions = questions[:num_samples]
        return questions

    # Built-in sample questions for common scenarios
    print(f"Using built-in sample questions for '{dataset}'")
    sample_questions = [
        "What is the relationship between photosynthesis and cellular respiration?",
        "If a train travels at 60 mph for 2 hours, how far does it travel?",
        "Explain why the sky appears blue during the day.",
        "What is the area of a triangle with base 6 cm and height 4 cm?",
        "How does vaccination protect against infectious diseases?",
        "If x + 5 = 12, what is the value of x?",
        "Why do objects fall toward the Earth?",
        "What happens when you mix an acid and a base?",
        "How does the human digestive system process food?",
        "If all squares are rectangles, and all rectangles have four sides, do all squares have four sides?",
        "What is the capital city of France?",
        "How many planets are in our solar system?",
        "What is Newton's first law of motion?",
        "If a bacteria doubles every 30 minutes, how many bacteria are there after 2 hours starting from 1?",
        "What is the difference between kinetic and potential energy?",
    ]

    # Repeat to get enough samples
    questions = (sample_questions * ((num_samples // len(sample_questions)) + 1))[:num_samples]
    return questions


def build_cot_prompts(questions: List[str], use_system_prompt: bool = True) -> List[str]:
    """Build CoT prompts using the chain-of-thought template."""
    prompts = []
    for q in questions:
        if use_system_prompt:
            prompt = f"{COT_SYSTEM_PROMPT}\n\n{COT_PROMPT_TEMPLATE.format(question=q)}"
        else:
            prompt = COT_PROMPT_TEMPLATE.format(question=q)
        prompts.append(prompt)
    return prompts


def build_non_cot_prompts(questions: List[str]) -> List[str]:
    """Build non-CoT (direct answer) prompts."""
    return [NON_COT_PROMPT_TEMPLATE.format(question=q) for q in questions]


def extract_and_save(
    llm_model: str,
    output_path: str,
    dataset: str = "default",
    data_path: Optional[str] = None,
    num_samples: int = 500,
    layer_idx: int = -1,
    cutoff_ratio: float = 0.1,
    target_hidden_size: Optional[int] = None,
    device: str = "cuda",
    batch_size: int = 4,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    alpha: float = 1.5,
    use_lat: bool = True,
    seed: int = 42,
) -> None:
    """
    Extract CoT direction vectors from an LLM and save to file.

    Args:
        llm_model: HuggingFace path to the source LLM.
        output_path: Path to save the CoT representations (.pt file).
        dataset: Dataset to use for generating questions.
        data_path: Optional path to custom question data.
        num_samples: Number of samples to use for extraction.
        layer_idx: LLM layer index to extract from (-1 = last layer).
        cutoff_ratio: Low-pass filter cutoff (fraction of frequencies to keep).
        target_hidden_size: VLM hidden size for resampling. If None, no resampling.
        device: Computation device.
        batch_size: Batch size for LLM inference.
        load_in_8bit: Enable 8-bit quantization.
        load_in_4bit: Enable 4-bit quantization.
        alpha: Scaling factor for the CoT representation.
        use_lat: If True, also run LAT analysis to validate separation.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # ── Load LLM ────────────────────────────────────────────────────────────
    model, tokenizer = load_llm(
        llm_model,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )

    # Determine actual layer index
    model_info = get_model_info(llm_model)
    n_layers = model_info.get("num_layers", 32)
    actual_layer_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
    print(f"LLM has {n_layers} layers. Extracting from layer {actual_layer_idx}.")

    # ── Load questions ───────────────────────────────────────────────────────
    questions = load_questions(dataset, data_path, num_samples)
    print(f"Loaded {len(questions)} questions.")

    cot_prompts = build_cot_prompts(questions)
    non_cot_prompts = build_non_cot_prompts(questions)

    # ── Collect hidden states ────────────────────────────────────────────────
    print("Collecting CoT hidden states...")
    cot_hidden = collect_hidden_states(
        model, tokenizer, cot_prompts,
        layer_indices=[actual_layer_idx],
        device=device,
        batch_size=batch_size,
    )

    print("Collecting non-CoT hidden states...")
    non_cot_hidden = collect_hidden_states(
        model, tokenizer, non_cot_prompts,
        layer_indices=[actual_layer_idx],
        device=device,
        batch_size=batch_size,
    )

    # ── Extract per-layer CoT representations ───────────────────────────────
    # cot_hidden: List of [tensor_for_layer_0, ...] per prompt
    # Group by layer
    cot_by_layer = {}
    non_cot_by_layer = {}
    for states in cot_hidden:
        for i, h in enumerate(states):
            l = actual_layer_idx if len(states) == 1 else i
            if l not in cot_by_layer:
                cot_by_layer[l] = []
            cot_by_layer[l].append(h)

    for states in non_cot_hidden:
        for i, h in enumerate(states):
            l = actual_layer_idx if len(states) == 1 else i
            if l not in non_cot_by_layer:
                non_cot_by_layer[l] = []
            non_cot_by_layer[l].append(h)

    representations = {}
    for l_idx in cot_by_layer:
        cot_h = torch.stack(cot_by_layer[l_idx], dim=0)  # (N, seq, hidden)
        non_cot_h = torch.stack(non_cot_by_layer[l_idx], dim=0)

        # ── LAT analysis (optional, for validation) ────────────────────────
        if use_lat:
            lat = LinearArtificialTomography(n_components=32, use_pca=True)
            lat.fit(cot_by_layer[l_idx], non_cot_by_layer[l_idx])
            accuracy = lat.score(cot_by_layer[l_idx], non_cot_by_layer[l_idx])
            print(f"Layer {l_idx}: LAT classification accuracy = {accuracy:.3f}")
            if accuracy < 0.6:
                print(
                    f"  Warning: Low LAT accuracy ({accuracy:.3f}). "
                    "CoT and non-CoT representations may not be well-separated "
                    "at this layer. Consider trying a different layer."
                )

        # ── Extract low-frequency CoT representation ───────────────────────
        cot_rep = extract_low_frequency_cot_representation(
            cot_h,
            non_cot_h,
            cutoff_ratio=cutoff_ratio,
            target_hidden_size=target_hidden_size,
            alpha=alpha,
        )
        representations[l_idx] = cot_rep
        print(f"Layer {l_idx}: CoT representation shape = {cot_rep.shape}")

    # ── Save representations ─────────────────────────────────────────────────
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "representations": representations,
        "metadata": {
            "llm_model": llm_model,
            "layer_idx": actual_layer_idx,
            "cutoff_ratio": cutoff_ratio,
            "alpha": alpha,
            "num_samples": len(questions),
            "target_hidden_size": target_hidden_size,
        },
    }
    torch.save(save_data, output_path)
    print(f"\nCoT representations saved to: {output_path}")
    print(f"Metadata: {save_data['metadata']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract CoT direction vectors from an LLM for L2V-CoT"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        required=True,
        help="HuggingFace path to the source LLM (e.g., meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cot_representations.pt",
        help="Path to save the extracted CoT representations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="default",
        choices=["scienceqa", "mmb", "mmstar", "mathvista", "default", "custom"],
        help="Dataset to use for question extraction",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to custom data file (JSONL with 'question' field)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to use for extraction",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="LLM layer index to extract from (-1 = last layer)",
    )
    parser.add_argument(
        "--cutoff_ratio",
        type=float,
        default=0.1,
        help="Low-pass filter cutoff ratio (fraction of frequencies to keep)",
    )
    parser.add_argument(
        "--target_hidden_size",
        type=int,
        default=None,
        help="VLM hidden size for resampling. If not set, no resampling is done.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="Scaling factor for the CoT representation strength",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for LLM inference",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load LLM in 8-bit quantization",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load LLM in 4-bit quantization",
    )
    parser.add_argument(
        "--no_lat",
        action="store_true",
        help="Disable LAT validation analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    extract_and_save(
        llm_model=args.llm_model,
        output_path=args.output_path,
        dataset=args.dataset,
        data_path=args.data_path,
        num_samples=args.num_samples,
        layer_idx=args.layer_idx,
        cutoff_ratio=args.cutoff_ratio,
        target_hidden_size=args.target_hidden_size,
        device=args.device,
        batch_size=args.batch_size,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        alpha=args.alpha,
        use_lat=not args.no_lat,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
