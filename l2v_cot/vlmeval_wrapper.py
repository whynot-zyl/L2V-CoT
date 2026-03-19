"""
VLMEvalKit integration for L2V-CoT evaluation.

This module provides a wrapper that integrates L2V-CoT's latent intervention
with VLMEvalKit's model evaluation framework. It wraps any VLM supported by
VLMEvalKit and applies CoT direction injection during inference.

Usage with VLMEvalKit:
    # Register the L2V-CoT wrapped model and run evaluation:
    python -m vlmeval.run \
        --model L2VCoT_LLaVA_1_5_7b \
        --data ScienceQA_VAL MMBench_DEV_EN \
        --work-dir results/l2v_cot

Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union

from l2v_cot.intervention import L2VCoTIntervention


class L2VCoTVLMWrapper:
    """
    Wrapper that applies L2V-CoT intervention to any VLMEvalKit-compatible VLM.

    This class wraps a VLMEvalKit model and applies latent intervention
    from pre-extracted CoT representations during inference.

    To use with VLMEvalKit, register this wrapper as a model class and
    pass it to the evaluation pipeline.
    """

    def __init__(
        self,
        base_model_name: str,
        cot_repr_path: str,
        target_layers: Optional[List[int]] = None,
        intervention_strength: float = 1.5,
        cutoff_ratio: float = 0.1,
        verbose: bool = False,
    ):
        """
        Args:
            base_model_name: VLMEvalKit model name (e.g., 'llava_v1.5_7b').
            cot_repr_path: Path to pre-extracted CoT representations (.pt file).
            target_layers: VLM transformer layer indices to intervene at.
            intervention_strength: Scaling factor for intervention.
            cutoff_ratio: Low-pass filter cutoff ratio.
            verbose: Enable verbose logging.
        """
        self.base_model_name = base_model_name
        self.cot_repr_path = cot_repr_path
        self.target_layers = target_layers
        self.intervention_strength = intervention_strength
        self.cutoff_ratio = cutoff_ratio
        self.verbose = verbose

        self._model = None
        self._intervention = None

    def _ensure_loaded(self):
        """Lazily load the model and intervention on first use."""
        if self._model is not None:
            return

        # Load VLMEvalKit model
        try:
            from vlmeval.vlm import supported_VLM
            model_cls = supported_VLM.get(self.base_model_name)
            if model_cls is None:
                raise ValueError(f"Model '{self.base_model_name}' not found in VLMEvalKit.")
            self._model = model_cls(self.base_model_name)
        except ImportError:
            raise ImportError(
                "VLMEvalKit not found. Install with: pip install -e VLMEvalKit/"
            )

        # Load pre-extracted CoT representations
        save_data = torch.load(self.cot_repr_path, map_location="cpu")
        representations = save_data.get("representations", save_data)

        # Set up intervention (LLM not needed since representations are loaded)
        self._intervention = L2VCoTIntervention(
            llm=None,
            vlm=self._model.model,  # underlying PyTorch model
            target_layers=self.target_layers,
            cutoff_ratio=self.cutoff_ratio,
            intervention_strength=self.intervention_strength,
        )
        self._intervention.load_cot_representations(representations)

        if self.verbose:
            print(f"L2VCoTVLMWrapper: Loaded model '{self.base_model_name}' with "
                  f"CoT representations from {self.cot_repr_path}")

    def generate(self, message: List[dict], dataset: Optional[str] = None) -> str:
        """
        Generate a response for a given message with L2V-CoT intervention.

        This overrides the standard VLMEvalKit generate() method.

        Args:
            message: VLMEvalKit message format (list with text/image entries).
            dataset: Dataset name (passed through to base model).

        Returns:
            Generated response string.
        """
        self._ensure_loaded()

        with self._intervention.apply():
            return self._model.generate(message, dataset=dataset)

    def __getattr__(self, name):
        """Delegate attribute access to the base model."""
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure_loaded()
        return getattr(self._model, name)


def build_vlmeval_model_registry() -> Dict[str, type]:
    """
    Build a dictionary of L2V-CoT wrapped model classes for VLMEvalKit.

    Returns:
        Dictionary mapping model name to model class.
    """
    registry = {}

    # Standard configurations from the L2V-CoT paper
    # These correspond to the models evaluated in the paper experiments
    configs = [
        {
            "name": "L2VCoT_LLaVA_1_5_7b",
            "base": "llava_v1.5_7b",
            "cot_repr": "outputs/cot_repr/llama3_8b_layer31.pt",
        },
        {
            "name": "L2VCoT_LLaVA_1_5_13b",
            "base": "llava_v1.5_13b",
            "cot_repr": "outputs/cot_repr/llama3_8b_layer31.pt",
        },
        {
            "name": "L2VCoT_LLaVA_Next_8b",
            "base": "llava_next_8b",
            "cot_repr": "outputs/cot_repr/llama3_8b_layer31.pt",
        },
    ]

    for cfg in configs:
        # Create a class dynamically for each configuration
        name = cfg["name"]

        def make_cls(base, cot_repr):
            class WrappedModel(L2VCoTVLMWrapper):
                def __init__(self, model_path=None):
                    super().__init__(
                        base_model_name=base,
                        cot_repr_path=cot_repr,
                    )
            WrappedModel.__name__ = name
            return WrappedModel

        registry[name] = make_cls(cfg["base"], cfg["cot_repr"])

    return registry
