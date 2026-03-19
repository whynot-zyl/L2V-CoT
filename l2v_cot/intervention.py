"""
Latent Intervention module for L2V-CoT.

This module implements the core inference-time latent intervention:
injecting low-frequency CoT representations extracted from an LLM
into the hidden states of a VLM at specific transformer layers.

The intervention is implemented via PyTorch forward hooks, enabling
architecture-agnostic, training-free CoT transfer.

Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

from .frequency import (
    extract_low_frequency_cot_representation,
    low_pass_filter,
    resample_hidden_states,
)


class LatentInterventionHook:
    """
    A PyTorch forward hook that injects CoT representations into a VLM layer.

    The hook intercepts the output of a transformer layer and adds the
    low-frequency CoT direction vector to the hidden states, steering
    the model's reasoning towards chain-of-thought behavior.
    """

    def __init__(
        self,
        cot_representation: torch.Tensor,
        intervention_strength: float = 1.0,
        token_positions: Optional[Union[str, List[int]]] = "last",
    ):
        """
        Args:
            cot_representation: The CoT direction vector to inject.
                                 Shape (hidden_size,).
            intervention_strength: Scaling factor for the intervention strength.
                                   Higher values = stronger CoT influence.
            token_positions: Which token positions to apply intervention to.
                             'last': only the last token.
                             'all': all tokens.
                             List of ints: specific positions.
        """
        self.cot_representation = cot_representation
        self.intervention_strength = intervention_strength
        self.token_positions = token_positions
        self._active = True

    def disable(self):
        """Disable this hook without removing it."""
        self._active = False

    def enable(self):
        """Re-enable this hook."""
        self._active = True

    def __call__(
        self,
        module: nn.Module,
        input: Tuple,
        output: Union[Tuple, torch.Tensor],
    ) -> Union[Tuple, torch.Tensor]:
        """
        Hook function called after the forward pass of a layer.

        Injects the CoT representation into the hidden states.
        """
        if not self._active:
            return output

        # Handle both tuple outputs (common in HuggingFace) and plain tensors
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Ensure the CoT representation is on the right device/dtype
        cot_rep = self.cot_representation.to(
            device=hidden.device, dtype=hidden.dtype
        )

        # Handle dimension mismatch by resampling
        if cot_rep.shape[-1] != hidden.shape[-1]:
            cot_rep = resample_hidden_states(
                cot_rep.unsqueeze(0), hidden.shape[-1]
            ).squeeze(0)

        # Apply intervention at specified token positions
        injection = cot_rep * self.intervention_strength

        if self.token_positions == "all":
            hidden = hidden + injection.unsqueeze(0).unsqueeze(0)
        elif self.token_positions == "last":
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + injection
        elif isinstance(self.token_positions, list):
            hidden = hidden.clone()
            for pos in self.token_positions:
                if pos < hidden.shape[1]:
                    hidden[:, pos, :] = hidden[:, pos, :] + injection

        if rest is not None:
            return (hidden,) + rest
        return hidden


class L2VCoTIntervention:
    """
    L2V-CoT Latent Intervention manager for a Vision-Language Model.

    This class manages the full pipeline for:
    1. Extracting CoT representations from an LLM
    2. Applying low-frequency filtering
    3. Injecting into VLM hidden states at specified layers during inference

    Example usage:
        # Load models
        llm, llm_tokenizer = load_llm("meta-llama/Llama-2-7b-hf")
        vlm, vlm_processor = load_vlm("llava-hf/llava-1.5-7b-hf")

        # Initialize intervention
        intervention = L2VCoTIntervention(
            llm=llm,
            vlm=vlm,
            target_layers=[15, 20, 25],
            cutoff_ratio=0.1,
            intervention_strength=1.5,
        )

        # Extract CoT representations from LLM
        intervention.extract_cot_representations(
            cot_prompts=["Let's think step by step. ..."],
            non_cot_prompts=["The answer is: ..."],
        )

        # Run VLM inference with intervention
        with intervention.apply():
            outputs = vlm.generate(input_ids, ...)
    """

    def __init__(
        self,
        llm: nn.Module,
        vlm: nn.Module,
        target_layers: Optional[List[int]] = None,
        cutoff_ratio: float = 0.1,
        intervention_strength: float = 1.5,
        token_positions: Union[str, List[int]] = "last",
        llm_layer_idx: Optional[int] = None,
    ):
        """
        Args:
            llm: The source LLM to extract CoT representations from.
            vlm: The target VLM to inject representations into.
            target_layers: Indices of VLM transformer layers to intervene in.
                           If None, defaults to the last third of all layers.
            cutoff_ratio: Fraction of low frequencies to keep in filtering.
            intervention_strength: Scaling factor for injection strength.
            token_positions: Which token positions to apply intervention to.
            llm_layer_idx: Which LLM layer to extract CoT representations from.
                           If None, uses the last layer.
        """
        self.llm = llm
        self.vlm = vlm
        self.target_layers = target_layers
        self.cutoff_ratio = cutoff_ratio
        self.intervention_strength = intervention_strength
        self.token_positions = token_positions
        self.llm_layer_idx = llm_layer_idx

        self.cot_representations: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._hook_handlers: List[LatentInterventionHook] = []

    def _get_vlm_layers(self) -> List[nn.Module]:
        """
        Extract the transformer layers from the VLM.

        Supports LLaVA, Qwen-VL, InstructBLIP, and generic HuggingFace VLMs.
        """
        # Try common attribute paths for VLM language model layers
        paths_to_try = [
            # LLaVA-style models
            lambda m: m.language_model.model.layers,
            lambda m: m.model.language_model.model.layers,
            # Qwen-VL style
            lambda m: m.transformer.h,
            # Generic HuggingFace
            lambda m: m.model.layers,
            lambda m: m.transformer.layers,
            lambda m: m.model.model.layers,
        ]
        for path_fn in paths_to_try:
            try:
                layers = path_fn(self.vlm)
                return list(layers)
            except AttributeError:
                continue
        raise ValueError(
            "Could not find transformer layers in VLM. "
            "Please override _get_vlm_layers() for your model architecture."
        )

    def _get_llm_layers(self) -> List[nn.Module]:
        """Extract the transformer layers from the LLM."""
        paths_to_try = [
            lambda m: m.model.layers,
            lambda m: m.transformer.h,
            lambda m: m.gpt_neox.layers,
        ]
        for path_fn in paths_to_try:
            try:
                layers = path_fn(self.llm)
                return list(layers)
            except AttributeError:
                continue
        raise ValueError(
            "Could not find transformer layers in LLM. "
            "Please override _get_llm_layers() for your model architecture."
        )

    def extract_cot_representations(
        self,
        cot_prompts: List[str],
        non_cot_prompts: List[str],
        tokenizer,
        device: str = "cuda",
        batch_size: int = 4,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract CoT representations from the LLM using contrastive prompting.

        This generates hidden states from CoT-prompted inputs (e.g., "Let's
        think step by step") and direct-answer inputs, then extracts the
        low-frequency difference to use as the CoT direction.

        Args:
            cot_prompts: Prompts that elicit CoT reasoning from the LLM.
            non_cot_prompts: Prompts for direct answers (non-CoT).
            tokenizer: LLM tokenizer.
            device: Device for computation.
            batch_size: Batch size for hidden state extraction.

        Returns:
            Dictionary mapping layer index to CoT representation tensor.
        """
        self.llm.eval()
        llm_layers = self._get_llm_layers()
        n_layers = len(llm_layers)

        if self.llm_layer_idx is not None:
            layer_indices = [self.llm_layer_idx]
        else:
            # Use the last layer by default (richest semantic representation)
            layer_indices = [n_layers - 1]

        cot_states_by_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}
        non_cot_states_by_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}

        # Collect CoT hidden states
        for prompts, states_dict in [
            (cot_prompts, cot_states_by_layer),
            (non_cot_prompts, non_cot_states_by_layer),
        ]:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                with torch.no_grad():
                    outputs = self.llm(
                        **inputs,
                        output_hidden_states=True,
                    )

                for l_idx in layer_indices:
                    # +1 because hidden_states[0] is the embedding layer
                    layer_hidden = outputs.hidden_states[l_idx + 1]  # (batch, seq, hidden)
                    states_dict[l_idx].append(layer_hidden.cpu())

        # Compute low-frequency CoT representations per layer
        vlm_layers = self._get_vlm_layers()
        vlm_hidden_size = self._get_vlm_hidden_size(vlm_layers)

        for l_idx in layer_indices:
            cot_h = torch.cat(cot_states_by_layer[l_idx], dim=0)  # (N, seq, llm_hidden)
            non_cot_h = torch.cat(non_cot_states_by_layer[l_idx], dim=0)

            cot_rep = extract_low_frequency_cot_representation(
                cot_h,
                non_cot_h,
                cutoff_ratio=self.cutoff_ratio,
                target_hidden_size=vlm_hidden_size,
                alpha=self.intervention_strength,
            )
            self.cot_representations[l_idx] = cot_rep

        return self.cot_representations

    def load_cot_representations(
        self,
        representations: Dict[int, torch.Tensor],
    ) -> None:
        """
        Load pre-computed CoT representations (e.g., from file).

        Args:
            representations: Dictionary mapping LLM layer index to CoT tensor.
        """
        self.cot_representations = representations

    def _get_vlm_hidden_size(self, vlm_layers: Optional[List[nn.Module]] = None) -> int:
        """Get the hidden size of the VLM's transformer layers."""
        if vlm_layers is None:
            vlm_layers = self._get_vlm_layers()

        # Try to get hidden size from model config
        try:
            return self.vlm.config.hidden_size
        except AttributeError:
            pass
        try:
            return self.vlm.config.text_config.hidden_size
        except AttributeError:
            pass

        # Fall back to inspecting the first layer's parameters
        for name, param in vlm_layers[0].named_parameters():
            if len(param.shape) >= 2:
                return param.shape[-1]

        raise ValueError("Could not determine VLM hidden size.")

    def _get_target_layer_indices(self) -> List[int]:
        """Get the VLM layer indices to intervene in."""
        if self.target_layers is not None:
            return self.target_layers

        # Default: apply to the last third of VLM layers
        n_layers = len(self._get_vlm_layers())
        start = n_layers * 2 // 3
        return list(range(start, n_layers))

    def _register_hooks(self) -> None:
        """Register forward hooks on VLM layers for intervention."""
        if not self.cot_representations:
            raise RuntimeError(
                "No CoT representations found. "
                "Call extract_cot_representations() first."
            )

        vlm_layers = self._get_vlm_layers()
        target_indices = self._get_target_layer_indices()

        # Use the single CoT representation for all target layers
        # (or per-layer if available)
        cot_rep_keys = sorted(self.cot_representations.keys())
        default_cot_rep = self.cot_representations[cot_rep_keys[-1]]

        self._hooks = []
        self._hook_handlers = []

        for layer_idx in target_indices:
            if layer_idx >= len(vlm_layers):
                continue

            # Use layer-specific representation if available, else default
            cot_rep = self.cot_representations.get(layer_idx, default_cot_rep)

            handler = LatentInterventionHook(
                cot_representation=cot_rep,
                intervention_strength=1.0,  # already scaled in extract
                token_positions=self.token_positions,
            )
            hook = vlm_layers[layer_idx].register_forward_hook(handler)
            self._hooks.append(hook)
            self._hook_handlers.append(handler)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._hook_handlers = []

    @contextmanager
    def apply(self):
        """
        Context manager that applies latent intervention during VLM inference.

        Usage:
            with intervention.apply():
                outputs = vlm.generate(input_ids, pixel_values=pixel_values)
        """
        self._register_hooks()
        try:
            yield self
        finally:
            self._remove_hooks()

    def save_cot_representations(self, path: str) -> None:
        """
        Save extracted CoT representations to a file.

        Args:
            path: File path to save (e.g., 'cot_representations.pt').
        """
        torch.save(self.cot_representations, path)
        print(f"CoT representations saved to {path}")

    @classmethod
    def load_from_file(
        cls,
        llm: nn.Module,
        vlm: nn.Module,
        path: str,
        **kwargs,
    ) -> "L2VCoTIntervention":
        """
        Create an L2VCoTIntervention instance with pre-computed representations.

        Args:
            llm: The source LLM.
            vlm: The target VLM.
            path: Path to saved CoT representations.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Initialized L2VCoTIntervention instance.
        """
        instance = cls(llm=llm, vlm=vlm, **kwargs)
        representations = torch.load(path, map_location="cpu")
        instance.load_cot_representations(representations)
        return instance
