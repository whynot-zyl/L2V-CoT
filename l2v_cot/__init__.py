"""
L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention

This package implements the L2V-CoT method from the paper:
  "L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention"
  https://arxiv.org/abs/2511.17910

The key components are:
  - lat.py: Linear Artificial Tomography for extracting CoT directions from LLM hidden states
  - frequency.py: Fourier-based low-pass filtering for low-frequency representation extraction
  - intervention.py: Latent intervention module for injecting CoT representations into VLMs
  - models.py: Model loading utilities for LLMs and VLMs
"""

from .lat import LinearArtificialTomography
from .frequency import low_pass_filter, resample_hidden_states
from .intervention import LatentInterventionHook, L2VCoTIntervention
from .models import load_llm, load_vlm

__all__ = [
    "LinearArtificialTomography",
    "low_pass_filter",
    "resample_hidden_states",
    "LatentInterventionHook",
    "L2VCoTIntervention",
    "load_llm",
    "load_vlm",
]
