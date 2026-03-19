"""
Fourier-based frequency domain utilities for L2V-CoT.

L2V-CoT uses low-pass filtering in the frequency domain to extract
low-frequency components from LLM hidden states. These low-frequency
components capture the essential CoT reasoning information that can
be transferred across model architectures.

Key operations:
  1. Apply FFT to hidden state vectors
  2. Zero out high-frequency components (low-pass filter)
  3. Apply IFFT to reconstruct filtered representation
  4. Resample to match target VLM hidden size

Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


def low_pass_filter(
    hidden_states: torch.Tensor,
    cutoff_ratio: float = 0.1,
    dim: int = -1,
) -> torch.Tensor:
    """
    Apply a low-pass filter to hidden states in the frequency domain.

    The filter removes high-frequency components from the hidden state vectors,
    retaining only the smooth, semantic low-frequency components that encode
    transferable reasoning capabilities.

    Args:
        hidden_states: Tensor of any shape, filtering applied along `dim`.
                       Typically shape (..., hidden_size).
        cutoff_ratio: Fraction of frequency components to keep (0 < ratio <= 1).
                      Lower values keep fewer (lower) frequencies.
                      Default 0.1 keeps the lowest 10% of frequencies.
        dim: Dimension along which to apply the filter.

    Returns:
        Filtered tensor of the same shape as input.
    """
    original_dtype = hidden_states.dtype
    h = hidden_states.float()

    # Apply FFT along the specified dimension
    h_freq = torch.fft.rfft(h, dim=dim)

    # Compute the cutoff index
    freq_size = h_freq.shape[dim]
    cutoff_idx = max(1, int(freq_size * cutoff_ratio))

    # Zero out high-frequency components beyond the cutoff
    mask = torch.zeros_like(h_freq)
    # Use index_put or slicing to set low frequencies to 1
    slices = [slice(None)] * len(h_freq.shape)
    slices[dim] = slice(0, cutoff_idx)
    mask[tuple(slices)] = 1.0
    h_freq_filtered = h_freq * mask

    # Apply inverse FFT to reconstruct the filtered signal
    # Use the original size to ensure the output shape matches
    orig_size = hidden_states.shape[dim]
    h_filtered = torch.fft.irfft(h_freq_filtered, n=orig_size, dim=dim)

    return h_filtered.to(original_dtype)


def high_pass_filter(
    hidden_states: torch.Tensor,
    cutoff_ratio: float = 0.1,
    dim: int = -1,
) -> torch.Tensor:
    """
    Apply a high-pass filter to hidden states (complement of low_pass_filter).

    Retains only the high-frequency components, which the L2V-CoT paper shows
    are NOT effective for cross-modal reasoning transfer.

    Args:
        hidden_states: Tensor of any shape.
        cutoff_ratio: Fraction of low frequencies to remove.
        dim: Dimension along which to apply the filter.

    Returns:
        High-pass filtered tensor of the same shape as input.
    """
    return hidden_states - low_pass_filter(hidden_states, cutoff_ratio, dim)


def resample_hidden_states(
    hidden_states: torch.Tensor,
    target_size: int,
    mode: str = "linear",
) -> torch.Tensor:
    """
    Resample hidden states to match a different model's hidden dimension.

    Since LLMs and VLMs may have different hidden sizes, this function
    resamples the frequency-filtered LLM representations to match the
    VLM's hidden state dimensionality.

    Args:
        hidden_states: Tensor to resample. Shape (..., hidden_size).
        target_size: The target hidden size for the VLM.
        mode: Interpolation mode for resampling ('linear', 'nearest').

    Returns:
        Resampled tensor of shape (..., target_size).
    """
    original_shape = hidden_states.shape
    hidden_size = original_shape[-1]

    if hidden_size == target_size:
        return hidden_states

    # Reshape to (batch, 1, hidden_size) for interpolation
    flat = hidden_states.view(-1, 1, hidden_size).float()

    # Apply 1D interpolation
    resampled = F.interpolate(
        flat,
        size=target_size,
        mode=mode,
        align_corners=False if mode == "linear" else None,
    )

    # Reshape back to original batch dimensions with new hidden_size
    new_shape = original_shape[:-1] + (target_size,)
    return resampled.view(new_shape).to(hidden_states.dtype)


def extract_low_frequency_cot_representation(
    cot_hidden_states: torch.Tensor,
    non_cot_hidden_states: torch.Tensor,
    cutoff_ratio: float = 0.1,
    target_hidden_size: Optional[int] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Extract the low-frequency CoT representation by contrasting CoT vs non-CoT
    hidden states in the frequency domain.

    This is the core operation of L2V-CoT:
    1. Compute the mean difference between CoT and non-CoT hidden states
    2. Apply low-pass filtering to extract low-frequency components
    3. Optionally resample to match VLM hidden size
    4. Scale by alpha

    Args:
        cot_hidden_states: Hidden states from CoT-prompted LLM.
                           Shape (n_samples, seq_len, hidden_size) or
                           (n_samples, hidden_size).
        non_cot_hidden_states: Hidden states from direct-answer LLM.
                               Same shape as cot_hidden_states.
        cutoff_ratio: Fraction of low frequencies to retain.
        target_hidden_size: Target hidden size for resampling. If None, no
                            resampling is applied.
        alpha: Scaling factor for the extracted representation.

    Returns:
        CoT representation tensor of shape (hidden_size,) or
        (target_hidden_size,) if target_hidden_size is specified.
    """
    # Average over samples
    if cot_hidden_states.dim() == 3:
        # (n_samples, seq_len, hidden_size) -> use last token
        cot_mean = cot_hidden_states[:, -1, :].float().mean(dim=0)
        non_cot_mean = non_cot_hidden_states[:, -1, :].float().mean(dim=0)
    else:
        cot_mean = cot_hidden_states.float().mean(dim=0)
        non_cot_mean = non_cot_hidden_states.float().mean(dim=0)

    # Compute the CoT direction (difference vector)
    cot_direction = cot_mean - non_cot_mean

    # Apply low-pass filter to isolate transferable low-frequency components
    cot_direction_filtered = low_pass_filter(
        cot_direction.unsqueeze(0), cutoff_ratio=cutoff_ratio, dim=-1
    ).squeeze(0)

    # Optionally resample to match VLM hidden size
    if target_hidden_size is not None and target_hidden_size != cot_direction_filtered.shape[-1]:
        cot_direction_filtered = resample_hidden_states(
            cot_direction_filtered.unsqueeze(0),
            target_hidden_size,
        ).squeeze(0)

    return cot_direction_filtered * alpha


def compute_frequency_spectrum(
    hidden_states: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute the power spectrum of hidden states for visualization/analysis.

    Args:
        hidden_states: Tensor of shape (..., hidden_size).
        dim: Dimension to analyze.

    Returns:
        Power spectrum tensor (magnitudes of FFT coefficients).
    """
    h_freq = torch.fft.rfft(hidden_states.float(), dim=dim)
    return torch.abs(h_freq)
