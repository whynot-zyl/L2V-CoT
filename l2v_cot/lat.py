"""
Linear Artificial Tomography (LAT) for extracting CoT reasoning directions.

LAT is used to empirically demonstrate that LLMs and VLMs share similar
low-frequency latent representations regarding CoT reasoning, and to extract
the "CoT direction" in the hidden state space.

Reference: L2V-CoT paper (https://arxiv.org/abs/2511.17910)
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class LinearArtificialTomography:
    """
    Linear Artificial Tomography (LAT) implementation.

    LAT extracts the CoT reasoning direction from LLM hidden states by
    contrasting hidden representations from CoT prompts vs. non-CoT prompts.
    This direction encodes the key reasoning information to be transferred.

    Usage:
        lat = LinearArtificialTomography(n_components=32)
        lat.fit(cot_hidden_states, non_cot_hidden_states)
        cot_direction = lat.get_cot_direction()
        projected = lat.project(hidden_states)
    """

    def __init__(
        self,
        n_components: int = 32,
        use_pca: bool = True,
        layer_idx: Optional[int] = None,
    ):
        """
        Args:
            n_components: Number of principal components for PCA reduction.
            use_pca: If True, apply PCA before computing the CoT direction.
            layer_idx: If set, only analyze a specific transformer layer.
        """
        self.n_components = n_components
        self.use_pca = use_pca
        self.layer_idx = layer_idx
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.classifier = LogisticRegression(max_iter=1000, C=1.0)
        self.cot_direction: Optional[np.ndarray] = None
        self.mean_cot: Optional[np.ndarray] = None
        self.mean_non_cot: Optional[np.ndarray] = None
        self._is_fitted = False

    def _prepare_hidden_states(
        self, hidden_states: List[torch.Tensor]
    ) -> np.ndarray:
        """
        Prepare hidden states for LAT analysis.

        Args:
            hidden_states: List of tensors of shape (seq_len, hidden_size)
                           or (batch, seq_len, hidden_size).

        Returns:
            Numpy array of shape (n_samples, hidden_size) using the last token.
        """
        processed = []
        for h in hidden_states:
            if h.dim() == 3:
                # (batch, seq_len, hidden_size) -> take last token mean
                h = h[:, -1, :]  # (batch, hidden_size)
                h = h.mean(dim=0)  # (hidden_size,)
            elif h.dim() == 2:
                # (seq_len, hidden_size) -> take last token
                h = h[-1, :]  # (hidden_size,)
            else:
                raise ValueError(f"Unexpected hidden state shape: {h.shape}")
            processed.append(h.float().cpu().numpy())
        return np.array(processed)  # (n_samples, hidden_size)

    def fit(
        self,
        cot_hidden_states: List[torch.Tensor],
        non_cot_hidden_states: List[torch.Tensor],
    ) -> "LinearArtificialTomography":
        """
        Fit the LAT model to extract the CoT reasoning direction.

        Args:
            cot_hidden_states: Hidden states from CoT-prompted LLM outputs.
            non_cot_hidden_states: Hidden states from direct-answer LLM outputs.

        Returns:
            self
        """
        X_cot = self._prepare_hidden_states(cot_hidden_states)
        X_non_cot = self._prepare_hidden_states(non_cot_hidden_states)

        self.mean_cot = X_cot.mean(axis=0)
        self.mean_non_cot = X_non_cot.mean(axis=0)

        X = np.concatenate([X_cot, X_non_cot], axis=0)
        y = np.array([1] * len(X_cot) + [0] * len(X_non_cot))

        if self.use_pca:
            X_reduced = self.pca.fit_transform(X)
        else:
            X_reduced = X

        self.classifier.fit(X_reduced, y)

        # The CoT direction is the difference between CoT and non-CoT means,
        # projected back through PCA if applicable
        diff = self.mean_cot - self.mean_non_cot
        if self.use_pca:
            # Project into PCA space and reconstruct
            diff_pca = self.pca.transform(diff.reshape(1, -1))  # (1, n_components)
            self.cot_direction = self.pca.inverse_transform(diff_pca)[0]  # (hidden_size,)
        else:
            self.cot_direction = diff

        # Normalize the direction vector
        norm = np.linalg.norm(self.cot_direction)
        if norm > 0:
            self.cot_direction = self.cot_direction / norm

        self._is_fitted = True
        return self

    def get_cot_direction(self) -> np.ndarray:
        """
        Return the extracted CoT reasoning direction vector.

        Returns:
            numpy array of shape (hidden_size,)
        """
        if not self._is_fitted:
            raise RuntimeError("LAT has not been fitted yet. Call fit() first.")
        return self.cot_direction

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states onto the CoT direction to measure CoT intensity.

        Args:
            hidden_states: Tensor of shape (..., hidden_size).

        Returns:
            Scalar projection values of the same batch dimensions.
        """
        if not self._is_fitted:
            raise RuntimeError("LAT has not been fitted yet. Call fit() first.")
        direction = torch.tensor(
            self.cot_direction, dtype=hidden_states.dtype, device=hidden_states.device
        )
        return (hidden_states * direction).sum(dim=-1)

    def get_cot_representation(
        self,
        hidden_states: List[torch.Tensor],
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the CoT representation to inject into the VLM.

        This computes the mean of CoT hidden states minus the component
        along the non-CoT direction, scaled by alpha.

        Args:
            hidden_states: Hidden states from CoT-prompted LLM.
            alpha: Scaling factor for the CoT representation strength.

        Returns:
            Tensor of shape (hidden_size,) representing the CoT direction.
        """
        if not self._is_fitted:
            raise RuntimeError("LAT has not been fitted yet. Call fit() first.")
        X_cot = self._prepare_hidden_states(hidden_states)
        mean_h = X_cot.mean(axis=0)
        cot_rep = torch.tensor(mean_h, dtype=torch.float32) * alpha
        return cot_rep

    def score(
        self,
        cot_hidden_states: List[torch.Tensor],
        non_cot_hidden_states: List[torch.Tensor],
    ) -> float:
        """
        Evaluate how well the LAT separates CoT from non-CoT representations.

        Returns:
            Classification accuracy of the linear probe (0.0 to 1.0).
        """
        if not self._is_fitted:
            raise RuntimeError("LAT has not been fitted yet. Call fit() first.")
        X_cot = self._prepare_hidden_states(cot_hidden_states)
        X_non_cot = self._prepare_hidden_states(non_cot_hidden_states)
        X = np.concatenate([X_cot, X_non_cot], axis=0)
        y = np.array([1] * len(X_cot) + [0] * len(X_non_cot))
        if self.use_pca:
            X_reduced = self.pca.transform(X)
        else:
            X_reduced = X
        return self.classifier.score(X_reduced, y)


def collect_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    layer_indices: Optional[List[int]] = None,
    device: str = "cuda",
    batch_size: int = 1,
) -> List[List[torch.Tensor]]:
    """
    Collect hidden states from an LLM for a list of prompts.

    Args:
        model: HuggingFace LLM (AutoModelForCausalLM).
        tokenizer: Corresponding tokenizer.
        prompts: List of text prompts to process.
        layer_indices: Which transformer layers to extract. None = all layers.
        device: Device to run inference on.
        batch_size: Batch size for processing.

    Returns:
        List of lists: outer = prompts, inner = one tensor per layer.
        Each tensor has shape (seq_len, hidden_size).
    """
    model.eval()
    all_hidden_states = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)

        for b in range(len(batch_prompts)):
            if layer_indices is None:
                layers = list(range(len(hidden_states)))
            else:
                layers = layer_indices
            prompt_states = [
                hidden_states[l][b].cpu() for l in layers
            ]
            all_hidden_states.append(prompt_states)

    return all_hidden_states
