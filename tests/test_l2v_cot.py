"""
Tests for the L2V-CoT core implementation.

These tests validate the key components of the L2V-CoT method:
- Fourier low-pass filtering
- Hidden state resampling
- LAT direction extraction
- Latent intervention hooks
"""

import pytest
import torch
import numpy as np


class TestFrequencyUtils:
    """Tests for the frequency domain utilities."""

    def test_low_pass_filter_preserves_shape(self):
        from l2v_cot.frequency import low_pass_filter
        h = torch.randn(4, 512)
        filtered = low_pass_filter(h, cutoff_ratio=0.1)
        assert filtered.shape == h.shape

    def test_low_pass_filter_3d_input(self):
        from l2v_cot.frequency import low_pass_filter
        h = torch.randn(2, 16, 512)
        filtered = low_pass_filter(h, cutoff_ratio=0.1)
        assert filtered.shape == h.shape

    def test_low_pass_filter_reduces_high_freq(self):
        from l2v_cot.frequency import low_pass_filter, compute_frequency_spectrum
        # Create a signal with both low and high frequencies
        t = torch.linspace(0, 1, 1024)
        # Low frequency (1 Hz) + high frequency (100 Hz)
        signal = torch.sin(2 * torch.pi * 1 * t) + 0.5 * torch.sin(2 * torch.pi * 100 * t)
        signal = signal.unsqueeze(0)  # (1, 1024)

        filtered = low_pass_filter(signal, cutoff_ratio=0.05)
        original_spectrum = compute_frequency_spectrum(signal)
        filtered_spectrum = compute_frequency_spectrum(filtered)

        # The filtered signal should have less energy at high frequencies
        mid = original_spectrum.shape[-1] // 2
        high_freq_original = original_spectrum[..., mid:].mean()
        high_freq_filtered = filtered_spectrum[..., mid:].mean()
        assert high_freq_filtered < high_freq_original

    def test_low_pass_full_cutoff_approx_identity(self):
        from l2v_cot.frequency import low_pass_filter
        h = torch.randn(8, 128)
        filtered = low_pass_filter(h, cutoff_ratio=1.0)
        # With cutoff=1.0, we keep all frequencies, so the output ≈ input
        assert torch.allclose(filtered, h, atol=1e-4)

    def test_resample_hidden_states_upscale(self):
        from l2v_cot.frequency import resample_hidden_states
        h = torch.randn(4, 512)
        resampled = resample_hidden_states(h, target_size=1024)
        assert resampled.shape == (4, 1024)

    def test_resample_hidden_states_downscale(self):
        from l2v_cot.frequency import resample_hidden_states
        h = torch.randn(4, 4096)
        resampled = resample_hidden_states(h, target_size=2048)
        assert resampled.shape == (4, 2048)

    def test_resample_same_size_no_change(self):
        from l2v_cot.frequency import resample_hidden_states
        h = torch.randn(4, 512)
        resampled = resample_hidden_states(h, target_size=512)
        assert torch.equal(resampled, h)

    def test_extract_low_frequency_cot_representation(self):
        from l2v_cot.frequency import extract_low_frequency_cot_representation
        # Create fake CoT and non-CoT hidden states
        hidden_size = 512
        n_samples = 10
        cot_h = torch.randn(n_samples, 8, hidden_size) + 1.0  # shifted by 1
        non_cot_h = torch.randn(n_samples, 8, hidden_size)

        cot_rep = extract_low_frequency_cot_representation(
            cot_h, non_cot_h, cutoff_ratio=0.1
        )
        assert cot_rep.shape == (hidden_size,)

    def test_extract_with_resampling(self):
        from l2v_cot.frequency import extract_low_frequency_cot_representation
        cot_h = torch.randn(5, 8, 4096)
        non_cot_h = torch.randn(5, 8, 4096)
        cot_rep = extract_low_frequency_cot_representation(
            cot_h, non_cot_h, cutoff_ratio=0.1, target_hidden_size=2048
        )
        assert cot_rep.shape == (2048,)


class TestLAT:
    """Tests for Linear Artificial Tomography."""

    def test_fit_and_get_direction(self):
        from l2v_cot.lat import LinearArtificialTomography
        hidden_size = 64
        n_samples = 30

        # Create separable CoT and non-CoT representations
        cot_states = [torch.randn(10, hidden_size) + 2.0 for _ in range(n_samples)]
        non_cot_states = [torch.randn(10, hidden_size) for _ in range(n_samples)]

        lat = LinearArtificialTomography(n_components=16, use_pca=True)
        lat.fit(cot_states, non_cot_states)

        direction = lat.get_cot_direction()
        assert direction.shape == (hidden_size,)
        # Direction should be normalized
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-5

    def test_score_separable_representations(self):
        from l2v_cot.lat import LinearArtificialTomography
        hidden_size = 64
        n_samples = 50

        # Create well-separated representations
        cot_states = [torch.randn(5, hidden_size) + 3.0 for _ in range(n_samples)]
        non_cot_states = [torch.randn(5, hidden_size) - 3.0 for _ in range(n_samples)]

        lat = LinearArtificialTomography(n_components=32, use_pca=False)
        lat.fit(cot_states, non_cot_states)

        # Well-separated data should have high classification accuracy
        score = lat.score(cot_states, non_cot_states)
        assert score > 0.9, f"Expected accuracy > 0.9, got {score:.3f}"

    def test_unfitted_raises_error(self):
        from l2v_cot.lat import LinearArtificialTomography
        lat = LinearArtificialTomography()
        with pytest.raises(RuntimeError, match="not been fitted"):
            lat.get_cot_direction()

    def test_project(self):
        from l2v_cot.lat import LinearArtificialTomography
        hidden_size = 64
        n_samples = 20
        cot_states = [torch.randn(5, hidden_size) + 2.0 for _ in range(n_samples)]
        non_cot_states = [torch.randn(5, hidden_size) for _ in range(n_samples)]

        lat = LinearArtificialTomography(n_components=16)
        lat.fit(cot_states, non_cot_states)

        h = torch.randn(3, hidden_size)
        projected = lat.project(h)
        assert projected.shape == (3,)


class TestLatentInterventionHook:
    """Tests for the LatentInterventionHook."""

    def test_hook_modifies_output(self):
        from l2v_cot.intervention import LatentInterventionHook

        hidden_size = 64
        cot_rep = torch.randn(hidden_size)
        hook = LatentInterventionHook(
            cot_representation=cot_rep,
            intervention_strength=1.0,
            token_positions="last",
        )

        # Simulate layer output
        batch, seq, hidden = 2, 10, hidden_size
        layer_output = torch.randn(batch, seq, hidden)
        original_last = layer_output[:, -1, :].clone()

        # Call hook manually
        result = hook(module=None, input=None, output=layer_output)

        # The last token should have changed
        assert result.shape == (batch, seq, hidden)
        assert not torch.equal(result[:, -1, :], original_last)
        # Other positions should not be changed
        assert torch.equal(result[:, :-1, :], layer_output[:, :-1, :])

    def test_hook_with_tuple_output(self):
        from l2v_cot.intervention import LatentInterventionHook

        hidden_size = 64
        cot_rep = torch.randn(hidden_size)
        hook = LatentInterventionHook(
            cot_representation=cot_rep,
            token_positions="last",
        )

        layer_output = torch.randn(2, 10, hidden_size)
        # Simulate HuggingFace-style tuple output
        tuple_output = (layer_output, torch.randn(2, 8, 64, 64))  # (hidden, attn_weights)

        result = hook(module=None, input=None, output=tuple_output)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 10, hidden_size)

    def test_hook_disable_enable(self):
        from l2v_cot.intervention import LatentInterventionHook

        hidden_size = 32
        cot_rep = torch.ones(hidden_size)
        hook = LatentInterventionHook(
            cot_representation=cot_rep,
            token_positions="last",
        )

        layer_output = torch.zeros(1, 5, hidden_size)

        # Disabled: output should be unchanged
        hook.disable()
        result = hook(module=None, input=None, output=layer_output.clone())
        assert torch.equal(result, layer_output)

        # Enabled: output should change
        hook.enable()
        result = hook(module=None, input=None, output=layer_output.clone())
        assert not torch.equal(result[:, -1, :], layer_output[:, -1, :])

    def test_hook_dimension_mismatch_resamples(self):
        from l2v_cot.intervention import LatentInterventionHook

        # CoT rep has 4096 dims but layer has 2048 dims
        cot_rep = torch.randn(4096)
        hook = LatentInterventionHook(
            cot_representation=cot_rep,
            token_positions="last",
        )

        layer_output = torch.randn(2, 10, 2048)
        result = hook(module=None, input=None, output=layer_output)
        # Should not raise, and output shape should match input shape
        assert result.shape == (2, 10, 2048)

    def test_hook_all_positions(self):
        from l2v_cot.intervention import LatentInterventionHook

        hidden_size = 32
        cot_rep = torch.ones(hidden_size)
        hook = LatentInterventionHook(
            cot_representation=cot_rep,
            token_positions="all",
        )

        layer_output = torch.zeros(1, 5, hidden_size)
        result = hook(module=None, input=None, output=layer_output.clone())
        # All positions should be modified
        assert torch.all(result != 0)

    def test_hook_specific_positions(self):
        from l2v_cot.intervention import LatentInterventionHook

        hidden_size = 32
        cot_rep = torch.ones(hidden_size)
        hook = LatentInterventionHook(
            cot_representation=cot_rep,
            token_positions=[1, 3],
        )

        layer_output = torch.zeros(1, 5, hidden_size)
        result = hook(module=None, input=None, output=layer_output.clone())

        # Positions 1 and 3 should be modified
        assert torch.all(result[:, 1, :] != 0)
        assert torch.all(result[:, 3, :] != 0)
        # Other positions should be unchanged
        assert torch.all(result[:, 0, :] == 0)
        assert torch.all(result[:, 2, :] == 0)
        assert torch.all(result[:, 4, :] == 0)


class TestL2VCoTIntervention:
    """Integration tests for L2VCoTIntervention."""

    def _make_simple_vlm(self, hidden_size=64, n_layers=4):
        """Create a minimal VLM-like model for testing."""
        import torch.nn as nn

        class SimpleLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                return (self.linear(x), None)

        class SimpleLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [SimpleLayer(hidden_size) for _ in range(n_layers)]
                )

        class SimpleVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleLM()

            def forward(self, x):
                h = x
                for layer in self.model.layers:
                    h, _ = layer(h)
                return h

        vlm = SimpleVLM()
        # Mock config
        vlm.config = type("Config", (), {"hidden_size": hidden_size})()
        return vlm

    def test_load_and_apply_representations(self):
        from l2v_cot.intervention import L2VCoTIntervention

        vlm = self._make_simple_vlm(hidden_size=64, n_layers=4)
        intervention = L2VCoTIntervention(
            llm=None,
            vlm=vlm,
            target_layers=[2, 3],
        )

        # Load fake representations
        representations = {31: torch.randn(64)}
        intervention.load_cot_representations(representations)

        # Test that apply() context manager works
        x = torch.randn(1, 5, 64)
        with intervention.apply():
            output = vlm(x)

        assert output.shape == (1, 5, 64)

    def test_hooks_removed_after_context(self):
        from l2v_cot.intervention import L2VCoTIntervention

        vlm = self._make_simple_vlm(hidden_size=64, n_layers=4)
        intervention = L2VCoTIntervention(
            llm=None,
            vlm=vlm,
            target_layers=[2, 3],
        )
        intervention.load_cot_representations({31: torch.randn(64)})

        with intervention.apply():
            assert len(intervention._hooks) > 0

        # Hooks should be cleaned up
        assert len(intervention._hooks) == 0

    def test_save_load_representations(self, tmp_path):
        from l2v_cot.intervention import L2VCoTIntervention

        vlm = self._make_simple_vlm()
        intervention = L2VCoTIntervention(llm=None, vlm=vlm)
        representations = {31: torch.randn(64), 30: torch.randn(64)}
        intervention.load_cot_representations(representations)

        save_path = str(tmp_path / "cot_repr.pt")
        intervention.save_cot_representations(save_path)

        # Load back
        loaded = torch.load(save_path)
        assert set(loaded.keys()) == set(representations.keys())
        for k in representations:
            assert torch.allclose(loaded[k], representations[k])
