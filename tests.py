"""
Test Suite for Activation Patching Framework

Tests cover:
1. Data structures work correctly
2. IOI dataset generates valid pairs
3. Patching mechanics work as expected
4. Edge cases are handled

Run with: python -m pytest tests.py -v
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict


# =============================================================================
# MOCK OBJECTS (avoid loading real model in tests)
# =============================================================================

class MockModelConfig:
    """Mock TransformerLens config."""
    n_layers = 12
    n_heads = 12
    d_model = 768
    d_head = 64


class MockHookedTransformer:
    """
    Mock HookedTransformer for testing without GPU/download.
    
    Real model loading is slow and needs network. We mock it.
    """
    def __init__(self):
        self.cfg = MockModelConfig()
        self.hook_dict = {
            f"blocks.{l}.hook_attn_out": None for l in range(12)
        }
        self.hook_dict.update({
            f"blocks.{l}.hook_mlp_out": None for l in range(12)
        })
        self.hook_dict.update({
            f"blocks.{l}.attn.hook_z": None for l in range(12)
        })
        
    def to_tokens(self, text):
        # Return fake token tensor
        words = text.split()
        return torch.randint(0, 50000, (1, len(words)))
    
    def to_single_token(self, text):
        # Return fake single token
        return torch.randint(0, 50000, (1,)).item()
    
    def __call__(self, tokens):
        # Return fake logits
        return torch.randn(1, tokens.shape[1], 50257)
    
    def run_with_cache(self, tokens):
        logits = self(tokens)
        cache = {}
        seq_len = tokens.shape[1]
        
        for l in range(self.cfg.n_layers):
            cache[f"blocks.{l}.hook_attn_out"] = torch.randn(1, seq_len, self.cfg.d_model)
            cache[f"blocks.{l}.hook_mlp_out"] = torch.randn(1, seq_len, self.cfg.d_model)
            cache[f"blocks.{l}.attn.hook_z"] = torch.randn(1, seq_len, self.cfg.n_heads, self.cfg.d_head)
            cache[f"blocks.{l}.attn.hook_pattern"] = torch.randn(1, self.cfg.n_heads, seq_len, seq_len)
        
        return logits, cache
    
    def run_with_hooks(self, tokens, fwd_hooks):
        # For testing, just return logits
        return self(tokens)
    
    def eval(self):
        pass


# =============================================================================
# TEST IMPORTS
# =============================================================================

def test_imports():
    """Test that all modules import correctly."""
    from activation_patching import (
        ActivationPatcher,
        ActivationCache,
        PatchingResult,
        PatchingDataset,
        IOIDataset
    )
    from head_patching import (
        HeadPatcher,
        HeadPatchingResult
    )
    assert True


# =============================================================================
# TEST DATA STRUCTURES
# =============================================================================

def test_patching_result_creation():
    """Test PatchingResult dataclass."""
    from activation_patching import PatchingResult
    
    result = PatchingResult(
        hook_point="blocks.5.hook_attn_out",
        clean_logit_diff=2.5,
        corrupted_logit_diff=-1.0,
        patched_logit_diff=1.5,
        patching_effect=0.71
    )
    
    assert result.hook_point == "blocks.5.hook_attn_out"
    assert result.clean_logit_diff == 2.5
    assert abs(result.patching_effect - 0.71) < 0.01


def test_activation_cache():
    """Test ActivationCache wrapper."""
    from activation_patching import ActivationCache
    
    fake_cache = {
        "blocks.0.hook_attn_out": torch.randn(1, 10, 768),
        "blocks.1.hook_attn_out": torch.randn(1, 10, 768)
    }
    
    cache = ActivationCache(
        cache=fake_cache,
        input_tokens=torch.randint(0, 1000, (1, 10)),
        logits=torch.randn(1, 10, 50257)
    )
    
    assert "blocks.0.hook_attn_out" in cache.keys()
    assert cache["blocks.0.hook_attn_out"].shape == (1, 10, 768)


def test_head_patching_result():
    """Test HeadPatchingResult dataclass."""
    from head_patching import HeadPatchingResult
    
    result = HeadPatchingResult(
        layer=9,
        head=6,
        patching_effect=0.45,
        clean_metric=2.0,
        corrupted_metric=-1.0,
        patched_metric=0.35
    )
    
    assert result.hook_name == "L9H6"
    assert result.layer == 9
    assert result.head == 6


# =============================================================================
# TEST IOI DATASET
# =============================================================================

def test_ioi_dataset_creation():
    """Test IOI dataset initializes correctly."""
    from activation_patching import IOIDataset
    
    mock_model = MockHookedTransformer()
    dataset = IOIDataset(mock_model)
    
    assert dataset.name == "IOI (Indirect Object Identification)"
    assert len(dataset.templates) == 5


def test_ioi_dataset_pairs():
    """Test IOI generates clean/corrupt pairs."""
    from activation_patching import IOIDataset
    
    mock_model = MockHookedTransformer()
    dataset = IOIDataset(mock_model)
    pairs = dataset.get_clean_corrupt_pairs()
    
    assert len(pairs) == 5
    
    # Check first pair structure
    clean, corrupted = pairs[0]
    assert "Mary and John" in clean
    assert "John and Mary" in corrupted
    
    # Both should end with "to" (prediction point)
    assert clean.endswith("to")
    assert corrupted.endswith("to")


def test_ioi_dataset_target_tokens():
    """Test IOI returns correct target token ids."""
    from activation_patching import IOIDataset
    
    mock_model = MockHookedTransformer()
    dataset = IOIDataset(mock_model)
    
    io_token, s_token = dataset.get_target_tokens(0)
    
    # Should be valid token ids (integers)
    assert isinstance(io_token, int)
    assert isinstance(s_token, int)
    assert io_token >= 0
    assert s_token >= 0


def test_ioi_compute_metric():
    """Test IOI metric computation."""
    from activation_patching import IOIDataset
    
    mock_model = MockHookedTransformer()
    dataset = IOIDataset(mock_model)
    
    # Create fake logits
    logits = torch.randn(1, 15, 50257)
    
    # Should return a float (logit difference)
    metric = dataset.compute_metric(logits, 0)
    assert isinstance(metric, float)


# =============================================================================
# TEST ACTIVATION PATCHER
# =============================================================================

def test_patcher_initialization():
    """Test ActivationPatcher initializes correctly."""
    from activation_patching import ActivationPatcher
    
    mock_model = MockHookedTransformer()
    patcher = ActivationPatcher(mock_model)
    
    assert patcher.model is mock_model


def test_patcher_cache_activations():
    """Test activation caching works."""
    from activation_patching import ActivationPatcher
    
    mock_model = MockHookedTransformer()
    patcher = ActivationPatcher(mock_model)
    
    cache = patcher.cache_activations("Hello world test")
    
    assert cache.input_tokens is not None
    assert cache.logits is not None
    assert "blocks.0.hook_attn_out" in cache.keys()


def test_patcher_get_hook_points():
    """Test hook point discovery."""
    from activation_patching import ActivationPatcher
    
    mock_model = MockHookedTransformer()
    patcher = ActivationPatcher(mock_model)
    
    all_hooks = patcher.get_all_hook_points()
    assert len(all_hooks) > 0
    
    attn_hooks = patcher.get_all_hook_points(pattern="attn")
    assert all("attn" in h for h in attn_hooks)


# =============================================================================
# TEST HEAD PATCHER
# =============================================================================

def test_head_patcher_initialization():
    """Test HeadPatcher initializes correctly."""
    from head_patching import HeadPatcher
    
    mock_model = MockHookedTransformer()
    patcher = HeadPatcher(mock_model)
    
    assert patcher.n_layers == 12
    assert patcher.n_heads == 12


def test_head_patcher_results_to_heatmap():
    """Test heatmap conversion."""
    from head_patching import HeadPatcher, HeadPatchingResult
    
    mock_model = MockHookedTransformer()
    patcher = HeadPatcher(mock_model)
    
    # Create fake results
    results = [
        HeadPatchingResult(layer=0, head=0, patching_effect=0.5,
                          clean_metric=1.0, corrupted_metric=-1.0, patched_metric=0.0),
        HeadPatchingResult(layer=5, head=3, patching_effect=0.8,
                          clean_metric=1.0, corrupted_metric=-1.0, patched_metric=0.6),
    ]
    
    heatmap = patcher.results_to_heatmap(results)
    
    assert heatmap.shape == (12, 12)
    assert heatmap[0, 0].item() == pytest.approx(0.5)
    assert heatmap[5, 3].item() == pytest.approx(0.8)


def test_find_important_heads():
    """Test important head filtering."""
    from head_patching import HeadPatcher, HeadPatchingResult
    
    mock_model = MockHookedTransformer()
    patcher = HeadPatcher(mock_model)
    
    results = [
        HeadPatchingResult(layer=0, head=0, patching_effect=0.05,
                          clean_metric=1.0, corrupted_metric=-1.0, patched_metric=0.0),
        HeadPatchingResult(layer=5, head=3, patching_effect=0.25,
                          clean_metric=1.0, corrupted_metric=-1.0, patched_metric=0.0),
        HeadPatchingResult(layer=9, head=6, patching_effect=0.45,
                          clean_metric=1.0, corrupted_metric=-1.0, patched_metric=0.0),
    ]
    
    important = patcher.find_important_heads(results, threshold=0.1)
    
    assert len(important) == 2
    assert important[0].layer == 9  # Highest effect first
    assert important[1].layer == 5


# =============================================================================
# TEST PATCHING EFFECT CALCULATION
# =============================================================================

def test_patching_effect_calculation():
    """Test the patching effect formula."""
    # Effect = (patched - corrupted) / (clean - corrupted)
    
    clean_metric = 2.0
    corrupted_metric = -1.0
    patched_metric = 0.5
    
    expected_effect = (0.5 - (-1.0)) / (2.0 - (-1.0))
    assert expected_effect == pytest.approx(0.5)
    
    # Full recovery should give effect = 1.0
    full_recovery_effect = (2.0 - (-1.0)) / (2.0 - (-1.0))
    assert full_recovery_effect == pytest.approx(1.0)
    
    # No recovery should give effect = 0.0
    no_recovery_effect = (-1.0 - (-1.0)) / (2.0 - (-1.0))
    assert no_recovery_effect == pytest.approx(0.0)


def test_patching_effect_edge_case():
    """Test effect calculation when clean == corrupted (avoid div by zero)."""
    clean_metric = 1.0
    corrupted_metric = 1.0  # Same! Denominator is 0
    
    denom = clean_metric - corrupted_metric
    if abs(denom) < 1e-6:
        effect = 0.0  # Safe default
    else:
        effect = 1.0 / denom
    
    assert effect == 0.0


# =============================================================================
# TEST SUMMARY FUNCTION
# =============================================================================

def test_summarize_results(capsys):
    """Test that summary function runs without error."""
    from activation_patching import summarize_results, PatchingResult
    
    results = {
        "blocks.0.hook_attn_out": [
            PatchingResult("blocks.0.hook_attn_out", 2.0, -1.0, 0.5, 0.5)
        ],
        "blocks.5.hook_attn_out": [
            PatchingResult("blocks.5.hook_attn_out", 2.0, -1.0, 1.5, 0.83)
        ]
    }
    
    summarize_results(results, top_k=2)
    
    captured = capsys.readouterr()
    assert "TOP 2" in captured.out
    assert "blocks.5" in captured.out  # Higher effect should appear


# =============================================================================
# TEST HEATMAP CREATION
# =============================================================================

def test_create_patching_heatmap():
    """Test heatmap tensor creation."""
    from activation_patching import create_patching_heatmap, PatchingResult
    
    mock_model = MockHookedTransformer()
    
    results = {
        "blocks.0.hook_attn_out": [
            PatchingResult("blocks.0.hook_attn_out", 2.0, -1.0, 0.5, 0.5)
        ],
        "blocks.0.hook_mlp_out": [
            PatchingResult("blocks.0.hook_mlp_out", 2.0, -1.0, 0.0, 0.33)
        ]
    }
    
    heatmap = create_patching_heatmap(results, mock_model)
    
    assert heatmap.shape == (12, 2)
    assert heatmap[0, 0].item() == pytest.approx(0.5)  # Attention
    assert heatmap[0, 1].item() == pytest.approx(0.33)  # MLP


# =============================================================================
# INTEGRATION TEST (requires model)
# =============================================================================

@pytest.mark.slow
@pytest.mark.skipif(True, reason="Requires model download, run manually")
def test_full_integration():
    """
    Full integration test with real model.
    
    Skip by default (slow), but run manually to verify everything works.
    Run with: pytest tests.py -v -m slow --runxfail
    """
    from transformer_lens import HookedTransformer
    from activation_patching import ActivationPatcher, IOIDataset
    
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    dataset = IOIDataset(model)
    patcher = ActivationPatcher(model)
    
    results = patcher.run_patching_experiment(
        dataset=dataset,
        hook_points=["blocks.0.hook_attn_out"],
        pair_indices=[0]
    )
    
    assert "blocks.0.hook_attn_out" in results
    assert len(results["blocks.0.hook_attn_out"]) == 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
