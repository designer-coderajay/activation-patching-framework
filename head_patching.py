"""
Head-Level Activation Patching

Layer-level patching tells you "layer 9 attention matters."
Head-level patching tells you "head 9.6 is the name mover."

This module patches individual attention heads, which is what the IOI paper
actually does to identify specific circuits.

The key insight: attention heads are the interpretable units. Each head
learns a specific "attention pattern" (what to attend to) and "OV circuit"
(what to copy from attended positions). By patching individual heads,
you can identify which specific computations matter.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from functools import partial
import einops

from activation_patching import (
    ActivationCache,
    PatchingDataset,
    PatchingResult,
    IOIDataset
)


@dataclass
class HeadPatchingResult:
    """
    Result from patching a single attention head.
    
    Attributes:
        layer: Which layer (0-indexed)
        head: Which head in that layer (0-indexed)
        patching_effect: Normalized recovery (0 = no effect, 1 = full recovery)
        clean_metric: Baseline correct behavior
        corrupted_metric: Baseline broken behavior
        patched_metric: After patching this head
    """
    layer: int
    head: int
    patching_effect: float
    clean_metric: float
    corrupted_metric: float
    patched_metric: float
    
    @property
    def hook_name(self) -> str:
        return f"L{self.layer}H{self.head}"


class HeadPatcher:
    """
    Patches individual attention heads for fine-grained circuit analysis.
    
    There are multiple ways to patch a head:
    1. Patch the output (z) - what the head writes to residual stream
    2. Patch the attention pattern - what positions the head attends to
    3. Patch the value vectors - what information gets moved
    
    For IOI, patching z (the output) is most informative because we want
    to know: "does this head's contribution to the residual stream matter?"
    """
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.model.eval()
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        
    def cache_activations(self, prompt: str) -> ActivationCache:
        """Cache all activations from a forward pass."""
        tokens = self.model.to_tokens(prompt)
        logits, cache = self.model.run_with_cache(tokens)
        return ActivationCache(cache=cache, input_tokens=tokens, logits=logits)
    
    def patch_head_output(
        self,
        corrupted_prompt: str,
        clean_cache: ActivationCache,
        layer: int,
        head: int,
        position: Optional[int] = None
    ) -> Tensor:
        """
        Patch a single head's output (z vector).
        
        The z vector is shape [batch, seq, n_heads, d_head].
        We patch just one head's slice: z[:, :, head, :].
        
        This answers: "If this head produced its clean output during
        the corrupted run, would the model recover?"
        """
        corrupted_tokens = self.model.to_tokens(corrupted_prompt)
        
        # The hook point for attention output before combining heads
        hook_point = f"blocks.{layer}.attn.hook_z"
        
        def head_patch_hook(
            z: Tensor,  # [batch, seq, n_heads, d_head]
            hook,
            clean_z: Tensor,
            target_head: int,
            pos: Optional[int]
        ) -> Tensor:
            """Patch just one head's output."""
            patched_z = z.clone()
            
            if pos is None:
                # Patch all positions for this head
                patched_z[:, :, target_head, :] = clean_z[:, :, target_head, :]
            else:
                # Patch specific position for this head
                patched_z[:, pos, target_head, :] = clean_z[:, pos, target_head, :]
            
            return patched_z
        
        # Get clean z for this layer
        clean_z = clean_cache[hook_point]
        
        hook_fn = partial(
            head_patch_hook,
            clean_z=clean_z,
            target_head=head,
            pos=position
        )
        
        patched_logits = self.model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_point, hook_fn)]
        )
        
        return patched_logits
    
    def patch_attention_pattern(
        self,
        corrupted_prompt: str,
        clean_cache: ActivationCache,
        layer: int,
        head: int
    ) -> Tensor:
        """
        Patch a head's attention pattern (which positions it attends to).
        
        The pattern is shape [batch, n_heads, seq_q, seq_k].
        This answers: "If this head attended to the same positions as in
        the clean run, would that fix things?"
        
        Note: This is less commonly used than patching z, but useful for
        understanding whether the issue is "wrong attention" vs "wrong OV".
        """
        corrupted_tokens = self.model.to_tokens(corrupted_prompt)
        
        hook_point = f"blocks.{layer}.attn.hook_pattern"
        
        def pattern_patch_hook(
            pattern: Tensor,  # [batch, n_heads, seq_q, seq_k]
            hook,
            clean_pattern: Tensor,
            target_head: int
        ) -> Tensor:
            """Patch one head's attention pattern."""
            patched = pattern.clone()
            patched[:, target_head, :, :] = clean_pattern[:, target_head, :, :]
            return patched
        
        clean_pattern = clean_cache[hook_point]
        
        hook_fn = partial(
            pattern_patch_hook,
            clean_pattern=clean_pattern,
            target_head=head
        )
        
        patched_logits = self.model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_point, hook_fn)]
        )
        
        return patched_logits
    
    def run_all_heads_experiment(
        self,
        dataset: PatchingDataset,
        pair_indices: Optional[List[int]] = None,
        position: Optional[int] = None,
        patch_type: str = "output"  # "output" or "pattern"
    ) -> List[HeadPatchingResult]:
        """
        Patch every head and measure effect.
        
        This creates the classic "head patching heatmap" from mech interp papers.
        
        Args:
            dataset: The task dataset (IOI, etc.)
            pair_indices: Which prompt pairs to use
            position: Specific position to patch (None = all)
            patch_type: "output" for z vectors, "pattern" for attention patterns
        
        Returns:
            List of HeadPatchingResult, one per head
        """
        pairs = dataset.get_clean_corrupt_pairs()
        if pair_indices is None:
            pair_indices = list(range(len(pairs)))
        
        results = []
        
        # For each head
        total_heads = self.n_layers * self.n_heads
        current = 0
        
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                current += 1
                
                effects = []
                clean_metrics = []
                corrupted_metrics = []
                patched_metrics = []
                
                # Average over prompt pairs
                for idx in pair_indices:
                    clean_prompt, corrupted_prompt = pairs[idx]
                    
                    # Cache clean run
                    clean_cache = self.cache_activations(clean_prompt)
                    clean_metric = dataset.compute_metric(clean_cache.logits, idx)
                    
                    # Get corrupted baseline
                    corrupted_cache = self.cache_activations(corrupted_prompt)
                    corrupted_metric = dataset.compute_metric(corrupted_cache.logits, idx)
                    
                    # Patch this head
                    if patch_type == "output":
                        patched_logits = self.patch_head_output(
                            corrupted_prompt, clean_cache, layer, head, position
                        )
                    else:
                        patched_logits = self.patch_attention_pattern(
                            corrupted_prompt, clean_cache, layer, head
                        )
                    
                    patched_metric = dataset.compute_metric(patched_logits, idx)
                    
                    # Compute effect
                    denom = clean_metric - corrupted_metric
                    if abs(denom) < 1e-6:
                        effect = 0.0
                    else:
                        effect = (patched_metric - corrupted_metric) / denom
                    
                    effects.append(effect)
                    clean_metrics.append(clean_metric)
                    corrupted_metrics.append(corrupted_metric)
                    patched_metrics.append(patched_metric)
                
                # Average results for this head
                avg_effect = sum(effects) / len(effects)
                avg_clean = sum(clean_metrics) / len(clean_metrics)
                avg_corrupted = sum(corrupted_metrics) / len(corrupted_metrics)
                avg_patched = sum(patched_metrics) / len(patched_metrics)
                
                results.append(HeadPatchingResult(
                    layer=layer,
                    head=head,
                    patching_effect=avg_effect,
                    clean_metric=avg_clean,
                    corrupted_metric=avg_corrupted,
                    patched_metric=avg_patched
                ))
                
                # Progress indicator
                if current % 12 == 0:
                    print(f"     Progress: {current}/{total_heads} heads patched...")
        
        return results
    
    def results_to_heatmap(self, results: List[HeadPatchingResult]) -> Tensor:
        """
        Convert results to [n_layers, n_heads] tensor for visualization.
        """
        heatmap = torch.zeros(self.n_layers, self.n_heads)
        
        for r in results:
            heatmap[r.layer, r.head] = r.patching_effect
        
        return heatmap
    
    def find_important_heads(
        self,
        results: List[HeadPatchingResult],
        threshold: float = 0.1
    ) -> List[HeadPatchingResult]:
        """
        Find heads with significant patching effect.
        
        These are the heads that are CAUSALLY IMPORTANT for the task.
        """
        important = [r for r in results if abs(r.patching_effect) > threshold]
        important.sort(key=lambda x: abs(x.patching_effect), reverse=True)
        return important


def print_head_results(results: List[HeadPatchingResult], top_k: int = 20):
    """Pretty print top heads by patching effect."""
    sorted_results = sorted(results, key=lambda x: abs(x.patching_effect), reverse=True)
    
    print(f"\n{'='*60}")
    print(f"TOP {top_k} ATTENTION HEADS BY PATCHING EFFECT")
    print(f"{'='*60}")
    print(f"\n{'Head':<10} {'Effect':>10} {'Visual'}")
    print("-" * 40)
    
    for r in sorted_results[:top_k]:
        bar = "█" * int(abs(r.patching_effect) * 30)
        sign = "+" if r.patching_effect >= 0 else "-"
        print(f"L{r.layer}H{r.head:<6} {sign}{abs(r.patching_effect):.3f}     {bar}")


def print_heatmap_ascii(heatmap: Tensor, n_layers: int, n_heads: int):
    """Print ASCII heatmap of head effects."""
    print(f"\n{'='*60}")
    print("HEAD PATCHING HEATMAP")
    print("(Brighter = higher patching effect = more causally important)")
    print(f"{'='*60}\n")
    
    # Header
    print("     ", end="")
    for h in range(n_heads):
        print(f"H{h:<2}", end=" ")
    print()
    print("     " + "-" * (n_heads * 4))
    
    # Each layer
    for layer in range(n_layers):
        print(f"L{layer:2d} |", end=" ")
        for head in range(n_heads):
            effect = heatmap[layer, head].item()
            # Convert to visual intensity
            if effect > 0.2:
                char = "██"
            elif effect > 0.1:
                char = "▓▓"
            elif effect > 0.05:
                char = "▒▒"
            elif effect > 0.02:
                char = "░░"
            else:
                char = "  "
            print(char, end=" ")
        print()


if __name__ == "__main__":
    print("Head-level patching module loaded.")
    print("Run head_demo.py for full head patching demonstration.")
