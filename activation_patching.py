"""
Activation Patching Framework for Mechanistic Interpretability

This module implements causal intervention techniques to identify which model
components are causally responsible for specific behaviors. The core methodology:

1. Run model on CLEAN input (correct behavior) → cache activations
2. Run model on CORRUPTED input (broken behavior) → get wrong output  
3. PATCH: During corrupted run, inject clean activation at specific point
4. MEASURE: If output improves, that component is causally important

This is causal science, not correlation. We're not asking "what activates?"
We're asking "what CAUSES the output?"
"""

import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import transformer_lens as tl
from transformer_lens import HookedTransformer
from functools import partial


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PatchingResult:
    """
    Container for results of a single patching experiment.
    
    Attributes:
        hook_point: Where we patched (e.g., "blocks.5.attn.hook_z")
        clean_logit_diff: Logit difference on clean input (baseline correct)
        corrupted_logit_diff: Logit difference on corrupted input (baseline broken)
        patched_logit_diff: Logit difference after patching
        patching_effect: How much patching recovered correct behavior (0 to 1 scale)
    """
    hook_point: str
    clean_logit_diff: float
    corrupted_logit_diff: float
    patched_logit_diff: float
    patching_effect: float  # (patched - corrupted) / (clean - corrupted)


@dataclass 
class ActivationCache:
    """
    Stores cached activations from a forward pass.
    
    This is a lightweight wrapper, TransformerLens does the heavy lifting.
    We store the cache plus metadata about the run.
    """
    cache: Dict[str, Tensor]
    input_tokens: Tensor
    logits: Tensor
    
    def __getitem__(self, key: str) -> Tensor:
        return self.cache[key]
    
    def keys(self) -> List[str]:
        return list(self.cache.keys())


# =============================================================================
# ABSTRACT BASE CLASS FOR DATASETS
# =============================================================================

class PatchingDataset(ABC):
    """
    Abstract base class for patching experiments.
    
    To create a new task, subclass this and implement:
    - get_clean_corrupt_pairs(): Returns paired inputs
    - get_target_tokens(): Returns what tokens we're measuring
    - compute_metric(): How to score model output
    
    This is the "swap out the dataset" part of the senior constraint.
    """
    
    @abstractmethod
    def get_clean_corrupt_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns list of (clean_prompt, corrupted_prompt) pairs.
        
        Clean prompt: Model should produce correct answer
        Corrupted prompt: Model should fail (different names, shuffled, etc.)
        """
        pass
    
    @abstractmethod
    def get_target_tokens(self, pair_idx: int) -> Tuple[int, int]:
        """
        Returns (correct_token_id, incorrect_token_id) for measuring logit diff.
        
        Logit difference = logit[correct] - logit[incorrect]
        Positive = model prefers correct answer
        """
        pass
    
    @abstractmethod
    def compute_metric(self, logits: Tensor, pair_idx: int) -> float:
        """
        Compute task-specific metric from model output.
        
        For IOI: logit_diff = logit[IO] - logit[S]
        (Indirect Object token minus Subject token)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name for logging."""
        pass


# =============================================================================
# IOI DATASET IMPLEMENTATION  
# =============================================================================

class IOIDataset(PatchingDataset):
    """
    Indirect Object Identification task.
    
    Template: "When [IO] and [S] went to the store, [S] gave a drink to"
    Correct answer: [IO] (the indirect object, person receiving)
    
    This tests whether the model tracks WHO did what to WHOM.
    
    Corruption method: Swap the names in the first clause.
    Clean:     "When Mary and John went to the store, John gave a drink to" → Mary
    Corrupted: "When John and Mary went to the store, John gave a drink to" → John (wrong!)
    
    Why this corruption works: The model relies on positional heuristics.
    By swapping names, we break the "first name = IO" pattern.
    """
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        
        # Classic IOI examples
        # Format: (IO_name, S_name, place, object)
        self.templates = [
            ("Mary", "John", "store", "drink"),
            ("Alice", "Bob", "park", "ball"),
            ("Emma", "James", "restaurant", "menu"),
            ("Sarah", "Michael", "library", "book"),
            ("Lisa", "David", "office", "report"),
        ]
        
        # Pre-compute token IDs for names
        self._token_cache = {}
        for io, s, _, _ in self.templates:
            # Add space prefix because that's how tokens work mid-sentence
            self._token_cache[io] = self.model.to_single_token(" " + io)
            self._token_cache[s] = self.model.to_single_token(" " + s)
    
    @property
    def name(self) -> str:
        return "IOI (Indirect Object Identification)"
    
    def _make_prompt(self, io: str, s: str, place: str, obj: str) -> str:
        """Generate IOI prompt from template."""
        return f"When {io} and {s} went to the {place}, {s} gave a {obj} to"
    
    def get_clean_corrupt_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate clean/corrupted pairs.
        
        Clean: Normal order (IO first in "IO and S")
        Corrupted: Swapped order (S first in "S and IO")
        
        This is "ABC → ABB" corruption style from the IOI paper.
        """
        pairs = []
        for io, s, place, obj in self.templates:
            clean = self._make_prompt(io, s, place, obj)
            # Swap IO and S positions in the first clause
            corrupted = self._make_prompt(s, io, place, obj)
            pairs.append((clean, corrupted))
        return pairs
    
    def get_target_tokens(self, pair_idx: int) -> Tuple[int, int]:
        """
        Returns (IO_token_id, S_token_id).
        
        We measure logit[IO] - logit[S].
        Positive = model correctly predicts IO.
        """
        io, s, _, _ = self.templates[pair_idx]
        return self._token_cache[io], self._token_cache[s]
    
    def compute_metric(self, logits: Tensor, pair_idx: int) -> float:
        """
        Compute logit difference for IOI task.
        
        logit_diff = logit[IO] - logit[S]
        
        We look at the LAST token position (prediction position).
        """
        io_token, s_token = self.get_target_tokens(pair_idx)
        
        # Get logits at final position
        final_logits = logits[0, -1, :]  # [vocab_size]
        
        logit_diff = final_logits[io_token] - final_logits[s_token]
        return logit_diff.item()


# =============================================================================
# THE MAIN PATCHING ENGINE
# =============================================================================

class ActivationPatcher:
    """
    The core activation patching engine.
    
    This is model-agnostic. You give it a HookedTransformer and a dataset,
    it runs the patching experiments. The "swap out the model" part.
    
    Usage:
        patcher = ActivationPatcher(model)
        results = patcher.run_patching_experiment(
            dataset=IOIDataset(model),
            hook_points=["blocks.5.attn.hook_z", "blocks.7.mlp.hook_post"]
        )
    """
    
    def __init__(self, model: HookedTransformer):
        """
        Initialize patcher with a TransformerLens model.
        
        Args:
            model: A HookedTransformer instance (GPT-2, etc.)
        """
        self.model = model
        self.model.eval()  # Always eval mode for interpretability
        
    def cache_activations(self, prompt: str) -> ActivationCache:
        """
        Run forward pass and cache all activations.
        
        This is the "clean run" or "corrupted run" depending on input.
        TransformerLens makes this trivial with run_with_cache().
        """
        tokens = self.model.to_tokens(prompt)
        logits, cache = self.model.run_with_cache(tokens)
        
        return ActivationCache(
            cache=cache,
            input_tokens=tokens,
            logits=logits
        )
    
    def patch_activation(
        self,
        corrupted_prompt: str,
        clean_cache: ActivationCache,
        hook_point: str,
        position: Optional[int] = None
    ) -> Tensor:
        """
        Run corrupted input while patching in clean activation at hook_point.
        
        This is the core causal intervention:
        1. Start corrupted forward pass
        2. At hook_point, swap in activation from clean_cache
        3. Continue forward pass with patched activation
        4. Return final logits
        
        Args:
            corrupted_prompt: The broken input
            clean_cache: Cached activations from clean run
            hook_point: Where to patch (e.g., "blocks.5.attn.hook_z")
            position: Which sequence position to patch (None = all positions)
        
        Returns:
            Logits after patched forward pass
        """
        corrupted_tokens = self.model.to_tokens(corrupted_prompt)
        
        def patching_hook(
            activation: Tensor,
            hook: Any,
            clean_activation: Tensor,
            pos: Optional[int]
        ) -> Tensor:
            """
            The actual hook function that does the swap.
            
            If pos is None, patch ALL positions.
            Otherwise, only patch the specified position.
            """
            if pos is None:
                return clean_activation
            else:
                # Clone to avoid modifying in place
                patched = activation.clone()
                patched[:, pos, ...] = clean_activation[:, pos, ...]
                return patched
        
        # Get the clean activation we're patching in
        clean_activation = clean_cache[hook_point]
        
        # Create the hook with clean activation bound
        hook_fn = partial(
            patching_hook,
            clean_activation=clean_activation,
            pos=position
        )
        
        # Run with the patching hook
        patched_logits = self.model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_point, hook_fn)]
        )
        
        return patched_logits
    
    def run_patching_experiment(
        self,
        dataset: PatchingDataset,
        hook_points: List[str],
        pair_indices: Optional[List[int]] = None,
        position: Optional[int] = None
    ) -> Dict[str, List[PatchingResult]]:
        """
        Run full patching experiment across multiple hook points.
        
        For each (clean, corrupted) pair in dataset:
        1. Get clean metric (baseline correct)
        2. Get corrupted metric (baseline broken)
        3. For each hook_point, patch and measure recovery
        
        Args:
            dataset: PatchingDataset instance (IOI, etc.)
            hook_points: List of activation names to patch
            pair_indices: Which prompt pairs to use (None = all)
            position: Sequence position to patch (None = all)
        
        Returns:
            Dict mapping hook_point -> list of PatchingResult
        """
        pairs = dataset.get_clean_corrupt_pairs()
        
        if pair_indices is None:
            pair_indices = list(range(len(pairs)))
        
        results = {hp: [] for hp in hook_points}
        
        for idx in pair_indices:
            clean_prompt, corrupted_prompt = pairs[idx]
            
            # Cache clean activations
            clean_cache = self.cache_activations(clean_prompt)
            clean_metric = dataset.compute_metric(clean_cache.logits, idx)
            
            # Get corrupted baseline
            corrupted_cache = self.cache_activations(corrupted_prompt)
            corrupted_metric = dataset.compute_metric(corrupted_cache.logits, idx)
            
            # Patch each hook point
            for hook_point in hook_points:
                patched_logits = self.patch_activation(
                    corrupted_prompt=corrupted_prompt,
                    clean_cache=clean_cache,
                    hook_point=hook_point,
                    position=position
                )
                patched_metric = dataset.compute_metric(patched_logits, idx)
                
                # Compute patching effect (normalized recovery)
                # 0 = no recovery (still corrupted)
                # 1 = full recovery (back to clean)
                denom = clean_metric - corrupted_metric
                if abs(denom) < 1e-6:
                    effect = 0.0  # Avoid division by zero
                else:
                    effect = (patched_metric - corrupted_metric) / denom
                
                results[hook_point].append(PatchingResult(
                    hook_point=hook_point,
                    clean_logit_diff=clean_metric,
                    corrupted_logit_diff=corrupted_metric,
                    patched_logit_diff=patched_metric,
                    patching_effect=effect
                ))
        
        return results
    
    def get_all_hook_points(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all available hook points in the model.
        
        Useful for discovering what you can patch.
        
        Args:
            pattern: Optional filter (e.g., "attn" for attention only)
        
        Returns:
            List of hook point names
        """
        all_hooks = list(self.model.hook_dict.keys())
        
        if pattern:
            all_hooks = [h for h in all_hooks if pattern in h]
        
        return sorted(all_hooks)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def summarize_results(
    results: Dict[str, List[PatchingResult]],
    top_k: int = 10
) -> None:
    """
    Print summary of patching results, sorted by effect size.
    
    Args:
        results: Output from run_patching_experiment
        top_k: How many top results to show
    """
    # Compute average effect per hook point
    avg_effects = {}
    for hook_point, result_list in results.items():
        effects = [r.patching_effect for r in result_list]
        avg_effects[hook_point] = sum(effects) / len(effects)
    
    # Sort by effect
    sorted_hooks = sorted(
        avg_effects.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    print(f"\n{'='*60}")
    print(f"TOP {top_k} PATCHING RESULTS (by average effect)")
    print(f"{'='*60}")
    print(f"Effect = 1.0 means full recovery, 0.0 means no recovery\n")
    
    for hook_point, effect in sorted_hooks[:top_k]:
        bar = "█" * int(abs(effect) * 20)
        print(f"{hook_point:40s} | {effect:+.3f} | {bar}")


def create_patching_heatmap(
    results: Dict[str, List[PatchingResult]],
    model: HookedTransformer
) -> Tensor:
    """
    Create a layer x component heatmap of patching effects.
    
    Useful for visualizing which layers/components matter most.
    
    Returns:
        Tensor of shape [n_layers, n_components] with avg patching effects
    """
    n_layers = model.cfg.n_layers
    
    # Components: attn_out, mlp_out (can expand later)
    component_names = ["hook_attn_out", "hook_mlp_out"]
    n_components = len(component_names)
    
    heatmap = torch.zeros(n_layers, n_components)
    
    for layer in range(n_layers):
        for comp_idx, comp_name in enumerate(component_names):
            hook_point = f"blocks.{layer}.{comp_name}"
            if hook_point in results:
                effects = [r.patching_effect for r in results[hook_point]]
                heatmap[layer, comp_idx] = sum(effects) / len(effects)
    
    return heatmap


if __name__ == "__main__":
    # Quick test that imports work
    print("Activation Patching Framework loaded successfully.")
    print("Run demo.py for full demonstration.")
