"""
Demo: Head-Level Activation Patching

This is what the IOI paper actually does. Instead of patching whole layers,
we patch individual attention heads to identify the specific circuits.

Expected findings for IOI on GPT-2 Small:
- Name Mover Heads (L9H6, L9H9, L10H0, etc.): HIGH effect
  These heads "move" the IO name to the output position
  
- Duplicate Token Heads (L0H1, etc.): Some effect
  These help identify that a name appeared twice
  
- Induction-style Heads: Moderate effect
  General pattern matching that helps with the task

This is the EXACT methodology you'll use in your thesis.
"""

import torch
from transformer_lens import HookedTransformer
from activation_patching import IOIDataset
from head_patching import (
    HeadPatcher,
    print_head_results,
    print_heatmap_ascii
)
import warnings
warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("HEAD-LEVEL ACTIVATION PATCHING: IOI Task")
    print("=" * 60)
    print("\nThis identifies WHICH SPECIFIC HEADS matter for IOI.")
    print("This is the methodology for your thesis.\n")
    
    # =========================================================================
    # Load Model
    # =========================================================================
    print("[1/3] Loading GPT-2 Small...")
    
    device = "cpu"  # MPS can be finicky, CPU is reliable
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    print(f"      {model.cfg.n_layers} layers × {model.cfg.n_heads} heads")
    print(f"      = {model.cfg.n_layers * model.cfg.n_heads} total heads to patch")
    
    # =========================================================================
    # Set up experiment
    # =========================================================================
    print("\n[2/3] Setting up IOI experiment...")
    
    dataset = IOIDataset(model)
    patcher = HeadPatcher(model)
    
    pairs = dataset.get_clean_corrupt_pairs()
    print(f"      Using {len(pairs)} prompt pairs")
    
    # =========================================================================
    # Run head patching
    # =========================================================================
    print("\n[3/3] Patching all attention heads...")
    print("      (This takes a minute, we're doing 144 experiments)\n")
    
    results = patcher.run_all_heads_experiment(
        dataset=dataset,
        pair_indices=[0, 1, 2],  # Use 3 pairs for speed
        patch_type="output"
    )
    
    # =========================================================================
    # Results
    # =========================================================================
    
    # Top heads
    print_head_results(results, top_k=15)
    
    # Heatmap
    heatmap = patcher.results_to_heatmap(results)
    print_heatmap_ascii(heatmap, model.cfg.n_layers, model.cfg.n_heads)
    
    # Important heads
    important = patcher.find_important_heads(results, threshold=0.05)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(important)} heads with >5% patching effect")
    print(f"{'='*60}")
    
    if important:
        print("\nThese are the CAUSALLY IMPORTANT heads for IOI:")
        for r in important[:10]:
            role = classify_ioi_head(r.layer, r.head)
            print(f"  L{r.layer}H{r.head}: effect={r.patching_effect:.3f}  ({role})")
    
    # =========================================================================
    # What this means for your thesis
    # =========================================================================
    print(f"\n{'='*60}")
    print("THESIS METHODOLOGY NOTES")
    print(f"{'='*60}")
    print("""
    What you just did is CAUSAL TRACING (activation patching):
    
    1. You ran the model on clean input → cached activations
    2. You ran the model on corrupted input → got wrong answer  
    3. You patched ONE HEAD at a time from clean into corrupted
    4. You measured: did patching this head recover the right answer?
    
    This is different from just looking at attention patterns or
    activation magnitudes. Those are CORRELATIONAL. This is CAUSAL.
    
    The heads with high patching effect are CAUSALLY NECESSARY for IOI.
    If you "break" these heads, the model fails the task.
    
    For your thesis on GPT-2 and induction heads:
    - Use this same methodology
    - Your "task" will be detecting induction-like patterns
    - You'll identify which heads implement induction circuits
    - You can then do weight inspection on those specific heads
    
    This is publishable methodology. The IOI paper (Anthropic, 2022)
    used exactly this approach.
    """)


def classify_ioi_head(layer: int, head: int) -> str:
    """
    Rough classification of IOI heads based on known circuit.
    
    These are approximations based on the IOI paper findings.
    Your actual results may vary slightly.
    """
    # Known name mover heads in GPT-2 Small (approximate)
    name_movers = [(9, 6), (9, 9), (10, 0), (10, 7), (10, 10), (11, 2), (11, 9)]
    
    # Known S-inhibition heads
    s_inhibition = [(7, 3), (7, 9), (8, 6), (8, 10)]
    
    # Duplicate token heads
    duplicate = [(0, 1), (0, 10), (3, 0)]
    
    # Induction heads  
    induction = [(5, 5), (5, 8), (5, 9), (6, 9)]
    
    if (layer, head) in name_movers:
        return "Name Mover"
    elif (layer, head) in s_inhibition:
        return "S-Inhibition"
    elif (layer, head) in duplicate:
        return "Duplicate Token"
    elif (layer, head) in induction:
        return "Induction-like"
    elif layer >= 9:
        return "Late layer (likely output)"
    elif layer <= 2:
        return "Early layer (likely general)"
    else:
        return "Middle layer"


if __name__ == "__main__":
    main()
