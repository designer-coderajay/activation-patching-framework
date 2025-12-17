"""
Demo: Activation Patching on IOI Task

This script demonstrates the full activation patching workflow:
1. Load GPT-2 Small with TransformerLens
2. Set up IOI dataset
3. Run patching experiments
4. Visualize which components are causally important

Expected findings (from the IOI paper):
- Name mover heads in layers 9-11 should have HIGH patching effect
- Early layers should have LOWER patching effect
- MLP contributions should be smaller than attention for this task
"""

import torch
from transformer_lens import HookedTransformer
from activation_patching import (
    ActivationPatcher,
    IOIDataset,
    summarize_results,
    create_patching_heatmap
)
import warnings
warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("ACTIVATION PATCHING DEMO: IOI Task on GPT-2 Small")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Load Model
    # =========================================================================
    print("\n[1/4] Loading GPT-2 Small...")
    
    # Use CPU for stability, MPS can be weird for interpretability work
    device = "cpu"  # Change to "cuda" if you have GPU
    
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device
    )
    print(f"     Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    print(f"     Device: {device}")
    
    # =========================================================================
    # STEP 2: Set up Dataset
    # =========================================================================
    print("\n[2/4] Setting up IOI Dataset...")
    
    dataset = IOIDataset(model)
    pairs = dataset.get_clean_corrupt_pairs()
    
    print(f"     Dataset: {dataset.name}")
    print(f"     Number of prompt pairs: {len(pairs)}")
    print(f"\n     Example pair:")
    print(f"     Clean:     '{pairs[0][0]}'")
    print(f"     Corrupted: '{pairs[0][1]}'")
    
    # Quick sanity check: does the model get IOI right?
    print("\n     Sanity check (clean prompt):")
    tokens = model.to_tokens(pairs[0][0])
    logits = model(tokens)
    probs = torch.softmax(logits[0, -1], dim=-1)
    
    io_token, s_token = dataset.get_target_tokens(0)
    io_prob = probs[io_token].item()
    s_prob = probs[s_token].item()
    
    print(f"     P(Mary) = {io_prob:.4f}  (should be higher)")
    print(f"     P(John) = {s_prob:.4f}  (should be lower)")
    print(f"     Logit diff = {(logits[0, -1, io_token] - logits[0, -1, s_token]).item():.2f}")
    
    # =========================================================================
    # STEP 3: Run Patching Experiment
    # =========================================================================
    print("\n[3/4] Running Activation Patching...")
    
    patcher = ActivationPatcher(model)
    
    # We'll patch attention output and MLP output at each layer
    # These are the "residual stream contributions" from each component
    hook_points = []
    for layer in range(model.cfg.n_layers):
        hook_points.append(f"blocks.{layer}.hook_attn_out")
        hook_points.append(f"blocks.{layer}.hook_mlp_out")
    
    print(f"     Patching {len(hook_points)} hook points...")
    print(f"     (This tests which layer/component causally affects IOI)")
    
    results = patcher.run_patching_experiment(
        dataset=dataset,
        hook_points=hook_points,
        pair_indices=[0, 1, 2]  # Use first 3 pairs for demo (faster)
    )
    
    # =========================================================================
    # STEP 4: Analyze Results
    # =========================================================================
    print("\n[4/4] Analyzing Results...")
    
    # Print top results
    summarize_results(results, top_k=15)
    
    # Create and display heatmap data
    print("\n" + "=" * 60)
    print("LAYER x COMPONENT HEATMAP")
    print("=" * 60)
    
    heatmap = create_patching_heatmap(results, model)
    
    print("\nLayer | Attention | MLP")
    print("-" * 30)
    for layer in range(model.cfg.n_layers):
        attn_effect = heatmap[layer, 0].item()
        mlp_effect = heatmap[layer, 1].item()
        
        # Visual bars
        attn_bar = "█" * int(abs(attn_effect) * 10)
        mlp_bar = "█" * int(abs(mlp_effect) * 10)
        
        print(f"  {layer:2d}  | {attn_effect:+.2f} {attn_bar:10s} | {mlp_effect:+.2f} {mlp_bar}")
    
    # =========================================================================
    # Interpretation Guide
    # =========================================================================
    print("\n" + "=" * 60)
    print("HOW TO INTERPRET THESE RESULTS")
    print("=" * 60)
    print("""
    Patching Effect = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
    
    • Effect ≈ 1.0: Patching this component FULLY RECOVERS correct behavior
                    → This component is CAUSALLY NECESSARY for the task
    
    • Effect ≈ 0.0: Patching this component has NO EFFECT
                    → This component doesn't matter for IOI
    
    • Effect < 0.0: Patching makes things WORSE
                    → Rare, but can happen with interference
    
    For IOI specifically, you should see:
    - Late layers (9-11) attention: HIGH effect (name mover heads)
    - Early layers: LOW effect (general processing, not IOI-specific)
    - MLPs: Generally lower than attention for this task
    
    This is CAUSAL evidence, not correlation. We're not asking "what fires?"
    We're asking "what CONTROLS the output?"
    """)


if __name__ == "__main__":
    main()
