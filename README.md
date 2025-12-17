# Activation Patching Framework

A modular library for **causal intervention** in transformer language models. This implements the core methodology from mechanistic interpretability research, specifically designed for tasks like the Indirect Object Identification (IOI) circuit analysis.

## What This Does

Activation patching (also called "causal tracing" or "causal scrubbing") answers a fundamental question: **which model components are causally responsible for a specific behavior?**

This is different from just looking at attention patterns or activation magnitudes. Those are correlational. This is causal.

### The Method

1. **Clean Run**: Process a prompt where the model behaves correctly. Cache all internal activations.
2. **Corrupted Run**: Process a modified prompt where the model fails. Different names, shuffled tokens, whatever breaks the behavior.
3. **Patch**: During the corrupted run, swap in ONE activation from the clean run at a specific location.
4. **Measure**: Did the output improve? If yes, that component is causally necessary.

If patching component X recovers correct behavior, then X is **causally responsible** for the task.

## Installation

```bash
# Clone the repo
git clone https://github.com/designer-coderajay/activation-patching-framework.git
cd activation-patching-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Layer-Level Patching

```python
from transformer_lens import HookedTransformer
from activation_patching import ActivationPatcher, IOIDataset

# Load model
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

# Create dataset and patcher
dataset = IOIDataset(model)
patcher = ActivationPatcher(model)

# Define what to patch
hook_points = [f"blocks.{l}.hook_attn_out" for l in range(12)]

# Run experiment
results = patcher.run_patching_experiment(
    dataset=dataset,
    hook_points=hook_points
)

# Analyze
from activation_patching import summarize_results
summarize_results(results, top_k=10)
```

### Head-Level Patching

```python
from head_patching import HeadPatcher, print_head_results

patcher = HeadPatcher(model)

# Patch every attention head
results = patcher.run_all_heads_experiment(
    dataset=dataset,
    patch_type="output"  # Patch the z vectors
)

# Find important heads
important = patcher.find_important_heads(results, threshold=0.1)
print_head_results(results, top_k=15)
```

### Visualization

```python
from visualize import create_head_heatmap, create_full_report

# Interactive heatmap
fig = create_head_heatmap(results, n_layers=12, n_heads=12)
fig.show()

# Generate all visualizations
create_full_report(
    head_results=results,
    layer_results=layer_results,
    n_layers=12,
    n_heads=12,
    save_dir="./figures"
)
```

## The IOI Task

The **Indirect Object Identification** task tests whether the model understands who did what to whom.

**Template**: "When [IO] and [S] went to the store, [S] gave a drink to ___"  
**Correct answer**: [IO] (the indirect object, the person receiving)

**Clean prompt**: "When Mary and John went to the store, John gave a drink to" → **Mary**  
**Corrupted prompt**: "When John and Mary went to the store, John gave a drink to" → **John** (wrong!)

The corruption swaps name positions. A model relying on superficial heuristics will fail. A model that truly tracks identity will succeed.

### Expected Findings

From the IOI paper (Anthropic, 2022), you should see:
- **Name Mover Heads** (L9-L11): HIGH patching effect. These copy the IO name to output.
- **S-Inhibition Heads** (L7-L8): Moderate effect. These suppress the subject name.
- **Duplicate Token Heads** (L0-L3): Some effect. These detect repeated names.
- **MLPs**: Generally lower effect than attention for this task.

## Project Structure

```
activation-patching-framework/
├── activation_patching.py   # Core patching engine
├── head_patching.py         # Head-level patching
├── visualize.py             # Plotting utilities
├── demo.py                  # Layer-level demo
├── head_demo.py             # Head-level demo
├── tests.py                 # Test suite
├── requirements.txt
└── README.md
```

## Creating Custom Datasets

Subclass `PatchingDataset` to create your own tasks:

```python
from activation_patching import PatchingDataset

class MyTask(PatchingDataset):
    def __init__(self, model):
        self.model = model
        # Your setup here
    
    @property
    def name(self) -> str:
        return "My Custom Task"
    
    def get_clean_corrupt_pairs(self):
        # Return [(clean_prompt, corrupted_prompt), ...]
        return [
            ("The cat sat on the mat", "The dog sat on the mat"),
            # ...
        ]
    
    def get_target_tokens(self, pair_idx):
        # Return (correct_token_id, incorrect_token_id)
        return (self.model.to_single_token(" cat"), 
                self.model.to_single_token(" dog"))
    
    def compute_metric(self, logits, pair_idx):
        # Compute your task-specific metric
        correct, incorrect = self.get_target_tokens(pair_idx)
        return (logits[0, -1, correct] - logits[0, -1, incorrect]).item()
```

## Interpreting Results

**Patching Effect** is normalized between 0 and 1:
- `effect ≈ 1.0`: Patching FULLY RECOVERS correct behavior. This component is necessary.
- `effect ≈ 0.0`: Patching has no effect. Component doesn't matter for this task.
- `effect < 0.0`: Patching makes things worse (rare, indicates interference).

The formula:
```
effect = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
```

## Running Tests

```bash
# Run all tests
python -m pytest tests.py -v

# Run only fast tests (no model loading)
python -m pytest tests.py -v -m "not slow"
```

## References

This implementation is based on:

1. **"Interpretability in the Wild"** (Wang et al., 2022) - The IOI paper
2. **"A Mathematical Framework for Transformer Circuits"** (Elhage et al., 2021) - The circuits framework
3. **TransformerLens** - The library that makes this possible

## Thesis Connection

This is the **exact methodology** for mechanistic interpretability research:

1. Define a task where you know the correct behavior
2. Use patching to identify causally important components
3. Analyze those components (attention patterns, weight inspection)
4. Reverse-engineer the algorithm the model learned

For induction head research, you would:
1. Create an induction task dataset (repeated patterns)
2. Run patching experiments
3. Identify heads with high patching effect
4. Verify they implement the induction circuit (attention to previous occurrence)

## License

MIT License - Use freely for research and education.
