"""
Visualization Module for Activation Patching

Creates interactive visualizations of patching results.
Uses Plotly for interactive heatmaps.
"""

import torch
from torch import Tensor
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from head_patching import HeadPatchingResult


def create_head_heatmap(
    results: List[HeadPatchingResult],
    n_layers: int,
    n_heads: int,
    title: str = "Head Patching Effects",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive heatmap of head patching effects.
    
    Args:
        results: List of HeadPatchingResult from patching experiment
        n_layers: Number of layers in model
        n_heads: Number of heads per layer
        title: Plot title
        save_path: Optional path to save HTML file
    
    Returns:
        Plotly Figure object
    """
    # Build matrix
    matrix = np.zeros((n_layers, n_heads))
    for r in results:
        matrix[r.layer, r.head] = r.patching_effect
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"H{i}" for i in range(n_heads)],
        y=[f"L{i}" for i in range(n_layers)],
        colorscale="RdBu",
        zmid=0,  # Center colorscale at 0
        colorbar=dict(title="Patching<br>Effect"),
        hovertemplate="Layer %{y}<br>Head %{x}<br>Effect: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Attention Head",
        yaxis_title="Layer",
        yaxis=dict(autorange="reversed"),  # Layer 0 at top
        width=800,
        height=600,
        font=dict(size=12)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved heatmap to {save_path}")
    
    return fig


def create_layer_comparison(
    results: Dict[str, List],
    title: str = "Layer-wise Patching Effects",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create bar chart comparing attention vs MLP effects per layer.
    
    Args:
        results: Dict from ActivationPatcher.run_patching_experiment
        title: Plot title
        save_path: Optional path to save
    
    Returns:
        Plotly Figure
    """
    # Extract layer-level effects
    attn_effects = {}
    mlp_effects = {}
    
    for hook_point, result_list in results.items():
        if "hook_attn_out" in hook_point:
            layer = int(hook_point.split(".")[1])
            effects = [r.patching_effect for r in result_list]
            attn_effects[layer] = sum(effects) / len(effects)
        elif "hook_mlp_out" in hook_point:
            layer = int(hook_point.split(".")[1])
            effects = [r.patching_effect for r in result_list]
            mlp_effects[layer] = sum(effects) / len(effects)
    
    layers = sorted(attn_effects.keys())
    attn_vals = [attn_effects[l] for l in layers]
    mlp_vals = [mlp_effects[l] for l in layers]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Attention",
        x=[f"L{l}" for l in layers],
        y=attn_vals,
        marker_color="steelblue"
    ))
    
    fig.add_trace(go.Bar(
        name="MLP",
        x=[f"L{l}" for l in layers],
        y=mlp_vals,
        marker_color="coral"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Layer",
        yaxis_title="Average Patching Effect",
        barmode="group",
        width=900,
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved comparison to {save_path}")
    
    return fig


def create_top_heads_bar(
    results: List[HeadPatchingResult],
    top_k: int = 15,
    title: str = "Top Attention Heads by Patching Effect",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create horizontal bar chart of top heads.
    """
    sorted_results = sorted(results, key=lambda x: abs(x.patching_effect), reverse=True)
    top_results = sorted_results[:top_k]
    
    labels = [f"L{r.layer}H{r.head}" for r in top_results]
    effects = [r.patching_effect for r in top_results]
    colors = ["steelblue" if e >= 0 else "coral" for e in effects]
    
    fig = go.Figure(go.Bar(
        x=effects,
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="Head: %{y}<br>Effect: %{x:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Patching Effect",
        yaxis_title="Attention Head",
        yaxis=dict(autorange="reversed"),
        width=700,
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved bar chart to {save_path}")
    
    return fig


def create_metric_scatter(
    results: List[HeadPatchingResult],
    title: str = "Clean vs Patched Metrics",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Scatter plot showing recovery from corrupted to clean metrics.
    
    Each point is a head. X-axis is corrupted metric, Y-axis is patched metric.
    Points above the diagonal = patching improved things.
    """
    fig = go.Figure()
    
    # Get range for reference lines
    all_metrics = []
    for r in results:
        all_metrics.extend([r.clean_metric, r.corrupted_metric, r.patched_metric])
    min_val, max_val = min(all_metrics), max(all_metrics)
    
    # Reference line (no change)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="No change",
        showlegend=True
    ))
    
    # Each head as a point
    colors = [r.patching_effect for r in results]
    
    fig.add_trace(go.Scatter(
        x=[r.corrupted_metric for r in results],
        y=[r.patched_metric for r in results],
        mode="markers",
        marker=dict(
            size=8,
            color=colors,
            colorscale="RdBu",
            colorbar=dict(title="Effect"),
            line=dict(width=1, color="black")
        ),
        text=[f"L{r.layer}H{r.head}" for r in results],
        hovertemplate="Head: %{text}<br>Corrupted: %{x:.2f}<br>Patched: %{y:.2f}<extra></extra>",
        name="Heads"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Corrupted Metric (baseline broken)",
        yaxis_title="Patched Metric",
        width=700,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved scatter to {save_path}")
    
    return fig


def create_full_report(
    head_results: List[HeadPatchingResult],
    layer_results: Dict[str, List],
    n_layers: int,
    n_heads: int,
    save_dir: str = "."
) -> None:
    """
    Generate all visualizations and save to directory.
    
    Creates:
    - head_heatmap.html
    - layer_comparison.html  
    - top_heads.html
    - metric_scatter.html
    """
    print(f"\nGenerating visualizations in {save_dir}/...")
    
    create_head_heatmap(
        head_results, n_layers, n_heads,
        title="Activation Patching: Head Effects",
        save_path=f"{save_dir}/head_heatmap.html"
    )
    
    create_layer_comparison(
        layer_results,
        title="Attention vs MLP by Layer",
        save_path=f"{save_dir}/layer_comparison.html"
    )
    
    create_top_heads_bar(
        head_results,
        title="Most Important Heads for IOI",
        save_path=f"{save_dir}/top_heads.html"
    )
    
    create_metric_scatter(
        head_results,
        title="Patching Recovery by Head",
        save_path=f"{save_dir}/metric_scatter.html"
    )
    
    print(f"\nAll visualizations saved to {save_dir}/")


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("Import and use with patching results.")
