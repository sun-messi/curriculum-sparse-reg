#!/usr/bin/env python3
"""
Feature Activation Map Visualization

Compare intermediate layer activations between Baseline and Curriculum DDPM models.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ddpm_torch import GaussianDiffusion, get_beta_schedule, DATASET_INFO
from ddpm_torch.models import UNet
from ddpm_torch.datasets import get_dataloader
from ddpm_torch.utils.feature_extractor import FeatureExtractor, get_target_layers


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize and compare feature activations between models'
    )

    # Checkpoint paths
    parser.add_argument('--baseline_ckpt', type=str, required=True,
                        help='Path to baseline checkpoint')
    parser.add_argument('--curriculum_ckpt', type=str, required=True,
                        help='Path to curriculum checkpoint')

    # Config paths
    parser.add_argument('--config_baseline', type=str, required=True,
                        help='Path to baseline config JSON')
    parser.add_argument('--config_curriculum', type=str, required=True,
                        help='Path to curriculum config JSON')

    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize (default: 5)')
    parser.add_argument('--timesteps', type=int, nargs='+', default=[100, 500, 900],
                        help='Timesteps for noisy visualization (default: 100 500 900)')
    parser.add_argument('--num_channels', type=int, default=64,
                        help='Number of top channels to visualize (default: 64)')
    parser.add_argument('--grid_size', type=int, default=8,
                        help='Grid size for channel display (default: 8)')

    # Output options
    parser.add_argument('--output_dir', type=str, default='feature_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights (default: True)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')

    # Input type options
    parser.add_argument('--visualize_clean', action='store_true', default=True,
                        help='Visualize features from clean images')
    parser.add_argument('--visualize_noisy', action='store_true', default=True,
                        help='Visualize features from noisy images')

    return parser.parse_args()


def load_model_and_diffusion(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True
) -> Tuple[nn.Module, GaussianDiffusion]:
    """
    Load model and diffusion process from config and checkpoint.

    Args:
        config_path: Path to JSON config file
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        use_ema: Whether to use EMA weights

    Returns:
        (model, diffusion) tuple
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load config (strip comments for JSON5-style configs)
    with open(config_path, 'r') as f:
        content = f.read()
        # Remove single-line comments
        lines = []
        for line in content.split('\n'):
            # Remove // comments but preserve JSON
            if '//' in line:
                # Find first occurrence outside of strings
                comment_idx = line.find('//')
                line = line[:comment_idx]
            lines.append(line)
        config = json.loads('\n'.join(lines))

    # Extract dataset info
    dataset = config.get('dataset', 'celeba')
    in_channels = DATASET_INFO[dataset]['channels']

    # Create diffusion process
    diffusion_kwargs = config['diffusion'].copy()
    beta_schedule = diffusion_kwargs.pop('beta_schedule')
    beta_start = diffusion_kwargs.pop('beta_start')
    beta_end = diffusion_kwargs.pop('beta_end')
    num_diffusion_timesteps = diffusion_kwargs.pop('timesteps')
    betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)
    diffusion = GaussianDiffusion(betas, **diffusion_kwargs)

    # Create model
    model = UNet(out_channels=in_channels, **config['model'])
    model.to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Extract EMA weights if available
    if use_ema:
        try:
            state_dict = state_dict['ema']['shadow']
            print("  Using EMA weights")
        except (KeyError, TypeError):
            print("  EMA weights not found, using model weights")
            try:
                state_dict = state_dict['model']
            except (KeyError, TypeError):
                print("  Using state dict directly")
    else:
        try:
            state_dict = state_dict['model']
            print("  Using model weights")
        except (KeyError, TypeError):
            print("  Using state dict directly")

    # Strip DDP wrapper if present
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k.split('.', maxsplit=1)[1]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    # Disable gradients
    for p in model.parameters():
        p.requires_grad_(False)

    print("  Model loaded successfully")
    return model, diffusion


def load_sample_images(
    dataset: str,
    num_samples: int,
    device: torch.device
) -> torch.Tensor:
    """
    Load sample images from dataset.

    Args:
        dataset: Dataset name (e.g., 'celeba')
        num_samples: Number of images to load
        device: Device to load on

    Returns:
        Tensor of shape [num_samples, C, H, W] in [-1, 1] range
    """
    print(f"Loading {num_samples} sample images from {dataset} dataset...")

    dataloader, _ = get_dataloader(
        dataset=dataset,
        batch_size=num_samples,
        split='test',
        num_workers=4,
        shuffle=False,
        raw=False,
    )

    # Get first batch
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]

    images = batch[:num_samples].to(device)
    print(f"  Loaded images with shape: {images.shape}")
    return images


def add_noise_to_images(
    x_0: torch.Tensor,
    t: int,
    diffusion: GaussianDiffusion,
    device: torch.device
) -> torch.Tensor:
    """
    Add noise to clean images at timestep t.

    Args:
        x_0: Clean images [B, C, H, W]
        t: Timestep value
        diffusion: GaussianDiffusion instance
        device: Device

    Returns:
        Noisy images x_t
    """
    B = x_0.shape[0]
    t_tensor = torch.full((B,), t, dtype=torch.int64, device=device)

    # Use q_sample to add noise
    x_t = diffusion.q_sample(x_0, t_tensor)

    return x_t


def extract_all_features(
    model: nn.Module,
    x_clean: torch.Tensor,
    x_noisy_dict: Dict[int, torch.Tensor],
    device: torch.device,
    visualize_clean: bool = True,
    visualize_noisy: bool = True
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract features for all input conditions.

    Args:
        model: UNet model
        x_clean: Clean images [B, C, H, W]
        x_noisy_dict: Dictionary mapping timesteps to noisy images
        device: Device
        visualize_clean: Whether to extract features from clean images
        visualize_noisy: Whether to extract features from noisy images

    Returns:
        Nested dict: {condition_name: {layer_name: activation_tensor}}
    """
    all_features = {}
    target_layers = get_target_layers()

    # Extract for clean images (t=0)
    if visualize_clean:
        print("  Extracting features for clean images (t=0)...")
        with FeatureExtractor(model) as extractor:
            for layer_name, layer_path in target_layers:
                extractor.register_layer(layer_name, layer_path)

            B = x_clean.shape[0]
            t_zero = torch.zeros(B, dtype=torch.int64, device=device)
            all_features['clean_t0'] = extractor.extract_features(x_clean, t_zero)

    # Extract for noisy images at each timestep
    if visualize_noisy:
        for timestep, x_noisy in x_noisy_dict.items():
            print(f"  Extracting features for noisy images (t={timestep})...")
            with FeatureExtractor(model) as extractor:
                for layer_name, layer_path in target_layers:
                    extractor.register_layer(layer_name, layer_path)

                B = x_noisy.shape[0]
                t_tensor = torch.full((B,), timestep, dtype=torch.int64, device=device)
                all_features[f'noisy_t{timestep}'] = extractor.extract_features(x_noisy, t_tensor)

    return all_features


def select_top_k_channels(
    activations: torch.Tensor,
    k: int
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Select top-k channels by variance across spatial dimensions.

    Args:
        activations: Feature map [B, C, H, W]
        k: Number of channels to select

    Returns:
        (selected_activations [B, k, H, W], channel_indices [k])
    """
    # Compute variance across spatial dimensions for each channel
    variance = activations.var(dim=[2, 3]).mean(dim=0)  # Average across batch

    # Get top-k channel indices
    k = min(k, activations.shape[1])
    _, top_indices = torch.topk(variance, k=k)
    top_indices = top_indices.cpu().numpy()

    # Select channels
    selected = activations[:, top_indices]

    return selected, top_indices


def normalize_activation_map(activation: torch.Tensor) -> np.ndarray:
    """
    Normalize single activation map to [0, 1] range for visualization.

    Args:
        activation: Single channel activation [H, W]

    Returns:
        Normalized array
    """
    arr = activation.numpy()

    # Robust normalization using percentiles to handle outliers
    p2, p98 = np.percentile(arr, [2, 98])
    if p98 > p2:
        arr_norm = np.clip((arr - p2) / (p98 - p2), 0, 1)
    else:
        # Handle edge case where all values are the same
        arr_norm = np.zeros_like(arr)

    return arr_norm


def compute_activation_stats(features: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive statistics for feature maps.

    Args:
        features: Feature tensor [B, C, H, W]

    Returns:
        Dictionary of statistics
    """
    features_flat = features.flatten()

    return {
        'mean': features_flat.mean().item(),
        'std': features_flat.std().item(),
        'min': features_flat.min().item(),
        'max': features_flat.max().item(),
        'median': features_flat.median().item(),
        'sparsity': (features_flat.abs() < 1e-3).float().mean().item(),
    }


def visualize_layer_comparison(
    baseline_features: torch.Tensor,
    curriculum_features: torch.Tensor,
    layer_name: str,
    condition_name: str,
    sample_idx: int,
    output_path: str,
    top_k: int = 64,
    grid_size: int = 8
):
    """
    Create side-by-side visualization of baseline vs curriculum for one layer.

    Args:
        baseline_features: Baseline activations [1, C, H, W]
        curriculum_features: Curriculum activations [1, C, H, W]
        layer_name: Name of the layer
        condition_name: Input condition
        sample_idx: Which sample
        output_path: Where to save figure
        top_k: Number of channels to visualize
        grid_size: Grid dimension (8x8 = 64)
    """
    # Select top-k channels for both
    baseline_selected, baseline_indices = select_top_k_channels(baseline_features, top_k)
    curriculum_selected, curriculum_indices = select_top_k_channels(curriculum_features, top_k)

    # Extract single sample (first in batch)
    baseline_maps = baseline_selected[0]  # [k, H, W]
    curriculum_maps = curriculum_selected[0]  # [k, H, W]

    # Compute statistics
    baseline_stats = compute_activation_stats(baseline_maps)
    curriculum_stats = compute_activation_stats(curriculum_maps)

    # Create figure with side-by-side grids
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(
        f'{layer_name} | {condition_name} | Sample {sample_idx}',
        fontsize=16, fontweight='bold', y=0.98
    )

    # Plot baseline
    ax_baseline = axes[0]
    ax_baseline.set_title(
        f'Baseline\nμ={baseline_stats["mean"]:.3f} σ={baseline_stats["std"]:.3f} '
        f'sparsity={baseline_stats["sparsity"]:.2%}',
        fontsize=14, fontweight='bold'
    )
    ax_baseline.axis('off')

    # Plot curriculum
    ax_curriculum = axes[1]
    ax_curriculum.set_title(
        f'Curriculum\nμ={curriculum_stats["mean"]:.3f} σ={curriculum_stats["std"]:.3f} '
        f'sparsity={curriculum_stats["sparsity"]:.2%}',
        fontsize=14, fontweight='bold'
    )
    ax_curriculum.axis('off')

    # Create 8x8 grids
    for model_name, maps, ax in [
        ('Baseline', baseline_maps, ax_baseline),
        ('Curriculum', curriculum_maps, ax_curriculum)
    ]:
        # Create grid
        num_maps = min(top_k, maps.shape[0])
        h, w = maps.shape[1], maps.shape[2]

        # Add padding between cells
        cell_padding = 2
        grid_h = grid_size * (h + cell_padding) - cell_padding
        grid_w = grid_size * (w + cell_padding) - cell_padding
        grid = np.ones((grid_h, grid_w)) * 0.5  # Gray background

        for idx in range(num_maps):
            if idx >= grid_size * grid_size:
                break

            row = idx // grid_size
            col = idx % grid_size

            # Normalize activation
            activation = normalize_activation_map(maps[idx].cpu())

            # Place in grid
            y_start = row * (h + cell_padding)
            x_start = col * (w + cell_padding)
            grid[y_start:y_start+h, x_start:x_start+w] = activation

        # Display grid with colormap
        im = ax.imshow(grid, cmap='viridis', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength', fontsize=12)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_input_image(
    image: torch.Tensor,
    output_path: str,
    title: str
):
    """
    Save input image for reference.

    Args:
        image: Image tensor [C, H, W] in [-1, 1] range
        output_path: Where to save
        title: Title for the image
    """
    # Convert from [-1, 1] to [0, 1]
    image_np = (image.cpu().numpy() + 1) / 2
    image_np = np.clip(image_np, 0, 1)

    # Transpose to [H, W, C]
    image_np = np.transpose(image_np, (1, 2, 0))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_np)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Feature Activation Map Visualization")
    print("=" * 80)

    # Load models
    print("\n[1/5] Loading models...")
    baseline_model, baseline_diffusion = load_model_and_diffusion(
        args.config_baseline, args.baseline_ckpt, device, args.use_ema
    )
    curriculum_model, curriculum_diffusion = load_model_and_diffusion(
        args.config_curriculum, args.curriculum_ckpt, device, args.use_ema
    )

    # Determine dataset from config
    with open(args.config_baseline, 'r') as f:
        config = json.load(f)
    dataset = config.get('dataset', 'celeba')

    # Load sample images
    print("\n[2/5] Loading sample images...")
    x_clean = load_sample_images(dataset, args.num_samples, device)

    # Generate noisy versions at different timesteps
    print("\n[3/5] Generating noisy images...")
    x_noisy_dict = {}
    for t in args.timesteps:
        x_noisy_dict[t] = add_noise_to_images(x_clean, t, baseline_diffusion, device)
        print(f"  Generated noisy images at t={t}")

    # Extract features for both models
    print("\n[4/5] Extracting features...")
    print("Baseline model:")
    baseline_all_features = extract_all_features(
        baseline_model, x_clean, x_noisy_dict, device,
        args.visualize_clean, args.visualize_noisy
    )
    print("Curriculum model:")
    curriculum_all_features = extract_all_features(
        curriculum_model, x_clean, x_noisy_dict, device,
        args.visualize_clean, args.visualize_noisy
    )

    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    target_layers = [name for name, _ in get_target_layers()]

    # Determine conditions to visualize
    conditions = []
    if args.visualize_clean:
        conditions.append('clean_t0')
    if args.visualize_noisy:
        conditions.extend([f'noisy_t{t}' for t in args.timesteps])

    total_plots = len(target_layers) * len(conditions) * args.num_samples
    plot_count = 0

    # Process each sample
    for sample_idx in range(args.num_samples):
        sample_dir = output_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(exist_ok=True)

        # Save input images
        save_input_image(
            x_clean[sample_idx],
            str(sample_dir / "input_clean.png"),
            f"Clean Image (Sample {sample_idx})"
        )

        for t, x_noisy in x_noisy_dict.items():
            save_input_image(
                x_noisy[sample_idx],
                str(sample_dir / f"input_noisy_t{t}.png"),
                f"Noisy Image t={t} (Sample {sample_idx})"
            )

        # Generate feature visualizations for each condition
        for condition in conditions:
            # Create condition subdirectory
            condition_dir = sample_dir / condition
            condition_dir.mkdir(exist_ok=True)

            for layer_name in target_layers:
                # Get features for this sample
                baseline_feat = baseline_all_features[condition][layer_name][sample_idx:sample_idx+1]
                curriculum_feat = curriculum_all_features[condition][layer_name][sample_idx:sample_idx+1]

                # Create filename
                filename = f"{layer_name}_baseline_vs_curriculum.png"
                output_path = condition_dir / filename

                # Visualize
                visualize_layer_comparison(
                    baseline_feat, curriculum_feat,
                    layer_name, condition, sample_idx,
                    str(output_path),
                    args.num_channels, args.grid_size
                )

                plot_count += 1
                if plot_count % 10 == 0:
                    print(f"  Progress: {plot_count}/{total_plots}")

    print(f"\n{plot_count} visualizations saved to {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
