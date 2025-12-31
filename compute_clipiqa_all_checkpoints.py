#!/usr/bin/env python
"""
Compute CLIPIQA scores for all checkpoints across different training methods.

CLIPIQA (CLIP Image Quality Assessment) uses CLIP embeddings to assess
image quality. Higher scores = better quality, range [0, 1].

This script evaluates:
    - Baseline
    - Curriculum (C)
    - Curriculum + Sparsity (CS)
"""

import os
import sys
import torch
import pyiqa
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Sample directories
SAMPLE_DIRS = {
    "Baseline": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small/20251226_025347",
    "Curriculum (C)": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_c/20251226_042556",
    "Curriculum+Sparsity (CS)": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs/20251226_035545",
}

# Checkpoints to evaluate (from 10k to 190k, step 10k)
CHECKPOINTS = [f"{i*10000}_ema" for i in range(1, 20)]


def compute_clipiqa_folder(folder_path, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=32):
    """
    Compute average CLIPIQA score for all images in a folder.

    Args:
        folder_path: Directory containing images
        device: Device for computation
        batch_size: Number of images to process in batch

    Returns:
        Average CLIPIQA score (higher is better, range [0, 1])
    """
    # Initialize CLIPIQA metric
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)

    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")

    print(f"  Found {len(image_files)} images")

    # Process images in batches
    scores = []
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    pbar = tqdm(total=len(image_files), desc="  Computing CLIPIQA", leave=False)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]

        # Load batch of images
        batch_tensors = []
        for img_file in batch_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    batch_tensors.append(img_tensor)
            except Exception as e:
                continue

        if len(batch_tensors) > 0:
            # Stack into batch and compute
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                batch_scores = clipiqa_metric(batch)

            # Handle both single value and batch output
            if batch_scores.dim() == 0:
                scores.append(float(batch_scores.item()))
            else:
                scores.extend([float(s.item()) for s in batch_scores])

        pbar.update(len(batch_files))

    pbar.close()

    if len(scores) == 0:
        raise ValueError(f"Failed to compute CLIPIQA for any images in {folder_path}")

    # Calculate average
    avg_score = float(np.mean(scores))

    return avg_score


def compute_all_clipiqa():
    """Compute CLIPIQA for all checkpoints."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    results = {method: {} for method in SAMPLE_DIRS.keys()}

    total_tasks = len(SAMPLE_DIRS) * len(CHECKPOINTS)
    task_num = 0

    for method_name, base_dir in SAMPLE_DIRS.items():
        print(f"\n{'='*80}")
        print(f"Processing: {method_name}")
        print(f"{'='*80}\n")

        for checkpoint in CHECKPOINTS:
            task_num += 1
            checkpoint_dir = os.path.join(base_dir, checkpoint)

            print(f"[{task_num}/{total_tasks}] {method_name} - {checkpoint}")

            if not os.path.exists(checkpoint_dir):
                print(f"  Warning: {checkpoint_dir} not found, skipping...")
                results[method_name][checkpoint] = None
                continue

            try:
                score = compute_clipiqa_folder(checkpoint_dir, device=device, batch_size=16)
                print(f"  ✓ {method_name} {checkpoint}: {score:.4f}\n")
                results[method_name][checkpoint] = score
            except Exception as e:
                import traceback
                print(f"  ✗ Error computing CLIPIQA for {checkpoint}: {e}")
                traceback.print_exc()
                results[method_name][checkpoint] = None

    return results


def format_output(results):
    """Format results into a comparison table."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("U-ViT CelebA-64 CLIPIQA Comparison (All Checkpoints)")
    lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Dataset: celeba")
    lines.append("Images per checkpoint: ~5000 generated images")
    lines.append("Evaluation Metric: CLIPIQA (CLIP Image Quality Assessment, higher is better)")
    lines.append("Score Range: [0, 1]")
    lines.append("=" * 80)
    lines.append("")

    # Table header
    method_names = list(SAMPLE_DIRS.keys())
    header = f"{'Checkpoint':<16}|"
    for name in method_names:
        col_width = max(len(name) + 2, 16)
        header += f"{name:>{col_width}} |"
    lines.append(header)
    lines.append("-" * 80)

    # Table rows
    for checkpoint in CHECKPOINTS:
        row = f"{checkpoint:<16}|"
        for method_name in method_names:
            score = results[method_name].get(checkpoint)
            if score is not None:
                row += f"{score:>16.4f} |"
            else:
                row += f"{'N/A':>16} |"
        lines.append(row)

    lines.append("")

    # Find best CLIPIQA for each method
    lines.append("BEST CLIPIQA (higher is better):")
    for method_name in method_names:
        valid_scores = [(ckpt, score) for ckpt, score in results[method_name].items() if score is not None]
        if valid_scores:
            best_ckpt, best_score = max(valid_scores, key=lambda x: x[1])  # max for higher is better
            iterations = int(best_ckpt.replace("_ema", "")) // 1000
            lines.append(f"  {method_name:<30} {best_score:.4f} @ {iterations}k iterations")
        else:
            lines.append(f"  {method_name:<30} N/A")

    lines.append("")

    # Analysis
    lines.append("Results Analysis:")
    baseline_scores = [s for s in results["Baseline"].values() if s is not None]
    curriculum_scores = [s for s in results["Curriculum (C)"].values() if s is not None]
    cs_scores = [s for s in results["Curriculum+Sparsity (CS)"].values() if s is not None]

    if baseline_scores and curriculum_scores:
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        curriculum_avg = sum(curriculum_scores) / len(curriculum_scores)
        improvement = (curriculum_avg - baseline_avg) / baseline_avg * 100
        lines.append(f"  • Curriculum (C) average improvement over Baseline: {improvement:+.1f}%")

    if baseline_scores and cs_scores:
        cs_avg = sum(cs_scores) / len(cs_scores)
        improvement = (cs_avg - baseline_avg) / baseline_avg * 100
        lines.append(f"  • Curriculum+Sparsity (CS) average improvement over Baseline: {improvement:+.1f}%")

    lines.append("")
    lines.append("Note: CLIPIQA uses CLIP (Contrastive Language-Image Pre-training) embeddings")
    lines.append("      to assess image quality by calculating similarity with quality-related")
    lines.append("      text prompts. Score range is [0, 1], higher is better.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("Source File Locations:")
    lines.append("=" * 80)
    lines.append("")
    for method_name, base_dir in SAMPLE_DIRS.items():
        lines.append(f"{method_name}:")
        lines.append(f"  Directory: {base_dir}")
        if os.path.exists(base_dir):
            timestamp = datetime.fromtimestamp(os.path.getmtime(base_dir))
            lines.append(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    """Main function."""
    print("=" * 80)
    print("CLIPIQA Comparison Script - All Checkpoints")
    print("=" * 80)
    print("\nThis will compute CLIPIQA scores for all checkpoints (10k-190k).")
    print("CLIPIQA uses CLIP embeddings to assess image quality.\n")

    # Compute CLIPIQA scores
    results = compute_all_clipiqa()

    # Format output
    output_text = format_output(results)

    # Print to console
    print("\n\n")
    print(output_text)

    # Save to file
    output_file = "uvit_celeba64_clipiqa_comparison.txt"
    with open(output_file, "w") as f:
        f.write(output_text)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
