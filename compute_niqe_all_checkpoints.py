#!/usr/bin/env python
"""
Compute NIQE scores for all checkpoints across different training methods.

This script evaluates:
    - Baseline
    - Curriculum (C)
    - Curriculum + Sparsity (CS)

And generates a comparison table similar to the FID comparison.
"""

import os
import sys
from datetime import datetime
from compute_niqe import compute_niqe_folder

# Sample directories (from uvit_celeba64_fid_comparison.txt)
SAMPLE_DIRS = {
    "Baseline": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small/20251226_025347",
    "Curriculum (C)": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_c/20251226_042556",
    "Curriculum+Sparsity (CS)": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs/20251226_035545",
}

# Checkpoints to evaluate (from 10k to 190k, step 10k)
CHECKPOINTS = [f"{i*10000}_ema" for i in range(1, 20)]  # 10000_ema to 190000_ema


def compute_all_niqe():
    """Compute NIQE for all checkpoints."""
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
                score = compute_niqe_folder(checkpoint_dir, patch_size=32)
                print(f"  ✓ {method_name} {checkpoint}: {score:.4f}\n")
                results[method_name][checkpoint] = score
            except Exception as e:
                import traceback
                print(f"  ✗ Error computing NIQE for {checkpoint}: {e}")
                traceback.print_exc()
                results[method_name][checkpoint] = None

    return results


def format_output(results):
    """Format results into a comparison table."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("U-ViT CelebA-64 NIQE Comparison (All Checkpoints)")
    lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Dataset: celeba")
    lines.append("Images per checkpoint: ~5000 generated images")
    lines.append("Evaluation Metric: NIQE (Natural Image Quality Evaluator, lower is better)")
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

    # Find best NIQE for each method
    lines.append("BEST NIQE (lower is better):")
    for method_name in method_names:
        valid_scores = [(ckpt, score) for ckpt, score in results[method_name].items() if score is not None]
        if valid_scores:
            best_ckpt, best_score = min(valid_scores, key=lambda x: x[1])  # min for lower is better
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
        improvement = (baseline_avg - curriculum_avg) / baseline_avg * 100
        lines.append(f"  • Curriculum (C) average improvement over Baseline: {improvement:.1f}%")

    if baseline_scores and cs_scores:
        cs_avg = sum(cs_scores) / len(cs_scores)
        improvement = (baseline_avg - cs_avg) / baseline_avg * 100
        lines.append(f"  • Curriculum+Sparsity (CS) average improvement over Baseline: {improvement:.1f}%")

    lines.append("")
    lines.append("Note: NIQE (Natural Image Quality Evaluator) measures how well generated")
    lines.append("      images conform to natural scene statistics. Lower scores indicate")
    lines.append("      images that are more similar to natural images, with fewer artifacts.")
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
    print("NIQE Comparison Script - All Checkpoints")
    print("=" * 80)
    print("\nThis will compute NIQE scores for all checkpoints (10k-190k).")
    print("NIQE is a no-reference metric sensitive to artifacts and naturalness.\n")

    # Compute NIQE scores
    results = compute_all_niqe()

    # Format output
    output_text = format_output(results)

    # Print to console
    print("\n\n")
    print(output_text)

    # Save to file
    output_file = "uvit_celeba64_niqe_comparison.txt"
    with open(output_file, "w") as f:
        f.write(output_text)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
