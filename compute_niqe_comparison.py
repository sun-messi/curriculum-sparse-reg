"""
Compute MANIQA scores for all checkpoints across different training methods.

MANIQA (Multi-dimension Attention Network for Image Quality Assessment) is a
no-reference deep learning metric. Higher scores = better quality.

This script evaluates:
    - Baseline
    - Curriculum (C)
    - Curriculum + Sparsity (CS)

And generates a comparison table similar to uvit_celeba64_fid_comparison.txt
"""

import os
import torch
import torch.multiprocessing as mp
from datetime import datetime
from ddpm_torch.metrics import compute_niqe_from_folder

# Sample directories (from uvit_celeba64_fid_comparison.txt)
SAMPLE_DIRS = {
    "Baseline": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small/20251226_025347",
    "Curriculum (C)": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_c/20251226_042556",
    "Curriculum+Sparsity (CS)": "/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs/20251226_035545",
}

# Checkpoints to evaluate (from 10k to 190k, step 10k)
CHECKPOINTS = [f"{i*10000}_ema" for i in range(1, 20)]  # 10000_ema to 190000_ema

def compute_all_niqe(num_gpus=None):
    """Compute MANIQA for all checkpoints sequentially on single GPU."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Import here to load model once
    from ddpm_torch.metrics import compute_niqe_from_folder

    results = {method: {} for method in SAMPLE_DIRS.keys()}

    total_tasks = len(SAMPLE_DIRS) * len(CHECKPOINTS)
    task_num = 0

    for method_name, base_dir in SAMPLE_DIRS.items():
        for checkpoint in CHECKPOINTS:
            task_num += 1
            checkpoint_dir = os.path.join(base_dir, checkpoint)

            print(f"[{task_num}/{total_tasks}] Processing: {method_name} - {checkpoint}")

            if not os.path.exists(checkpoint_dir):
                print(f"  Warning: {checkpoint_dir} not found, skipping...")
                results[method_name][checkpoint] = None
                continue

            try:
                score = compute_niqe_from_folder(checkpoint_dir, verbose=True, device=device, batch_size=16)
                print(f"  ✓ {method_name} {checkpoint}: {score:.4f}")
                results[method_name][checkpoint] = score
            except Exception as e:
                import traceback
                print(f"  Error computing MANIQA for {checkpoint}: {e}")
                traceback.print_exc()
                results[method_name][checkpoint] = None

    return results


def format_output(results):
    """Format results into a comparison table."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("U-ViT CelebA-64 MANIQA Comparison (Three Training Methods)")
    lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Dataset: celeba")
    lines.append("Ref size: ~5000 generated images per checkpoint")
    lines.append("Evaluation Metric: MANIQA (Multi-dimension Attention NR-IQA, higher is better)")
    lines.append("=" * 80)
    lines.append("")

    # Table header
    method_names = list(SAMPLE_DIRS.keys())
    header = f"{'Checkpoint':<16}|"
    for name in method_names:
        # Adjust column width based on method name length
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

    # Find best MANIQA for each method
    lines.append("BEST MANIQA (higher is better):")
    for method_name in method_names:
        valid_scores = [(ckpt, score) for ckpt, score in results[method_name].items() if score is not None]
        if valid_scores:
            best_ckpt, best_score = max(valid_scores, key=lambda x: x[1])  # max for higher is better
            iterations = int(best_ckpt.replace("_ema", "")) // 1000
            lines.append(f"  {method_name:<30} {best_score:.4f} @ {iterations}k iterations")
        else:
            lines.append(f"  {method_name:<30} N/A")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Source File Locations:")
    lines.append("=" * 80)
    lines.append("")
    for method_name, base_dir in SAMPLE_DIRS.items():
        lines.append(f"{method_name}:")
        lines.append(f"  Directory: {base_dir}")
        # Get directory timestamp if available
        if os.path.exists(base_dir):
            timestamp = datetime.fromtimestamp(os.path.getmtime(base_dir))
            lines.append(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Compute MANIQA scores for all checkpoints')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use (default: all available)')
    args = parser.parse_args()

    print("=" * 80)
    print("MANIQA Comparison Script")
    print("=" * 80)
    print("\nThis will compute MANIQA scores for all checkpoints.")
    print("MANIQA is a no-reference metric sensitive to perceptual quality.\n")

    # Compute MANIQA scores
    results = compute_all_niqe(num_gpus=args.num_gpus)

    # Format output
    output_text = format_output(results)

    # Print to console
    print("\n\n")
    print(output_text)

    # Save to file
    output_file = "uvit_celeba64_maniqa_comparison.txt"
    with open(output_file, "w") as f:
        f.write(output_text)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
