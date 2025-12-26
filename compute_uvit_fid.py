"""
Compute FID scores for U-ViT generated samples.

Usage:

python compute_uvit_fid.py \
  --uvit-dir /home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_c/20251226_042556 \
  --dataset celeba \
  --device cuda:0 \
  --ref-size 5000
  
"""

import os
import sys
from argparse import ArgumentParser
from datetime import datetime

# Import FID computation functions
from compare_fid import compute_fid, compute_real_stats, compute_is, compute_lpips


def main():
    parser = ArgumentParser()
    parser.add_argument("--uvit-dir", required=True, type=str,
                        help="Path to U-ViT eval_samples directory")
    parser.add_argument("--dataset", default="celeba", type=str,
                        help="Dataset name for real image stats (celeba, cifar10, etc.)")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--ref-size", default=2000, type=int,
                        help="Number of real images for reference stats")
    parser.add_argument("--calc-is", action="store_true",
                        help="Also calculate Inception Score")
    parser.add_argument("--calc-lpips", action="store_true",
                        help="Also calculate LPIPS")
    parser.add_argument("--output", default=None, type=str,
                        help="Output file (default: results/uvit_fid_comparison.txt)")
    args = parser.parse_args()

    uvit_dir = args.uvit_dir
    if not os.path.exists(uvit_dir):
        print(f"Error: Directory not found: {uvit_dir}")
        sys.exit(1)

    # Detect directory structure: single checkpoint, model dir with checkpoints, or parent dir with models

    # First check: does this dir directly contain image files? (single checkpoint)
    direct_samples = [f for f in os.listdir(uvit_dir)
                     if os.path.isfile(os.path.join(uvit_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]

    if direct_samples:
        # Direct checkpoint directory - compute FID for just this one
        print("=" * 80)
        print(f"FID Computation for Single Checkpoint")
        print(f"Checkpoint: {os.path.basename(uvit_dir)}")
        print(f"Samples: {len(direct_samples)}")
        print("=" * 80)

        # Compute FID directly
        print(f"\n[Precomputing real stats for {args.dataset}]")
        real_stats = compute_real_stats(args.dataset, device=args.device, ref_size=args.ref_size)
        print("  Real stats computed successfully.\n")

        print(f"Computing FID...")
        fid = compute_fid(uvit_dir, args.dataset, device=args.device,
                        ref_size=args.ref_size, real_stats=real_stats)
        print(f"FID: {fid:.2f}")

        # Save result
        os.makedirs("results", exist_ok=True)
        output_file = args.output or "results/uvit_fid_single.txt"
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(output_file, "a") as f:
            f.write(f"\n{timestamp} | {os.path.basename(uvit_dir)} | FID: {fid:.2f}\n")
        print(f"\nResult appended to {output_file}")
        return

    # Second check: subdirs contain image files (model dir with checkpoints)
    subdirs = [d for d in os.listdir(uvit_dir) if os.path.isdir(os.path.join(uvit_dir, d))]
    has_samples = False
    if subdirs:
        first_subdir = os.path.join(uvit_dir, subdirs[0])
        sample_files = [f for f in os.listdir(first_subdir)
                       if f.endswith(('.png', '.jpg', '.jpeg'))]
        has_samples = len(sample_files) > 0

    if has_samples:
        # We're in a model directory with checkpoint subdirs (e.g., celeba64_uvit_small_cs/)
        model_name = os.path.basename(uvit_dir)
        model_dirs = {model_name: uvit_dir}
        print("=" * 80)
        print(f"FID Computation for U-ViT Generated Samples")
        print(f"Model: {model_name}")
        print(f"Checkpoints found: {subdirs}")
        print("=" * 80)
    else:
        # We're in a parent directory with model subdirs
        model_dirs = {d: os.path.join(uvit_dir, d) for d in subdirs}
        if not model_dirs:
            print(f"Error: No model directories found in {uvit_dir}")
            sys.exit(1)
        print("=" * 80)
        print(f"FID Computation for U-ViT Generated Samples")
        print(f"U-ViT directory: {uvit_dir}")
        print(f"Dataset: {args.dataset}")
        print(f"Models found: {list(model_dirs.keys())}")
        print("=" * 80)

    # Precompute real stats once
    print(f"\n[Precomputing real stats for {args.dataset}]")
    real_stats = compute_real_stats(args.dataset, device=args.device, ref_size=args.ref_size)
    print("  Real stats computed successfully.\n")

    # Results: {model: {checkpoint: {fid, is, lpips}}}
    results = {}

    for model_name in sorted(model_dirs.keys()):
        model_path = model_dirs[model_name]

        # Find all checkpoint directories (50k, 100k, etc.)
        checkpoint_dirs = [d for d in os.listdir(model_path)
                          if os.path.isdir(os.path.join(model_path, d))]

        if not checkpoint_dirs:
            print(f"[SKIP] {model_name}: no checkpoint directories found")
            continue

        print(f"\n[Model: {model_name}]")
        results[model_name] = {}

        def parse_checkpoint_name(name):
            """Extract numeric value from checkpoint name (e.g., '40000_ema' -> 40000, '50k' -> 50000)"""
            import re
            # Try to extract number from formats like '40000_ema', '50k', '100000_ema', etc.
            match = re.search(r'(\d+)([k]?)', name.lower())
            if match:
                num = int(match.group(1))
                if match.group(2) == 'k':
                    num *= 1000
                return num
            return 0

        for chkpt_name in sorted(checkpoint_dirs, key=parse_checkpoint_name):
            sample_dir = os.path.join(model_path, chkpt_name)

            # Count samples
            sample_files = [f for f in os.listdir(sample_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
            num_samples = len(sample_files)

            if num_samples == 0:
                print(f"  {chkpt_name}: no samples found, skipping")
                continue

            print(f"  {chkpt_name}: {num_samples} samples")

            # Compute FID
            print(f"    Computing FID...")
            fid = compute_fid(sample_dir, args.dataset, device=args.device,
                            ref_size=args.ref_size, real_stats=real_stats)

            results[model_name][chkpt_name] = {"fid": fid, "num_samples": num_samples}
            print(f"    FID: {fid:.2f}")

            # Compute IS if requested
            if args.calc_is:
                print(f"    Computing IS...")
                is_mean, is_std = compute_is(sample_dir, device=args.device)
                results[model_name][chkpt_name]["is_mean"] = is_mean
                results[model_name][chkpt_name]["is_std"] = is_std
                print(f"    IS: {is_mean:.2f} +/- {is_std:.2f}")

            # Compute LPIPS if requested
            if args.calc_lpips:
                print(f"    Computing LPIPS...")
                lpips = compute_lpips(sample_dir, args.dataset, device=args.device,
                                     ref_size=min(args.ref_size, num_samples))
                results[model_name][chkpt_name]["lpips"] = lpips
                print(f"    LPIPS: {lpips:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY - FID (lower is better)")
    print("=" * 80)

    # Get all unique checkpoint names across all models
    def parse_checkpoint_name(name):
        """Extract numeric value from checkpoint name (e.g., '40000_ema' -> 40000, '50k' -> 50000)"""
        import re
        match = re.search(r'(\d+)([k]?)', name.lower())
        if match:
            num = int(match.group(1))
            if match.group(2) == 'k':
                num *= 1000
            return num
        return 0

    all_checkpoints = set()
    for model_name in results:
        all_checkpoints.update(results[model_name].keys())
    all_checkpoints = sorted(all_checkpoints, key=parse_checkpoint_name)

    # Header
    header = f"{'Checkpoint':<15}"
    for model_name in sorted(results.keys()):
        header += f" | {model_name[:20]:>20}"
    print(header)
    print("-" * len(header))

    # FID rows
    for chkpt in all_checkpoints:
        row = f"{chkpt:<15}"
        for model_name in sorted(results.keys()):
            if chkpt in results[model_name]:
                fid = results[model_name][chkpt]["fid"]
                row += f" | {fid:>20.2f}"
            else:
                row += f" | {'N/A':>20}"
        print(row)

    # IS table if computed
    if args.calc_is:
        print("\n" + "=" * 80)
        print("SUMMARY - IS (higher is better)")
        print("=" * 80)
        print(header)
        print("-" * len(header))

        for chkpt in all_checkpoints:
            row = f"{chkpt:<15}"
            for model_name in sorted(results.keys()):
                if chkpt in results[model_name] and "is_mean" in results[model_name][chkpt]:
                    is_mean = results[model_name][chkpt]["is_mean"]
                    row += f" | {is_mean:>20.2f}"
                else:
                    row += f" | {'N/A':>20}"
            print(row)

    # LPIPS table if computed
    if args.calc_lpips:
        print("\n" + "=" * 80)
        print("SUMMARY - LPIPS (lower is better)")
        print("=" * 80)
        print(header)
        print("-" * len(header))

        for chkpt in all_checkpoints:
            row = f"{chkpt:<15}"
            for model_name in sorted(results.keys()):
                if chkpt in results[model_name] and "lpips" in results[model_name][chkpt]:
                    lpips = results[model_name][chkpt]["lpips"]
                    row += f" | {lpips:>20.4f}"
                else:
                    row += f" | {'N/A':>20}"
            print(row)

    print("=" * 80)

    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    output_file = args.output or f"{results_dir}/uvit_fid_comparison.txt"

    with open(output_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"U-ViT FID Comparison\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"U-ViT directory: {uvit_dir}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Ref size: {args.ref_size}\n")
        metrics_str = "FID"
        if args.calc_is:
            metrics_str += ", IS"
        if args.calc_lpips:
            metrics_str += ", LPIPS"
        f.write(f"Metrics: {metrics_str}\n")
        f.write("-" * 80 + "\n\n")

        # Write FID table
        f.write("FID (lower is better):\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for chkpt in all_checkpoints:
            row = f"{chkpt:<15}"
            for model_name in sorted(results.keys()):
                if chkpt in results[model_name]:
                    fid = results[model_name][chkpt]["fid"]
                    row += f" | {fid:>20.2f}"
                else:
                    row += f" | {'N/A':>20}"
            f.write(row + "\n")

        # Write IS table if computed
        if args.calc_is:
            f.write("\nIS (higher is better):\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for chkpt in all_checkpoints:
                row = f"{chkpt:<15}"
                for model_name in sorted(results.keys()):
                    if chkpt in results[model_name] and "is_mean" in results[model_name][chkpt]:
                        is_mean = results[model_name][chkpt]["is_mean"]
                        row += f" | {is_mean:>20.2f}"
                    else:
                        row += f" | {'N/A':>20}"
                f.write(row + "\n")

        # Write LPIPS table if computed
        if args.calc_lpips:
            f.write("\nLPIPS (lower is better):\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for chkpt in all_checkpoints:
                row = f"{chkpt:<15}"
                for model_name in sorted(results.keys()):
                    if chkpt in results[model_name] and "lpips" in results[model_name][chkpt]:
                        lpips = results[model_name][chkpt]["lpips"]
                        row += f" | {lpips:>20.4f}"
                    else:
                        row += f" | {'N/A':>20}"
                f.write(row + "\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults appended to {output_file}")

    # Save CSV
    csv_file = output_file.replace(".txt", ".csv")
    csv_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    with open(csv_file, "a") as f:
        if not csv_exists:
            # Header
            csv_header = ["checkpoint"]
            for model_name in sorted(results.keys()):
                csv_header.append(f"{model_name}_fid")
                if args.calc_is:
                    csv_header.append(f"{model_name}_is")
                if args.calc_lpips:
                    csv_header.append(f"{model_name}_lpips")
            f.write(",".join(csv_header) + "\n")

        # Data rows
        for chkpt in all_checkpoints:
            row = [chkpt]
            for model_name in sorted(results.keys()):
                if chkpt in results[model_name]:
                    row.append(f"{results[model_name][chkpt]['fid']:.2f}")
                    if args.calc_is:
                        row.append(f"{results[model_name][chkpt].get('is_mean', ''):.2f}"
                                  if "is_mean" in results[model_name][chkpt] else "")
                    if args.calc_lpips:
                        row.append(f"{results[model_name][chkpt].get('lpips', ''):.4f}"
                                  if "lpips" in results[model_name][chkpt] else "")
                else:
                    row.append("")
                    if args.calc_is:
                        row.append("")
                    if args.calc_lpips:
                        row.append("")
            f.write(",".join(row) + "\n")

    print(f"CSV appended to {csv_file}")

    # Auto-append to uvit_all_methods_comparison.txt
    comparison_file = f"{results_dir}/uvit_all_methods_comparison.txt"
    if os.path.exists(comparison_file):
        # Map model directories to comparison file column names
        model_name_map = {
            'celeba64_uvit_small': 'Baseline',
            'celeba64_uvit_small_c': 'Curriculum',
            'celeba64_uvit_small_cs': 'CS Mode',
            'celeba64_uvit_small_c_20251225_170001': 'Curriculum (new)'
        }

        for model_name in results.keys():
            # Match model directory name to comparison column
            comparison_col = None
            for dir_pattern, col_name in model_name_map.items():
                if dir_pattern in model_name:
                    comparison_col = col_name
                    break

            if comparison_col:
                print(f"\n[Auto-updating {comparison_file}]")
                print(f"  Model: {model_name} → {comparison_col}")

                # Update each checkpoint
                for chkpt_name, chkpt_data in results[model_name].items():
                    # Parse checkpoint number
                    import re
                    match = re.search(r'(\d+)', chkpt_name)
                    if match:
                        step = int(match.group(1))
                        if 'k' in chkpt_name.lower() and step < 1000:
                            step *= 1000

                        fid_val = chkpt_data['fid']

                        # Update comparison file
                        try:
                            from append_fid_to_comparison import update_comparison_file
                            update_comparison_file(comparison_file, comparison_col, step, fid_val)
                        except Exception as e:
                            print(f"  Warning: Could not auto-update comparison file: {e}")
                            print(f"  You can manually update using:")
                            print(f"    python append_fid_to_comparison.py --model-name '{comparison_col}' --checkpoint {step} --fid {fid_val:.2f}")


if __name__ == "__main__":
    main()
