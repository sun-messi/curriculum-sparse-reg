"""
Compare metrics (FID, IS, LPIPS) across different training stages (epochs).

Usage:
    # Compare all 4 models at epochs 1,5,10
    conda run -n csdiff python compare_fid_stages.py --dataset celeba64 --epochs 1,5,10 --num-gpus 6

    # Compare all epochs
    conda run -n csdiff python compare_fid_stages.py --dataset celeba64 --epochs all --num-gpus 6

    # Compare specific models
    conda run -n csdiff python compare_fid_stages.py --dataset celeba64 --models c,cs --epochs 1,5,10 --num-gpus 6

    # Disable specific metrics
    conda run -n csdiff python compare_fid_stages.py --dataset celeba64 --epochs 1,5,10 --no-is --no-lpips
"""

import os
import shutil
from argparse import ArgumentParser
from datetime import datetime

# Import from compare_fid.py
from compare_fid import (
    generate_samples,
    compute_fid,
    compute_real_stats,
    compute_is,
    compute_lpips,
    load_json_with_comments,
)


def find_checkpoint_by_epoch(chkpt_dir, prefix, target_epoch):
    """Find checkpoint file for a specific epoch.

    Supports both formats:
    - New: prefix_epoch_YYYYMMDD_HHMMSS.pt
    - Old: prefix_YYYYMMDD_HHMMSS_epoch.pt
    """
    if not os.path.exists(chkpt_dir):
        return None

    files = [f for f in os.listdir(chkpt_dir) if f.startswith(prefix) and f.endswith(".pt")]
    if not files:
        return None

    # Find files matching target epoch
    matching = []
    for f in files:
        name = f.replace(prefix + "_", "").replace(".pt", "")
        parts = name.split("_")

        try:
            if len(parts) >= 3:
                # Detect format by first part length
                if len(parts[0]) == 8:  # Date format YYYYMMDD
                    # New format: YYYYMMDD_HHMMSS_epoch
                    epoch = int(parts[2])
                    timestamp = parts[0] + parts[1]
                else:
                    # Old format: epoch_YYYYMMDD_HHMMSS
                    epoch = int(parts[0])
                    timestamp = parts[1] + parts[2]
            else:
                # Very old format: just epoch
                epoch = int(parts[0])
                timestamp = "0"

            if epoch == target_epoch:
                matching.append((f, timestamp))
        except (ValueError, IndexError):
            continue

    if not matching:
        return None

    # Return the latest one (by timestamp)
    matching.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(chkpt_dir, matching[0][0])


def eval_metrics(config_path, chkpt_path, total_size=1000, ref_size=1000,
                 num_gpus=1, device="cuda:0", subseq_size=50,
                 real_stats=None, calc_is=True, calc_lpips=True):
    """
    Evaluate all metrics (FID, IS, LPIPS) for a single checkpoint.

    Args:
        config_path: Path to config JSON file
        chkpt_path: Path to checkpoint file
        total_size: Number of samples to generate
        ref_size: Number of real images for reference stats (ignored if real_stats provided)
        num_gpus: Number of GPUs for generation
        device: Device for FID computation
        subseq_size: DDIM subsequence size
        real_stats: Precomputed (mean, var) tuple for real images. If provided, skips
                    real stats computation for faster evaluation.
        calc_is: Whether to compute Inception Score
        calc_lpips: Whether to compute LPIPS

    Returns:
        dict: {"fid": float, "is_mean": float, "is_std": float, "lpips": float}
    """
    # Extract model name from config path (e.g., "celeba64_c" from "configs/celeba64_c.json")
    model_name = os.path.basename(config_path).replace('.json', '')

    # Create organized directory structure
    checkpoint_name = os.path.basename(chkpt_path)[:-3]  # Remove .pt extension
    sample_dir = f"./generated/{model_name}/{checkpoint_name}"

    # Generate samples only if they don't exist
    os.makedirs(sample_dir, exist_ok=True)
    if len(os.listdir(sample_dir)) == 0:
        generate_samples(
            config_path=config_path,
            chkpt_path=chkpt_path,
            save_dir=sample_dir,
            total_size=total_size,
            batch_size=128,
            device=device,
            use_ddim=True,
            subseq_size=subseq_size,
            num_gpus=num_gpus,
        )
    else:
        print(f"    Reusing {len(os.listdir(sample_dir))} existing samples")

    # Get dataset name
    meta_config = load_json_with_comments(config_path)
    dataset = meta_config.get("dataset", "celeba")

    results = {}

    # Compute FID
    results["fid"] = compute_fid(sample_dir, dataset, device=device, ref_size=ref_size, real_stats=real_stats)

    # Compute IS
    if calc_is:
        is_mean, is_std = compute_is(sample_dir, device=device)
        results["is_mean"] = is_mean
        results["is_std"] = is_std
    else:
        results["is_mean"] = None
        results["is_std"] = None

    # Compute LPIPS
    if calc_lpips:
        results["lpips"] = compute_lpips(sample_dir, dataset, device=device, ref_size=min(ref_size, total_size))
    else:
        results["lpips"] = None

    return results


def get_model_info(dataset, model_suffix=None):
    """Get config path and checkpoint directory for a model."""
    if model_suffix is None:
        # Baseline model
        if dataset == "celeba64":
            return "configs/celeba.json", "chkpts/celeba", "celeba"
        else:
            return f"configs/{dataset}.json", f"chkpts/{dataset}", dataset
    else:
        # Curriculum variant
        name = f"{dataset}_{model_suffix}"
        return f"configs/{name}.json", f"chkpts/{name}", name


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="celeba64", type=str,
                        help="Base dataset name (celeba32, celeba64)")
    parser.add_argument("--models", default="baseline,c,cs,cr", type=str,
                        help="Models to compare (comma-separated: baseline,c,cs,cr)")
    parser.add_argument("--epochs", default="1,5,10", type=str,
                        help="Epochs to evaluate (comma-separated or 'all')")
    parser.add_argument("--total-size", default=2000, type=int,
                        help="Number of samples to generate for FID")
    parser.add_argument("--ref-size", default=2000, type=int,
                        help="Number of real images for reference stats")
    parser.add_argument("--num-gpus", default=1, type=int,
                        help="Number of GPUs for parallel generation")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--subseq-size", default=50, type=int,
                        help="DDIM subsequence size")
    parser.add_argument("--output", default=None, type=str,
                        help="Output file (default: metrics_stages_{dataset}.txt)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate metric curve plots")
    parser.add_argument("--no-is", action="store_true",
                        help="Disable Inception Score calculation")
    parser.add_argument("--no-lpips", action="store_true",
                        help="Disable LPIPS calculation")
    args = parser.parse_args()

    dataset = args.dataset
    models = [m.strip() for m in args.models.split(",")]
    calc_is = not args.no_is
    calc_lpips = not args.no_lpips

    # Parse epochs
    if args.epochs.lower() == "all":
        # Auto-detect max epoch from checkpoint files
        max_epoch = 10  # default
        for suffix in [None, "c", "cs", "cr"]:
            config_path, chkpt_dir, prefix = get_model_info(dataset, suffix)
            if os.path.exists(chkpt_dir):
                files = [f for f in os.listdir(chkpt_dir) if f.startswith(prefix) and f.endswith(".pt")]
                for f in files:
                    name = f.replace(prefix + "_", "").replace(".pt", "")
                    parts = name.split("_")
                    try:
                        if len(parts) >= 3:
                            if len(parts[0]) == 8:  # Date format YYYYMMDD
                                epoch = int(parts[2])
                            else:
                                epoch = int(parts[0])
                        else:
                            epoch = int(parts[0])
                        max_epoch = max(max_epoch, epoch)
                    except (ValueError, IndexError):
                        continue
        epochs = list(range(1, max_epoch + 1))
    else:
        epochs = [int(e.strip()) for e in args.epochs.split(",")]

    metrics_str = "FID"
    if calc_is:
        metrics_str += ", IS"
    if calc_lpips:
        metrics_str += ", LPIPS"

    print("=" * 80)
    print(f"Metrics Comparison by Stage for {dataset}")
    print(f"Metrics: {metrics_str}")
    print(f"Models: {models}")
    print(f"Epochs: {epochs}")
    print(f"Params: total_size={args.total_size}, ref_size={args.ref_size}, num_gpus={args.num_gpus}")
    print("=" * 80)

    # Precompute real stats once (celeba64 uses "celeba" dataset)
    fid_dataset = "celeba" if "celeba" in dataset else dataset
    print(f"\n[Precomputing real stats for {fid_dataset}]")
    real_stats = compute_real_stats(fid_dataset, device=args.device, ref_size=args.ref_size)
    print("  Real stats computed successfully.\n")

    # Results: {model: {epoch: {fid, is_mean, is_std, lpips}}}
    results = {m: {} for m in models}

    for model in models:
        suffix = None if model == "baseline" else model
        config_path, chkpt_dir, prefix = get_model_info(dataset, suffix)

        if not os.path.exists(config_path):
            print(f"[SKIP] {model}: config not found ({config_path})")
            continue

        print(f"\n[Model: {model}]")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint dir: {chkpt_dir}")

        for epoch in epochs:
            chkpt_path = find_checkpoint_by_epoch(chkpt_dir, prefix, epoch)

            if chkpt_path is None:
                print(f"  Epoch {epoch}: checkpoint not found")
                continue

            print(f"  Epoch {epoch}: {os.path.basename(chkpt_path)}")
            print(f"    Generating {args.total_size} samples...")

            metrics = eval_metrics(
                config_path=config_path,
                chkpt_path=chkpt_path,
                total_size=args.total_size,
                ref_size=args.ref_size,
                num_gpus=args.num_gpus,
                device=args.device,
                subseq_size=args.subseq_size,
                real_stats=real_stats,
                calc_is=calc_is,
                calc_lpips=calc_lpips,
            )

            # Print where samples are stored
            print(f"    Samples saved to: ./generated/{os.path.basename(config_path)[:-5]}/{os.path.basename(chkpt_path)[:-3]}")

            results[model][epoch] = metrics
            print(f"    FID: {metrics['fid']:.2f}")
            if metrics['is_mean'] is not None:
                print(f"    IS: {metrics['is_mean']:.2f} +/- {metrics['is_std']:.2f}")
            if metrics['lpips'] is not None:
                print(f"    LPIPS: {metrics['lpips']:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY - FID (lower is better)")
    print("=" * 80)

    # Header
    header = f"{'Epoch':<8}"
    for model in models:
        header += f" | {model:>12}"
    print(header)
    print("-" * len(header))

    # FID rows
    for epoch in epochs:
        row = f"{epoch:<8}"
        for model in models:
            m = results[model].get(epoch)
            if m is not None:
                row += f" | {m['fid']:>12.2f}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    # IS table (if enabled)
    if calc_is:
        print("\n" + "=" * 80)
        print("SUMMARY - IS (higher is better)")
        print("=" * 80)
        print(header)
        print("-" * len(header))
        for epoch in epochs:
            row = f"{epoch:<8}"
            for model in models:
                m = results[model].get(epoch)
                if m is not None and m['is_mean'] is not None:
                    row += f" | {m['is_mean']:>12.2f}"
                else:
                    row += f" | {'N/A':>12}"
            print(row)

    # LPIPS table (if enabled)
    if calc_lpips:
        print("\n" + "=" * 80)
        print("SUMMARY - LPIPS (lower is better)")
        print("=" * 80)
        print(header)
        print("-" * len(header))
        for epoch in epochs:
            row = f"{epoch:<8}"
            for model in models:
                m = results[model].get(epoch)
                if m is not None and m['lpips'] is not None:
                    row += f" | {m['lpips']:>12.4f}"
                else:
                    row += f" | {'N/A':>12}"
            print(row)

    print("=" * 80)

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    output_file = args.output or f"{results_dir}/metrics_stages_{dataset}.txt"

    with open(output_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Metrics Comparison by Stage for {dataset}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Metrics: {metrics_str}\n")
        f.write(f"Models: {models}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Params: total_size={args.total_size}, ref_size={args.ref_size}, "
                f"DDIM_steps={args.subseq_size}, num_gpus={args.num_gpus}\n")
        f.write("-" * 80 + "\n\n")

        # FID Table
        f.write("FID (lower is better):\n")
        header = f"{'Epoch':<8}"
        for model in models:
            header += f" | {model:>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for epoch in epochs:
            row = f"{epoch:<8}"
            for model in models:
                m = results[model].get(epoch)
                if m is not None:
                    row += f" | {m['fid']:>12.2f}"
                else:
                    row += f" | {'N/A':>12}"
            f.write(row + "\n")

        # IS Table
        if calc_is:
            f.write("\nIS (higher is better):\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for epoch in epochs:
                row = f"{epoch:<8}"
                for model in models:
                    m = results[model].get(epoch)
                    if m is not None and m['is_mean'] is not None:
                        row += f" | {m['is_mean']:>12.2f}"
                    else:
                        row += f" | {'N/A':>12}"
                f.write(row + "\n")

        # LPIPS Table
        if calc_lpips:
            f.write("\nLPIPS (lower is better):\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for epoch in epochs:
                row = f"{epoch:<8}"
                for model in models:
                    m = results[model].get(epoch)
                    if m is not None and m['lpips'] is not None:
                        row += f" | {m['lpips']:>12.4f}"
                    else:
                        row += f" | {'N/A':>12}"
                f.write(row + "\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults appended to {output_file}")

    # Save CSV (with all metrics) - append new results to existing file
    csv_file = output_file.replace(".txt", ".csv")

    # Check if CSV file exists and has header
    csv_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    with open(csv_file, "a") as f:
        # Header: epoch, model1_fid, model1_is, model1_lpips, model2_fid, ...
        if not csv_exists:
            csv_header = ["epoch"]
            for model in models:
                csv_header.append(f"{model}_fid")
                if calc_is:
                    csv_header.append(f"{model}_is")
                if calc_lpips:
                    csv_header.append(f"{model}_lpips")
            f.write(",".join(csv_header) + "\n")

        for epoch in epochs:
            row = [str(epoch)]
            for model in models:
                m = results[model].get(epoch)
                if m is not None:
                    row.append(f"{m['fid']:.2f}")
                    if calc_is:
                        row.append(f"{m['is_mean']:.2f}" if m['is_mean'] else "")
                    if calc_lpips:
                        row.append(f"{m['lpips']:.4f}" if m['lpips'] else "")
                else:
                    row.append("")
                    if calc_is:
                        row.append("")
                    if calc_lpips:
                        row.append("")
            f.write(",".join(row) + "\n")
    print(f"CSV appended to {csv_file}")

    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            # Count number of plots needed
            num_plots = 1  # FID always
            if calc_is:
                num_plots += 1
            if calc_lpips:
                num_plots += 1

            fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
            if num_plots == 1:
                axes = [axes]

            plot_idx = 0

            # FID plot
            ax = axes[plot_idx]
            for model in models:
                model_epochs = sorted(results[model].keys())
                fids = [results[model][e]['fid'] for e in model_epochs if results[model][e] is not None]
                valid_epochs = [e for e in model_epochs if results[model][e] is not None]
                if fids:
                    ax.plot(valid_epochs, fids, marker='o', label=model)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("FID")
            ax.set_title(f"FID vs Epoch ({dataset})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

            # IS plot
            if calc_is:
                ax = axes[plot_idx]
                for model in models:
                    model_epochs = sorted(results[model].keys())
                    is_vals = [results[model][e]['is_mean'] for e in model_epochs
                               if results[model][e] is not None and results[model][e]['is_mean'] is not None]
                    valid_epochs = [e for e in model_epochs
                                    if results[model][e] is not None and results[model][e]['is_mean'] is not None]
                    if is_vals:
                        ax.plot(valid_epochs, is_vals, marker='o', label=model)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("IS")
                ax.set_title(f"IS vs Epoch ({dataset})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1

            # LPIPS plot
            if calc_lpips:
                ax = axes[plot_idx]
                for model in models:
                    model_epochs = sorted(results[model].keys())
                    lpips_vals = [results[model][e]['lpips'] for e in model_epochs
                                  if results[model][e] is not None and results[model][e]['lpips'] is not None]
                    valid_epochs = [e for e in model_epochs
                                    if results[model][e] is not None and results[model][e]['lpips'] is not None]
                    if lpips_vals:
                        ax.plot(valid_epochs, lpips_vals, marker='o', label=model)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("LPIPS")
                ax.set_title(f"LPIPS vs Epoch ({dataset})")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = output_file.replace(".txt", ".png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
