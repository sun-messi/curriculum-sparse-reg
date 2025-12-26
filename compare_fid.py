"""
Compare FID scores across different curriculum training modes.

Usage:
    # Single GPU
    python compare_fid.py --dataset celeba32

    # Multi-GPU (4 GPUs)
    python compare_fid.py --dataset celeba32 --num-gpus 4

    # Select specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python compare_fid.py --dataset celeba32 --num-gpus 4

This will evaluate:
    - celeba32 (baseline)
    - celeba32_c (curriculum only)
    - celeba32_cs (curriculum + sparsity)
    - celeba32_cr (curriculum + regularization)
"""

import json
import math
import os
import shutil
import torch
import torch.multiprocessing as mp
import uuid
import numpy as np
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.sharedctypes import Synchronized
from PIL import Image
from tqdm import tqdm
import time

from ddim import DDIM, get_selection_schedule
from ddpm_torch import *
from ddpm_torch.metrics import InceptionStatistics, calc_fd, get_precomputed, compute_is_from_folder
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import re


def load_json_with_comments(filepath):
    """Load JSON file that may contain // comments."""
    with open(filepath, "r") as f:
        content = f.read()
    # Remove // comments (but not inside strings)
    content = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)
    return json.loads(content)


class ImageFolder(Dataset):
    """Dataset for loading generated images from a folder."""
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = [
            img for img in os.listdir(img_dir)
            if img.split(".")[-1] in {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
        ]
        self.transform = transforms.PILToTensor()

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.img_dir, self.img_list[idx])) as im:
            return self.transform(im)

    def __len__(self):
        return len(self.img_list)


def generate_samples_single_gpu(
    config_path,
    chkpt_path,
    save_dir,
    total_size=1000,
    batch_size=128,
    device="cuda:0",
    use_ema=True,
    use_ddim=True,
    eta=0.0,
    subseq_size=50,
):
    """Generate samples from a checkpoint."""
    meta_config = load_json_with_comments(config_path)

    dataset = meta_config.get("dataset", "celeba32")
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"][0]
    input_shape = (in_channels, image_res, image_res)

    # Setup diffusion
    diffusion_kwargs = meta_config["diffusion"].copy()
    beta_schedule = diffusion_kwargs.pop("beta_schedule")
    beta_start = diffusion_kwargs.pop("beta_start")
    beta_end = diffusion_kwargs.pop("beta_end")
    num_timesteps = diffusion_kwargs.pop("timesteps")
    betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps)

    if use_ddim:
        diffusion_kwargs["model_var_type"] = "fixed-small"
        subsequence = get_selection_schedule("linear", size=subseq_size, timesteps=num_timesteps)
        diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
    else:
        diffusion = GaussianDiffusion(betas, **diffusion_kwargs)

    # Load checkpoint first to infer model config
    device = torch.device(device)
    state_dict = torch.load(chkpt_path, map_location=device, weights_only=False)
    try:
        if use_ema:
            state_dict = state_dict["ema"]["shadow"]
        else:
            state_dict = state_dict["model"]
    except KeyError:
        pass  # Direct state dict

    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)

    # Auto-detect hid_channels from checkpoint
    model_config = meta_config["model"].copy()
    if "in_conv.weight" in state_dict:
        inferred_hid = state_dict["in_conv.weight"].shape[0]
        if model_config.get("hid_channels") != inferred_hid:
            print(f"  Auto-detected hid_channels={inferred_hid} from checkpoint")
            model_config["hid_channels"] = inferred_hid

    # Setup model
    block_size = model_config.pop("block_size", 1)
    model = UNet(out_channels=in_channels, **model_config)
    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)
        post_transform = torch.nn.PixelShuffle(block_size)
        model = ModelWrapper(model, pre_transform, post_transform)
    model.to(device)

    model.load_state_dict(state_dict)
    del state_dict
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Generate samples
    os.makedirs(save_dir, exist_ok=True)
    num_batches = math.ceil(total_size / batch_size)

    def save_image(arr):
        with Image.fromarray(arr, mode="RGB") as im:
            im.save(f"{save_dir}/{uuid.uuid4()}.png")

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in tqdm(range(num_batches), desc="Generating"):
            current_batch = min(batch_size, total_size - i * batch_size)
            shape = (current_batch, *input_shape)
            x = diffusion.p_sample(
                model, shape=shape, device=device,
                noise=torch.randn(shape, device=device)
            ).cpu()
            x = (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            x = x.permute(0, 2, 3, 1).numpy()
            pool.map(save_image, list(x))

    return save_dir


def _generate_worker(rank, args, counter):
    """Worker function for multi-GPU generation."""
    world_size = args.world_size
    local_total_size = args.total_size // world_size
    if rank < args.total_size % world_size:
        local_total_size += 1

    device = f"cuda:{rank}"

    meta_config = load_json_with_comments(args.config_path)

    dataset = meta_config.get("dataset", "celeba32")
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"][0]
    input_shape = (in_channels, image_res, image_res)

    # Setup diffusion
    diffusion_kwargs = meta_config["diffusion"].copy()
    beta_schedule = diffusion_kwargs.pop("beta_schedule")
    beta_start = diffusion_kwargs.pop("beta_start")
    beta_end = diffusion_kwargs.pop("beta_end")
    num_timesteps = diffusion_kwargs.pop("timesteps")
    betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps)

    if args.use_ddim:
        diffusion_kwargs["model_var_type"] = "fixed-small"
        subsequence = get_selection_schedule("linear", size=args.subseq_size, timesteps=num_timesteps)
        diffusion = DDIM(betas, **diffusion_kwargs, eta=0.0, subsequence=subsequence)
    else:
        diffusion = GaussianDiffusion(betas, **diffusion_kwargs)

    # Load checkpoint first to infer model config
    state_dict = torch.load(args.chkpt_path, map_location=device, weights_only=False)
    try:
        if args.use_ema:
            state_dict = state_dict["ema"]["shadow"]
        else:
            state_dict = state_dict["model"]
    except KeyError:
        pass

    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)

    # Auto-detect hid_channels from checkpoint
    model_config = meta_config["model"].copy()
    if "in_conv.weight" in state_dict:
        inferred_hid = state_dict["in_conv.weight"].shape[0]
        if model_config.get("hid_channels") != inferred_hid:
            model_config["hid_channels"] = inferred_hid

    # Setup model
    block_size = model_config.pop("block_size", 1)
    model = UNet(out_channels=in_channels, **model_config)
    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)
        post_transform = torch.nn.PixelShuffle(block_size)
        model = ModelWrapper(model, pre_transform, post_transform)
    model.to(device)

    model.load_state_dict(state_dict)
    del state_dict
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Generate samples
    batch_size = args.batch_size
    num_batches = math.ceil(local_total_size / batch_size)

    def save_image(arr):
        with Image.fromarray(arr, mode="RGB") as im:
            im.save(f"{args.save_dir}/{uuid.uuid4()}.png")

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(num_batches):
            current_batch = min(batch_size, local_total_size - i * batch_size)
            shape = (current_batch, *input_shape)
            x = diffusion.p_sample(
                model, shape=shape, device=device,
                noise=torch.randn(shape, device=device)
            ).cpu()
            x = (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            x = x.permute(0, 2, 3, 1).numpy()
            pool.map(save_image, list(x))

            if isinstance(counter, Synchronized):
                with counter.get_lock():
                    counter.value += 1


def _progress_monitor(total, counter):
    """Monitor progress across all GPUs."""
    pbar = tqdm(total=total, desc="Generating")
    while pbar.n < total:
        if pbar.n < counter.value:
            pbar.update(counter.value - pbar.n)
        time.sleep(0.1)
    pbar.close()


def generate_samples(
    config_path,
    chkpt_path,
    save_dir,
    total_size=1000,
    batch_size=128,
    device="cuda:0",
    use_ema=True,
    use_ddim=True,
    eta=0.0,
    subseq_size=50,
    num_gpus=1,
):
    """Generate samples with optional multi-GPU support."""
    os.makedirs(save_dir, exist_ok=True)

    if num_gpus <= 1:
        # Single GPU mode
        return generate_samples_single_gpu(
            config_path=config_path,
            chkpt_path=chkpt_path,
            save_dir=save_dir,
            total_size=total_size,
            batch_size=batch_size,
            device=device,
            use_ema=use_ema,
            use_ddim=use_ddim,
            eta=eta,
            subseq_size=subseq_size,
        )

    # Multi-GPU mode
    from argparse import Namespace
    args = Namespace(
        config_path=config_path,
        chkpt_path=chkpt_path,
        save_dir=save_dir,
        total_size=total_size,
        batch_size=batch_size,
        use_ema=use_ema,
        use_ddim=use_ddim,
        subseq_size=subseq_size,
        world_size=num_gpus,
    )

    # Calculate total batches for progress bar
    local_total_size = total_size // num_gpus
    remainder = total_size % num_gpus
    num_batches = math.ceil((local_total_size + 1) / batch_size) * remainder
    num_batches += math.ceil(local_total_size / batch_size) * (num_gpus - remainder)

    mp.set_start_method("spawn", force=True)
    counter = mp.Value("i", 0)

    # Start progress monitor
    monitor = mp.Process(target=_progress_monitor, args=(num_batches, counter), daemon=True)
    monitor.start()

    # Spawn workers
    mp.spawn(_generate_worker, args=(args, counter), nprocs=num_gpus)

    monitor.join(timeout=1)

    return save_dir


def compute_real_stats(dataset, device="cuda:0", ref_size=None, precomputed_dir="./precomputed"):
    """Compute or load real image statistics for FID calculation.

    Args:
        dataset: Dataset name (e.g., "celeba")
        device: Device for computation
        ref_size: Number of real images to use. None = use all (and cache).
        precomputed_dir: Directory for cached stats

    Returns:
        (mean, var): Tuple of numpy arrays for mean and covariance
    """
    import numpy as np

    istats = InceptionStatistics(device=device, input_transform=lambda im: (im - 127.5) / 127.5)

    # Load precomputed stats or compute from raw data
    precomputed_path = os.path.join(precomputed_dir, f"fid_stats_{dataset}.npz")

    if ref_size is None and os.path.exists(precomputed_path):
        # Use cached full-dataset stats
        print(f"  Loading precomputed stats from {precomputed_path}")
        precomputed_data = np.load(precomputed_path)
        return precomputed_data["mu"], precomputed_data["sigma"]

    # Compute stats from raw data
    from torch.utils.data import Subset
    full_dataset = get_dataloader(
        dataset, batch_size=256, split="all", val_size=0.,
        root=os.path.expanduser("~/datasets"),
        pin_memory=True, drop_last=False, num_workers=4, raw=True
    )[0].dataset

    if ref_size and ref_size < len(full_dataset):
        # Use random subset for faster computation
        indices = torch.randperm(len(full_dataset))[:ref_size].tolist()
        subset = Subset(full_dataset, indices)
        print(f"  Computing real stats from {ref_size} samples...")
    else:
        subset = full_dataset
        print(f"  Computing real stats from {len(full_dataset)} samples...")

    dataloader = DataLoader(
        subset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    for x in tqdm(dataloader, desc="Computing real stats"):
        istats(x.to(device))
    true_mean, true_var = istats.get_statistics()

    # Only cache if using full dataset
    if ref_size is None:
        os.makedirs(precomputed_dir, exist_ok=True)
        np.savez(precomputed_path, mu=true_mean, sigma=true_var)
        print(f"  Saved precomputed stats to {precomputed_path}")

    return true_mean, true_var


def compute_fid(sample_dir, dataset, precomputed_dir="./precomputed", device="cuda:0",
                ref_size=None, real_stats=None):
    """Compute FID score for generated samples.

    Args:
        sample_dir: Directory containing generated samples
        dataset: Dataset name for real stats (ignored if real_stats provided)
        precomputed_dir: Directory for cached stats
        device: Device for computation
        ref_size: Number of real images to use for reference stats.
                  None = use all (and cache), otherwise compute on-the-fly with subset.
        real_stats: Precomputed (mean, var) tuple. If provided, skips real stats computation.

    Returns:
        fid: FID score (float)
    """
    import numpy as np

    imagefolder = ImageFolder(sample_dir)
    imageloader = DataLoader(
        imagefolder, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    istats = InceptionStatistics(device=device, input_transform=lambda im: (im - 127.5) / 127.5)

    # Get real stats (use precomputed if provided)
    if real_stats is not None:
        true_mean, true_var = real_stats
    else:
        true_mean, true_var = compute_real_stats(
            dataset, device=device, ref_size=ref_size, precomputed_dir=precomputed_dir
        )

    # Compute generated stats
    for x in tqdm(imageloader, desc="Computing gen stats"):
        istats(x.to(device))
    gen_mean, gen_var = istats.get_statistics()

    fid = calc_fd(gen_mean, gen_var, true_mean, true_var)
    return fid


def compute_is(sample_dir, device="cuda:0", splits=10):
    """Compute Inception Score for generated samples.

    Args:
        sample_dir: Directory containing generated images
        device: Device for computation
        splits: Number of splits for mean/std calculation

    Returns:
        (mean, std): IS mean and standard deviation
    """
    return compute_is_from_folder(sample_dir, device=device, splits=splits)


def compute_lpips(sample_dir, dataset, device="cuda:0", ref_size=1000):
    """Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Compares generated images with random real images.

    Args:
        sample_dir: Directory containing generated images
        dataset: Dataset name for loading real images
        device: Device for computation
        ref_size: Number of image pairs to compare

    Returns:
        mean_lpips: Average LPIPS distance
    """
    try:
        import lpips
    except ImportError:
        print("  [WARN] lpips not installed. Run: pip install lpips")
        return None

    # Load LPIPS model
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    # Load generated images
    gen_dataset = ImageFolder(sample_dir)

    # Load real images
    real_dataset = get_dataloader(
        dataset, batch_size=256, split="all", val_size=0.,
        root=os.path.expanduser("~/datasets"),
        pin_memory=True, drop_last=False, num_workers=4, raw=True
    )[0].dataset

    # Sample pairs
    num_pairs = min(ref_size, len(gen_dataset), len(real_dataset))
    gen_indices = torch.randperm(len(gen_dataset))[:num_pairs].tolist()
    real_indices = torch.randperm(len(real_dataset))[:num_pairs].tolist()

    # Compute LPIPS
    lpips_scores = []
    batch_size = 32

    for i in tqdm(range(0, num_pairs, batch_size), desc="Computing LPIPS"):
        batch_end = min(i + batch_size, num_pairs)

        # Load generated images
        gen_batch = torch.stack([
            gen_dataset[gen_indices[j]] for j in range(i, batch_end)
        ]).to(device)

        # Load real images
        real_batch = torch.stack([
            real_dataset[real_indices[j]] for j in range(i, batch_end)
        ]).to(device)

        # Normalize to [-1, 1] for LPIPS
        # gen_dataset returns [0, 255] uint8, real_dataset returns [0, 255] uint8
        if gen_batch.dtype == torch.uint8:
            gen_batch = gen_batch.float() / 127.5 - 1
        elif gen_batch.max() > 1:
            gen_batch = gen_batch / 127.5 - 1

        if real_batch.dtype == torch.uint8:
            real_batch = real_batch.float() / 127.5 - 1
        elif real_batch.max() > 1:
            real_batch = real_batch / 127.5 - 1

        with torch.no_grad():
            dist = loss_fn(gen_batch, real_batch)
            lpips_scores.extend(dist.cpu().squeeze().tolist())

    return float(np.mean(lpips_scores))


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="celeba32", type=str,
                        help="Base dataset name (celeba32, celeba64, or celeba)")
    parser.add_argument("--total-size", default=200, type=int,
                        help="Number of samples to generate for FID")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--use-ddim", action="store_true", default=True)
    parser.add_argument("--subseq-size", default=50, type=int,
                        help="DDIM subsequence size")
    parser.add_argument("--num-gpus", default=1, type=int,
                        help="Number of GPUs for parallel generation")
    parser.add_argument("--ref-size", default=5000, type=int,
                        help="Number of real images for reference stats (None=use all & cache)")
    args = parser.parse_args()

    dataset = args.dataset
    # celeba32 -> celeba, celeba64 -> celeba
    base_name = dataset.replace("32", "").replace("64", "")

    def find_latest_checkpoint(chkpt_dir, prefix):
        """Find the latest checkpoint file by epoch and timestamp.

        Supports both old format (prefix_epoch.pt) and new format (prefix_epoch_timestamp.pt)
        """
        if not os.path.exists(chkpt_dir):
            return None
        files = [f for f in os.listdir(chkpt_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not files:
            return None

        def get_sort_key(f):
            # Remove prefix and .pt extension
            name = f.replace(prefix + "_", "").replace(".pt", "")
            parts = name.split("_")
            try:
                if len(parts) >= 3:
                    # Detect format by first part length
                    if len(parts[0]) == 8:  # Date format YYYYMMDD
                        # New format: timestamp_epoch (YYYYMMDD_HHMMSS_epoch)
                        timestamp = parts[0] + parts[1]  # YYYYMMDDHHMMSS
                        epoch = int(parts[2])
                    else:
                        # Old format: epoch_timestamp (epoch_YYYYMMDD_HHMMSS)
                        epoch = int(parts[0])
                        timestamp = parts[1] + parts[2]  # YYYYMMDDHHMMSS
                    return (timestamp, epoch)
                # Very old format: just epoch
                epoch = int(parts[0])
                return ("0", epoch)
            except (ValueError, IndexError):
                return ("0", 0)

        files.sort(key=get_sort_key, reverse=True)
        return os.path.join(chkpt_dir, files[0])

    # For celeba64, use "celeba" as baseline config but "celeba64_*" for curriculum variants
    if dataset == "celeba64":
        baseline_config = "configs/celeba.json"
        baseline_chkpt_dir = "chkpts/celeba"
        baseline_prefix = "celeba"
    else:
        baseline_config = f"configs/{dataset}.json"
        baseline_chkpt_dir = f"chkpts/{dataset}"
        baseline_prefix = dataset

    # Define models to compare
    models = [
        {
            "name": f"{base_name} (baseline)",
            "config": baseline_config,
            "chkpt": find_latest_checkpoint(baseline_chkpt_dir, baseline_prefix),
        },
        {
            "name": f"{dataset}_c (curriculum)",
            "config": f"configs/{dataset}_c.json",
            "chkpt": find_latest_checkpoint(f"chkpts/{dataset}_c", f"{dataset}_c"),
        },
        {
            "name": f"{dataset}_cs (curriculum+sparsity)",
            "config": f"configs/{dataset}_cs.json",
            "chkpt": find_latest_checkpoint(f"chkpts/{dataset}_cs", f"{dataset}_cs"),
        },
        {
            "name": f"{dataset}_cr (curriculum+regularization)",
            "config": f"configs/{dataset}_cr.json",
            "chkpt": find_latest_checkpoint(f"chkpts/{dataset}_cr", f"{dataset}_cr"),
        },
    ]

    results = []
    print("=" * 60)
    print(f"FID Comparison for {dataset}")
    print("=" * 60)

    for model_info in models:
        name = model_info["name"]
        config_path = model_info["config"]
        chkpt_path = model_info["chkpt"]

        # Check if files exist
        if not os.path.exists(config_path):
            print(f"[SKIP] {name}: config not found ({config_path})")
            continue
        if chkpt_path is None or not os.path.exists(chkpt_path):
            print(f"[SKIP] {name}: checkpoint not found")
            continue

        print(f"\n[{name}]")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {chkpt_path}")

        # Extract model name from config path for organized folder structure
        model_name = os.path.basename(config_path).replace('.json', '')
        checkpoint_name = os.path.basename(chkpt_path)[:-3]  # Remove .pt
        sample_dir = f"./generated/{model_name}/{checkpoint_name}"

        # Generate samples only if they don't exist
        os.makedirs(sample_dir, exist_ok=True)
        if len(os.listdir(sample_dir)) == 0:
            print(f"  Generating {args.total_size} samples (GPUs: {args.num_gpus})...")
            generate_samples(
                config_path=config_path,
                chkpt_path=chkpt_path,
                save_dir=sample_dir,
                total_size=args.total_size,
                batch_size=args.batch_size,
                device=args.device,
                use_ddim=args.use_ddim,
                subseq_size=args.subseq_size,
                num_gpus=args.num_gpus,
            )
        else:
            print(f"  Reusing {len(os.listdir(sample_dir))} existing samples")

        # Compute FID
        print("  Computing FID...")
        # Use the actual dataset name for precomputed stats
        fid_dataset = dataset if "32" in dataset else base_name
        fid = compute_fid(sample_dir, fid_dataset, device=args.device, ref_size=args.ref_size)
        results.append({"name": name, "fid": fid, "chkpt": chkpt_path, "config": config_path})
        print(f"  FID: {fid:.2f}")
        print(f"  Samples saved to: {sample_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<45} {'FID':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["fid"]):
        print(f"{r['name']:<45} {r['fid']:>10.2f}")
    print("=" * 60)

    # Save results (append mode with timestamp and details)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = f"{results_dir}/fid_comparison_{dataset}.txt"
    with open(results_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"FID Comparison for {dataset}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Eval Params: total_size={args.total_size}, ref_size={args.ref_size}, "
                f"DDIM_steps={args.subseq_size}, num_gpus={args.num_gpus}\n")
        f.write("-" * 80 + "\n")

        for r in sorted(results, key=lambda x: x["fid"]):
            chkpt_name = os.path.basename(r.get('chkpt', 'N/A')) if r.get('chkpt') else 'N/A'
            config_path = r.get('config', '')

            f.write(f"\n[{r['name']}]\n")
            f.write(f"  Checkpoint: {chkpt_name}\n")
            f.write(f"  FID: {r['fid']:.2f}\n")

            # Load and write model config
            if config_path and os.path.exists(config_path):
                try:
                    config = load_json_with_comments(config_path)
                    model_cfg = config.get("model", {})
                    train_cfg = config.get("train", {})
                    curriculum_cfg = config.get("curriculum", {})

                    f.write(f"  Model: hid_channels={model_cfg.get('hid_channels')}, "
                            f"ch_multipliers={model_cfg.get('ch_multipliers')}, "
                            f"apply_attn={model_cfg.get('apply_attn')}\n")
                    f.write(f"  Train: lr={train_cfg.get('lr')}, batch_size={train_cfg.get('batch_size')}\n")

                    if curriculum_cfg.get("enabled"):
                        stages = curriculum_cfg.get("stages", [])
                        total_epochs = sum(s.get("epochs", 1) for s in stages)
                        f.write(f"  Curriculum: {len(stages)} stages, {total_epochs} total epochs\n")
                except Exception as e:
                    f.write(f"  Config: (error reading: {e})\n")

        f.write("\n" + "=" * 80 + "\n")
    print(f"\nResults appended to {results_file}")


if __name__ == "__main__":
    main()
