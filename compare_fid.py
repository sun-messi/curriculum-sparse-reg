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
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.sharedctypes import Synchronized
from PIL import Image
from tqdm import tqdm
import time

from ddim import DDIM, get_selection_schedule
from ddpm_torch import *
from ddpm_torch.metrics import InceptionStatistics, calc_fd, get_precomputed
from torch.utils.data import Dataset, DataLoader
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


def compute_fid(sample_dir, dataset, precomputed_dir="./precomputed", device="cuda:0", ref_size=None):
    """Compute FID score for generated samples.

    Args:
        ref_size: Number of real images to use for reference stats.
                  None = use all (and cache), otherwise compute on-the-fly with subset.
    """
    import numpy as np

    imagefolder = ImageFolder(sample_dir)
    imageloader = DataLoader(
        imagefolder, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    istats = InceptionStatistics(device=device, input_transform=lambda im: (im - 127.5) / 127.5)

    # Load precomputed stats or compute from raw data
    precomputed_path = os.path.join(precomputed_dir, f"fid_stats_{dataset}.npz")

    if ref_size is None and os.path.exists(precomputed_path):
        # Use cached full-dataset stats
        print(f"  Loading precomputed stats from {precomputed_path}")
        precomputed_data = np.load(precomputed_path)
        true_mean, true_var = precomputed_data["mu"], precomputed_data["sigma"]
    else:
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

        istats.reset()

    # Compute generated stats
    for x in tqdm(imageloader, desc="Computing gen stats"):
        istats(x.to(device))
    gen_mean, gen_var = istats.get_statistics()

    fid = calc_fd(gen_mean, gen_var, true_mean, true_var)
    return fid


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="celeba32", type=str,
                        help="Base dataset name (celeba32 or celeba)")
    parser.add_argument("--total-size", default=200, type=int,
                        help="Number of samples to generate for FID")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--use-ddim", action="store_true", default=True)
    parser.add_argument("--subseq-size", default=50, type=int,
                        help="DDIM subsequence size")
    parser.add_argument("--keep-samples", action="store_true",
                        help="Keep generated samples after evaluation")
    parser.add_argument("--num-gpus", default=1, type=int,
                        help="Number of GPUs for parallel generation")
    parser.add_argument("--ref-size", default=5000, type=int,
                        help="Number of real images for reference stats (None=use all & cache)")
    args = parser.parse_args()

    dataset = args.dataset
    base_name = dataset.replace("32", "")  # celeba32 -> celeba

    def find_latest_checkpoint(chkpt_dir, prefix):
        """Find the latest checkpoint file by epoch number."""
        if not os.path.exists(chkpt_dir):
            return None
        files = [f for f in os.listdir(chkpt_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not files:
            return None
        # Sort by epoch number (e.g., celeba32_c_15.pt -> 15)
        def get_epoch(f):
            try:
                return int(f.replace(prefix + "_", "").replace(".pt", ""))
            except ValueError:
                return 0
        files.sort(key=get_epoch, reverse=True)
        return os.path.join(chkpt_dir, files[0])

    # Define models to compare
    models = [
        {
            "name": f"{dataset} (baseline)",
            "config": f"configs/{dataset}.json",
            "chkpt": find_latest_checkpoint(f"chkpts/{dataset}", dataset),
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

        # Generate samples
        sample_dir = f"./tmp_fid_samples/{os.path.basename(chkpt_path)[:-3]}"
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)

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

        # Compute FID
        print("  Computing FID...")
        # Use the actual dataset name for precomputed stats
        fid_dataset = dataset if "32" in dataset else base_name
        fid = compute_fid(sample_dir, fid_dataset, device=args.device, ref_size=args.ref_size)
        results.append({"name": name, "fid": fid})
        print(f"  FID: {fid:.2f}")

        # Cleanup
        if not args.keep_samples:
            shutil.rmtree(sample_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<45} {'FID':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["fid"]):
        print(f"{r['name']:<45} {r['fid']:>10.2f}")
    print("=" * 60)

    # Save results
    results_file = f"fid_comparison_{dataset}.txt"
    with open(results_file, "w") as f:
        f.write(f"FID Comparison for {dataset}\n")
        f.write(f"Samples: {args.total_size}, DDIM steps: {args.subseq_size}\n")
        f.write("-" * 60 + "\n")
        for r in sorted(results, key=lambda x: x["fid"]):
            f.write(f"{r['name']:<45} {r['fid']:>10.2f}\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
