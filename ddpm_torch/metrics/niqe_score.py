"""
MANIQA (Multi-dimension Attention Network for Image Quality Assessment) implementation.

MANIQA is a no-reference deep learning image quality metric using multi-dimension
attention. Higher scores = better quality.

MANIQA is sensitive to:
- Distortion artifacts (blur, noise, compression)
- Structural degradation
- Perceptual quality issues

Reference:
    Yang et al., "MANIQA: Multi-dimension Attention Network for No-Reference Image
    Quality Assessment", CVPR 2022 Workshop.
"""

import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm, trange
import pyiqa


def compute_niqe_single_image(image_path, metric_model=None, device='cuda'):
    """
    Compute MANIQA score for a single image (no-reference).

    Args:
        image_path: Path to the image file
        metric_model: Pre-initialized MANIQA model
        device: Device for computation

    Returns:
        MANIQA score (float, higher is better)
    """
    try:
        # Initialize metric if not provided
        if metric_model is None:
            metric_model = pyiqa.create_metric('maniqa', device=device)

        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Convert to tensor (C, H, W) in range [0, 1]
            img_array = np.array(img)

            # Check if image is valid
            if img_array.size == 0 or len(img_array.shape) != 3:
                raise ValueError(f"Invalid image shape: {img_array.shape}")

            img_array = img_array.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

            # Move to device
            img_tensor = img_tensor.to(device)

            # Compute MANIQA
            with torch.no_grad():
                score = metric_model(img_tensor)

            return float(score.item())
    except Exception as e:
        print(f"Warning: Failed to compute MANIQA for {image_path}: {e}")
        return None


def compute_niqe_from_folder(folder_path, verbose=True, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=32):
    """
    Compute average MANIQA score for all images in a folder.

    Args:
        folder_path: Directory containing images
        verbose: Whether to show progress bar
        device: Device for computation ('cuda' or 'cpu')
        batch_size: Number of images to process in batch

    Returns:
        Average MANIQA score (float, higher is better)
    """
    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")

    # Initialize MANIQA metric once
    metric_model = pyiqa.create_metric('maniqa', device=device)

    # Process images in batches for speed
    scores = []
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    if verbose:
        pbar = tqdm(total=len(image_files), desc="Computing MANIQA")

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
                if verbose:
                    print(f"\nWarning: Failed to load {img_path}: {e}")
                continue

        if len(batch_tensors) > 0:
            # Stack into batch and compute
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                batch_scores = metric_model(batch)

            # Handle both single value and batch output
            if batch_scores.dim() == 0:
                scores.append(float(batch_scores.item()))
            else:
                scores.extend([float(s.item()) for s in batch_scores])

        if verbose:
            pbar.update(len(batch_files))

    if verbose:
        pbar.close()

    if len(scores) == 0:
        raise ValueError(f"Failed to compute MANIQA for any images in {folder_path}")

    # Calculate average
    avg_score = float(np.mean(scores))

    if verbose:
        success_rate = len(scores) / len(image_files) * 100
        print(f"Successfully computed MANIQA for {len(scores)}/{len(image_files)} images ({success_rate:.1f}%)")
        print(f"Average MANIQA: {avg_score:.4f}")

    return avg_score


if __name__ == "__main__":
    # Test the implementation
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        score = compute_niqe_from_folder(folder)
        print(f"\nFinal MANIQA score: {score:.4f}")
    else:
        print("Usage: python niqe_score.py <image_folder>")
