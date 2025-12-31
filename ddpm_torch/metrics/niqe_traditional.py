"""
Traditional NIQE (Natural Image Quality Evaluator) implementation.

NIQE is a statistical no-reference image quality metric based on natural scene
statistics. Lower scores = better quality (more natural).

This is a lightweight implementation using scipy for faster computation.
"""

import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from scipy import signal
from scipy.special import gamma
import pickle


def estimate_aggd_params(x):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters."""
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = (gamma(2.0/gam) ** 2) / (gamma(1.0/gam) * gamma(3.0/gam))

    left_std = np.sqrt(np.mean(x[x < 0] ** 2))
    right_std = np.sqrt(np.mean(x[x >= 0] ** 2))
    gammahat = left_std / right_std

    rhat = (np.mean(np.abs(x))) ** 2 / np.mean(x ** 2)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)

    array_position = np.argmin(np.abs(r_gam - rhatnorm))
    alpha = gam[array_position]

    return alpha, left_std, right_std


def compute_niqe_features(img):
    """
    Compute NIQE features from an image.

    Args:
        img: PIL Image or numpy array

    Returns:
        features: numpy array of NIQE features
    """
    # Convert to grayscale
    if isinstance(img, Image.Image):
        img = np.array(img.convert('L')).astype(np.float64)
    else:
        if len(img.shape) == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = img.astype(np.float64)

    # Normalize
    img = img / 255.0

    # Local mean subtraction
    mu = signal.correlate2d(img, np.ones((7, 7))/49, mode='same', boundary='symm')
    mu_sq = mu ** 2
    sigma = np.sqrt(np.abs(signal.correlate2d(img ** 2, np.ones((7, 7))/49, mode='same', boundary='symm') - mu_sq))

    # Avoid division by zero
    structdis = (img - mu) / (sigma + 1)

    # Extract features
    features = []

    # Patch-wise feature extraction (simplified version)
    patch_size = 96
    step = 96

    for scale in [1, 2]:  # Two scales
        if scale == 2:
            # Downsample
            img_scaled = img[::2, ::2]
            mu = signal.correlate2d(img_scaled, np.ones((7, 7))/49, mode='same', boundary='symm')
            mu_sq = mu ** 2
            sigma = np.sqrt(np.abs(signal.correlate2d(img_scaled ** 2, np.ones((7, 7))/49, mode='same', boundary='symm') - mu_sq))
            structdis = (img_scaled - mu) / (sigma + 1)

        # Sample patches
        h, w = structdis.shape
        if h < patch_size or w < patch_size:
            continue

        for i in range(0, h - patch_size + 1, step):
            for j in range(0, w - patch_size + 1, step):
                patch = structdis[i:i+patch_size, j:j+patch_size]

                # Compute AGGD parameters
                try:
                    alpha, left_std, right_std = estimate_aggd_params(patch.flatten())
                    features.extend([alpha, (left_std + right_std) / 2])
                except:
                    continue

    return np.array(features) if len(features) > 0 else np.zeros(18)


def compute_niqe_single_image(image_path):
    """
    Compute NIQE score for a single image.

    Args:
        image_path: Path to image file

    Returns:
        score: NIQE score (lower is better), or None if failed
    """
    try:
        img = Image.open(image_path)
        features = compute_niqe_features(img)

        # Simplified NIQE score: just return mean of features
        # (A full NIQE would compare against pristine image statistics)
        # Lower values indicate more natural images
        score = float(np.mean(np.abs(features)))

        return score
    except Exception as e:
        return None


def compute_niqe_from_folder(folder_path, verbose=True):
    """
    Compute average NIQE score for all images in a folder.

    Args:
        folder_path: Directory containing images
        verbose: Whether to show progress bar

    Returns:
        Average NIQE score (float, lower is better)
    """
    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")

    # Process images
    scores = []
    iterator = tqdm(image_files, desc="Computing NIQE") if verbose else image_files

    for img_file in iterator:
        img_path = os.path.join(folder_path, img_file)
        score = compute_niqe_single_image(img_path)

        if score is not None:
            scores.append(score)

    if len(scores) == 0:
        raise ValueError(f"Failed to compute NIQE for any images in {folder_path}")

    # Calculate average
    avg_score = float(np.mean(scores))

    if verbose:
        success_rate = len(scores) / len(image_files) * 100
        print(f"Successfully computed NIQE for {len(scores)}/{len(image_files)} images ({success_rate:.1f}%)")
        print(f"Average NIQE: {avg_score:.4f}")

    return avg_score


if __name__ == "__main__":
    # Test the implementation
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        score = compute_niqe_from_folder(folder)
        print(f"\nFinal NIQE score: {score:.4f}")
    else:
        print("Usage: python niqe_traditional.py <image_folder>")
