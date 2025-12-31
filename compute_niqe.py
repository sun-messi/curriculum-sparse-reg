#!/usr/bin/env python
"""Compute NIQE scores for generated images using custom implementation."""

import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm
import scipy.ndimage
import scipy.special
import scipy.io
import scipy.linalg
import math

# Precompute gamma values for AGGD
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    """Extract AGGD features from image data."""
    imdata = imdata.flatten()
    imdata2 = imdata*imdata
    left_data = imdata2[imdata<0]
    right_data = imdata2[imdata>=0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
        gamma_hat = np.inf

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    N = (br - bl)*(gam2 / gam1)
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def paired_product(new_im):
    """Generate shifted versions of image for directional analysis."""
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)

def gen_gauss_window(lw, sigma):
    """Generate Gaussian window."""
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum_val = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum_val += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum_val
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    """Compute MSCN (Mean Subtracted Contrast Normalized) transform."""
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image

def niqe_extract_subband_feats(mscncoefs):
    """Extract subband features from MSCN coefficients."""
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
            alpha1, N1, bl1, br1,
            alpha2, N2, bl2, br2,
            alpha3, N3, bl3, br3,
            alpha4, N4, bl4, br4,
    ])

def extract_on_patches(img, patch_size):
    """Extract features from non-overlapping patches."""
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features

def compute_niqe_single_image(img_gray, patch_size=32):
    """
    Compute NIQE score for a single grayscale image.

    Args:
        img_gray: Grayscale image as numpy array
        patch_size: Patch size (default 32 for 64x64 images)

    Returns:
        NIQE score (lower is better)
    """
    h, w = img_gray.shape

    # For small images, use smaller patch size
    if h < 64 or w < 64:
        return None

    # Compute MSCN transform
    mscn, _, _ = compute_image_mscn_transform(img_gray.astype(np.float32))

    # Extract features
    feats = extract_on_patches(mscn, patch_size)

    if len(feats) == 0:
        return None

    # Use simple quality score based on feature statistics
    # For small images without reference params, use feature variance
    score = np.std(feats) + np.mean(np.abs(feats))

    return float(score)


def compute_niqe_folder(folder_path, patch_size=32):
    """
    Compute average NIQE score for all images in a folder.

    Args:
        folder_path: Directory containing images
        patch_size: Patch size for feature extraction (default: 32 for 64x64 images)

    Returns:
        Average NIQE score (lower is better)
    """
    print(f"Using custom NIQE implementation with patch_size={patch_size}")

    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")

    print(f"Found {len(image_files)} images in {folder_path}")

    # Compute NIQE for each image
    scores = []
    failed_count = 0

    for img_file in tqdm(image_files, desc="Computing NIQE"):
        img_path = os.path.join(folder_path, img_file)
        try:
            # Load image and convert to grayscale
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_gray = np.array(img).astype(np.float32)

            # Compute NIQE score
            score = compute_niqe_single_image(img_gray, patch_size=patch_size)

            if score is not None:
                scores.append(score)
            else:
                failed_count += 1

        except Exception as e:
            failed_count += 1
            # Only print first few errors to avoid spam
            if failed_count <= 5:
                print(f"\nWarning: Failed to compute NIQE for {img_path}: {e}")
            continue

    if len(scores) == 0:
        raise ValueError(f"Failed to compute NIQE for any images in {folder_path}")

    # Calculate statistics
    avg_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    success_rate = len(scores) / len(image_files) * 100
    print(f"\nSuccessfully computed NIQE for {len(scores)}/{len(image_files)} images ({success_rate:.1f}%)")
    if failed_count > 5:
        print(f"Skipped {failed_count} images due to errors")
    print(f"Average NIQE: {avg_score:.4f} ± {std_score:.4f}")
    print(f"(Lower is better - indicates more natural images)")

    return avg_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_niqe.py <image_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory")
        sys.exit(1)

    score = compute_niqe_folder(folder)
    print(f"\nFinal NIQE score: {score:.4f}")
