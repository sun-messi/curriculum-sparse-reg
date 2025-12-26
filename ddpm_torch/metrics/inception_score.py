"""
Inception Score (IS) implementation.

IS = exp(E[KL(p(y|x) || p(y))])

Higher IS indicates better image quality and diversity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision
import os
from PIL import Image
from torchvision import transforms


class InceptionScoreModel(nn.Module):
    """InceptionV3 model for computing Inception Score.

    Unlike InceptionV3 for FID (which outputs 2048-dim features),
    this outputs 1000-class softmax probabilities.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        # Use torchvision's pretrained InceptionV3 with classification head
        self.model = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False  # We'll handle normalization ourselves
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, x):
        """
        Args:
            x: Images tensor (N, 3, H, W) in range [0, 1] or [-1, 1]

        Returns:
            Softmax probabilities (N, 1000)
        """
        # Resize to 299x299
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize: InceptionV3 expects input in [-1, 1]
        # If input is in [0, 1], convert to [-1, 1]
        if x.min() >= 0:
            x = 2 * x - 1

        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        return probs


def compute_inception_score(images, device="cuda:0", batch_size=64, splits=10):
    """
    Compute Inception Score for a set of images.

    IS = exp(E[KL(p(y|x) || p(y))])

    Args:
        images: Either a tensor (N, C, H, W) or a DataLoader
        device: Device for computation
        batch_size: Batch size for processing (if images is tensor)
        splits: Number of splits for computing mean and std

    Returns:
        (mean, std): Inception Score mean and standard deviation
    """
    model = InceptionScoreModel(device=device)

    # Collect all softmax predictions
    all_probs = []

    if isinstance(images, DataLoader):
        for batch in tqdm(images, desc="Computing IS"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Handle (image, label) pairs
            batch = batch.to(device)
            # Normalize from [0, 255] uint8 to [0, 1] if needed
            if batch.dtype == torch.uint8:
                batch = batch.float() / 255.0
            probs = model(batch)
            all_probs.append(probs.cpu().numpy())
    else:
        # Tensor input
        images = images.to(device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        num_batches = (len(images) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Computing IS"):
            start = i * batch_size
            end = min(start + batch_size, len(images))
            batch = images[start:end]
            probs = model(batch)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Compute IS using splits
    scores = []
    N = len(all_probs)
    split_size = N // splits

    for i in range(splits):
        start = i * split_size
        end = start + split_size if i < splits - 1 else N
        part = all_probs[start:end]

        # p(y|x) for each image
        py_given_x = part

        # p(y) = average of p(y|x) across all images in this split
        py = np.mean(py_given_x, axis=0, keepdims=True)

        # KL divergence: sum(p(y|x) * log(p(y|x) / p(y)))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        kl = py_given_x * (np.log(py_given_x + eps) - np.log(py + eps))
        kl = np.sum(kl, axis=1)  # Sum over classes

        # IS for this split = exp(mean KL)
        is_score = np.exp(np.mean(kl))
        scores.append(is_score)

    return float(np.mean(scores)), float(np.std(scores))


class ImageFolder(Dataset):
    """Dataset for loading images from a folder."""

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = [
            img for img in os.listdir(img_dir)
            if img.split(".")[-1].lower() in {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
        ]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1]
        ])

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.img_dir, self.img_list[idx])) as im:
            return self.transform(im.convert('RGB'))

    def __len__(self):
        return len(self.img_list)


def compute_is_from_folder(sample_dir, device="cuda:0", batch_size=64, splits=10):
    """
    Compute Inception Score from a folder of generated images.

    Args:
        sample_dir: Directory containing generated images
        device: Device for computation
        batch_size: Batch size
        splits: Number of splits

    Returns:
        (mean, std): IS mean and std
    """
    dataset = ImageFolder(sample_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )

    return compute_inception_score(dataloader, device=device, splits=splits)
