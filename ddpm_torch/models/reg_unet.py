"""
RegUNet: UNet with Group L1 regularization support.

Implements soft sparsity through Group L1 penalty on conv layers
for Curriculum + Regularization (CR) training mode.
"""

import torch
import torch.nn as nn

from .unet import UNet

# Import custom Conv2d from modules (used in UNet instead of nn.Conv2d)
try:
    from ..modules import Conv2d as CustomConv2d
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ddpm_torch.modules import Conv2d as CustomConv2d


class RegUNet(UNet):
    """
    UNet with Group L1 regularization support.

    Provides methods to compute Group L1 penalty on Conv layers from the
    last ResBlock of the last downsample level (before bottleneck),
    which encourages structured sparsity (entire output channels
    becoming zero) in the deepest feature layers.

    Group L1 formula:
        L_reg = lambda * sum_layers sum_c ||W[c,:,:,:]||_2

    Where W[c,:,:,:] is the weight for output channel c, and we
    compute the L2 norm of each channel's weights then sum them.

    Note: Regularization is only applied to the last ResBlock of
    downsamples.level_N-1 (the ResBlock directly feeding into bottleneck).

    Args:
        **unet_kwargs: Arguments passed to parent UNet class
    """

    def __init__(self, **unet_kwargs):
        super().__init__(**unet_kwargs)

        # Cache conv layers for efficiency
        self._conv_layers = None
        self._bottleneck_adjacent_conv_layers = None

    def get_conv_layers(self):
        """
        Find all Conv2d and ConvTranspose2d layers in the model.

        Returns:
            List of (name, module) tuples for all conv layers
        """
        if self._conv_layers is not None:
            return self._conv_layers

        conv_layers = []
        for name, module in self.named_modules():
            # Check for custom Conv2d (from ddpm_torch.modules) and standard nn.Conv2d
            if isinstance(module, (CustomConv2d, nn.Conv2d, nn.ConvTranspose2d)):
                conv_layers.append((name, module))

        self._conv_layers = conv_layers
        return conv_layers

    def get_bottleneck_adjacent_conv_layers(self):
        """
        Get only the conv layers from the last ResBlock of the last downsample level.

        This ResBlock's output directly feeds into the bottleneck/middle block.

        Returns:
            List of (name, module) tuples for conv layers
        """
        if self._bottleneck_adjacent_conv_layers is not None:
            return self._bottleneck_adjacent_conv_layers

        last_level = self.levels - 1
        conv_layers = []

        # Last ResBlock of downsample (输出直接进bottleneck)
        downsample_module = self.downsamples[f"level_{last_level}"]
        last_res_idx = self.num_res_blocks - 1
        last_resblock = downsample_module[last_res_idx]

        for name, module in last_resblock.named_modules():
            if isinstance(module, (CustomConv2d, nn.Conv2d, nn.ConvTranspose2d)):
                full_name = f"downsamples.level_{last_level}.{last_res_idx}.{name}" if name else f"downsamples.level_{last_level}.{last_res_idx}"
                conv_layers.append((full_name, module))

        self._bottleneck_adjacent_conv_layers = conv_layers
        return conv_layers

    def get_group_l1_penalty(self, lambda_val):
        """
        Compute Group L1 regularization penalty.

        This penalty encourages entire output channels to become zero,
        leading to structured sparsity that can be pruned.

        Formula: L_reg = lambda * sum_layers sum_c ||W[c,:,:,:]||_2

        Note: Only applied to the last ResBlock of the last downsample level
        (the ResBlock directly feeding into bottleneck) for targeted
        regularization of high-level features.

        Args:
            lambda_val: Regularization strength coefficient

        Returns:
            Scalar tensor with penalty value
        """
        device = next(self.parameters()).device

        if lambda_val == 0:
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=device)
        conv_layers = self.get_bottleneck_adjacent_conv_layers()

        for name, layer in conv_layers:
            weight = layer.weight
            # weight shape: (out_channels, in_channels, kH, kW)
            # Compute L2 norm per output channel
            weight_flat = weight.view(weight.size(0), -1)
            channel_norms = torch.norm(weight_flat, p=2, dim=1)
            # Sum of L2 norms (Group L1)
            penalty = penalty + channel_norms.sum()

        return lambda_val * penalty

    def get_channel_norms(self):
        """
        Get L2 norms for each output channel in all conv layers.

        Returns:
            Dict mapping layer name to tensor of channel norms
        """
        norms = {}
        conv_layers = self.get_conv_layers()

        for name, layer in conv_layers:
            weight = layer.weight
            weight_flat = weight.view(weight.size(0), -1)
            channel_norms = torch.norm(weight_flat, p=2, dim=1)
            norms[name] = channel_norms.detach()

        return norms

    def get_channel_sparsity_stats(self, threshold=1e-4):
        """
        Analyze channel sparsity induced by regularization.

        Channels with L2 norm below threshold are considered sparse/inactive.

        Args:
            threshold: Norm threshold below which channels are sparse

        Returns:
            Dict with sparsity statistics:
            - total_channels: Total number of output channels
            - sparse_channels: Number of channels with norm < threshold
            - overall_sparsity: Fraction of sparse channels
            - layer_stats: Per-layer statistics
        """
        conv_layers = self.get_conv_layers()
        total_channels = 0
        sparse_channels = 0

        layer_stats = {}
        for name, layer in conv_layers:
            weight = layer.weight
            weight_flat = weight.view(weight.size(0), -1)
            channel_norms = torch.norm(weight_flat, p=2, dim=1)

            num_sparse = (channel_norms < threshold).sum().item()
            total = weight.size(0)

            layer_stats[name] = {
                "total": total,
                "sparse": num_sparse,
                "sparsity": num_sparse / total if total > 0 else 0
            }

            total_channels += total
            sparse_channels += num_sparse

        return {
            "total_channels": total_channels,
            "sparse_channels": sparse_channels,
            "overall_sparsity": sparse_channels / total_channels if total_channels > 0 else 0,
            "layer_stats": layer_stats
        }

    def get_norm_statistics(self):
        """
        Get statistical summary of channel norms across all conv layers.

        Returns:
            Dict with norm statistics:
            - mean: Mean of all channel norms
            - std: Standard deviation
            - min: Minimum norm
            - max: Maximum norm
        """
        all_norms = []
        conv_layers = self.get_conv_layers()

        for name, layer in conv_layers:
            weight = layer.weight
            weight_flat = weight.view(weight.size(0), -1)
            channel_norms = torch.norm(weight_flat, p=2, dim=1)
            all_norms.append(channel_norms)

        if not all_norms:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        all_norms = torch.cat(all_norms)
        return {
            "mean": all_norms.mean().item(),
            "std": all_norms.std().item(),
            "min": all_norms.min().item(),
            "max": all_norms.max().item()
        }

    def print_reg_info(self, threshold=1e-4):
        """
        Print regularization and sparsity information.

        Args:
            threshold: Norm threshold for sparsity analysis
        """
        stats = self.get_channel_sparsity_stats(threshold)
        norm_stats = self.get_norm_statistics()

        print(f"[RegUNet] Overall sparsity: {stats['overall_sparsity']:.1%} "
              f"({stats['sparse_channels']}/{stats['total_channels']} channels < {threshold})")
        print(f"[RegUNet] Norm stats: mean={norm_stats['mean']:.4f}, "
              f"std={norm_stats['std']:.4f}, "
              f"min={norm_stats['min']:.6f}, max={norm_stats['max']:.4f}")
