"""
SparseUNet: UNet with channel-level sparsity at bottleneck.

Implements hard channel masking with gradient-based regrowth for
Curriculum + Sparsity (CS) training mode.
"""

import torch
import torch.nn as nn

from .unet import UNet
try:
    from ..functions import get_timestep_embedding
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ddpm_torch.functions import get_timestep_embedding


class SparseUNet(UNet):
    """
    UNet with channel-level sparsity at the bottleneck.

    Channel masking is applied after the middle (bottleneck) block.
    Supports gradient-based or random channel regrowth during
    curriculum learning stages.

    Args:
        initial_sparsity: Initial fraction of channels to mask (0.0 to 1.0)
        regrowth_method: Method for selecting channels to regrow
            - "gradient": Select channels with highest accumulated gradient
            - "random": Randomly select channels
        **unet_kwargs: Arguments passed to parent UNet class
    """

    def __init__(
            self,
            initial_sparsity=0.8,
            regrowth_method="gradient",
            **unet_kwargs
    ):
        super().__init__(**unet_kwargs)

        # Bottleneck channels = ch_multipliers[-1] * hid_channels
        self.bottleneck_channels = self.ch_multipliers[-1] * self.hid_channels

        # Sparsity configuration
        self.initial_sparsity = initial_sparsity
        self.regrowth_method = regrowth_method

        # Channel mask: 1 = active, 0 = masked
        self.register_buffer(
            "channel_mask",
            torch.ones(self.bottleneck_channels)
        )

        # Track which stage each channel was activated
        self.register_buffer(
            "channel_birth_stage",
            torch.zeros(self.bottleneck_channels, dtype=torch.long)
        )

        # Accumulate gradients for regrowth decisions
        self.register_buffer(
            "channel_grad_accum",
            torch.zeros(self.bottleneck_channels)
        )

        # Initialize mask based on initial sparsity
        self._initialize_mask(initial_sparsity)

    def _initialize_mask(self, sparsity):
        """
        Initialize channel mask with given sparsity.

        Args:
            sparsity: Fraction of channels to mask (0.0 to 1.0)
        """
        num_channels = self.bottleneck_channels
        num_active = int(num_channels * (1 - sparsity))
        num_active = max(1, num_active)  # At least 1 channel

        # Randomly select channels to activate
        indices = torch.randperm(num_channels)[:num_active]
        self.channel_mask.zero_()
        self.channel_mask[indices] = 1.0

        # Mark all initially active channels as born in stage 0
        self.channel_birth_stage.zero_()

    def forward(self, x, t):
        """
        Forward pass with channel masking at bottleneck.

        The mask is applied after the middle (bottleneck) block,
        zeroing out masked channel activations.
        """
        t_emb = get_timestep_embedding(t, self.hid_channels)
        t_emb = self.embed(t_emb)

        # Downsample
        hs = [self.in_conv(x)]
        for i in range(self.levels):
            downsample = self.downsamples[f"level_{i}"]
            for j, layer in enumerate(downsample):
                h = hs[-1]
                if j != self.num_res_blocks:
                    hs.append(layer(h, t_emb=t_emb))
                else:
                    hs.append(layer(h))

        # Middle (bottleneck)
        h = self.middle(hs[-1], t_emb=t_emb)

        # Apply channel mask at bottleneck
        # h shape: (B, C, H, W), channel_mask shape: (C,)
        h = h * self.channel_mask.view(1, -1, 1, 1)

        # Upsample
        for i in range(self.levels - 1, -1, -1):
            upsample = self.upsamples[f"level_{i}"]
            for j, layer in enumerate(upsample):
                if j != self.num_res_blocks + 1:
                    h = layer(torch.cat([h, hs.pop()], dim=1), t_emb=t_emb)
                else:
                    h = layer(h)

        h = self.out_conv(h)
        return h

    def accumulate_gradients(self):
        """
        Accumulate gradient magnitudes for masked channels.

        Should be called after loss.backward() during training.
        The accumulated gradients are used for gradient-based regrowth.
        """
        if self.regrowth_method != "gradient":
            return

        # Get gradients from the first ResidualBlock in middle
        # middle[0] is the first ResidualBlock
        first_res_block = self.middle[0]
        if hasattr(first_res_block, 'conv2') and first_res_block.conv2.weight.grad is not None:
            grad = first_res_block.conv2.weight.grad
            # Sum absolute gradients per output channel
            # grad shape: (out_channels, in_channels, kH, kW)
            grad_magnitude = grad.abs().sum(dim=(1, 2, 3))

            # Accumulate only for masked (inactive) channels
            masked_indices = (self.channel_mask == 0)
            self.channel_grad_accum[masked_indices] += grad_magnitude[masked_indices]

    def regrow_channels(self, target_sparsity, current_stage):
        """
        Regrow channels to reach target sparsity.

        Selects channels to activate based on regrowth_method:
        - "gradient": Channels with highest accumulated gradient
        - "random": Random selection

        Args:
            target_sparsity: Target sparsity level (0.0 to 1.0)
            current_stage: Current curriculum stage index

        Returns:
            Number of channels actually regrown
        """
        num_channels = self.bottleneck_channels
        target_active = int(num_channels * (1 - target_sparsity))
        target_active = max(1, target_active)

        current_active = int(self.channel_mask.sum().item())
        num_to_regrow = target_active - current_active

        if num_to_regrow <= 0:
            return 0  # No regrowth needed

        # Find masked (inactive) channels
        masked_indices = torch.where(self.channel_mask == 0)[0]

        if len(masked_indices) == 0:
            return 0  # All channels already active

        # Limit regrowth to available channels
        num_to_regrow = min(num_to_regrow, len(masked_indices))

        if self.regrowth_method == "gradient":
            # Select channels with highest accumulated gradients
            grad_scores = self.channel_grad_accum[masked_indices]
            _, top_indices = torch.topk(
                grad_scores,
                min(num_to_regrow, len(masked_indices))
            )
            regrow_indices = masked_indices[top_indices]
        else:  # random
            perm = torch.randperm(len(masked_indices), device=masked_indices.device)
            regrow_indices = masked_indices[perm[:num_to_regrow]]

        # Activate selected channels
        self.channel_mask[regrow_indices] = 1.0
        self.channel_birth_stage[regrow_indices] = current_stage

        # Reset gradient accumulator for next stage
        self.channel_grad_accum.zero_()

        return len(regrow_indices)

    def get_current_sparsity(self):
        """Return current sparsity ratio (fraction of masked channels)."""
        num_active = self.channel_mask.sum().item()
        return 1.0 - (num_active / self.bottleneck_channels)

    def get_active_channels(self):
        """Return number of active (unmasked) channels."""
        return int(self.channel_mask.sum().item())

    def get_channel_stats_by_stage(self):
        """
        Get statistics of channels grouped by birth stage.

        Returns:
            Dict mapping stage index to number of channels born in that stage
        """
        active_indices = torch.where(self.channel_mask == 1)[0]
        stages = self.channel_birth_stage[active_indices]

        stats = {}
        for stage in stages.unique().tolist():
            count = (stages == stage).sum().item()
            stats[stage] = count
        return stats

    def print_sparsity_info(self):
        """Print detailed sparsity information."""
        sparsity = self.get_current_sparsity()
        active = self.get_active_channels()
        total = self.bottleneck_channels
        stage_stats = self.get_channel_stats_by_stage()

        print(f"[SparseUNet] Sparsity: {sparsity:.1%} "
              f"({active}/{total} channels active)")
        print(f"[SparseUNet] Channels by birth stage: {stage_stats}")
