"""
Curriculum Learning support for DDPM

Reference: minDiffusion_curriculum project
"""

import torch
from .diffusion import GaussianDiffusion


class CurriculumDiffusion(GaussianDiffusion):
    """
    GaussianDiffusion with curriculum learning support.

    Curriculum strategy:
    - Start training only on high noise (t near timesteps)
    - Gradually expand to low noise (t near 0)
    - Use set_time_range() to dynamically adjust training range
    """

    def __init__(self, betas, model_mean_type, model_var_type, loss_type, **kwargs):
        super().__init__(betas, model_mean_type, model_var_type, loss_type, **kwargs)

        # Time range as indices (0 to timesteps)
        self.t_min = 0
        self.t_max = self.timesteps

    def set_time_range(self, t_min_ratio: float, t_max_ratio: float):
        """
        Set training time range using ratios (0.0 to 1.0).

        Args:
            t_min_ratio: Minimum time ratio (0.0 = clean image, 1.0 = pure noise)
            t_max_ratio: Maximum time ratio

        Examples:
            set_time_range(0.8, 1.0)  # Stage 1: high noise only
            set_time_range(0.5, 1.0)  # Mid stage: medium range
            set_time_range(0.0, 1.0)  # Final: full range
        """
        # Clamp ratios to [0, 1]
        t_min_ratio = max(0.0, min(t_min_ratio, 1.0))
        t_max_ratio = max(0.0, min(t_max_ratio, 1.0))

        # Convert to indices
        self.t_min = int(t_min_ratio * self.timesteps)
        self.t_max = int(t_max_ratio * self.timesteps)

        # Ensure valid range (at least 1 step)
        if self.t_min >= self.t_max:
            self.t_min = max(0, self.t_max - 1)

    def get_time_range(self):
        """Get current time range as indices."""
        return (self.t_min, self.t_max)

    def get_time_range_ratio(self):
        """Get current time range as ratios."""
        return (self.t_min / self.timesteps, self.t_max / self.timesteps)

    def sample_timesteps(self, batch_size, device, generator=None):
        """
        Sample timesteps within curriculum range.

        Returns:
            Tensor of shape (batch_size,) with timesteps in [t_min, t_max)
        """
        t_min = self.t_min
        t_max = max(t_min + 1, self.t_max)  # Ensure at least 1 step

        return torch.randint(
            t_min, t_max, (batch_size,),
            device=device, dtype=torch.int64,
            generator=generator
        )
