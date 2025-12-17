"""
CurriculumTrainer: Trainer with Curriculum Learning support

Inherits from Trainer and adds:
- Stage-based training management
- Time range control for diffusion
- Sparsity support (CS mode)
- Regularization support (CR mode)
- Logging for curriculum stages
"""

import os
import re
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .utils.train import Trainer, save_image


class CurriculumTrainer(Trainer):
    """
    Trainer with curriculum learning, sparsity, and regularization support.

    Curriculum configuration format:
    {
        "stages": [
            {"t_min": 0.8, "t_max": 1.0, "epochs": 10},
            {"t_min": 0.6, "t_max": 1.0, "epochs": 15},
            ...
        ]
    }

    Sparsity configuration (CS mode):
    {
        "enabled": true,
        "initial_sparsity": 0.8,
        "regrowth_method": "gradient"
    }

    Regularization configuration (CR mode):
    {
        "enabled": true,
        "lambda_max": 0.00003
    }
    """

    def __init__(
            self,
            curriculum_config=None,
            sparsity_config=None,
            regularization_config=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.curriculum_config = curriculum_config or {}
        self.stages = self.curriculum_config.get("stages", [])
        self.current_stage = -1
        self.current_stage_name = "default"

        # Sparsity configuration (CS mode)
        self.sparsity_config = sparsity_config or {}
        self.sparsity_enabled = self.sparsity_config.get("enabled", False)
        self.initial_sparsity = self.sparsity_config.get("initial_sparsity", 0.8)

        # Regularization configuration (CR mode)
        self.reg_config = regularization_config or {}
        self.reg_enabled = self.reg_config.get("enabled", False)
        self.lambda_max = self.reg_config.get("lambda_max", 0.00003)
        self.current_lambda = 0.0

        # Build epoch-to-stage mapping
        self._build_stage_mapping()

        # Override epochs with curriculum total if stages are defined
        if self.stages and self.curriculum_config.get("enabled", False):
            curriculum_total_epochs = sum(s.get("epochs", 10) for s in self.stages)
            if self.is_leader:
                print(f"[Curriculum] Overriding train.epochs ({self.epochs}) "
                      f"with curriculum total ({curriculum_total_epochs})")
            self.epochs = curriculum_total_epochs

        # Log mode
        if self.is_leader:
            if self.sparsity_enabled:
                print(f"[Mode] CS - Curriculum + Sparsity (initial={self.initial_sparsity:.0%})")
            elif self.reg_enabled:
                print(f"[Mode] CR - Curriculum + Regularization (lambda_max={self.lambda_max})")
            else:
                print(f"[Mode] C - Curriculum only")

    def _build_stage_mapping(self):
        """Build mapping from epoch number to stage."""
        self.epoch_to_stage = {}
        self.stage_start_epochs = []

        cumulative_epoch = 0
        for stage_idx, stage in enumerate(self.stages):
            self.stage_start_epochs.append(cumulative_epoch)
            stage_epochs = stage.get("epochs", 10)
            for e in range(stage_epochs):
                self.epoch_to_stage[cumulative_epoch + e] = stage_idx
            cumulative_epoch += stage_epochs

        # If no stages defined, default to full range
        if not self.stages:
            self.epoch_to_stage = {}

    def _get_stage_for_epoch(self, epoch):
        """Get stage index for a given epoch."""
        if not self.stages:
            return -1  # No curriculum
        return self.epoch_to_stage.get(epoch, len(self.stages) - 1)

    def _get_model(self):
        """Get underlying model (handles DDP wrapper)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def _get_sparsity_for_stage(self, stage_idx):
        """
        Get target sparsity for given stage using linear decay.

        Formula: sparsity = initial * (num_stages - 1 - stage_idx) / (num_stages - 1)
        Last stage is always 0 (all channels active).
        """
        if not self.stages:
            return 0.0
        num_stages = len(self.stages)
        if num_stages <= 1:
            return 0.0
        return self.initial_sparsity * (num_stages - 1 - stage_idx) / (num_stages - 1)

    def _get_lambda_for_stage(self, stage_idx):
        """
        Get regularization lambda for given stage using linear decay.

        Formula: lambda = lambda_max * (num_stages - 1 - stage_idx) / (num_stages - 1)
        Last stage is always 0 (no regularization).
        """
        if not self.stages:
            return 0.0
        num_stages = len(self.stages)
        if num_stages <= 1:
            return 0.0
        return self.lambda_max * (num_stages - 1 - stage_idx) / (num_stages - 1)

    def _update_curriculum(self, epoch):
        """Update diffusion time range, sparsity, and regularization based on epoch."""
        if not self.stages:
            return  # No curriculum configured

        stage_idx = self._get_stage_for_epoch(epoch)

        if stage_idx != self.current_stage:
            self.current_stage = stage_idx
            stage = self.stages[stage_idx]
            t_min = stage.get("t_min", 0.0)
            t_max = stage.get("t_max", 1.0)
            self.current_stage_name = stage.get("name", f"stage_{stage_idx + 1}")

            # Update diffusion time range
            if hasattr(self.diffusion, 'set_time_range'):
                self.diffusion.set_time_range(t_min, t_max)
                if self.is_leader:
                    t_min_idx, t_max_idx = self.diffusion.get_time_range()
                    print(f"\n[Curriculum] Stage {stage_idx + 1}/{len(self.stages)}: "
                          f"t_range=[{t_min:.2f}, {t_max:.2f}] "
                          f"(indices: [{t_min_idx}, {t_max_idx}])")

            # Update sparsity if enabled (CS mode)
            if self.sparsity_enabled:
                model = self._get_model()
                if hasattr(model, 'regrow_channels'):
                    target_sparsity = self._get_sparsity_for_stage(stage_idx)
                    num_regrown = model.regrow_channels(target_sparsity, stage_idx)
                    if self.is_leader:
                        actual_sparsity = model.get_current_sparsity()
                        print(f"[Sparsity] target={target_sparsity:.0%}, "
                              f"actual={actual_sparsity:.0%}, regrown={num_regrown}")

            # Update regularization lambda if enabled (CR mode)
            if self.reg_enabled:
                self.current_lambda = self._get_lambda_for_stage(stage_idx)
                if self.is_leader:
                    print(f"[Regularization] lambda={self.current_lambda:.6f}")

    def get_input(self, x):
        """
        Override: Sample timesteps within curriculum range.
        """
        x = x.to(self.device)

        # Use curriculum-aware timestep sampling if available
        if hasattr(self.diffusion, 'sample_timesteps'):
            t = self.diffusion.sample_timesteps(
                x.shape[0], self.device, generator=self.generator
            )
        else:
            # Fallback to standard sampling
            t = torch.empty(
                (x.shape[0],), dtype=torch.int64, device=self.device
            ).random_(to=self.timesteps, generator=self.generator)

        return {
            "x_0": x,
            "t": t,
            "noise": torch.empty_like(x).normal_(generator=self.generator)
        }

    def loss(self, x):
        """
        Override: Compute loss with optional regularization penalty (CR mode).

        Returns:
            tuple: (total_loss, mse_loss, reg_loss) where losses are per-sample tensors.
                   For non-CR mode, reg_loss is 0.
        """
        mse_loss = self.diffusion.train_losses(self.model, **self.get_input(x))
        assert mse_loss.shape == (x.shape[0],)

        # Add regularization penalty for CR mode
        reg_loss = torch.zeros(1, device=self.device)
        if self.reg_enabled and self.current_lambda > 0:
            model = self._get_model()
            if hasattr(model, 'get_group_l1_penalty'):
                reg_loss = model.get_group_l1_penalty(self.current_lambda)

        # Total loss = MSE + reg_penalty (distributed across batch)
        total_loss = mse_loss + reg_loss / x.shape[0]

        return total_loss, mse_loss, reg_loss

    def step(self, x, global_steps=1):
        """
        Override: Training step with gradient accumulation for sparsity (CS mode).
        Tracks MSE and reg losses separately for CR mode.
        """
        total_loss, mse_loss, reg_loss = self.loss(x)
        loss = total_loss.mean()
        loss.div(self.num_accum).backward()

        # Accumulate gradients for sparse channel regrowth (CS mode)
        if self.sparsity_enabled:
            model = self._get_model()
            if hasattr(model, 'accumulate_gradients'):
                model.accumulate_gradients()

        if global_steps % self.num_accum == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            if self.use_ema and hasattr(self.ema, "update"):
                self.ema.update()

        # Detach losses for logging
        loss = loss.detach()
        mse_loss_mean = mse_loss.mean().detach()
        reg_loss_val = reg_loss.detach()

        if self.distributed:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(mse_loss_mean, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(reg_loss_val, dst=0, op=dist.ReduceOp.SUM)
            loss.div_(self.world_size)
            mse_loss_mean.div_(self.world_size)
            reg_loss_val.div_(self.world_size)

        # Update stats - for CR mode, show separate mse and reg
        if self.reg_enabled:
            self.stats.update(
                x.shape[0],
                loss=loss.item() * x.shape[0],
                mse=mse_loss_mean.item() * x.shape[0],
                reg=reg_loss_val.item() * x.shape[0]
            )
        else:
            self.stats.update(x.shape[0], loss=loss.item() * x.shape[0])

    def train(self, evaluator=None, chkpt_path=None, image_dir=None):
        """
        Training loop with curriculum support.
        """
        nrow = math.floor(math.sqrt(self.num_samples))
        if self.num_samples:
            assert self.num_samples % self.world_size == 0, \
                "Number of samples should be divisible by WORLD_SIZE!"

        if self.dry_run:
            self.start_epoch, self.epochs = 0, 1

        global_steps = 0

        for e in range(self.start_epoch, self.epochs):
            # Update curriculum stage at the start of each epoch
            self._update_curriculum(e)

            self.stats.reset()
            self.model.train()
            results = dict()

            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)

            # Progress bar description includes stage info
            if self.current_stage >= 0:
                desc = f"{e + 1}/{self.epochs} epochs [{self.current_stage_name}]"
            else:
                desc = f"{e + 1}/{self.epochs} epochs"

            with tqdm(self.trainloader, desc=desc, disable=not self.is_leader) as t:
                for i, x in enumerate(t):
                    if isinstance(x, (list, tuple)):
                        x = x[0]  # unconditional model -> discard labels
                    global_steps += 1
                    self.step(x.to(self.device), global_steps=global_steps)
                    t.set_postfix(self.current_stats)
                    results.update(self.current_stats)
                    if self.dry_run and not global_steps % self.num_accum:
                        break

            if not (e + 1) % self.image_intv and self.num_samples and image_dir:
                self.model.eval()
                x = self.sample_fn(
                    sample_size=self.num_samples, sample_seed=self.sample_seed
                ).cpu()
                if self.is_leader:
                    # Include stage info in filename: s{stage}_{epoch}.jpg
                    stage_num = self.current_stage + 1 if self.current_stage >= 0 else 0
                    filename = f"s{stage_num}_{e + 1}.jpg"
                    save_image(x, os.path.join(image_dir, filename), nrow=nrow)

            if not (e + 1) % self.chkpt_intv and chkpt_path:
                self.model.eval()
                if evaluator is not None:
                    eval_results = evaluator.eval(self.sample_fn, is_leader=self.is_leader)
                else:
                    eval_results = dict()
                results.update(eval_results)
                # Add curriculum info to checkpoint
                results["curriculum_stage"] = self.current_stage
                if self.is_leader:
                    self.save_checkpoint(chkpt_path, epoch=e + 1, **results)

            if self.distributed:
                dist.barrier()  # synchronize all processes here

    def save_checkpoint(self, chkpt_path, **extra_info):
        """Override to include curriculum state."""
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))

        # Add curriculum state
        chkpt.append(("curriculum_stage", self.current_stage))
        chkpt.append(("curriculum_config", self.curriculum_config))

        for k, v in extra_info.items():
            chkpt.append((k, v))

        if "epoch" in extra_info:
            chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", chkpt_path)
        torch.save(dict(chkpt), chkpt_path)

    def load_checkpoint(self, chkpt_path, map_location):
        """Override to restore curriculum state."""
        super().load_checkpoint(chkpt_path, map_location)

        chkpt = torch.load(chkpt_path, map_location=map_location)
        if "curriculum_stage" in chkpt:
            self.current_stage = chkpt["curriculum_stage"]
            if self.is_leader:
                print(f"[Curriculum] Resumed at stage {self.current_stage + 1}")
