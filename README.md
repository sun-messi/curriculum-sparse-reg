<p align="center"><img alt="banner" src="./assets/banner.webp" width="100%"></p>

---

# DDPM-Torch with Curriculum Learning & Sparsity

PyTorch Implementation of Denoising Diffusion Probabilistic Models with **Curriculum Learning**, **Sparsity**, and **Regularization** extensions.

Based on [[DDPM paper]](https://arxiv.org/abs/2006.11239) [[official repo]](https://github.com/hojonathanho/diffusion)

## Features

- [x] Original DDPM training & sampling
- [x] DDIM sampler
- [x] Standard evaluation metrics (FID, Precision & Recall)
- [x] Distributed Data Parallel (DDP) multi-GPU training
- [x] **Curriculum Learning (C)** - Progressive training from high noise to low noise
- [x] **Curriculum + Sparsity (CS)** - Hard mask with channel regrowth
- [x] **Curriculum + Regularization (CR)** - Group L1 soft sparsity

---

## Quick Start

### Training Commands

```bash
# ============================================================
# Mode C: Curriculum Only
# ============================================================
# Single GPU
python train_curriculum.py --config-path configs/celeba32_c.json

# Multi-GPU (6 GPUs)
python train_curriculum.py --config-path configs/celeba32_c.json \
    --num-gpus 6 --distributed --rigid-launch

# Select specific GPUs (0,1,3,5)
CUDA_VISIBLE_DEVICES=0,1,3,5 python train_curriculum.py \
    --config-path configs/celeba32_c.json \
    --num-gpus 4 --distributed --rigid-launch

# Using torchrun (recommended for elastic launch)
torchrun --standalone --nproc_per_node 4 train_curriculum.py \
    --config-path configs/celeba32_c.json --distributed

# ============================================================
# Mode CS: Curriculum + Sparsity
# ============================================================
python train_curriculum.py --config-path configs/celeba32_cs.json \
    --num-gpus 6 --distributed --rigid-launch

# ============================================================
# Mode CR: Curriculum + Regularization
# ============================================================
python train_curriculum.py --config-path configs/celeba32_cr.json \
    --num-gpus 6 --distributed --rigid-launch
```

---

## Curriculum Learning Extensions

### Core Idea

Standard diffusion training treats all timesteps equally, but this ignores an important structure:

| Timestep | Noise Level | Learnable Content |
|----------|-------------|-------------------|
| t ≈ 1000 | Very high | Only **major features** (coarse structure) |
| t ≈ 500 | Medium | Medium features |
| t ≈ 1 | Very low | **Fine details** (minor features) |

**Key Insight**: At high noise levels, fine details are completely masked. Training on high-noise samples first forces the network to focus on major features.

### Three Training Modes

| Config File | Mode | Model | Description |
|-------------|------|-------|-------------|
| `celeba32_c.json` | C | `UNet` | Curriculum only |
| `celeba32_cs.json` | CS | `SparseUNet` | Curriculum + Hard mask at bottleneck |
| `celeba32_cr.json` | CR | `RegUNet` | Curriculum + Group L1 regularization |

### Mode Selection (via JSON config)

```json
// Mode C: Both disabled
"sparsity": { "enabled": false },
"regularization": { "enabled": false }

// Mode CS: Sparsity enabled
"sparsity": { "enabled": true, "initial_sparsity": 0.8 }

// Mode CR: Regularization enabled
"regularization": { "enabled": true, "lambda_max": 0.00003 }
```

---

## Architecture

### CurriculumDiffusion

Extends `GaussianDiffusion` with dynamic timestep range control.

```python
from ddpm_torch.curriculum import CurriculumDiffusion

# Create diffusion with curriculum support
diffusion = CurriculumDiffusion(betas=betas, **config)

# Stage 1: High noise only (t ∈ [800, 1000])
diffusion.set_time_range(0.8, 1.0)

# Stage 5: Full range (t ∈ [0, 1000])
diffusion.set_time_range(0.0, 1.0)
```

### Curriculum Stages

```
Stage 1: t ∈ [0.8, 1.0] → High noise - Learn coarse structure
Stage 2: t ∈ [0.6, 1.0] → Expand range
Stage 3: t ∈ [0.4, 1.0] → Continue expanding
Stage 4: t ∈ [0.2, 1.0] → Near full range
Stage 5: t ∈ [0.0, 1.0] → Full range - Learn fine details
```

### Sparsity + Curriculum Synergy

| Component | Controls | Purpose |
|-----------|----------|---------|
| **Curriculum** | Data difficulty (noise: high → low) | **What** to learn (major → minor features) |
| **Sparsity** | Model capacity (sparse → dense) | **Who** learns (reserve capacity for minor features) |

**Problem**: If all neurons are active from the start, they all get "contaminated" by major features, making it harder to learn fine details later.

**Solution**: Use sparsity to **reserve clean model capacity** for later-stage features.

---

## CS Mode: Curriculum + Sparsity

### SparseUNet Implementation

Channel-level sparsity at the UNet bottleneck with gradient-based regrowth.

**File**: `ddpm_torch/models/sparse_unet.py`

```python
class SparseUNet(UNet):
    """UNet with channel-level sparsity at the bottleneck."""

    def __init__(self, initial_sparsity=0.8, regrowth_method="gradient", **unet_kwargs):
        super().__init__(**unet_kwargs)

        # Bottleneck channels = ch_multipliers[-1] * hid_channels
        # For celeba32: 2 * 64 = 128 channels
        self.bottleneck_channels = self.ch_multipliers[-1] * self.hid_channels

        # Registered buffers (saved with model state)
        self.register_buffer("channel_mask", torch.ones(self.bottleneck_channels))
        self.register_buffer("channel_birth_stage", torch.zeros(...))
        self.register_buffer("channel_grad_accum", torch.zeros(...))

    def forward(self, x, t):
        # ... downsample ...

        # Middle (bottleneck)
        h = self.middle(hs[-1], t_emb=t_emb)

        # KEY: Apply channel mask at bottleneck
        h = h * self.channel_mask.view(1, -1, 1, 1)

        # ... upsample ...
        return h

    def accumulate_gradients(self):
        """Called after backward() to track gradient for regrowth."""
        # Accumulate gradient magnitudes for masked (inactive) channels
        grad = first_res_block.conv2.weight.grad
        grad_magnitude = grad.abs().sum(dim=(1, 2, 3))
        masked_indices = (self.channel_mask == 0)
        self.channel_grad_accum[masked_indices] += grad_magnitude[masked_indices]

    def regrow_channels(self, target_sparsity, current_stage):
        """Activate channels based on accumulated gradients."""
        # Select channels with highest gradient magnitude
        # Mark them as active (channel_mask = 1)
        # Record birth stage for analysis
```

### Sparsity Schedule (Linear Decay)

```python
# Formula: sparsity = initial * (num_stages - 1 - stage_idx) / (num_stages - 1)
# With initial_sparsity=0.8 and 5 stages:

Stage 0: 0.8 * (5-1-0)/(5-1) = 0.8 * 4/4 = 80%  (25/128 active)
Stage 1: 0.8 * (5-1-1)/(5-1) = 0.8 * 3/4 = 60%  (51/128 active)
Stage 2: 0.8 * (5-1-2)/(5-1) = 0.8 * 2/4 = 40%  (77/128 active)
Stage 3: 0.8 * (5-1-3)/(5-1) = 0.8 * 1/4 = 20%  (102/128 active)
Stage 4: 0.8 * (5-1-4)/(5-1) = 0.8 * 0/4 = 0%   (128/128 active)
```

### Regrowth Methods

| Method | Selection Criteria |
|--------|-------------------|
| `gradient` | Channels with highest accumulated gradient magnitude |
| `random` | Random selection from masked channels |

---

## CR Mode: Curriculum + Regularization

### RegUNet Implementation

Soft sparsity via Group L1 regularization on all Conv layers.

**File**: `ddpm_torch/models/reg_unet.py`

```python
class RegUNet(UNet):
    """UNet with Group L1 regularization support."""

    def get_conv_layers(self):
        """Find all Conv2d layers (including custom Conv2d from modules.py)."""
        conv_layers = []
        for name, module in self.named_modules():
            if isinstance(module, (CustomConv2d, nn.Conv2d, nn.ConvTranspose2d)):
                conv_layers.append((name, module))
        return conv_layers  # Returns 52 layers for celeba32 config

    def get_group_l1_penalty(self, lambda_val):
        """
        Compute Group L1 regularization penalty.

        Formula: L_reg = λ × Σ_layers Σ_c ||W[c,:,:,:]||_2

        Where W[c,:,:,:] is the weight for output channel c.
        """
        if lambda_val == 0:
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=device)
        for name, layer in self.get_conv_layers():
            weight = layer.weight  # (out_channels, in_channels, kH, kW)
            weight_flat = weight.view(weight.size(0), -1)
            channel_norms = torch.norm(weight_flat, p=2, dim=1)
            penalty = penalty + channel_norms.sum()

        return lambda_val * penalty
```

### Lambda Schedule (Linear Decay)

```python
# Formula: lambda = lambda_max * (num_stages - 1 - stage_idx) / (num_stages - 1)
# With lambda_max=0.00003 and 5 stages:

Stage 0: λ = 0.000030  (strong regularization → drive sparsity)
Stage 1: λ = 0.000023
Stage 2: λ = 0.000015
Stage 3: λ = 0.000008
Stage 4: λ = 0.000000  (no regularization → release capacity)
```

### Loss Computation (CR Mode)

```python
def loss(self, x):
    mse_loss = self.diffusion.train_losses(self.model, **self.get_input(x))

    # Add regularization penalty for CR mode
    reg_loss = torch.zeros(1, device=self.device)
    if self.reg_enabled and self.current_lambda > 0:
        reg_loss = model.get_group_l1_penalty(self.current_lambda)

    total_loss = mse_loss + reg_loss / x.shape[0]
    return total_loss, mse_loss, reg_loss
```

**CR Mode Training Output** shows separate losses:
```
1/10 epochs [stage1_high_noise]: loss=0.0234, mse=0.0210, reg=0.0024
```

---

## CurriculumTrainer

### Key Modifications from Base Trainer

**File**: `ddpm_torch/curriculum_trainer.py`

```python
class CurriculumTrainer(Trainer):
    def __init__(
        self,
        curriculum_config=None,
        sparsity_config=None,      # CS mode
        regularization_config=None, # CR mode
        **kwargs
    ):
        # Sparsity configuration
        self.sparsity_enabled = sparsity_config.get("enabled", False)
        self.initial_sparsity = sparsity_config.get("initial_sparsity", 0.8)

        # Regularization configuration
        self.reg_enabled = reg_config.get("enabled", False)
        self.lambda_max = reg_config.get("lambda_max", 0.00003)
        self.current_lambda = 0.0

    def _update_curriculum(self, epoch):
        """Called at start of each epoch to update stage if needed."""
        # Update diffusion time range
        self.diffusion.set_time_range(t_min_idx, t_max_idx)

        # CS mode: Regrow channels
        if self.sparsity_enabled:
            target_sparsity = self._get_sparsity_for_stage(new_stage)
            model.regrow_channels(target_sparsity, new_stage)

        # CR mode: Update lambda
        if self.reg_enabled:
            self.current_lambda = self._get_lambda_for_stage(new_stage)

    def step(self, x, global_steps=1):
        """Training step with sparsity gradient accumulation."""
        loss = self.loss(x).mean()
        loss.backward()

        # CS mode: Accumulate gradients for regrowth
        if self.sparsity_enabled:
            model.accumulate_gradients()

        # ... optimizer step ...
```

---

## CS vs CR Comparison

| Feature | CS (Hard Mask) | CR (Group L1) |
|---------|----------------|---------------|
| **Mechanism** | `channel_mask` binary mask | Regularization penalty in loss |
| **Location** | Bottleneck only (128 channels) | All Conv layers (52 layers, 5827 channels) |
| **Control** | Explicit regrowth at stage transition | λ decay gradually releases capacity |
| **Flexibility** | Discrete (0 or 1) | Continuous (soft sparsity) |
| **Overhead** | Minimal (mask multiplication) | Moderate (norm computation) |

---

## Configuration Reference

### Config Files

**32x32 Resolution:**

| File | Mode | sparsity.enabled | regularization.enabled |
|------|------|:----------------:|:----------------------:|
| `celeba32_c.json` | C | `false` | `false` |
| `celeba32_cs.json` | CS | `true` | `false` |
| `celeba32_cr.json` | CR | `false` | `true` |

**64x64 Resolution:**

| File | Mode | sparsity.enabled | regularization.enabled |
|------|------|:----------------:|:----------------------:|
| `celeba64_c.json` | C | `false` | `false` |
| `celeba64_cs.json` | CS | `true` | `false` |
| `celeba64_cr.json` | CR | `false` | `true` |

> **Note**: 64x64 configs use `"dataset": "celeba"` (existing CelebA class) with `batch_size: 128`.

### Full Config Example (`celeba32_cs.json`)

```json
{
  "_comment": "Mode CS: Curriculum + Sparsity",
  "_usage": [
    "Single GPU:  python train_curriculum.py --config-path configs/celeba32_cs.json",
    "Multi-GPU:   python train_curriculum.py --config-path configs/celeba32_cs.json --num-gpus 6 --distributed --rigid-launch",
    "Select GPUs: CUDA_VISIBLE_DEVICES=0,1,3,5 python train_curriculum.py --config-path configs/celeba32_cs.json --num-gpus 4 --distributed --rigid-launch"
  ],
  "dataset": "celeba32",
  "diffusion": {
    "timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "model_mean_type": "eps",
    "model_var_type": "fixed-small",
    "loss_type": "mse"
  },
  "model": {
    "in_channels": 3,
    "hid_channels": 64,
    "ch_multipliers": [1, 2, 2],
    "num_res_blocks": 2,
    "apply_attn": [false, false, false],
    "drop_rate": 0.0
  },
  "train": {
    "lr": 2e-4,
    "batch_size": 256,
    "grad_norm": 1.0,
    "epochs": 100,
    "warmup": 5000,
    "use_ema": true,
    "ema_decay": 0.9999,
    "num_samples": 12,
    "image_intv": 1
  },
  "curriculum": {
    "enabled": true,
    "stages": [
      {"t_min": 0.8, "t_max": 1.0, "epochs": 1, "name": "stage1_high_noise"},
      {"t_min": 0.6, "t_max": 1.0, "epochs": 1, "name": "stage2_expand"},
      {"t_min": 0.4, "t_max": 1.0, "epochs": 1, "name": "stage3_expand"},
      {"t_min": 0.2, "t_max": 1.0, "epochs": 2, "name": "stage4_expand"},
      {"t_min": 0.0, "t_max": 1.0, "epochs": 5, "name": "stage5_full_range"}
    ]
  },
  "sparsity": {
    "enabled": true,
    "initial_sparsity": 0.8,
    "regrowth_method": "gradient"
  },
  "regularization": {
    "enabled": false,
    "lambda_max": 0.00003
  }
}
```

---

## Project Structure

```
ddpm-torch/
├── configs/
│   ├── celeba32.json           # Base config (standard training)
│   ├── celeba32_c.json         # Mode C: Curriculum only (32x32)
│   ├── celeba32_cs.json        # Mode CS: Curriculum + Sparsity (32x32)
│   ├── celeba32_cr.json        # Mode CR: Curriculum + Regularization (32x32)
│   ├── celeba64_c.json         # Mode C: Curriculum only (64x64)
│   ├── celeba64_cs.json        # Mode CS: Curriculum + Sparsity (64x64)
│   ├── celeba64_cr.json        # Mode CR: Curriculum + Regularization (64x64)
│   └── ...
├── ddpm_torch/
│   ├── diffusion.py            # GaussianDiffusion
│   ├── curriculum.py           # CurriculumDiffusion (time range control)
│   ├── curriculum_trainer.py   # CurriculumTrainer (stage management)
│   ├── models/
│   │   ├── __init__.py         # Exports: UNet, SparseUNet, RegUNet
│   │   ├── unet.py             # Base UNet architecture
│   │   ├── sparse_unet.py      # SparseUNet (CS mode)
│   │   └── reg_unet.py         # RegUNet (CR mode)
│   ├── modules.py              # Custom Conv2d, Linear, etc.
│   ├── utils/train.py          # Base Trainer
│   └── metrics/                # FID, Precision/Recall
├── train.py                    # Standard training
├── train_curriculum.py         # Curriculum training (C/CS/CR)
├── generate.py                 # Image generation
├── eval.py                     # Evaluation
└── ddim.py                     # DDIM sampler
```

---

## Standard Training

```bash
# Single GPU
python train.py --dataset cifar10 --train-device cuda:0 --epochs 50

# Multi-GPU
torchrun --standalone --nproc_per_node 2 train.py --dataset celeba --distributed
```

### Generation & Evaluation

```bash
# Generate samples
python generate.py --dataset cifar10 --chkpt-path ./chkpts/cifar10.pt --use-ddim

# Evaluate FID
python eval.py --dataset cifar10 --sample-folder ./images/eval/cifar10
```

---

## Requirements

- torch>=1.12.0
- torchvision>=0.13.0
- scipy>=1.7.3

---

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)
- [minDiffusion](https://github.com/cloneofsimo/minDiffusion) - Minimal diffusion implementation
