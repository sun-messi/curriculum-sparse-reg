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
- [ ] **Curriculum + Sparsity (CS)** - Hard mask with channel regrowth
- [ ] **Curriculum + Regularization (CR)** - Group L1 soft sparsity

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

### Three Training Variants

| Script | Curriculum | Sparsity | Method |
|--------|:----------:|:--------:|--------|
| `train_curriculum.py` | ✓ | ✗ | Curriculum only (C) |
| `train_curriculum.py --use-sparse` | ✓ | ✓ | Hard mask + regrowth (CS) |
| `train_curriculum.py --use-reg` | ✓ | ✓ | Group L1 regularization (CR) |

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

## CS: Curriculum + Sparsity (Hard Mask)

Channel-level sparsity at the UNet bottleneck with gradient-based regrowth.

```python
# Bottleneck forward pass
thro = self.to_vec(down3)              # (B, 256, 1, 1)
thro = thro * self.channel_mask        # Apply mask
thro = self.up0(thro + temb)
```

**Sparsity Schedule**:
```
Stage 1: 80% sparse → Only 20% channels active (learn M₁)
Stage 2: 60% sparse → Regrow 20% clean channels
Stage 3: 40% sparse → Continue regrowth
Stage 4: 20% sparse
Stage 5: 0% sparse  → All channels active
```

**Regrowth Methods**:
- `random`: Randomly select masked channels to activate
- `gradient`: Select channels with highest gradient magnitude

---

## CR: Curriculum + Regularization (Group L1)

Soft sparsity via Group L1 regularization on all Conv layers.

```python
# Group L1 penalty
L_reg = λ × Σ_c ||W[c,:,:,:]||_2

# Total loss
L_total = MSE(noise_pred, noise) + λ(stage) × L_reg
```

**λ Schedule** (linear decay):
```
Stage 1: λ = λ_max (strong regularization → drive sparsity)
Stage 5: λ ≈ 0 (no regularization → release capacity)
```

### CS vs CR Comparison

| Feature | CS (Hard Mask) | CR (Group L1) |
|---------|----------------|---------------|
| Mechanism | `channel_mask` hard mask | Regularization penalty |
| Location | Bottleneck only | All Conv layers |
| Control | Explicit regrowth | λ decay auto-release |
| Flexibility | Discrete (0/1) | Continuous (soft) |

---

## Usage

### Curriculum Learning (C)

```bash
# Single GPU
python train_curriculum.py --config-path configs/celeba32_c.json

# Multi-GPU with DDP
python train_curriculum.py --config-path configs/celeba32_c.json \
    --num-gpus 4 --distributed --rigid-launch

# Elastic Launch (recommended)
torchrun --standalone --nproc_per_node 4 train_curriculum.py \
    --config-path configs/celeba32_c.json --distributed
```

### Standard Training

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

## Configuration

### Curriculum Config Example (`configs/celeba32_c.json`)

```json
{
  "dataset": "celeba32",
  "diffusion": {
    "timesteps": 1000,
    "beta_schedule": "linear",
    "model_mean_type": "eps",
    "model_var_type": "fixed-small",
    "loss_type": "mse"
  },
  "model": {
    "hid_channels": 64,
    "ch_multipliers": [1, 2, 2],
    "num_res_blocks": 2
  },
  "train": {
    "lr": 2e-4,
    "batch_size": 128,
    "epochs": 100,
    "use_ema": true
  },
  "curriculum": {
    "enabled": true,
    "stages": [
      {"t_min": 0.8, "t_max": 1.0, "epochs": 1, "name": "stage1_high_noise"},
      {"t_min": 0.6, "t_max": 1.0, "epochs": 1, "name": "stage2_expand"},
      {"t_min": 0.4, "t_max": 1.0, "epochs": 2, "name": "stage3_expand"},
      {"t_min": 0.2, "t_max": 1.0, "epochs": 2, "name": "stage4_expand"},
      {"t_min": 0.0, "t_max": 1.0, "epochs": 5, "name": "stage5_full_range"}
    ]
  }
}
```

---

## Project Structure

```
ddpm-torch/
├── configs/
│   ├── celeba32.json          # Base config
│   ├── celeba32_c.json        # Curriculum config
│   ├── cifar10.json
│   └── ...
├── ddpm_torch/
│   ├── diffusion.py           # GaussianDiffusion
│   ├── curriculum.py          # CurriculumDiffusion
│   ├── curriculum_trainer.py  # CurriculumTrainer
│   ├── models/unet.py         # UNet architecture
│   ├── utils/train.py         # Base Trainer
│   └── metrics/               # FID, Precision/Recall
├── train.py                   # Standard training
├── train_curriculum.py        # Curriculum training
├── generate.py                # Image generation
├── eval.py                    # Evaluation
└── ddim.py                    # DDIM sampler
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
