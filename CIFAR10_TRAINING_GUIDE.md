# CIFAR10 Training Guide - 6 GPU Distributed Training

This guide provides terminal commands to train DDPM models on CIFAR10 with 4 different configurations using 6 GPUs.

## Configuration Variants

1. **cifar10.json** - Baseline DDPM (standard training)
2. **cifar10_c.json** - Curriculum Learning only
3. **cifar10_cr.json** - Curriculum + Regularization (L2 weight decay)
4. **cifar10_cs.json** - Curriculum + Sparsity (channel pruning)

All configurations use the same training hyperparameters:
- Learning rate: 2e-4
- Batch size: 128
- Epochs: 2040 (baseline) or 15 (curriculum variants)
- EMA decay: 0.9999
- Warmup steps: 5000
- Model: 128 hidden channels, [1,2,2,2] multipliers

---

## Single GPU Training (for testing/debugging)

### Baseline DDPM
```bash
python train.py --config-path configs/cifar10.json
```

### Curriculum Learning Only
```bash
python train_curriculum.py --config-path configs/cifar10_c.json
```

### Curriculum + Regularization
```bash
python train_curriculum.py --config-path configs/cifar10_cr.json
```

### Curriculum + Sparsity
```bash
python train_curriculum.py --config-path configs/cifar10_cs.json
```

---

## 6 GPU Distributed Training (Recommended)

All commands use `--distributed --rigid-launch --num-gpus 6` for synchronous multi-GPU training.

### Baseline DDPM (6 GPUs)
```bash
python train.py --config-path configs/cifar10.json --num-gpus 6 --distributed --rigid-launch
```

### Curriculum Only (6 GPUs)
```bash
python train_curriculum.py --config-path configs/cifar10_c.json --num-gpus 6 --distributed --rigid-launch
```

### Curriculum + Regularization (6 GPUs)
```bash
python train_curriculum.py --config-path configs/cifar10_cr.json --num-gpus 6 --distributed --rigid-launch
```

### Curriculum + Sparsity (6 GPUs)
```bash
python train_curriculum.py --config-path configs/cifar10_cs.json --num-gpus 6 --distributed --rigid-launch
```

---

## Distributed Training with GPU Selection

If you want to use specific GPUs (e.g., GPUs 0, 1, 2, 3, 4, 5):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_curriculum.py --config-path configs/cifar10_c.json --num-gpus 6 --distributed --rigid-launch
```

If you want to use different GPUs (e.g., GPUs 1, 2, 4, 5, 6, 7):

```bash
CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 python train_curriculum.py --config-path configs/cifar10_c.json --num-gpus 6 --distributed --rigid-launch
```

---

## Using torchrun (Elastic Launch)

Alternative method using PyTorch's elastic launcher (more robust for multi-node):

### Baseline DDPM
```bash
torchrun --standalone --nproc_per_node 6 train.py --config-path configs/cifar10.json --distributed
```

### Curriculum Only
```bash
torchrun --standalone --nproc_per_node 6 train_curriculum.py --config-path configs/cifar10_c.json --distributed
```

### Curriculum + Regularization
```bash
torchrun --standalone --nproc_per_node 6 train_curriculum.py --config-path configs/cifar10_cr.json --distributed
```

### Curriculum + Sparsity
```bash
torchrun --standalone --nproc_per_node 6 train_curriculum.py --config-path configs/cifar10_cs.json --distributed
```

---

## Training from Checkpoint (Resume)

Add `--resume` flag to continue from the latest checkpoint:

```bash
python train_curriculum.py --config-path configs/cifar10_c.json --num-gpus 6 --distributed --rigid-launch --resume
```

---

## Output Structure

Training will save:
- **Checkpoints**: `./chkpts/cifar10_c_[timestamp]_[epoch]/`
- **Generated samples**: `./images/cifar10_c_[timestamp]_[epoch]/`
- **Metrics**: Logged to stdout (loss, learning rate, etc.)

---

## Key Arguments Explained

| Argument | Description |
|----------|-------------|
| `--config-path` | Path to configuration JSON file |
| `--num-gpus 6` | Use 6 GPUs for distributed training |
| `--distributed` | Enable distributed training mode |
| `--rigid-launch` | Use torch.multiprocessing.spawn (synchronous) |
| `--resume` | Resume from last checkpoint |
| `CUDA_VISIBLE_DEVICES` | Comma-separated list of GPU indices to use |

---

## Monitoring Training

Watch GPU usage:
```bash
nvidia-smi -l 1  # Updates every 1 second
```

View checkpoint directory:
```bash
ls -lah chkpts/cifar10_c_*/
```

View generated samples:
```bash
ls -lah images/cifar10_c_*/
```

---

## Configuration Differences

### Baseline (cifar10.json)
- No curriculum learning
- Standard 2040 epochs of continuous training
- Best for baseline comparison

### Curriculum Only (cifar10_c.json)
- Progressive noise level filtering (t_min: 0.3 → 0.0)
- 9 stages totaling 15 epochs
- Faster convergence on coarse features first
- Stages: 1, 1, 1, 1, 1, 1, 2, 3, 4 epochs each

### Curriculum + Regularization (cifar10_cr.json)
- Adds L2 weight regularization (Group L1 norm)
- Lambda decays across stages: 0.00005 → 0.0
- Encourages sparse weight matrices
- Same 9-stage curriculum as cifar10_c.json

### Curriculum + Sparsity (cifar10_cs.json)
- Adds channel-level sparsity with hard masking
- Sparsity decays: 80% → 0% across stages
- Uses gradient-based regrowth for pruned channels
- Same 9-stage curriculum as cifar10_c.json

---

## Tips for Best Results

1. **6 GPUs recommended**: Provides good balance of throughput and memory usage
2. **Use rigid-launch**: More stable than elastic launch for fixed GPU count
3. **Monitor first epoch**: Check GPU memory and convergence speed
4. **Save checkpoints**: Default is every 1 epoch (`chkpt_intv: 1`)
5. **Resume training**: Can safely stop and resume with `--resume`

---

## Quick Start

To start training all 4 configurations in sequence:

```bash
# Start baseline
python train.py --config-path configs/cifar10.json --num-gpus 6 --distributed --rigid-launch &
BASELINE_PID=$!

# Wait for baseline to complete, then start curriculum
wait $BASELINE_PID
python train_curriculum.py --config-path configs/cifar10_c.json --num-gpus 6 --distributed --rigid-launch &
C_PID=$!

wait $C_PID
python train_curriculum.py --config-path configs/cifar10_cr.json --num-gpus 6 --distributed --rigid-launch &
CR_PID=$!

wait $CR_PID
python train_curriculum.py --config-path configs/cifar10_cs.json --num-gpus 6 --distributed --rigid-launch
```

Or run them in parallel (requires 24 GPUs total):

```bash
python train.py --config-path configs/cifar10.json --num-gpus 6 --distributed --rigid-launch &
python train_curriculum.py --config-path configs/cifar10_c.json --num-gpus 6 --distributed --rigid-launch &
python train_curriculum.py --config-path configs/cifar10_cr.json --num-gpus 6 --distributed --rigid-launch &
python train_curriculum.py --config-path configs/cifar10_cs.json --num-gpus 6 --distributed --rigid-launch
```
