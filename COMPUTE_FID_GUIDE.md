# FID 计算指南

## 脚本功能

`compute_uvit_fid.py` 可以计算 U-ViT 模型不同 checkpoint 的 FID 分数，支持两种目录结构。

## 使用方法

### 1. 计算单个模型的所有 checkpoint

```bash
python compute_uvit_fid.py \
  --uvit-dir /home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs \
  --dataset celeba \
  --device cuda:0 \
  --ref-size 5000
```

**目录结构**:
```
celeba64_uvit_small_cs/
├── 40000_ema/  (包含 1000 张 .png/.jpg 样本)
├── 50000_ema/
├── 60000_ema/
├── 70000_ema/
├── 80000_ema/
├── 90000_ema/
└── 100000_ema/
```

**输出**:
```
Checkpoint      | celeba64_uvit_small_cs
----------------|------------------------
40000_ema       |                  45.23
50000_ema       |                  42.18
60000_ema       |                  39.87
70000_ema       |                  38.54
80000_ema       |                  37.21
90000_ema       |                  36.45
100000_ema      |                  35.89
```

### 2. 比较多个模型

```bash
python compute_uvit_fid.py \
  --uvit-dir /home/sunj11/Documents/U-ViT/eval_samples \
  --dataset celeba \
  --device cuda:0 \
  --ref-size 5000
```

**目录结构**:
```
eval_samples/
├── celeba64_uvit_small/        (baseline)
│   ├── 50k/
│   ├── 100k/
│   └── ...
├── celeba64_uvit_small_c/      (curriculum only)
│   ├── 10000_ema/
│   └── ...
└── celeba64_uvit_small_cs/     (curriculum + sparsity)
    ├── 40000_ema/
    └── ...
```

**输出**:
```
Checkpoint  | celeba64_uvit_small | celeba64_uvit_small_c | celeba64_uvit_small_cs
------------|---------------------|----------------------|------------------------
50k         |               42.15 |                40.23 |                  45.23
100k        |               38.90 |                36.54 |                  38.12
...
```

### 3. 计算额外指标

```bash
python compute_uvit_fid.py \
  --uvit-dir /path/to/model \
  --dataset celeba \
  --device cuda:0 \
  --calc-is        # Inception Score
  --calc-lpips     # LPIPS (diversity)
```

### 4. CPU vs GPU

```bash
# GPU (推荐 - 快 10-20 倍)
--device cuda:0

# CPU (当 GPU 不可用时)
--device cpu
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--uvit-dir` | U-ViT 样本目录 (必需) | - |
| `--dataset` | 数据集名称 | `celeba` |
| `--device` | 计算设备 | `cuda:0` |
| `--ref-size` | 真实图像参考数量 | 5000 |
| `--calc-is` | 计算 Inception Score | False |
| `--calc-lpips` | 计算 LPIPS | False |
| `--output` | 输出文件路径 | `results/uvit_fid_comparison.txt` |

## 输出文件

脚本会生成两个文件:

1. **TXT 文件**: `results/uvit_fid_comparison.txt`
   - 包含完整的表格和元数据
   - 每次运行会追加 (append)

2. **CSV 文件**: `results/uvit_fid_comparison.csv`
   - 便于导入 Excel 或 Python/pandas
   - 每次运行会追加新行

## 支持的 Checkpoint 命名格式

脚本自动识别以下格式:

- `40000_ema` → 40000 步
- `50k` → 50000 步
- `100000_ema` → 100000 步
- `200k` → 200000 步

## 示例

### 当前运行的命令

```bash
# 计算 CS 模式的所有 checkpoint FID
python compute_uvit_fid.py \
  --uvit-dir /home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs \
  --dataset celeba \
  --device cpu \
  --ref-size 5000
```

**进度**:
- 预计算真实图像统计: ~3-5 分钟 (一次性)
- 每个 checkpoint FID 计算: ~2-3 分钟
- 总共 7 个 checkpoint: ~20-25 分钟

**结果保存在**:
- `results/uvit_fid_comparison.txt`
- `results/uvit_fid_comparison.csv`

## 常见问题

### Q1: "AssertionError: Torch not compiled with CUDA enabled"
**A**: 使用 `--device cpu` 代替 `--device cuda:0`

### Q2: FID 值很高 (>100)
**A**: 检查:
- 样本图像质量
- 样本数量 (建议 ≥1000)
- 数据集是否匹配

### Q3: 如何加速计算?
**A**:
- 使用 GPU (`--device cuda:0`)
- 减少 `--ref-size` (但会降低准确性)
- 只计算 FID (不加 `--calc-is` 和 `--calc-lpips`)

### Q4: 如何比较 C、CS、CR 三种模式?
**A**: 将所有模型放在同一父目录下，然后运行:
```bash
python compute_uvit_fid.py \
  --uvit-dir /home/sunj11/Documents/U-ViT/eval_samples \
  --dataset celeba \
  --device cuda:0
```
