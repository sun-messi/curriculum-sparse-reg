# Feature Activation Map Visualizations

Comparison of intermediate layer activations between **Baseline** and **Curriculum** DDPM models on CelebA-64 images.

## Generated Files

**Total**: 140 visualization files
- 20 input images (5 samples × 4 conditions)
- 120 feature comparisons (5 samples × 4 conditions × 6 layers)

## Directory Structure

```
feature_visualizations/
├── sample_0/
│   ├── input_clean.png                   # Original clean image
│   ├── input_noisy_t100.png              # Image with noise at t=100
│   ├── input_noisy_t500.png              # Image with noise at t=500
│   ├── input_noisy_t900.png              # Image with noise at t=900
│   ├── clean_t0/                         # Features from clean input
│   │   ├── in_conv_baseline_vs_curriculum.png
│   │   ├── encoder_level_0_baseline_vs_curriculum.png
│   │   ├── encoder_level_1_baseline_vs_curriculum.png
│   │   ├── encoder_level_2_baseline_vs_curriculum.png
│   │   ├── encoder_level_3_baseline_vs_curriculum.png
│   │   └── bottleneck_baseline_vs_curriculum.png
│   ├── noisy_t100/                       # Features from t=100 noisy input
│   ├── noisy_t500/                       # Features from t=500 noisy input
│   └── noisy_t900/                       # Features from t=900 noisy input
├── sample_1/
├── sample_2/
├── sample_3/
└── sample_4/
```

## Layer Information

Each visualization shows side-by-side 8×8 grids of the top 64 feature channels (selected by variance):

| Layer | Output Shape | Spatial Size | Channels | Description |
|-------|--------------|--------------|----------|-------------|
| `in_conv` | (128, 64, 64) | 64×64 | 128 | Initial convolution features |
| `encoder_level_0` | (128, 64, 64) | 64×64 | 128 | Shallow features |
| `encoder_level_1` | (256, 32, 32) | 32×32 | 256 | Mid-shallow features |
| `encoder_level_2` | (256, 16, 16) | 16×16 | 256 | Mid-deep features (with attention) |
| `encoder_level_3` | (256, 8, 8) | 8×8 | 256 | Deep features |
| `bottleneck` | (256, 4, 4) | 4×4 | 256 | Most abstract features |

## How to Read the Visualizations

### Layout

Each PNG file shows:
```
┌─────────────────────────────────────────────────────┐
│     encoder_level_2 | noisy_t500 | Sample 0        │
├─────────────────────────┬───────────────────────────┤
│      Baseline           │      Curriculum           │
│  μ=0.123 σ=0.456       │  μ=0.089 σ=0.234         │
│  sparsity=23.5%         │  sparsity=45.2%           │
│                         │                           │
│   [8×8 grid of         │   [8×8 grid of           │
│    64 channels]         │    64 channels]           │
│                         │                           │
└─────────────────────────┴───────────────────────────┘
```

### Statistics

- **μ (mean)**: Average activation magnitude
- **σ (std)**: Standard deviation of activations
- **sparsity**: Percentage of near-zero activations (|x| < 0.001)

### Color Map

- **Purple/Dark**: Low activation
- **Yellow/Bright**: High activation
- Uses 'viridis' colormap for perceptually uniform visualization

## Expected Observations

Based on the Curriculum Learning hypothesis:

### Curriculum Model Features
✅ **Clearer edge responses** (especially in encoder_level_0/1)
✅ **Higher activation sparsity** (more near-zero values)
✅ **More structured activation patterns** (less noise)
✅ **Better semantic region separation** (in deeper layers)

### Baseline Model Features
⚠️ More diffuse/blurry features
⚠️ Lower sparsity (more non-zero activations)
⚠️ More noisy activation patterns
⚠️ Less structured responses

## Recommended Viewing Order

### 1. Start with Input Images
Look at `sample_X/input_clean.png` and `sample_X/input_noisy_t500.png` to understand the inputs.

### 2. Compare Clean Features
Open `sample_X/clean_t0/encoder_level_2_baseline_vs_curriculum.png`
- This shows how models process clean images
- Look for differences in feature clarity

### 3. Compare Noisy Features at Mid-Diffusion
Open `sample_X/noisy_t500/encoder_level_2_baseline_vs_curriculum.png`
- t=500 represents mid-point of diffusion process
- Most representative of model behavior during denoising

### 4. Examine Layer Progression
For a single sample and condition, view:
1. `in_conv` → Initial feature extraction
2. `encoder_level_0` → Shallow features (edges, textures)
3. `encoder_level_1` → Mid-level features (parts)
4. `encoder_level_2` → Semantic features (with attention)
5. `encoder_level_3` → Abstract features
6. `bottleneck` → Most compressed representation

### 5. Compare Across Timesteps
For a single layer (e.g., `encoder_level_2`), compare:
- `noisy_t100` (low noise - nearly clean)
- `noisy_t500` (medium noise - mid-diffusion)
- `noisy_t900` (high noise - early diffusion)

## Quick Analysis Examples

### Example 1: Check Sparsity Improvement
```bash
# View encoder_level_2 at t=500 for sample 0
open feature_visualizations/sample_0/noisy_t500/encoder_level_2_baseline_vs_curriculum.png
```
Look at the sparsity percentage in the title. Curriculum should have higher sparsity.

### Example 2: Compare Bottleneck Features
```bash
# View bottleneck features for clean input
open feature_visualizations/sample_0/clean_t0/bottleneck_baseline_vs_curriculum.png
```
Curriculum's 4×4 features should be more concentrated and meaningful.

### Example 3: Examine All Samples for One Layer
```bash
# View encoder_level_2 at t=500 for all samples
open feature_visualizations/sample_*/noisy_t500/encoder_level_2_baseline_vs_curriculum.png
```
Check if the pattern (Curriculum > Baseline) is consistent across different faces.

## Key Insights to Look For

1. **Sparsity**: Curriculum should consistently show higher sparsity across layers
2. **Structure**: Curriculum features should have clearer patterns (not random noise)
3. **Edges**: In shallow layers (encoder_level_0/1), Curriculum should show sharper edge responses
4. **Consistency**: Differences should be consistent across all 5 samples

## Regenerating Visualizations

To regenerate with different parameters:

```bash
python visualize_features.py \
    --baseline_ckpt chkpts/celeba/celeba_20251219_113720_16.pt \
    --curriculum_ckpt chkpts/celeba64_c/celeba64_c_20251219_132723_16.pt \
    --config_baseline configs/celeba.json \
    --config_curriculum configs/celeba64_c.json \
    --num_samples 5 \
    --timesteps 100 500 900 \
    --num_channels 64 \
    --output_dir feature_visualizations \
    --device cuda:0
```

## Checkpoints Used

- **Baseline**: `chkpts/celeba/celeba_20251219_113720_16.pt` (Epoch 16)
- **Curriculum**: `chkpts/celeba64_c/celeba64_c_20251219_132723_16.pt` (Epoch 16)

Both models use EMA (Exponential Moving Average) weights for more stable features.
