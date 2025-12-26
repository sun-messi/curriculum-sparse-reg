"""
Plot three-way comparison: Baseline vs Curriculum vs Curriculum+Sparsity
Horizontal layout showing FID curves and sample images for all three methods
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Data from results
steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
steps_k = [s * 1000 for s in steps]

# FID scores
baseline = [402.39, 364.07, 339.82, 178.20, 100.95, 49.67, 28.11, 85.83, 61.25, 35.97,
            None, 27.28, None, 26.70, None, 26.24, None, 25.34, None, 24.48]

curriculum = [393.93, 382.88, 292.35, 170.63, 126.90, 95.68, 43.87, 33.72, 30.13, 29.47,
              None, 32.47, None, 23.34, None, 26.40, None, 24.71, None, 23.03]

cs = [422.93, 411.90, 377.10, 430.50, 343.50, 90.19, 75.70, 39.41, 34.66, 32.52,
      31.68, 23.88, 25.42, 17.32, 16.31, 15.90, 15.84, 15.81, 15.90, None]

# Create figure with GridSpec for FID curve + sample images
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(4, 1, height_ratios=[2.5, 1, 1, 1], hspace=0.4)

# Top: FID curve
ax = fig.add_subplot(gs[0])

# Plot lines
# Filter out None values for plotting
def filter_data(steps, values):
    filtered_steps = []
    filtered_values = []
    for s, v in zip(steps, values):
        if v is not None:
            filtered_steps.append(s)
            filtered_values.append(v)
    return filtered_steps, filtered_values

baseline_steps, baseline_fid = filter_data(steps_k, baseline)
curriculum_steps, curriculum_fid = filter_data(steps_k, curriculum)
cs_steps, cs_fid = filter_data(steps_k, cs)

# Plot with distinct colors and styles
ax.plot(baseline_steps, baseline_fid, 'o-', linewidth=2.5, markersize=8,
        label='Baseline', color='#1f77b4', alpha=0.8)
ax.plot(curriculum_steps, curriculum_fid, 's-', linewidth=2.5, markersize=8,
        label='Curriculum (C)', color='#ff7f0e', alpha=0.8)
ax.plot(cs_steps, cs_fid, '^-', linewidth=2.5, markersize=8,
        label='Curriculum+Sparsity (CS)', color='#2ca02c', alpha=0.8)

# Highlight best points
best_baseline_idx = baseline_fid.index(min(baseline_fid))
best_curriculum_idx = curriculum_fid.index(min(curriculum_fid))
best_cs_idx = cs_fid.index(min(cs_fid))

ax.scatter([baseline_steps[best_baseline_idx]], [baseline_fid[best_baseline_idx]],
           s=200, marker='*', color='#1f77b4', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Baseline: {baseline_fid[best_baseline_idx]:.2f}')
ax.scatter([curriculum_steps[best_curriculum_idx]], [curriculum_fid[best_curriculum_idx]],
           s=200, marker='*', color='#ff7f0e', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Curriculum: {curriculum_fid[best_curriculum_idx]:.2f}')
ax.scatter([cs_steps[best_cs_idx]], [cs_fid[best_cs_idx]],
           s=200, marker='*', color='#2ca02c', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best CS: {cs_fid[best_cs_idx]:.2f}')

# Styling
ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('FID Score (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('U-ViT: Baseline vs Curriculum vs Curriculum+Sparsity\nFID Comparison on CelebA 64×64',
             fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')

# Y-axis limit for better visualization
ax.set_ylim([0, 450])

# Format x-axis
ax.set_xlim([0, 210000])
ax.set_xticks([0, 50000, 100000, 150000, 200000])
ax.set_xticklabels(['0', '50k', '100k', '150k', '200k'])

# Add annotation for key insight
ax.annotate('CS achieves 15.81\n(31% better than C)',
            xy=(cs_steps[best_cs_idx], cs_fid[best_cs_idx]),
            xytext=(cs_steps[best_cs_idx] - 30000, 50),
            fontsize=11, fontweight='bold', color='#2ca02c',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2ca02c', linewidth=2),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2, connectionstyle='arc3,rad=0.3'))

# Add shaded region showing where CS overtakes others
overtake_step = 120000
ax.axvspan(overtake_step, 200000, alpha=0.1, color='#2ca02c',
           label='CS superior region')

# Add vertical line at overtake point
ax.axvline(x=overtake_step, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.text(overtake_step + 5000, 400, '120k: CS overtakes\nBaseline & C',
        fontsize=10, color='gray', style='italic')

# Sample image paths - key checkpoints to show
sample_checkpoints = [40000, 80000, 120000, 160000, 180000, 190000]
base_dirs = {
    'Baseline': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small/20251224_161010',
    'Curriculum': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_c/20251224_155455',
    'CS': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs'
}

def load_sample_grid(checkpoint_dir, num_samples=16):
    """Load and create a grid of sample images"""
    if not os.path.exists(checkpoint_dir):
        return None

    # Find image files
    img_files = [f for f in os.listdir(checkpoint_dir)
                 if f.endswith(('.png', '.jpg', '.jpeg'))]

    if len(img_files) == 0:
        return None

    # Load first num_samples images
    imgs = []
    for i, f in enumerate(sorted(img_files)[:num_samples]):
        try:
            img = Image.open(os.path.join(checkpoint_dir, f))
            imgs.append(np.array(img))
        except:
            pass
        if len(imgs) >= num_samples:
            break

    if len(imgs) == 0:
        return None

    # Create grid
    grid_size = int(np.sqrt(len(imgs)))
    h, w = imgs[0].shape[:2]
    grid = np.zeros((grid_size * h, grid_size * w, 3), dtype=np.uint8)

    for idx, img in enumerate(imgs[:grid_size**2]):
        row = idx // grid_size
        col = idx % grid_size
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

    return grid

# Row 1: Baseline samples
ax_base = fig.add_subplot(gs[1])
ax_base.axis('off')
ax_base.text(0.02, 0.95, 'Baseline', transform=ax_base.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))

# Row 2: Curriculum samples
ax_curr = fig.add_subplot(gs[2])
ax_curr.axis('off')
ax_curr.text(0.02, 0.95, 'Curriculum (C)', transform=ax_curr.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))

# Row 3: CS samples
ax_cs = fig.add_subplot(gs[3])
ax_cs.axis('off')
ax_cs.text(0.02, 0.95, 'Curriculum+Sparsity (CS)', transform=ax_cs.transAxes,
          fontsize=12, fontweight='bold', va='top',
          bbox=dict(boxstyle='round', facecolor='#2ca02c', alpha=0.3))

# Load and display samples - create horizontal concatenation
sample_grids = {'base': [], 'curr': [], 'cs': []}

for ckpt in sample_checkpoints:
    # Baseline samples
    base_dir_ode = os.path.join(base_dirs['Baseline'], f'{ckpt}_ema_ode50')
    base_grid = load_sample_grid(base_dir_ode, num_samples=16)
    if base_grid is not None:
        sample_grids['base'].append(base_grid)

    # Curriculum samples
    curr_dir_ode = os.path.join(base_dirs['Curriculum'], f'{ckpt}_ema_ode50')
    curr_grid = load_sample_grid(curr_dir_ode, num_samples=16)
    if curr_grid is not None:
        sample_grids['curr'].append(curr_grid)

    # CS samples
    cs_dir = os.path.join(base_dirs['CS'], f'{ckpt}_ema')
    cs_grid = load_sample_grid(cs_dir, num_samples=16)
    if cs_grid is not None:
        sample_grids['cs'].append(cs_grid)

# Concatenate grids horizontally
if sample_grids['base']:
    base_concat = np.concatenate(sample_grids['base'], axis=1)
    ax_base.imshow(base_concat, interpolation='bilinear')
    ax_base.axis('off')

    # Add checkpoint labels and FID scores
    n_samples = len(sample_grids['base'])
    grid_width = sample_grids['base'][0].shape[1]
    for i, ckpt in enumerate(sample_checkpoints[:n_samples]):
        x_center = (i + 0.5) * grid_width
        # Checkpoint label (below)
        ax_base.text(x_center, base_concat.shape[0] + 10, f'{ckpt//1000}k',
                    ha='center', va='top', fontsize=9)
        # FID score (above)
        ckpt_idx = steps.index(ckpt//1000) if ckpt//1000 in steps else None
        if ckpt_idx is not None and baseline[ckpt_idx] is not None:
            ax_base.text(x_center, -10, f'FID: {baseline[ckpt_idx]:.1f}',
                        ha='center', va='bottom', fontsize=8, color='#1f77b4',
                        fontweight='bold')

if sample_grids['curr']:
    curr_concat = np.concatenate(sample_grids['curr'], axis=1)
    ax_curr.imshow(curr_concat, interpolation='bilinear')
    ax_curr.axis('off')

    # Add checkpoint labels and FID scores
    n_samples = len(sample_grids['curr'])
    grid_width = sample_grids['curr'][0].shape[1]
    for i, ckpt in enumerate(sample_checkpoints[:n_samples]):
        x_center = (i + 0.5) * grid_width
        # Checkpoint label (below)
        ax_curr.text(x_center, curr_concat.shape[0] + 10, f'{ckpt//1000}k',
                    ha='center', va='top', fontsize=9)
        # FID score (above)
        ckpt_idx = steps.index(ckpt//1000) if ckpt//1000 in steps else None
        if ckpt_idx is not None and curriculum[ckpt_idx] is not None:
            ax_curr.text(x_center, -10, f'FID: {curriculum[ckpt_idx]:.1f}',
                        ha='center', va='bottom', fontsize=8, color='#ff7f0e',
                        fontweight='bold')

if sample_grids['cs']:
    cs_concat = np.concatenate(sample_grids['cs'], axis=1)
    ax_cs.imshow(cs_concat, interpolation='bilinear')
    ax_cs.axis('off')

    # Add checkpoint labels and FID scores
    n_samples = len(sample_grids['cs'])
    grid_width = sample_grids['cs'][0].shape[1]
    for i, ckpt in enumerate(sample_checkpoints[:n_samples]):
        x_center = (i + 0.5) * grid_width
        # Checkpoint label (below)
        ax_cs.text(x_center, cs_concat.shape[0] + 10, f'{ckpt//1000}k',
                  ha='center', va='top', fontsize=9)
        # FID score (above)
        ckpt_idx = steps.index(ckpt//1000) if ckpt//1000 in steps else None
        if ckpt_idx is not None and cs[ckpt_idx] is not None:
            ax_cs.text(x_center, -10, f'FID: {cs[ckpt_idx]:.1f}',
                      ha='center', va='bottom', fontsize=8, color='#2ca02c',
                      fontweight='bold')

plt.savefig('results/three_way_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to results/three_way_comparison.png")
plt.show()
