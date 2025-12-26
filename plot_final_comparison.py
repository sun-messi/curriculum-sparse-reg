"""
Plot final comparison: CS Mode vs Curriculum (new) vs Baseline (new)
Shows the three best performing methods with FID curves and sample images
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib.gridspec import GridSpec

# Data from results
steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
steps_k = [s * 1000 for s in steps]

# FID scores for the three best methods
cs = [422.93, 411.90, 377.10, 430.50, 343.50, 90.19, 75.70, 39.41, 34.66, 32.52,
      31.68, 23.88, 25.42, 17.32, 16.31, 15.90, 15.84, 15.81, 15.90, None]

curriculum_new = [393.74, 382.04, 296.81, 155.92, 105.68, 89.22, 74.27, 63.97, 47.74, 40.27,
                  None, 29.99, None, 16.27, None, 14.80, None, 14.48, 14.56, None]

baseline_new = [401.06, 371.21, 319.34, 187.71, 111.15, 54.62, 28.90, 67.40, 45.64, 21.52,
                None, 15.00, None, 14.49, None, 14.33, None, 13.94, 13.93, None]

# Create figure with GridSpec for FID curve + sample images
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 0.3], hspace=0.35)

# Top: FID curve
ax = fig.add_subplot(gs[0])

# Filter out None values for plotting
def filter_data(steps, values):
    filtered_steps = []
    filtered_values = []
    for s, v in zip(steps, values):
        if v is not None:
            filtered_steps.append(s)
            filtered_values.append(v)
    return filtered_steps, filtered_values

cs_steps, cs_fid = filter_data(steps_k, cs)
curriculum_new_steps, curriculum_new_fid = filter_data(steps_k, curriculum_new)
baseline_new_steps, baseline_new_fid = filter_data(steps_k, baseline_new)

# Plot with distinct colors and styles
ax.plot(cs_steps, cs_fid, '^-', linewidth=2.5, markersize=8,
        label='CS Mode', color='#2ca02c', alpha=0.8)
ax.plot(curriculum_new_steps, curriculum_new_fid, 's-', linewidth=2.5, markersize=8,
        label='Curriculum (new)', color='#ff7f0e', alpha=0.8)
ax.plot(baseline_new_steps, baseline_new_fid, 'o-', linewidth=2.5, markersize=8,
        label='Baseline (new)', color='#1f77b4', alpha=0.8)

# Highlight best points
best_cs_idx = cs_fid.index(min(cs_fid))
best_curriculum_new_idx = curriculum_new_fid.index(min(curriculum_new_fid))
best_baseline_new_idx = baseline_new_fid.index(min(baseline_new_fid))

ax.scatter([cs_steps[best_cs_idx]], [cs_fid[best_cs_idx]],
           s=200, marker='*', color='#2ca02c', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best CS: {cs_fid[best_cs_idx]:.2f}')
ax.scatter([curriculum_new_steps[best_curriculum_new_idx]], [curriculum_new_fid[best_curriculum_new_idx]],
           s=200, marker='*', color='#ff7f0e', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Curriculum: {curriculum_new_fid[best_curriculum_new_idx]:.2f}')
ax.scatter([baseline_new_steps[best_baseline_new_idx]], [baseline_new_fid[best_baseline_new_idx]],
           s=200, marker='*', color='#1f77b4', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Baseline: {baseline_new_fid[best_baseline_new_idx]:.2f}')

# Styling
ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('FID Score (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('U-ViT Final Comparison: Top 3 Methods\\nFID Convergence on CelebA 64×64',
             fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')

# Y-axis limit for better visualization
ax.set_ylim([0, 450])

# Format x-axis
ax.set_xlim([0, 200000])
ax.set_xticks([0, 50000, 100000, 150000, 200000])
ax.set_xticklabels(['0', '50k', '100k', '150k', '200k'])

# Add annotation for the winner
ax.annotate('NEW CHAMPION!\\nBaseline: 13.93',
            xy=(baseline_new_steps[best_baseline_new_idx], baseline_new_fid[best_baseline_new_idx]),
            xytext=(baseline_new_steps[best_baseline_new_idx] - 30000, 60),
            fontsize=12, fontweight='bold', color='#1f77b4',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#1f77b4', linewidth=2.5),
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2.5, connectionstyle='arc3,rad=0.3'))

# Sample image paths - key checkpoints to show
sample_checkpoints = [40000, 80000, 120000, 160000, 180000, 190000]
base_dirs = {
    'Curriculum (new)': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_c_20251225_170001',
    'CS': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs',
    'Baseline (new)': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small/20251225_172649'
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

# Row 1: CS samples
ax_cs = fig.add_subplot(gs[1])
ax_cs.axis('off')
ax_cs.text(0.01, 0.95, 'CS Mode', transform=ax_cs.transAxes,
          fontsize=12, fontweight='bold', va='top',
          bbox=dict(boxstyle='round', facecolor='#2ca02c', alpha=0.3))

# Row 2: Curriculum (new) samples
ax_curr = fig.add_subplot(gs[2])
ax_curr.axis('off')
ax_curr.text(0.01, 0.95, 'Curriculum (new)', transform=ax_curr.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))

# Row 3: Baseline (new) samples
ax_base = fig.add_subplot(gs[3])
ax_base.axis('off')
ax_base.text(0.01, 0.95, 'Baseline (new)', transform=ax_base.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))

# Load and display samples - create horizontal concatenation
sample_grids = {'cs': [], 'curr': [], 'base': []}

for ckpt in sample_checkpoints:
    # CS samples
    cs_dir = os.path.join(base_dirs['CS'], f'{ckpt}_ema')
    cs_grid = load_sample_grid(cs_dir, num_samples=16)
    if cs_grid is not None:
        sample_grids['cs'].append(cs_grid)

    # Curriculum (new) samples
    curr_dir = os.path.join(base_dirs['Curriculum (new)'], f'{ckpt}_ema')
    curr_grid = load_sample_grid(curr_dir, num_samples=16)
    if curr_grid is not None:
        sample_grids['curr'].append(curr_grid)

    # Baseline (new) samples
    base_dir = os.path.join(base_dirs['Baseline (new)'], f'{ckpt}_ema')
    base_grid = load_sample_grid(base_dir, num_samples=16)
    if base_grid is not None:
        sample_grids['base'].append(base_grid)

# Concatenate grids horizontally with spacing between checkpoints
def concat_with_spacing(grids, spacing=10):
    """Concatenate grids horizontally with white spacing between them"""
    if not grids:
        return None

    h = grids[0].shape[0]
    # Create white spacer
    spacer = np.ones((h, spacing, 3), dtype=np.uint8) * 255

    # Interleave grids with spacers
    result = []
    for i, grid in enumerate(grids):
        result.append(grid)
        if i < len(grids) - 1:  # Don't add spacer after last grid
            result.append(spacer)

    return np.concatenate(result, axis=1)

if sample_grids['cs']:
    cs_concat = concat_with_spacing(sample_grids['cs'], spacing=20)
    ax_cs.imshow(cs_concat, interpolation='bilinear')

if sample_grids['curr']:
    curr_concat = concat_with_spacing(sample_grids['curr'], spacing=20)
    ax_curr.imshow(curr_concat, interpolation='bilinear')

if sample_grids['base']:
    base_concat = concat_with_spacing(sample_grids['base'], spacing=20)
    ax_base.imshow(base_concat, interpolation='bilinear')

# Row 4: Unified checkpoint labels at the bottom (shared across all three rows)
ax_labels = fig.add_subplot(gs[4])
ax_labels.axis('off')
ax_labels.set_xlim([0, 1])
ax_labels.set_ylim([0, 1])

# Add unified checkpoint labels
n_checkpoints = len(sample_checkpoints)
for i, ckpt in enumerate(sample_checkpoints):
    x_pos = (i + 0.5) / n_checkpoints
    ax_labels.text(x_pos, 0.5, f'{ckpt//1000}k',
                  ha='center', va='center', fontsize=12, fontweight='bold',
                  transform=ax_labels.transAxes)

plt.savefig('results/final_three_way_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to results/final_three_way_comparison.png")
plt.show()
