"""
Plot final comparison: CS Mode vs Curriculum (new) vs Baseline (new)
Shows the three best performing methods with FID curves and sample images
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib.gridspec import GridSpec

# Data from results - Updated 2025-12-26
steps_full = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
steps_k_full = [s * 1000 for s in steps_full]

baseline_modified_full = [402.39, 364.07, 339.82, 178.20, 100.95, 95.91, 90.87, 85.83, 61.25, 35.97, 27.28]
curriculum_full = [393.93, 382.88, 292.35, 170.63, 126.90, 95.68, 43.87, 33.72, 30.13, 29.47, 32.47]
cs_mode_full = [422.93, 411.90, 377.10, 430.50, 343.50, 90.19, 75.70, 39.41, 34.66, 32.52, 23.88]

# Create figure with GridSpec for FID curve + sample images
fig = plt.figure(figsize=(28, 18))
gs = GridSpec(4, 5, height_ratios=[2, 1.7, 1.7, 1.7], hspace=0.2, wspace=0.08)

# Top: FID curve (spanning all columns)
ax = fig.add_subplot(gs[0, :])

# Plot all data from 50k onwards
start_step = 50000
mask = [s >= start_step for s in steps_k_full]
steps_k = [s for s, m in zip(steps_k_full, mask) if m]
baseline_modified = [v for v, m in zip(baseline_modified_full, mask) if m]
curriculum = [v for v, m in zip(curriculum_full, mask) if m]
cs_mode = [v for v, m in zip(cs_mode_full, mask) if m]

ax.plot(steps_k, baseline_modified, 'o-', linewidth=2.5, markersize=8,
        label='Baseline (Modified)', color='#1f77b4', alpha=0.8)
ax.plot(steps_k, curriculum, 's-', linewidth=2.5, markersize=8,
        label='Curriculum', color='#ff7f0e', alpha=0.8)
ax.plot(steps_k, cs_mode, '^-', linewidth=2.5, markersize=8,
        label='CS Mode', color='#2ca02c', alpha=0.8)

# Highlight best points
best_baseline_idx = baseline_modified.index(min(baseline_modified))
best_curriculum_idx = curriculum.index(min(curriculum))
best_cs_idx = cs_mode.index(min(cs_mode))

ax.scatter([steps_k[best_baseline_idx]], [baseline_modified[best_baseline_idx]],
           s=200, marker='*', color='#1f77b4', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Baseline: {baseline_modified[best_baseline_idx]:.2f}')
ax.scatter([steps_k[best_curriculum_idx]], [curriculum[best_curriculum_idx]],
           s=200, marker='*', color='#ff7f0e', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Curriculum: {curriculum[best_curriculum_idx]:.2f}')
ax.scatter([steps_k[best_cs_idx]], [cs_mode[best_cs_idx]],
           s=200, marker='*', color='#2ca02c', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best CS: {cs_mode[best_cs_idx]:.2f}')

# Styling
ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('FID Score (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('U-ViT Final Comparison: Top 3 Methods\nFID Convergence on CelebA 64×64',
             fontsize=16, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')

# Y-axis limit
ax.set_ylim([0, 400])

# Format x-axis
ax.set_xlim([50000, 120000])
ax.set_xticks([50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000])
ax.set_xticklabels(['50k', '60k', '70k', '80k', '90k', '100k', '110k', '120k'])

# Sample image paths - show 50k, 70k, 80k, 100k, 120k checkpoints
sample_checkpoints = [50000, 70000, 80000, 100000, 120000]
base_dirs = {
    'Baseline (Modified)': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small/20251225_225757',
    'Curriculum': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_c/20251226_042556',
    'CS Mode': '/home/sunj11/Documents/U-ViT/eval_samples/celeba64_uvit_small_cs/20251226_035545'
}

def load_sample_grid(checkpoint_dir, num_samples=25):
    """Load and create a grid of sample images (5x5 grid)"""
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

    # Create 5x5 grid
    grid_size = 5
    h, w = imgs[0].shape[:2]
    grid = np.zeros((grid_size * h, grid_size * w, 3), dtype=np.uint8)

    for idx, img in enumerate(imgs[:grid_size**2]):
        row = idx // grid_size
        col = idx % grid_size
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

    return grid

# Row labels
row_labels = ['Baseline (Modified)', 'Curriculum', 'CS Mode']
row_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Display sample images in grid
for row_idx, (method, color) in enumerate(zip(row_labels, row_colors)):
    for col_idx, ckpt in enumerate(sample_checkpoints):
        ax_img = fig.add_subplot(gs[row_idx + 1, col_idx])
        ax_img.axis('off')

        # Add label for first column
        if col_idx == 0:
            ax_img.text(-0.1, 0.5, method, transform=ax_img.transAxes,
                       fontsize=11, fontweight='bold', va='center', ha='right',
                       rotation=90)

        # Add checkpoint label for first row
        if row_idx == 0:
            ax_img.text(0.5, 1.05, f'{ckpt//1000}k steps', transform=ax_img.transAxes,
                       fontsize=11, fontweight='bold', va='bottom', ha='center')

        # Load and display image
        checkpoint_dir = os.path.join(base_dirs[method], f'{ckpt}_ema')
        grid = load_sample_grid(checkpoint_dir, num_samples=25)

        if grid is not None:
            ax_img.imshow(grid, interpolation='bilinear')

plt.savefig('results/final_three_way_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to results/final_three_way_comparison.png")
plt.show()
