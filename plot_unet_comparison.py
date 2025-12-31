"""
Plot U-Net CNN-based comparison: Baseline vs Curriculum vs CS Mode
Shows the three training methods with FID curves and sample images
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib.gridspec import GridSpec

# Data from celeba64_comparison.txt (excluding CR)
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

baseline_fid = [442.22, 289.47, 144.92, 111.96, 93.18, 78.80, 68.06, 56.69, 50.25, 44.00, 40.20, 37.22, 45.40, 30.94, 28.79]
curriculum_fid = [446.09, 283.13, 173.15, 129.34, 96.06, 75.75, 71.15, 53.22, 48.20, 35.02, 30.55, 31.15, 46.48, 27.18, 26.57]
cs_mode_fid = [445.74, 326.68, 234.30, 179.81, 128.59, 94.13, 91.73, 58.43, 51.43, 35.52, 30.93, 31.01, 45.14, 28.37, 26.06]

# Create figure with GridSpec for FID curve + sample images
fig = plt.figure(figsize=(28, 18))
gs = GridSpec(4, 5, height_ratios=[2, 1.7, 1.7, 1.7], hspace=0.2, wspace=0.08)

# Top: FID curve (spanning all columns)
ax = fig.add_subplot(gs[0, :])

# Plot all data from epoch 5 onwards
start_epoch = 5
mask = [e >= start_epoch for e in epochs]
epochs_plot = [e for e, m in zip(epochs, mask) if m]
baseline_plot = [v for v, m in zip(baseline_fid, mask) if m]
curriculum_plot = [v for v, m in zip(curriculum_fid, mask) if m]
cs_mode_plot = [v for v, m in zip(cs_mode_fid, mask) if m]

ax.plot(epochs_plot, baseline_plot, 'o-', linewidth=2.5, markersize=8,
        label='Baseline', color='#1f77b4', alpha=0.8)
ax.plot(epochs_plot, curriculum_plot, 's-', linewidth=2.5, markersize=8,
        label='Curriculum', color='#ff7f0e', alpha=0.8)
ax.plot(epochs_plot, cs_mode_plot, '^-', linewidth=2.5, markersize=8,
        label='CS Mode', color='#2ca02c', alpha=0.8)

# Highlight best points
best_baseline_idx = baseline_plot.index(min(baseline_plot))
best_curriculum_idx = curriculum_plot.index(min(curriculum_plot))
best_cs_idx = cs_mode_plot.index(min(cs_mode_plot))

ax.scatter([epochs_plot[best_baseline_idx]], [baseline_plot[best_baseline_idx]],
           s=200, marker='*', color='#1f77b4', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Baseline: {baseline_plot[best_baseline_idx]:.2f}')
ax.scatter([epochs_plot[best_curriculum_idx]], [curriculum_plot[best_curriculum_idx]],
           s=200, marker='*', color='#ff7f0e', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best Curriculum: {curriculum_plot[best_curriculum_idx]:.2f}')
ax.scatter([epochs_plot[best_cs_idx]], [cs_mode_plot[best_cs_idx]],
           s=200, marker='*', color='#2ca02c', edgecolors='black', linewidths=2,
           zorder=5, label=f'Best CS: {cs_mode_plot[best_cs_idx]:.2f}')

# Styling
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('FID Score (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('U-Net CNN-based Comparison: Top 3 Methods\nFID Convergence on CelebA 64×64',
             fontsize=16, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')

# Y-axis limit
ax.set_ylim([0, 150])

# Format x-axis
ax.set_xlim([5, 15])
ax.set_xticks([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
ax.set_xticklabels(['5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])

# Sample image paths - show epochs 5, 8, 10, 12, 15
sample_epochs = [5, 8, 10, 12, 15]
base_dirs = {
    'Baseline': '/home/sunj11/Documents/ddpm-torch/generated/celeba',
    'Curriculum': '/home/sunj11/Documents/ddpm-torch/generated/celeba64_c',
    'CS Mode': '/home/sunj11/Documents/ddpm-torch/generated/celeba64_cs'
}

# Mapping from epoch to directory timestamp (based on directory listings)
epoch_dirs = {
    'Baseline': {
        3: 'celeba_20251219_101242_3',
        5: 'celeba_20251219_102231_5',
        8: 'celeba_20251219_104231_8',
        10: 'celeba_20251219_105618_10',
        12: 'celeba_20251219_111007_12',
        15: 'celeba_20251219_113026_15'
    },
    'Curriculum': {
        3: 'celeba64_c_20251219_115818_3',
        5: 'celeba64_c_20251219_121206_5',
        8: 'celeba64_c_20251219_123230_8',
        10: 'celeba64_c_20251219_124620_10',
        12: 'celeba64_c_20251219_130007_12',
        15: 'celeba64_c_20251219_132027_15'
    },
    'CS Mode': {
        3: 'celeba64_cs_20251219_153724_3',
        5: 'celeba64_cs_20251219_155115_5',
        8: 'celeba64_cs_20251219_161140_8',
        10: 'celeba64_cs_20251219_162529_10',
        12: 'celeba64_cs_20251219_164312_12',
        15: 'celeba64_cs_20251219_171135_15'
    }
}

def load_sample_grid(checkpoint_dir, num_samples=25):
    """Load and create a grid of sample images (5x5 grid)"""
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Directory not found: {checkpoint_dir}")
        return None

    # Find image files
    img_files = [f for f in os.listdir(checkpoint_dir)
                 if f.endswith(('.png', '.jpg', '.jpeg'))]

    if len(img_files) == 0:
        print(f"Warning: No images found in {checkpoint_dir}")
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
row_labels = ['Baseline', 'Curriculum', 'CS Mode']
row_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Display sample images in grid
for row_idx, (method, color) in enumerate(zip(row_labels, row_colors)):
    for col_idx, epoch in enumerate(sample_epochs):
        ax_img = fig.add_subplot(gs[row_idx + 1, col_idx])
        ax_img.axis('off')

        # Add label for first column
        if col_idx == 0:
            ax_img.text(-0.1, 0.5, method, transform=ax_img.transAxes,
                       fontsize=11, fontweight='bold', va='center', ha='right',
                       rotation=90)

        # Add epoch label for first row
        if row_idx == 0:
            ax_img.text(0.5, 1.05, f'Epoch {epoch}', transform=ax_img.transAxes,
                       fontsize=11, fontweight='bold', va='bottom', ha='center')

        # Load and display image
        if epoch in epoch_dirs[method]:
            checkpoint_dir = os.path.join(base_dirs[method], epoch_dirs[method][epoch])
            grid = load_sample_grid(checkpoint_dir, num_samples=25)

            if grid is not None:
                ax_img.imshow(grid, interpolation='bilinear')

plt.savefig('results/unet_three_way_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to results/unet_three_way_comparison.png")
plt.show()
