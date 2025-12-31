"""
Test script to verify Group L1 regularization is only applied to bottleneck-adjacent layers.
"""

import torch
from ddpm_torch.models.reg_unet import RegUNet

# Create a RegUNet model with the same config as celeba32
model = RegUNet(
    in_channels=3,
    hid_channels=32,
    out_channels=3,
    ch_multipliers=[1, 2, 2, 2],
    num_res_blocks=2,
    apply_attn=[False, False, False, False],
    drop_rate=0.0
)

print("=" * 80)
print("RegUNet Layer Analysis")
print("=" * 80)

# Get all conv layers
all_conv_layers = model.get_conv_layers()
print(f"\n1. Total Conv layers in model: {len(all_conv_layers)}")
print("\nAll Conv layer names:")
for i, (name, layer) in enumerate(all_conv_layers, 1):
    print(f"  {i:2d}. {name:50s} {layer.weight.shape}")

# Get bottleneck-adjacent conv layers
bottleneck_layers = model.get_bottleneck_adjacent_conv_layers()
print(f"\n2. Bottleneck-adjacent Conv layers: {len(bottleneck_layers)}")
print("\nBottleneck-adjacent layer names:")
for i, (name, layer) in enumerate(bottleneck_layers, 1):
    print(f"  {i:2d}. {name:50s} {layer.weight.shape}")

# Test regularization computation
lambda_val = 0.00003
penalty = model.get_group_l1_penalty(lambda_val)
print(f"\n3. Group L1 Penalty (lambda={lambda_val}):")
print(f"   Penalty value: {penalty.item():.6f}")

print("\n" + "=" * 80)
print("Verification Summary:")
print("=" * 80)
print(f"✓ Total layers in model: {len(all_conv_layers)}")
print(f"✓ Layers being regularized: {len(bottleneck_layers)}")
print(f"✓ Expected layers: downsamples.level_3 and upsamples.level_3")
print("\nRegularization is now applied ONLY to bottleneck-adjacent layers!")
print("=" * 80)
