"""
Analyze U-Net structure and channel dimensions
"""

import torch
from ddpm_torch.models.reg_unet import RegUNet

# Create model with celeba32 config
model = RegUNet(
    in_channels=3,
    hid_channels=32,
    out_channels=3,
    ch_multipliers=[1, 2, 2, 2],  # 4 levels
    num_res_blocks=2,
    apply_attn=[False, False, False, False],
    drop_rate=0.0
)

print("=" * 100)
print("U-Net Architecture Analysis")
print("=" * 100)

print("\n配置参数:")
print(f"  hid_channels: 32")
print(f"  ch_multipliers: [1, 2, 2, 2]")
print(f"  num_res_blocks: 2 (每个level有2个ResidualBlock)")
print(f"  levels: 4")

print("\n" + "=" * 100)
print("每个Level的通道数:")
print("=" * 100)
ch_multipliers = [1, 2, 2, 2]
hid_channels = 32

for i, mult in enumerate(ch_multipliers):
    channels = mult * hid_channels
    print(f"  Level {i}: {channels} channels (multiplier={mult})")

print("\n" + "=" * 100)
print("U-Net 结构流程:")
print("=" * 100)

print("""
输入图像 (3 channels, 32x32)
    ↓
in_conv: 3 → 32 channels
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Downsampling Path (编码器)                                    │
├─────────────────────────────────────────────────────────────┤
│ Level 0: 32 channels (空间尺寸: 32x32 → 16x16)              │
│   - ResBlock #1: 32 → 32                                     │
│   - ResBlock #2: 32 → 32                                     │
│   - Downsample: 32 → 32 (stride=2, 降采样)                  │
│                                                              │
│ Level 1: 64 channels (空间尺寸: 16x16 → 8x8)                │
│   - ResBlock #1: 32 → 64 (通道数增加)                       │
│   - ResBlock #2: 64 → 64                                     │
│   - Downsample: 64 → 64 (stride=2)                          │
│                                                              │
│ Level 2: 64 channels (空间尺寸: 8x8 → 4x4)                  │
│   - ResBlock #1: 64 → 64                                     │
│   - ResBlock #2: 64 → 64                                     │
│   - Downsample: 64 → 64 (stride=2)                          │
│                                                              │
│ Level 3: 64 channels (空间尺寸: 4x4, 最深层) ← 距离bottleneck最近 │
│   - ResBlock #1: 64 → 64                                     │
│   - ResBlock #2: 64 → 64                                     │
│   (没有downsample，因为是最后一层)                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Middle/Bottleneck (瓶颈层, 空间尺寸: 4x4)                    │
├─────────────────────────────────────────────────────────────┤
│   - ResBlock + Attention + ResBlock                         │
│   - 64 channels (最抽象的特征表示)                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Upsampling Path (解码器)                                     │
├─────────────────────────────────────────────────────────────┤
│ Level 3: 64 channels (空间尺寸: 4x4) ← 距离bottleneck最近     │
│   - ResBlock #1: 128 → 64 (concat skip connection from down) │
│   - ResBlock #2: 128 → 64                                    │
│   - ResBlock #3: 128 → 64                                    │
│   - Upsample: 64 → 64 (scale=2, 上采样)                     │
│   (空间尺寸: 4x4 → 8x8)                                      │
│                                                              │
│ Level 2: 64 channels (空间尺寸: 8x8 → 16x16)                │
│   - ResBlock #1: 128 → 64                                    │
│   - ResBlock #2: 128 → 64                                    │
│   - ResBlock #3: 128 → 64                                    │
│   - Upsample: 64 → 64 (scale=2)                             │
│                                                              │
│ Level 1: 64 channels (空间尺寸: 16x16 → 32x32)              │
│   - ResBlock #1: 128 → 64                                    │
│   - ResBlock #2: 128 → 64                                    │
│   - ResBlock #3: 96 → 64 (最后一个ResBlock输入稍小)         │
│   - Upsample: 64 → 64 (scale=2)                             │
│                                                              │
│ Level 0: 32 channels (空间尺寸: 32x32)                       │
│   - ResBlock #1: 96 → 32                                     │
│   - ResBlock #2: 64 → 32                                     │
│   - ResBlock #3: 64 → 32                                     │
│   (没有upsample，因为已经恢复到原始尺寸)                     │
└─────────────────────────────────────────────────────────────┘
    ↓
out_conv: 32 → 3 channels
    ↓
输出图像 (3 channels, 32x32)
""")

print("\n" + "=" * 100)
print("被正则化的层 (Bottleneck-Adjacent Layers):")
print("=" * 100)

bottleneck_layers = model.get_bottleneck_adjacent_conv_layers()

print("\n1. downsamples.level_3 (下采样最后一层，进入bottleneck之前):")
print("   空间尺寸: 4x4")
down_count = 0
for name, layer in bottleneck_layers:
    if 'downsamples.level_3' in name:
        down_count += 1
        weight_shape = layer.weight.shape
        print(f"   {down_count}. {name:50s}")
        print(f"      Shape: {weight_shape} (out={weight_shape[0]}, in={weight_shape[1]}, k={weight_shape[2]}x{weight_shape[3]})")

print(f"\n   小计: {down_count}个卷积层")

print("\n2. upsamples.level_3 (上采样第一层，离开bottleneck之后):")
print("   空间尺寸: 4x4 → 8x8")
up_count = 0
for name, layer in bottleneck_layers:
    if 'upsamples.level_3' in name:
        up_count += 1
        weight_shape = layer.weight.shape
        print(f"   {up_count}. {name:50s}")
        print(f"      Shape: {weight_shape} (out={weight_shape[0]}, in={weight_shape[1]}, k={weight_shape[2]}x{weight_shape[3]})")

print(f"\n   小计: {up_count}个卷积层")

print("\n" + "=" * 100)
print("ResidualBlock 内部结构:")
print("=" * 100)
print("""
每个ResidualBlock包含:
  1. conv1: 3x3卷积
  2. conv2: 3x3卷积
  3. skip: 1x1卷积 (仅当输入输出通道数不同时)

对于downsamples.level_3:
  - 有2个ResidualBlock
  - 每个有conv1 (3x3) 和 conv2 (3x3)
  - 输入输出都是64 channels，所以没有skip connection
  - 总计: 2 blocks × 2 convs = 4个卷积层

对于upsamples.level_3:
  - 有3个ResidualBlock + 1个upsample
  - ResBlock输入是concat后的特征 (64+64=128 channels)
  - 每个ResBlock: conv1 (3x3) + conv2 (3x3) + skip (1x1)
  - Upsample后的卷积: 3x3
  - 总计: 3 blocks × 3 convs + 1 upsample conv = 10个卷积层
""")

print("\n" + "=" * 100)
print("为什么选择这两层进行正则化?")
print("=" * 100)
print("""
1. **特征抽象程度最高**:
   - 这两层处于网络最深处，特征最抽象
   - 空间分辨率最低 (4x4)，通道数最多 (64)

2. **对生成质量影响最大**:
   - Bottleneck附近的特征决定了图像的全局结构和高级语义
   - 浅层特征主要负责细节和纹理

3. **计算效率**:
   - 只正则化14/67 ≈ 21%的层
   - 减少了约79%的正则化计算开销

4. **结构化稀疏的最佳位置**:
   - 在最深层进行通道剪枝不会损失太多细节信息
   - 可以显著减少bottleneck附近的计算量
""")

print("=" * 100)
