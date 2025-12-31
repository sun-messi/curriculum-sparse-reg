import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import OrderedDict


class FeatureExtractor:
    """
    Extract intermediate feature activations from UNet layers using forward hooks.

    Supports extracting from nested ModuleList structures like downsamples.level_X.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize FeatureExtractor.

        Args:
            model: UNet model to extract features from
        """
        self.model = model
        self.hooks = []  # Store hook handles for cleanup
        self.features = OrderedDict()  # Preserve insertion order

    def _make_hook(self, name: str):
        """
        Factory function to create a hook that captures activations.

        Args:
            name: Identifier for this layer's activations

        Returns:
            Hook function with closure over name
        """
        def hook(module, input, output):
            # Detach and clone to avoid keeping computation graph
            # Move to CPU immediately to free GPU memory
            self.features[name] = output.detach().cpu().clone()
        return hook

    def register_layer(self, layer_name: str, layer_path: str):
        """
        Register a hook on a specific layer.

        Args:
            layer_name: Human-readable name for storage (e.g., "encoder_level_2")
            layer_path: Dot-notation path to the layer (e.g., "downsamples.level_2")

        Example:
            extractor.register_layer("encoder_level_0", "downsamples.level_0")
        """
        # Navigate to the target module
        parts = layer_path.split('.')
        module = self.model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                raise ValueError(f"Module path '{layer_path}' not found in model")

        # For downsample/upsample levels, hook the last module in ModuleList
        if isinstance(module, nn.ModuleList):
            module = module[-1]

        # Register the hook
        hook_handle = module.register_forward_hook(self._make_hook(layer_name))
        self.hooks.append(hook_handle)

    def extract_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and extract registered features.

        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep tensor [B]

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self.features.clear()

        with torch.no_grad():
            _ = self.model(x, t)

        # Return copy to avoid reference issues
        return self.features.copy()

    def remove_hooks(self):
        """Clean up all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure hooks are cleaned up even if exception occurs."""
        self.remove_hooks()
        return False


def get_target_layers() -> List[Tuple[str, str]]:
    """
    Define target layers for feature extraction from UNet.

    Returns:
        List of (layer_name, layer_path) tuples
    """
    return [
        ("in_conv", "in_conv"),
        ("encoder_level_0", "downsamples.level_0"),
        ("encoder_level_1", "downsamples.level_1"),
        ("encoder_level_2", "downsamples.level_2"),
        ("encoder_level_3", "downsamples.level_3"),
        ("bottleneck", "middle"),
    ]
