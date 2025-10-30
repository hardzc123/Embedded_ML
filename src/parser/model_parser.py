"""
PyTorch Model Parser - Extract architecture and weights from PyTorch models
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import OrderedDict


class LayerInfo:
    """Container for layer information"""
    def __init__(self, name: str, layer_type: str, params: Dict[str, Any], weights: Dict[str, np.ndarray]):
        self.name = name
        self.layer_type = layer_type
        self.params = params
        self.weights = weights
        self.input_shape = None
        self.output_shape = None


class ModelParser:
    """Parse PyTorch models and extract architecture for code generation"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()  # Set to evaluation mode
        self.layers: List[LayerInfo] = []
        self.input_shape = None
        self.output_shape = None

    def parse(self, sample_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Parse the model and extract all necessary information

        Args:
            sample_input: Optional sample input to infer shapes

        Returns:
            Dictionary containing model architecture and weights
        """
        # Extract layers
        self._extract_layers()

        # Infer shapes if sample input provided
        if sample_input is not None:
            self._infer_shapes(sample_input)

        return {
            'layers': self.layers,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'num_parameters': self._count_parameters(),
            'model_type': self._infer_model_type()
        }

    def _extract_layers(self):
        """Extract all layers from the model"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = self._parse_module(name, module)
                if layer_info:
                    self.layers.append(layer_info)

    def _parse_module(self, name: str, module: nn.Module) -> Optional[LayerInfo]:
        """Parse individual module and extract information"""

        # Linear layer
        if isinstance(module, nn.Linear):
            return LayerInfo(
                name=name,
                layer_type='Linear',
                params={
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None
                },
                weights={
                    'weight': module.weight.detach().cpu().numpy(),
                    'bias': module.bias.detach().cpu().numpy() if module.bias is not None else None
                }
            )

        # Conv2D layer
        elif isinstance(module, nn.Conv2d):
            return LayerInfo(
                name=name,
                layer_type='Conv2d',
                params={
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'dilation': module.dilation,
                    'groups': module.groups,
                    'bias': module.bias is not None
                },
                weights={
                    'weight': module.weight.detach().cpu().numpy(),
                    'bias': module.bias.detach().cpu().numpy() if module.bias is not None else None
                }
            )

        # Conv1D layer
        elif isinstance(module, nn.Conv1d):
            return LayerInfo(
                name=name,
                layer_type='Conv1d',
                params={
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'dilation': module.dilation,
                    'groups': module.groups,
                    'bias': module.bias is not None
                },
                weights={
                    'weight': module.weight.detach().cpu().numpy(),
                    'bias': module.bias.detach().cpu().numpy() if module.bias is not None else None
                }
            )

        # BatchNorm2D
        elif isinstance(module, nn.BatchNorm2d):
            return LayerInfo(
                name=name,
                layer_type='BatchNorm2d',
                params={
                    'num_features': module.num_features,
                    'eps': module.eps,
                    'momentum': module.momentum,
                    'affine': module.affine,
                    'track_running_stats': module.track_running_stats
                },
                weights={
                    'weight': module.weight.detach().cpu().numpy() if module.weight is not None else None,
                    'bias': module.bias.detach().cpu().numpy() if module.bias is not None else None,
                    'running_mean': module.running_mean.detach().cpu().numpy() if module.running_mean is not None else None,
                    'running_var': module.running_var.detach().cpu().numpy() if module.running_var is not None else None
                }
            )

        # BatchNorm1D
        elif isinstance(module, nn.BatchNorm1d):
            return LayerInfo(
                name=name,
                layer_type='BatchNorm1d',
                params={
                    'num_features': module.num_features,
                    'eps': module.eps,
                    'momentum': module.momentum,
                    'affine': module.affine,
                    'track_running_stats': module.track_running_stats
                },
                weights={
                    'weight': module.weight.detach().cpu().numpy() if module.weight is not None else None,
                    'bias': module.bias.detach().cpu().numpy() if module.bias is not None else None,
                    'running_mean': module.running_mean.detach().cpu().numpy() if module.running_mean is not None else None,
                    'running_var': module.running_var.detach().cpu().numpy() if module.running_var is not None else None
                }
            )

        # LSTM
        elif isinstance(module, nn.LSTM):
            # Extract LSTM weights (complex structure)
            weights = {}
            for i in range(module.num_layers):
                suffix = f'_l{i}'
                if hasattr(module, f'weight_ih_l{i}'):
                    weights[f'weight_ih{suffix}'] = getattr(module, f'weight_ih_l{i}').detach().cpu().numpy()
                    weights[f'weight_hh{suffix}'] = getattr(module, f'weight_hh_l{i}').detach().cpu().numpy()
                    if module.bias:
                        weights[f'bias_ih{suffix}'] = getattr(module, f'bias_ih_l{i}').detach().cpu().numpy()
                        weights[f'bias_hh{suffix}'] = getattr(module, f'bias_hh_l{i}').detach().cpu().numpy()

            return LayerInfo(
                name=name,
                layer_type='LSTM',
                params={
                    'input_size': module.input_size,
                    'hidden_size': module.hidden_size,
                    'num_layers': module.num_layers,
                    'bias': module.bias,
                    'batch_first': module.batch_first,
                    'dropout': module.dropout,
                    'bidirectional': module.bidirectional
                },
                weights=weights
            )

        # Activation layers (no weights)
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.GELU)):
            layer_type = module.__class__.__name__
            params = {}
            if isinstance(module, nn.LeakyReLU):
                params['negative_slope'] = module.negative_slope
            elif isinstance(module, nn.Softmax):
                params['dim'] = module.dim

            return LayerInfo(
                name=name,
                layer_type=layer_type,
                params=params,
                weights={}
            )

        # Pooling layers
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            layer_type = module.__class__.__name__
            params = {}
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                params['kernel_size'] = module.kernel_size
                params['stride'] = module.stride
                params['padding'] = module.padding
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                params['output_size'] = module.output_size

            return LayerInfo(
                name=name,
                layer_type=layer_type,
                params=params,
                weights={}
            )

        # Dropout (for inference, just pass-through)
        elif isinstance(module, nn.Dropout):
            return LayerInfo(
                name=name,
                layer_type='Dropout',
                params={'p': module.p},
                weights={}
            )

        # Flatten
        elif isinstance(module, nn.Flatten):
            return LayerInfo(
                name=name,
                layer_type='Flatten',
                params={'start_dim': module.start_dim, 'end_dim': module.end_dim},
                weights={}
            )

        return None

    def _infer_shapes(self, sample_input: torch.Tensor):
        """Infer input/output shapes for each layer using a forward pass"""
        self.input_shape = tuple(sample_input.shape)

        # Hook to capture shapes
        shapes = {}

        def hook_fn(name):
            def hook(module, input, output):
                shapes[name] = {
                    'input': tuple(input[0].shape) if isinstance(input, tuple) else tuple(input.shape),
                    'output': tuple(output.shape) if isinstance(output, torch.Tensor) else None
                }
            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            output = self.model(sample_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Assign shapes to layers
        for layer in self.layers:
            if layer.name in shapes:
                layer.input_shape = shapes[layer.name]['input']
                layer.output_shape = shapes[layer.name]['output']

        self.output_shape = tuple(output.shape)

    def _count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())

    def _infer_model_type(self) -> str:
        """Infer the type of model based on layers"""
        has_conv = any(layer.layer_type.startswith('Conv') for layer in self.layers)
        has_lstm = any(layer.layer_type == 'LSTM' for layer in self.layers)
        has_linear = any(layer.layer_type == 'Linear' for layer in self.layers)

        if has_lstm:
            return 'RNN'
        elif has_conv:
            return 'CNN'
        elif has_linear:
            return 'MLP'
        else:
            return 'Unknown'

    def export_summary(self) -> str:
        """Generate a human-readable summary of the model"""
        summary = []
        summary.append("Model Architecture Summary")
        summary.append("=" * 80)
        summary.append(f"Model Type: {self._infer_model_type()}")
        summary.append(f"Total Parameters: {self._count_parameters():,}")
        summary.append(f"Input Shape: {self.input_shape}")
        summary.append(f"Output Shape: {self.output_shape}")
        summary.append("\nLayers:")
        summary.append("-" * 80)

        for i, layer in enumerate(self.layers):
            summary.append(f"{i+1}. {layer.name} ({layer.layer_type})")
            summary.append(f"   Parameters: {layer.params}")
            if layer.input_shape:
                summary.append(f"   Input Shape: {layer.input_shape}")
            if layer.output_shape:
                summary.append(f"   Output Shape: {layer.output_shape}")
            summary.append("")

        return "\n".join(summary)
