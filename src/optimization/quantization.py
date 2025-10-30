"""
Quantization tools for model compression
Reduce precision to int8/int16 for embedded deployment
"""

import numpy as np
from typing import Dict, Tuple, Optional


class Quantizer:
    """Quantize model weights and activations for embedded deployment"""

    def __init__(self, bits: int = 8):
        """
        Args:
            bits: Number of bits for quantization (8 or 16)
        """
        if bits not in [8, 16]:
            raise ValueError("Only 8-bit and 16-bit quantization supported")
        self.bits = bits
        self.dtype = np.int8 if bits == 8 else np.int16
        self.max_val = 2 ** (bits - 1) - 1
        self.min_val = -(2 ** (bits - 1))

    def quantize_tensor(self, tensor: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Quantize a tensor to int8/int16

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (quantized_tensor, scale, zero_point)
        """
        # Calculate min and max
        min_val = tensor.min()
        max_val = tensor.max()

        # Calculate scale and zero point
        scale = (max_val - min_val) / (self.max_val - self.min_val)
        zero_point = self.min_val - min_val / scale

        # Quantize
        quantized = np.clip(
            np.round(tensor / scale + zero_point),
            self.min_val,
            self.max_val
        ).astype(self.dtype)

        return quantized, scale, zero_point

    def dequantize_tensor(self, quantized: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
        """
        Dequantize tensor back to float

        Args:
            quantized: Quantized tensor
            scale: Quantization scale
            zero_point: Quantization zero point

        Returns:
            Dequantized float tensor
        """
        return (quantized.astype(np.float32) - zero_point) * scale

    def quantize_model_weights(self, model_info: Dict) -> Dict:
        """
        Quantize all weights in a model

        Args:
            model_info: Model information from parser

        Returns:
            Dictionary with quantized weights and metadata
        """
        quantized_model = {
            'layers': [],
            'metadata': {
                'quantization_bits': self.bits,
                'original_size_mb': 0,
                'quantized_size_mb': 0,
            }
        }

        original_size = 0
        quantized_size = 0

        for layer in model_info['layers']:
            quantized_layer = {
                'name': layer.name,
                'type': layer.layer_type,
                'params': layer.params,
                'weights': {},
                'quantization_params': {}
            }

            # Quantize each weight tensor
            for weight_name, weight_data in layer.weights.items():
                if weight_data is not None:
                    original_size += weight_data.nbytes

                    quant_weight, scale, zero_point = self.quantize_tensor(weight_data)
                    quantized_size += quant_weight.nbytes + 8  # +8 for scale and zero_point

                    quantized_layer['weights'][weight_name] = quant_weight
                    quantized_layer['quantization_params'][weight_name] = {
                        'scale': scale,
                        'zero_point': zero_point
                    }

            quantized_model['layers'].append(quantized_layer)

        quantized_model['metadata']['original_size_mb'] = original_size / (1024 * 1024)
        quantized_model['metadata']['quantized_size_mb'] = quantized_size / (1024 * 1024)
        quantized_model['metadata']['compression_ratio'] = original_size / quantized_size if quantized_size > 0 else 0

        return quantized_model

    def quantize_symmetric(self, tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Symmetric quantization (zero_point = 0)
        Useful for weights

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (quantized_tensor, scale)
        """
        max_abs = np.abs(tensor).max()
        scale = max_abs / self.max_val

        if scale == 0:
            scale = 1.0

        quantized = np.clip(
            np.round(tensor / scale),
            self.min_val,
            self.max_val
        ).astype(self.dtype)

        return quantized, scale

    def quantize_per_channel(self, tensor: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-channel quantization (better accuracy for conv layers)

        Args:
            tensor: Input tensor
            axis: Channel axis

        Returns:
            Tuple of (quantized_tensor, scales)
        """
        # Move channel axis to first position
        tensor = np.moveaxis(tensor, axis, 0)
        original_shape = tensor.shape
        num_channels = original_shape[0]

        # Flatten each channel
        tensor_flat = tensor.reshape(num_channels, -1)

        quantized_flat = np.zeros_like(tensor_flat, dtype=self.dtype)
        scales = np.zeros(num_channels, dtype=np.float32)

        # Quantize each channel separately
        for c in range(num_channels):
            max_abs = np.abs(tensor_flat[c]).max()
            scale = max_abs / self.max_val if max_abs > 0 else 1.0
            scales[c] = scale

            quantized_flat[c] = np.clip(
                np.round(tensor_flat[c] / scale),
                self.min_val,
                self.max_val
            ).astype(self.dtype)

        # Reshape back
        quantized = quantized_flat.reshape(original_shape)
        quantized = np.moveaxis(quantized, 0, axis)

        return quantized, scales


class DynamicQuantizer:
    """Dynamic quantization - quantize activations at runtime based on their range"""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.max_val = 2 ** (bits - 1) - 1
        self.min_val = -(2 ** (bits - 1))

    def calibrate(self, activations: np.ndarray) -> Tuple[float, float]:
        """
        Calibrate quantization parameters from activation statistics

        Args:
            activations: Sample activations

        Returns:
            Tuple of (scale, zero_point)
        """
        min_val = activations.min()
        max_val = activations.max()

        scale = (max_val - min_val) / (self.max_val - self.min_val)
        zero_point = self.min_val - min_val / scale

        return scale, zero_point


def calculate_quantization_error(original: np.ndarray, quantized: np.ndarray) -> Dict[str, float]:
    """
    Calculate quantization error metrics

    Args:
        original: Original float tensor
        quantized: Dequantized tensor

    Returns:
        Dictionary with error metrics
    """
    mse = np.mean((original - quantized) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - quantized))

    # Signal-to-noise ratio
    signal_power = np.mean(original ** 2)
    snr = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'snr_db': float(snr)
    }
