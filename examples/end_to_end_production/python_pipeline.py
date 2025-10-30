"""
Complete End-to-End Production Pipeline in Python

This example shows a realistic production workflow including:
1. Data loading and preprocessing
2. Model inference
3. Postprocessing and result formatting
4. All the operations a researcher typically writes in Python

We'll then convert this ENTIRE pipeline to C++/Rust for embedded deployment.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import json


class DataPreprocessor:
    """
    Example preprocessing pipeline that researchers typically write in Python
    This needs to be converted to C++/Rust for production
    """

    def __init__(self, config: Dict[str, Any]):
        self.mean = config.get('mean', 0.0)
        self.std = config.get('std', 1.0)
        self.min_val = config.get('min_val', -10.0)
        self.max_val = config.get('max_val', 10.0)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        return (data - self.mean) / self.std

    def clip_outliers(self, data: np.ndarray) -> np.ndarray:
        """Clip outliers to prevent extreme values"""
        return np.clip(data, self.min_val, self.max_val)

    def handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with mean value"""
        data = np.copy(data)
        mask = ~np.isfinite(data)
        if np.any(mask):
            data[mask] = self.mean
        return data

    def apply_transforms(self, data: np.ndarray) -> np.ndarray:
        """Apply mathematical transformations"""
        # Common in financial/sensor data
        data = np.log1p(np.abs(data)) * np.sign(data)  # Log transform with sign preservation
        return data

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline
        This is what needs to run on embedded device
        """
        # Step 1: Handle missing values
        data = self.handle_missing_values(raw_data)

        # Step 2: Clip outliers
        data = self.clip_outliers(data)

        # Step 3: Normalize
        data = self.normalize(data)

        # Step 4: Apply transforms
        data = self.apply_transforms(data)

        return data.astype(np.float32)


class ResultPostprocessor:
    """
    Post-processing operations after model inference
    Common operations researchers write in Python
    """

    def __init__(self, config: Dict[str, Any]):
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.top_k = config.get('top_k', 5)
        self.use_temperature_scaling = config.get('temperature_scaling', False)
        self.temperature = config.get('temperature', 1.0)

    def apply_softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply softmax to convert logits to probabilities
        """
        # Temperature scaling (calibration)
        if self.use_temperature_scaling:
            logits = logits / self.temperature

        # Softmax with numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def get_top_k_predictions(self, probabilities: np.ndarray,
                               labels: List[str]) -> List[Dict[str, Any]]:
        """
        Get top-k predictions with labels
        """
        # Get top k indices
        top_k_indices = np.argsort(probabilities)[-self.top_k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append({
                'label': labels[idx] if idx < len(labels) else f'class_{idx}',
                'probability': float(probabilities[idx]),
                'confidence': float(probabilities[idx]),
                'index': int(idx)
            })

        return results

    def filter_by_confidence(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter predictions below confidence threshold
        """
        return [p for p in predictions if p['probability'] >= self.confidence_threshold]

    def apply_moving_average(self, new_prediction: np.ndarray,
                            history: List[np.ndarray],
                            window_size: int = 5) -> np.ndarray:
        """
        Temporal smoothing using moving average
        Common in real-time applications
        """
        history.append(new_prediction)
        if len(history) > window_size:
            history.pop(0)

        return np.mean(history, axis=0)

    def postprocess(self, model_output: np.ndarray,
                   labels: List[str]) -> Dict[str, Any]:
        """
        Complete postprocessing pipeline
        """
        # Step 1: Apply softmax
        probabilities = self.apply_softmax(model_output)

        # Step 2: Get top-k predictions
        top_predictions = self.get_top_k_predictions(probabilities, labels)

        # Step 3: Filter by confidence
        confident_predictions = self.filter_by_confidence(top_predictions)

        # Step 4: Determine final prediction
        if len(confident_predictions) > 0:
            final_prediction = confident_predictions[0]
        else:
            final_prediction = top_predictions[0] if top_predictions else None

        return {
            'prediction': final_prediction,
            'top_k': top_predictions[:3],  # Return top 3
            'probabilities': probabilities.tolist(),
            'confidence': final_prediction['probability'] if final_prediction else 0.0
        }


class ProductionPipeline:
    """
    Complete production pipeline
    Everything a researcher writes in Python that needs to run on embedded device
    """

    def __init__(self, model_path: str, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(self.config['preprocessing'])

        # Load model
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()

        # Initialize postprocessor
        self.postprocessor = ResultPostprocessor(self.config['postprocessing'])

        # Load labels
        self.labels = self.config.get('labels', [f'class_{i}' for i in range(10)])

    def predict(self, raw_input: np.ndarray) -> Dict[str, Any]:
        """
        Complete inference pipeline
        This entire function needs to run on embedded device
        """
        # STEP 1: Preprocessing
        preprocessed = self.preprocessor.preprocess(raw_input)

        # STEP 2: Model Inference
        with torch.no_grad():
            input_tensor = torch.from_numpy(preprocessed).unsqueeze(0)
            output = self.model(input_tensor)
            output_np = output.squeeze(0).numpy()

        # STEP 3: Postprocessing
        result = self.postprocessor.postprocess(output_np, self.labels)

        return result

    def batch_predict(self, raw_inputs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Batch inference
        """
        results = []
        for raw_input in raw_inputs:
            result = self.predict(raw_input)
            results.append(result)
        return results

    def save_pipeline_config(self, output_path: str):
        """
        Save all preprocessing/postprocessing parameters
        These need to be hardcoded in C++/Rust for deployment
        """
        deployment_config = {
            'preprocessing': {
                'mean': self.preprocessor.mean,
                'std': self.preprocessor.std,
                'min_val': self.preprocessor.min_val,
                'max_val': self.preprocessor.max_val,
            },
            'postprocessing': {
                'confidence_threshold': self.postprocessor.confidence_threshold,
                'top_k': self.postprocessor.top_k,
                'temperature': self.postprocessor.temperature,
            },
            'labels': self.labels
        }

        with open(output_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)

        print(f"Saved deployment config to {output_path}")
        return deployment_config


def create_example_config():
    """Create example configuration file"""
    config = {
        'preprocessing': {
            'mean': 0.5,
            'std': 0.2,
            'min_val': -5.0,
            'max_val': 5.0
        },
        'postprocessing': {
            'confidence_threshold': 0.7,
            'top_k': 5,
            'temperature_scaling': True,
            'temperature': 1.5
        },
        'labels': ['cat', 'dog', 'bird', 'fish', 'hamster',
                  'rabbit', 'turtle', 'snake', 'lizard', 'frog']
    }

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("Created config.json")
    return config


def demonstrate_pipeline():
    """
    Demonstrate the complete pipeline
    """
    print("="*70)
    print("PYTHON PRODUCTION PIPELINE DEMONSTRATION")
    print("="*70)

    # Create example model (simple classifier)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    # Save example model
    model = SimpleModel()
    torch.save(model, 'example_model.pt')
    print("✓ Created example model")

    # Create config
    config = create_example_config()
    print("✓ Created configuration")

    # Initialize pipeline
    pipeline = ProductionPipeline('example_model.pt', 'config.json')
    print("✓ Initialized pipeline")

    # Create sample input
    raw_input = np.random.randn(10).astype(np.float32)
    print(f"\n📥 Input data shape: {raw_input.shape}")
    print(f"Input range: [{raw_input.min():.3f}, {raw_input.max():.3f}]")

    # Run prediction
    result = pipeline.predict(raw_input)

    print(f"\n📤 Prediction Results:")
    print(f"Predicted class: {result['prediction']['label']}")
    print(f"Confidence: {result['prediction']['probability']:.4f}")
    print(f"\nTop 3 predictions:")
    for i, pred in enumerate(result['top_k'], 1):
        print(f"  {i}. {pred['label']}: {pred['probability']:.4f}")

    # Save deployment config
    pipeline.save_pipeline_config('deployment_config.json')

    print("\n" + "="*70)
    print("NEXT STEPS FOR EMBEDDED DEPLOYMENT:")
    print("="*70)
    print("1. Convert model to C++/Rust:")
    print("   python src/cli/convert.py --model example_model.pt --format both --output ./embedded")
    print("\n2. Implement preprocessing in C++ using:")
    print("   - embedded_ml::preprocessing::StandardScaler")
    print("   - embedded_ml::math::clip, log1p, etc.")
    print("\n3. Implement postprocessing in C++ using:")
    print("   - embedded_ml::postprocessing::argmax")
    print("   - embedded_ml::postprocessing::topk")
    print("   - embedded_ml::postprocessing::temperature_scaling")
    print("\n4. Use deployment_config.json for hardcoded parameters")
    print("="*70)


if __name__ == '__main__':
    demonstrate_pipeline()
