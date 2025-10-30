"""
Example 1: Simple MLP with Preprocessing
Complete conversion from Python to C++ and Rust with verification

This example shows a simple neural network with preprocessing and postprocessing.
We verify that Python, C++, and Rust produce identical outputs.
"""

import numpy as np
import torch
import torch.nn as nn
import json


# ============================================================================
# PYTHON IMPLEMENTATION
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple 3-layer MLP"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def preprocess_python(data, mean, std):
    """Preprocessing in Python"""
    # Clip outliers
    data = np.clip(data, -5.0, 5.0)
    # Normalize
    data = (data - mean) / std
    return data


def postprocess_python(output):
    """Postprocessing in Python"""
    # Softmax
    exp_output = np.exp(output - np.max(output))
    probs = exp_output / np.sum(exp_output)

    # Argmax
    predicted_class = np.argmax(probs)

    return {
        'probabilities': probs.tolist(),
        'predicted_class': int(predicted_class),
        'confidence': float(probs[predicted_class])
    }


def python_pipeline(raw_input, model, mean, std):
    """Complete Python pipeline"""
    # Preprocess
    preprocessed = preprocess_python(raw_input, mean, std)

    # Model inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(preprocessed).float().unsqueeze(0)
        output = model(input_tensor)
        output_np = output.squeeze(0).numpy()

    # Postprocess
    result = postprocess_python(output_np)

    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("VERIFIED CONVERSION EXAMPLE: Simple MLP with Pre/Post Processing")
    print("="*70)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create and save model
    model = SimpleMLP()
    model.eval()

    # Save model
    torch.save(model, 'verified_mlp.pt')
    print("\n✓ Created and saved model")

    # Define preprocessing parameters
    MEAN = 0.5
    STD = 0.2

    # Create test input
    raw_input = np.random.randn(10).astype(np.float32)

    # Save test input
    np.save('test_input.npy', raw_input)
    print("✓ Created test input")

    # Run Python pipeline
    result_python = python_pipeline(raw_input, model, MEAN, STD)

    print("\n" + "="*70)
    print("PYTHON RESULTS:")
    print("="*70)
    print(f"Input: {raw_input}")
    print(f"Predicted class: {result_python['predicted_class']}")
    print(f"Confidence: {result_python['confidence']:.6f}")
    print(f"Probabilities: {[f'{p:.6f}' for p in result_python['probabilities']]}")

    # Save results for verification
    results = {
        'input': raw_input.tolist(),
        'mean': MEAN,
        'std': STD,
        'python': result_python
    }

    with open('verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved verification results to verification_results.json")

    # Extract and save model weights for C++/Rust
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy().tolist()

    with open('model_weights.json', 'w') as f:
        json.dump(weights, f, indent=2)

    print("✓ Saved model weights to model_weights.json")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Convert model:")
    print("   python ../../../src/cli/convert.py --model verified_mlp.pt --format both --output ./output")
    print("\n2. Build C++ version:")
    print("   cd output/cpp && mkdir build && cd build")
    print("   cmake .. && make")
    print("   ./run_verification")
    print("\n3. Build Rust version:")
    print("   cd output/rust")
    print("   cargo run --release --example verification")
    print("\n4. Verify all outputs match:")
    print("   python verify_conversion.py")
    print("="*70)


if __name__ == '__main__':
    main()
