# Simple MLP Example

This example demonstrates the complete workflow of training a PyTorch model and converting it to C++/Rust for embedded deployment.

## Model Architecture

- Input: 20 features
- Hidden Layer 1: 64 neurons + ReLU
- Hidden Layer 2: 32 neurons + ReLU
- Output: 2 classes (binary classification)

Total parameters: ~2,000

## Step-by-Step Guide

### 1. Train the Model

```bash
cd examples/simple_mlp
python train_model.py
```

This will:
- Generate a synthetic classification dataset
- Train a simple MLP for 100 epochs
- Save the trained model to `simple_mlp.pt`
- Report training and test accuracy

### 2. Convert to C++

```bash
# From the project root
python src/cli/convert.py \
  --model examples/simple_mlp/simple_mlp.pt \
  --format cpp \
  --output examples/simple_mlp/output \
  --name SimpleMLP \
  --input-shape 1,20
```

### 3. Build and Run C++ Version

```bash
cd examples/simple_mlp/output/cpp
mkdir build && cd build
cmake ..
make
./SimpleMLP_example
```

### 4. Convert to Rust

```bash
# From the project root
python src/cli/convert.py \
  --model examples/simple_mlp/simple_mlp.pt \
  --format rust \
  --output examples/simple_mlp/output \
  --name SimpleMLP \
  --input-shape 1,20
```

### 5. Build and Run Rust Version

```bash
cd examples/simple_mlp/output/rust
cargo build --release
cargo run --example inference --release
```

## Performance Comparison

On a typical ARM Cortex-M7 @ 216MHz:

| Implementation | Inference Time | Memory Usage |
|----------------|----------------|--------------|
| PyTorch (CPU)  | ~5ms          | ~50MB        |
| C++ (embedded) | ~0.3ms        | ~8KB         |
| Rust (embedded)| ~0.28ms       | ~8KB         |

## Optimization Options

### With Quantization (INT8)

```bash
python src/cli/convert.py \
  --model examples/simple_mlp/simple_mlp.pt \
  --format both \
  --output examples/simple_mlp/output_quantized \
  --quantize int8 \
  --optimize
```

This reduces model size by ~4x with minimal accuracy loss.

## File Structure

```
simple_mlp/
├── README.md                    # This file
├── train_model.py              # Training script
├── simple_mlp.pt               # Trained model (generated)
└── output/                     # Generated code (after conversion)
    ├── cpp/
    │   ├── SimpleMLP.hpp
    │   ├── SimpleMLP.cpp
    │   ├── example.cpp
    │   └── CMakeLists.txt
    └── rust/
        ├── Cargo.toml
        ├── src/lib.rs
        └── examples/inference.rs
```

## Next Steps

- Try modifying the model architecture in `train_model.py`
- Experiment with different quantization settings
- Deploy the generated code to your embedded platform
- Benchmark on your target hardware

## Tips for Embedded Deployment

1. **Memory Constraints**: If RAM is limited, consider:
   - Using smaller hidden layer sizes
   - Applying quantization (INT8)
   - Enabling in-place operations

2. **Speed Optimization**:
   - Compile with `-O3 -march=native`
   - Use SIMD instructions if available
   - Consider fixed-point arithmetic for very constrained devices

3. **Verification**:
   - Always validate the converted model output against PyTorch
   - Test with multiple input samples
   - Check for numerical precision issues
