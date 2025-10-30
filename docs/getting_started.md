# Getting Started with Embedded ML

This guide will help you get started with converting PyTorch models to C++/Rust for embedded deployment.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- C++ compiler (GCC 7+ or Clang 5+) for C++ output
- Rust toolchain (1.70+) for Rust output

### Install from Source

```bash
git clone https://github.com/yourusername/Embedded_ML.git
cd Embedded_ML
pip install -r requirements.txt
python setup.py install
```

### Install for Development

```bash
pip install -r requirements-dev.txt
pip install -e .
```

## Quick Start

### 1. Prepare Your Model

First, ensure your PyTorch model is saved as a complete model (not just state dict):

```python
import torch
import torch.nn as nn

# Your model definition
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Train your model...
model = MyModel()
# ... training code ...

# Save the complete model
torch.save(model, 'my_model.pt')
```

### 2. Convert to C++

```bash
python src/cli/convert.py \
  --model my_model.pt \
  --format cpp \
  --output ./output \
  --name MyModel \
  --input-shape 1,10
```

This will generate:
- `MyModel.hpp` - Header file with model class
- `MyModel.cpp` - Implementation file
- `example.cpp` - Example usage
- `CMakeLists.txt` - Build configuration

### 3. Build C++ Code

```bash
cd output/cpp
mkdir build && cd build
cmake ..
make
./MyModel_example
```

### 4. Convert to Rust

```bash
python src/cli/convert.py \
  --model my_model.pt \
  --format rust \
  --output ./output \
  --name MyModel \
  --input-shape 1,10
```

This will generate:
- `Cargo.toml` - Rust package configuration
- `src/lib.rs` - Model implementation
- `examples/inference.rs` - Example usage
- `README.md` - Documentation

### 5. Build Rust Code

```bash
cd output/rust
cargo build --release
cargo run --example inference --release
```

## Python API

You can also use the Python API directly:

```python
from embedded_ml import ModelParser, CppCodeGenerator, RustCodeGenerator
import torch

# Load model
model = torch.load('my_model.pt')

# Parse model
parser = ModelParser(model)
sample_input = torch.randn(1, 10)
model_info = parser.parse(sample_input)

# Print summary
print(parser.export_summary())

# Generate C++ code
cpp_gen = CppCodeGenerator(model_info, './output/cpp', optimize=True)
cpp_gen.generate('MyModel')

# Generate Rust code
rust_gen = RustCodeGenerator(model_info, './output/rust', optimize=True)
rust_gen.generate('MyModel')
```

## Supported Layers

### Fully Supported
- Linear (Fully Connected)
- Conv1D, Conv2D
- MaxPool2D, AvgPool2D
- BatchNorm1D, BatchNorm2D
- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- GELU, SiLU/Swish
- Flatten, Dropout (inference mode)

### Experimental
- LSTM, GRU (basic support)
- Conv3D
- TransformerEncoder (coming soon)

## Optimization Options

### Quantization

Reduce model size and improve inference speed:

```bash
python src/cli/convert.py \
  --model my_model.pt \
  --format cpp \
  --output ./output \
  --quantize int8 \
  --optimize
```

Quantization reduces model size by ~4x with minimal accuracy loss.

### Memory Optimization

For very constrained devices:

```bash
# Use smaller data types
python src/cli/convert.py \
  --model my_model.pt \
  --format cpp \
  --output ./output \
  --optimize
```

The `--optimize` flag enables:
- In-place operations where possible
- Static memory allocation
- Loop unrolling
- SIMD hints

## Target Platforms

### Tested Platforms

- **ARM Cortex-M4/M7**: STM32, NXP i.MX RT
- **ARM Cortex-A**: Raspberry Pi, NVIDIA Jetson
- **RISC-V**: SiFive FE310, ESP32-C3
- **x86/x64**: General purpose, development

### Compiler Flags

For embedded ARM:

```cmake
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_CXX_FLAGS "-O3 -mcpu=cortex-m7 -mthumb -mfloat-abi=hard")
```

For RISC-V:

```cmake
set(CMAKE_C_COMPILER riscv32-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER riscv32-unknown-elf-g++)
set(CMAKE_CXX_FLAGS "-O3 -march=rv32imac -mabi=ilp32")
```

## Troubleshooting

### Model Loading Issues

**Problem**: "Model state dict found but architecture is not saved"

**Solution**: Save the complete model, not just state dict:
```python
# ❌ Don't do this
torch.save(model.state_dict(), 'model.pt')

# ✅ Do this instead
torch.save(model, 'model.pt')
```

### Unsupported Layers

**Problem**: "Layer type X is not supported"

**Solution**: Check supported layers or implement custom conversion:
1. Check if layer is in supported list
2. Implement layer in runtime library
3. Add parsing in `model_parser.py`
4. Add code generation in `generator.py`

### Build Errors

**Problem**: Compilation errors in generated code

**Solution**:
1. Ensure you're using C++17 or later
2. Check compiler version (GCC 7+ or Clang 5+)
3. Enable verbose output: `cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON`

## Next Steps

- [API Reference](api_reference.md)
- [Optimization Guide](optimization_guide.md)
- [Architecture Overview](architecture.md)
- [Examples](../examples/)

## Getting Help

- 📖 [Documentation](https://embedded-ml.readthedocs.io)
- 💬 [Discussions](https://github.com/yourusername/Embedded_ML/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/Embedded_ML/issues)
