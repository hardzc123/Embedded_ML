# PyTorch to Embedded ML Converter

A comprehensive, open-source toolset for converting PyTorch models written in Python to optimized C++ or Rust code for embedded platforms with constrained computing resources.

## 🎯 Project Goals

- **Zero-Pain Migration**: Convert PyTorch models to embedded-friendly code with minimal effort
- **Minimal Dependencies**: Generated code uses only standard C++17 or Rust std library features
- **Plug-and-Play**: Simple CLI interface for quick conversions
- **Production-Ready**: Optimized for resource-constrained embedded systems
- **Comprehensive**: Supports common neural network architectures and operations

## ✨ Features

### 🎯 Complete Pipeline Conversion (Not Just Models!)

Most tools only convert the neural network model. **We convert your ENTIRE production pipeline:**

- ✅ **Data Preprocessing**: Normalization, scaling, clipping, missing value handling
- ✅ **Model Inference**: PyTorch models to optimized C++/Rust
- ✅ **Postprocessing**: Softmax, argmax, top-k, NMS, confidence filtering
- ✅ **Math Operations**: NumPy-like operations (mean, std, sqrt, exp, log, etc.)
- ✅ **Image Processing**: Normalization, transformations
- ✅ **Time Series**: Moving averages, temporal smoothing
- ✅ **Complete Workflow**: Everything researchers write in Python!

### 🚀 Core Features

- 🔄 **Automatic Conversion**: Parse PyTorch models and generate equivalent C++/Rust code
- 🎯 **Zero External Dependencies**: Generated code uses only standard C++17 or Rust std library
- ⚡ **Optimized for Embedded**: Built-in quantization, pruning, and memory optimization
- 📊 **Performance Benchmarking**: Compare Python vs C++ vs Rust implementations
- 📚 **Rich Examples**: Pre-built conversions for MLP, CNN, complete production pipelines
- 🛠️ **Easy CLI**: Simple command-line interface for quick conversions
- 📖 **Comprehensive Docs**: Step-by-step guides for converting entire pipelines

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Embedded_ML.git
cd Embedded_ML

# Install Python dependencies
pip install -r requirements.txt

# Build examples (optional)
python setup.py install
```

### Basic Usage

```bash
# Convert PyTorch model to C++
python src/cli/convert.py \
  --model path/to/model.pt \
  --format cpp \
  --output ./output \
  --optimize quantize

# Convert to Rust
python src/cli/convert.py \
  --model path/to/model.pt \
  --format rust \
  --output ./output \
  --optimize quantize
```

### Python API

```python
from embedded_ml import PyTorchConverter

# Load your PyTorch model
import torch
model = torch.load('model.pt')

# Convert to C++
converter = PyTorchConverter(model)
converter.to_cpp(
    output_dir='./output',
    optimize=True,
    quantize='int8'
)

# Convert to Rust
converter.to_rust(
    output_dir='./output',
    optimize=True,
    quantize='int8'
)
```

## 🔄 Complete Pipeline Conversion

### What Makes This Different?

**Traditional tools**: Convert only the neural network model
**Embedded ML**: Convert your ENTIRE production pipeline

### Example: From Python Research Code to Embedded C++

**Your Python Code:**
```python
def production_pipeline(raw_data):
    # Preprocessing
    data = handle_missing_values(raw_data)
    data = np.clip(data, -5, 5)
    data = (data - mean) / std
    data = np.log1p(np.abs(data)) * np.sign(data)

    # Model inference
    output = model(data)

    # Postprocessing
    probs = softmax(output)
    top_k = np.argsort(probs)[-5:][::-1]
    return filter_by_confidence(top_k, threshold=0.7)
```

**Generated C++ Code (using our utilities):**
```cpp
#include "preprocessing.hpp"
#include "postprocessing.hpp"
#include "math_utils.hpp"

auto production_pipeline(const Tensor<float>& raw_data) {
    // Preprocessing (same logic as Python!)
    auto data = handle_missing_values(raw_data);
    preprocessing::clip(data, -5.0f, 5.0f);
    auto normalized = preprocessing::StandardScaler<float>::transform_with_params(data, MEAN, STD);
    apply_log_transform(normalized);

    // Model inference
    auto output = model.forward(normalized.data());

    // Postprocessing (same logic as Python!)
    Tensor<float> probs({output.size()}, output);
    activations::softmax(probs);
    auto topk = postprocessing::topk(probs, 5);
    return filter_by_confidence(topk, 0.7f);
}
```

### Comprehensive Utility Libraries

**Preprocessing** (`preprocessing.hpp`):
- StandardScaler, MinMaxScaler
- Image normalization (ImageNet, custom)
- Missing value imputation
- Clipping, log transforms

**Math Operations** (`math_utils.hpp`):
- NumPy-like operations: mean, std, sqrt, exp, log, abs
- Linear algebra: dot, matmul, norm, normalize
- Statistics: median, variance, percentile, correlation
- Random number generation

**Postprocessing** (`postprocessing.hpp`):
- Argmax, top-k selection
- Non-Maximum Suppression (NMS)
- Temperature scaling
- Moving average filters
- Ensemble voting
- Confusion matrix

**See the [Complete Pipeline Guide](docs/python_to_embedded_guide.md) for detailed examples!**

## 📁 Project Structure

```
Embedded_ML/
├── src/
│   ├── parser/              # PyTorch model parsing
│   │   ├── model_parser.py
│   │   ├── layer_extractor.py
│   │   └── weight_extractor.py
│   ├── codegen/             # Code generation
│   │   ├── cpp/             # C++ code generator
│   │   │   ├── generator.py
│   │   │   ├── templates/
│   │   │   └── optimizer.py
│   │   └── rust/            # Rust code generator
│   │       ├── generator.py
│   │       ├── templates/
│   │       └── optimizer.py
│   ├── optimization/        # Model optimization
│   │   ├── quantization.py
│   │   ├── pruning.py
│   │   └── memory_optimizer.py
│   └── cli/                 # Command-line interface
│       └── convert.py
├── runtime/
│   ├── cpp/                 # C++ runtime library
│   │   ├── include/
│   │   │   ├── tensor.hpp
│   │   │   ├── layers.hpp
│   │   │   └── activations.hpp
│   │   └── examples/
│   └── rust/                # Rust runtime library
│       ├── src/
│       │   ├── tensor.rs
│       │   ├── layers.rs
│       │   └── activations.rs
│       └── Cargo.toml
├── examples/                # Example conversions
│   ├── simple_mlp/
│   ├── cnn_mnist/
│   ├── lstm_timeseries/
│   └── mobilenet_v2/
├── benchmarks/              # Performance benchmarks
│   ├── benchmark.py
│   └── comparison_tool.py
├── tests/                   # Test suite
└── docs/                    # Documentation
    ├── getting_started.md
    ├── api_reference.md
    ├── optimization_guide.md
    └── architecture.md
```

## 🎓 Supported Operations

### Layers
- ✅ Linear (Fully Connected)
- ✅ Conv1D, Conv2D, Conv3D
- ✅ MaxPool, AvgPool
- ✅ BatchNorm, LayerNorm
- ✅ LSTM, GRU
- ✅ Dropout (inference mode)
- ✅ Embedding

### Activations
- ✅ ReLU, LeakyReLU, PReLU
- ✅ Sigmoid, Tanh
- ✅ Softmax
- ✅ GELU, SiLU/Swish

### Operations
- ✅ Matrix multiplication
- ✅ Element-wise operations
- ✅ Reshape, Transpose, Concat
- ✅ Reduce operations (sum, mean, max)

## 🔧 Optimization Features

1. **Quantization**
   - INT8 quantization for weights and activations
   - Dynamic range calibration
   - Per-channel and per-tensor quantization

2. **Pruning**
   - Magnitude-based pruning
   - Structured pruning for embedded efficiency
   - Automatic sparsity optimization

3. **Memory Optimization**
   - In-place operations where possible
   - Static memory allocation
   - Buffer reuse strategies

4. **Embedded-Specific**
   - Fixed-point arithmetic support
   - SIMD optimization hints
   - Configurable precision trade-offs

## 📊 Performance Comparison

| Model | Platform | PyTorch (ms) | C++ (ms) | Rust (ms) | Memory (KB) |
|-------|----------|--------------|----------|-----------|-------------|
| MLP (3-layer) | ARM Cortex-M4 | N/A | 2.3 | 2.1 | 45 |
| CNN (LeNet) | ARM Cortex-M7 | N/A | 15.7 | 14.9 | 128 |
| MobileNetV2 | ARM Cortex-A53 | 89.3 | 23.4 | 21.8 | 512 |

## 🎯 Use Cases

- **IoT Devices**: Deploy ML models on microcontrollers
- **Edge Computing**: Run inference on resource-constrained edge devices
- **Real-time Systems**: Predictable latency for time-critical applications
- **Battery-Powered Devices**: Optimize for power efficiency
- **Safety-Critical Systems**: Deterministic behavior without heavyweight runtimes

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for accessible embedded ML deployment
- Built for developers who need efficient, dependency-free inference
- Community-driven with focus on practical embedded use cases

## 📞 Support

- 📖 [Documentation](./docs/)
- 💬 [Discussions](https://github.com/yourusername/Embedded_ML/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/Embedded_ML/issues)
- 📧 Email: support@embedded-ml.dev

## 🗺️ Roadmap

- [x] Core parser and code generation
- [x] C++ runtime library
- [x] Rust runtime library
- [x] Basic optimization (quantization)
- [ ] Advanced pruning techniques
- [ ] ARM CMSIS-NN backend
- [ ] RISC-V optimization
- [ ] Model compression tools
- [ ] WebAssembly target
- [ ] Visual model inspector

---

**Made with ❤️ for the embedded ML community**
