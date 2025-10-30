# Architecture Overview

This document explains the architecture and design decisions of Embedded ML.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PyTorch Model (.pt)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Parser                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Extract layers and parameters                      │  │
│  │  • Extract weights and biases                         │  │
│  │  • Infer input/output shapes                          │  │
│  │  • Analyze model structure                            │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Model Info    │
              │  (JSON-like)   │
              └───┬────────┬───┘
                  │        │
        ┌─────────┘        └─────────┐
        ▼                            ▼
┌──────────────────┐        ┌──────────────────┐
│  C++ Generator   │        │  Rust Generator  │
│  ┌────────────┐  │        │  ┌────────────┐  │
│  │ Templates  │  │        │  │ Templates  │  │
│  │ Optimizer  │  │        │  │ Optimizer  │  │
│  └────────────┘  │        │  └────────────┘  │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│   C++ Code       │        │   Rust Code      │
│   • .hpp/.cpp    │        │   • Cargo.toml   │
│   • CMakeLists   │        │   • lib.rs       │
│   • Example      │        │   • Example      │
└──────────────────┘        └──────────────────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
         ┌───────────────────────┐
         │  Embedded Platform    │
         │  • ARM Cortex         │
         │  • RISC-V             │
         │  • ESP32              │
         └───────────────────────┘
```

## Component Details

### 1. Model Parser

**Location**: `src/parser/model_parser.py`

**Responsibilities**:
- Parse PyTorch nn.Module objects
- Extract layer-by-layer information
- Extract weights and parameters
- Infer tensor shapes through forward pass
- Generate model summary

**Key Classes**:
- `ModelParser`: Main parser class
- `LayerInfo`: Container for layer information

**Supported Layers**:
- Linear, Conv1D/2D/3D
- BatchNorm1D/2D
- LSTM, GRU
- All activation functions
- Pooling layers

### 2. Code Generators

#### C++ Generator

**Location**: `src/codegen/cpp/generator.py`

**Outputs**:
- Header file (.hpp) with class declaration
- Implementation file (.cpp) with forward pass
- Example file with usage demo
- CMakeLists.txt for building

**Design Decisions**:
- Uses only C++17 standard library
- Template-based for flexibility
- Optimized for embedded (no exceptions/RTTI by default)
- Weights embedded in code for simplicity

#### Rust Generator

**Location**: `src/codegen/rust/generator.py`

**Outputs**:
- Cargo.toml with project configuration
- lib.rs with model implementation
- Example usage code
- README.md

**Design Decisions**:
- Uses only Rust std library
- no_std compatible
- Zero-cost abstractions
- Optimized for size with release profiles

### 3. Runtime Libraries

#### C++ Runtime

**Location**: `runtime/cpp/include/`

**Components**:
- `tensor.hpp`: Lightweight tensor implementation
- `layers.hpp`: Neural network layers
- `activations.hpp`: Activation functions

**Features**:
- Header-only for easy integration
- Template-based for type flexibility
- SIMD optimization hints
- Memory-efficient operations

#### Rust Runtime

**Location**: `runtime/rust/src/`

**Components**:
- `tensor.rs`: Tensor implementation
- `layers.rs`: Layer implementations
- `activations.rs`: Activation functions

**Features**:
- Zero dependencies
- no_std support for bare metal
- Memory-safe by design
- Optimized for embedded

### 4. Optimization Tools

**Location**: `src/optimization/`

**Components**:
- `quantization.py`: Weight and activation quantization
- `pruning.py`: Network pruning (future)
- `memory_optimizer.py`: Memory optimization (future)

**Quantization Methods**:
- Asymmetric quantization (weights + activations)
- Symmetric quantization (weights only)
- Per-channel quantization (better accuracy)
- Dynamic quantization (runtime)

### 5. CLI Tool

**Location**: `src/cli/convert.py`

**Features**:
- Simple command-line interface
- Progress reporting
- Error handling
- Multiple output formats

**Usage Flow**:
```
User Input → Validation → Model Loading → Parsing →
Code Generation → File Writing → Build Instructions
```

## Data Flow

### 1. Model Information Structure

```python
{
    'layers': [
        {
            'name': 'fc1',
            'type': 'Linear',
            'params': {'in_features': 10, 'out_features': 20},
            'weights': {
                'weight': np.array(...),
                'bias': np.array(...)
            },
            'input_shape': (1, 10),
            'output_shape': (1, 20)
        },
        # ... more layers
    ],
    'input_shape': (1, 10),
    'output_shape': (1, 2),
    'num_parameters': 1234,
    'model_type': 'MLP'
}
```

### 2. Generated C++ Structure

```cpp
class Model {
private:
    std::vector<float> fc1_weight;
    std::vector<float> fc1_bias;
    // ... more weights

public:
    Model();  // Constructor loads weights
    std::vector<float> forward(const std::vector<float>& input);
};
```

### 3. Generated Rust Structure

```rust
pub struct Model {
    fc1_weight: Vec<f32>,
    fc1_bias: Vec<f32>,
    // ... more weights
}

impl Model {
    pub fn new() -> Self { /* load weights */ }
    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> { /* inference */ }
}
```

## Design Principles

### 1. Zero Dependencies

**Why**: Embedded systems often can't use external libraries

**How**:
- Implement all operations from scratch
- Use only std library features
- No dynamic linking

### 2. Memory Efficiency

**Why**: Embedded devices have limited RAM

**How**:
- In-place operations where possible
- Static allocation when feasible
- Avoid unnecessary copies
- Weight quantization

### 3. Type Safety

**Why**: Catch errors at compile time

**How**:
- Strong typing in generated code
- Const correctness in C++
- Ownership system in Rust
- Template/generic programming

### 4. Portability

**Why**: Support many embedded platforms

**How**:
- Standard C++17/Rust 2021
- No platform-specific code
- Configurable via compilation flags
- No assembly (unless optional)

### 5. Ease of Use

**Why**: Lower barrier to entry

**How**:
- Simple CLI interface
- Python API for automation
- Clear documentation
- Working examples

## Extension Points

### Adding New Layer Types

1. **Parser** (`model_parser.py`):
```python
elif isinstance(module, nn.NewLayer):
    return LayerInfo(
        name=name,
        layer_type='NewLayer',
        params={...},
        weights={...}
    )
```

2. **C++ Runtime** (`layers.hpp`):
```cpp
template<typename T>
class NewLayer {
    Tensor<T> forward(const Tensor<T>& input) const;
};
```

3. **Rust Runtime** (`layers.rs`):
```rust
pub struct NewLayer { }
impl NewLayer {
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32>;
}
```

4. **Code Generation** (both generators):
```python
if layer.layer_type == 'NewLayer':
    # Generate code for this layer
```

### Adding Optimization Passes

1. Create optimizer in `src/optimization/`
2. Add CLI flag in `convert.py`
3. Apply optimization in generator
4. Document in optimization guide

### Adding Target Platforms

1. Add platform-specific compiler flags
2. Test on target hardware
3. Document platform quirks
4. Add to supported platforms list

## Performance Considerations

### Code Generation

- **Template vs Direct**: We use templates for flexibility but inline for performance
- **Loop Unrolling**: Applied automatically for small, fixed-size loops
- **SIMD Hints**: Compiler can vectorize operations with proper flags

### Runtime

- **Tensor Layout**: Row-major (C-style) for cache efficiency
- **Memory Access**: Sequential access patterns where possible
- **Operation Fusion**: Combine operations to reduce memory traffic

### Optimization Levels

1. **-O0** (Debug): No optimization, fast compilation
2. **-O2** (Default): Balanced optimization
3. **-O3** (Performance): Maximum speed
4. **-Os/-Oz** (Size): Minimize binary size

## Testing Strategy

### Unit Tests
- Test individual layers
- Test tensor operations
- Test activation functions

### Integration Tests
- Test full model conversion
- Test C++ generation
- Test Rust generation

### Validation Tests
- Compare outputs with PyTorch
- Verify numerical accuracy
- Test quantization error

### Platform Tests
- Test on different architectures
- Verify cross-compilation
- Benchmark performance

## Future Enhancements

### Planned Features
- [ ] Dynamic shapes support
- [ ] Advanced pruning
- [ ] Model compression
- [ ] ARM CMSIS-NN backend
- [ ] WebAssembly target
- [ ] Visual model inspector
- [ ] Auto-tuning for target hardware

### Under Consideration
- [ ] TensorFlow/JAX support
- [ ] Transformer optimizations
- [ ] GPU support for embedded
- [ ] Real-time profiling
- [ ] Model distillation

## Resources

- [Code Repository](https://github.com/yourusername/Embedded_ML)
- [API Documentation](api_reference.md)
- [Getting Started Guide](getting_started.md)
- [C++ vs Rust Comparison](cpp_vs_rust.md)
