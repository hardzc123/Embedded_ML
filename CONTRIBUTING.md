# Contributing to Embedded ML

Thank you for your interest in contributing to Embedded ML! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version)
- Model architecture if relevant

### Suggesting Features

We welcome feature suggestions! Please open an issue with:
- Clear description of the feature
- Use case and motivation
- Proposed API or interface (if applicable)

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/Embedded_ML.git
   cd Embedded_ML
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Make your changes**
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

5. **Run tests**
   ```bash
   pytest tests/
   ```

6. **Format code**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

7. **Commit and push**
   ```bash
   git add .
   git commit -m "Add feature: description"
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request**
   - Describe your changes
   - Reference any related issues
   - Ensure CI passes

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use Black for automatic formatting
- Use type hints where appropriate
- Write docstrings for public APIs

Example:
```python
def convert_model(model: nn.Module, format: str) -> Dict[str, Any]:
    """
    Convert PyTorch model to target format.

    Args:
        model: PyTorch model to convert
        format: Target format ('cpp' or 'rust')

    Returns:
        Dictionary with conversion results
    """
    pass
```

### Testing

- Write unit tests for new features
- Ensure existing tests pass
- Aim for high code coverage
- Test on multiple platforms if possible

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update docs/ for significant features
- Include examples for new features

### C++ Guidelines

- Use modern C++17 features
- Avoid external dependencies
- Write portable code
- Comment complex algorithms
- Keep headers self-contained

### Rust Guidelines

- Follow Rust idioms and best practices
- Use cargo fmt for formatting
- Use cargo clippy for linting
- Minimize unsafe code
- Document public APIs

## Project Structure

```
Embedded_ML/
├── src/                    # Python source code
│   ├── parser/            # Model parsing
│   ├── codegen/           # Code generation
│   │   ├── cpp/          # C++ generator
│   │   └── rust/         # Rust generator
│   ├── optimization/      # Optimization tools
│   └── cli/              # CLI tools
├── runtime/               # Runtime libraries
│   ├── cpp/              # C++ runtime
│   └── rust/             # Rust runtime
├── examples/             # Example models
├── tests/                # Test suite
├── docs/                 # Documentation
└── benchmarks/           # Performance benchmarks
```

## Adding Support for New Layers

To add support for a new layer type:

1. **Add parsing in `model_parser.py`**:
   ```python
   elif isinstance(module, nn.NewLayer):
       return LayerInfo(
           name=name,
           layer_type='NewLayer',
           params={...},
           weights={...}
       )
   ```

2. **Implement in C++ runtime** (`runtime/cpp/include/layers.hpp`):
   ```cpp
   template<typename T>
   class NewLayer {
   public:
       Tensor<T> forward(const Tensor<T>& input) const;
   };
   ```

3. **Implement in Rust runtime** (`runtime/rust/src/layers.rs`):
   ```rust
   pub struct NewLayer { }
   impl NewLayer {
       pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
           // Implementation
       }
   }
   ```

4. **Add code generation** in both `cpp/generator.py` and `rust/generator.py`

5. **Add tests** in `tests/test_layers.py`

6. **Update documentation**

## Release Process

1. Update version in `setup.py` and `src/__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish to PyPI (maintainers only)

## Community

- Be respectful and inclusive
- Help others in discussions
- Share your use cases and success stories
- Provide constructive feedback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion if you have questions!
