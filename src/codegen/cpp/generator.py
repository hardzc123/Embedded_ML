"""
C++ Code Generator - Generate optimized C++ code from PyTorch models
"""
import numpy as np
from typing import Dict, List, Any
import os


class CppCodeGenerator:
    """Generate C++ inference code from parsed PyTorch models"""

    def __init__(self, model_info: Dict[str, Any], output_dir: str, optimize: bool = True):
        self.model_info = model_info
        self.output_dir = output_dir
        self.optimize = optimize
        self.layers = model_info['layers']
        self.model_name = "Model"

    def generate(self, model_name: str = "Model") -> None:
        """Generate complete C++ inference code"""
        self.model_name = model_name

        os.makedirs(self.output_dir, exist_ok=True)

        # Generate header file
        self._generate_header()

        # Generate implementation file
        self._generate_implementation()

        # Generate weights file
        self._generate_weights()

        # Generate example usage
        self._generate_example()

        # Generate CMakeLists.txt
        self._generate_cmake()

    def _generate_header(self) -> None:
        """Generate C++ header file"""
        header_code = []
        header_code.append(f"/**")
        header_code.append(f" * Auto-generated C++ inference code")
        header_code.append(f" * Model: {self.model_name}")
        header_code.append(f" * No external dependencies - uses only standard C++17")
        header_code.append(f" */")
        header_code.append("")
        header_code.append(f"#ifndef {self.model_name.upper()}_HPP")
        header_code.append(f"#define {self.model_name.upper()}_HPP")
        header_code.append("")
        header_code.append("#include <vector>")
        header_code.append("#include <array>")
        header_code.append("#include <memory>")
        header_code.append("")
        header_code.append("// Forward declarations")
        header_code.append("template<typename T> class Tensor;")
        header_code.append("")
        header_code.append(f"class {self.model_name} {{")
        header_code.append("public:")
        header_code.append(f"    {self.model_name}();")
        header_code.append(f"    ~{self.model_name}() = default;")
        header_code.append("")
        header_code.append("    // Run inference")
        header_code.append("    std::vector<float> forward(const std::vector<float>& input);")
        header_code.append("")
        header_code.append("    // Model info")
        header_code.append("    size_t get_input_size() const;")
        header_code.append("    size_t get_output_size() const;")
        header_code.append("")
        header_code.append("private:")
        header_code.append("    void load_weights();")
        header_code.append("")

        # Generate member variables for weights
        for i, layer in enumerate(self.layers):
            if layer.weights and any(w is not None for w in layer.weights.values()):
                header_code.append(f"    // Layer {i}: {layer.name} ({layer.layer_type})")
                for weight_name, weight_data in layer.weights.items():
                    if weight_data is not None:
                        header_code.append(f"    std::vector<float> {layer.name}_{weight_name};")

        header_code.append("};")
        header_code.append("")
        header_code.append(f"#endif // {self.model_name.upper()}_HPP")

        # Write to file
        with open(os.path.join(self.output_dir, f"{self.model_name}.hpp"), 'w') as f:
            f.write('\n'.join(header_code))

    def _generate_implementation(self) -> None:
        """Generate C++ implementation file"""
        impl_code = []
        impl_code.append(f"#include \"{self.model_name}.hpp\"")
        impl_code.append("#include <cmath>")
        impl_code.append("#include <algorithm>")
        impl_code.append("#include <numeric>")
        impl_code.append("#include <stdexcept>")
        impl_code.append("")

        # Simple tensor class inline
        impl_code.append("// Lightweight tensor for inference")
        impl_code.append("template<typename T>")
        impl_code.append("class Tensor {")
        impl_code.append("public:")
        impl_code.append("    std::vector<T> data;")
        impl_code.append("    std::vector<size_t> shape;")
        impl_code.append("    Tensor(const std::vector<size_t>& s) : shape(s) {")
        impl_code.append("        size_t size = 1;")
        impl_code.append("        for (auto dim : shape) size *= dim;")
        impl_code.append("        data.resize(size);")
        impl_code.append("    }")
        impl_code.append("    Tensor(const std::vector<size_t>& s, const std::vector<T>& d) : shape(s), data(d) {}")
        impl_code.append("};")
        impl_code.append("")

        # Helper functions
        impl_code.append("// Helper functions")
        impl_code.append("namespace {")
        impl_code.append("")

        # ReLU
        impl_code.append("inline float relu(float x) { return std::max(0.0f, x); }")
        impl_code.append("")

        # Sigmoid
        impl_code.append("inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }")
        impl_code.append("")

        # Tanh
        impl_code.append("inline float tanh_act(float x) { return std::tanh(x); }")
        impl_code.append("")

        # Softmax
        impl_code.append("inline void softmax(std::vector<float>& x) {")
        impl_code.append("    float max_val = *std::max_element(x.begin(), x.end());")
        impl_code.append("    float sum = 0.0f;")
        impl_code.append("    for (auto& val : x) {")
        impl_code.append("        val = std::exp(val - max_val);")
        impl_code.append("        sum += val;")
        impl_code.append("    }")
        impl_code.append("    for (auto& val : x) val /= sum;")
        impl_code.append("}")
        impl_code.append("")

        # Matrix multiplication
        impl_code.append("// Matrix multiply: (m, k) @ (k, n) -> (m, n)")
        impl_code.append("inline void matmul(const std::vector<float>& a, const std::vector<float>& b,")
        impl_code.append("                    std::vector<float>& out, size_t m, size_t k, size_t n) {")
        impl_code.append("    for (size_t i = 0; i < m; ++i) {")
        impl_code.append("        for (size_t j = 0; j < n; ++j) {")
        impl_code.append("            float sum = 0.0f;")
        impl_code.append("            for (size_t p = 0; p < k; ++p) {")
        impl_code.append("                sum += a[i * k + p] * b[p * n + j];")
        impl_code.append("            }")
        impl_code.append("            out[i * n + j] = sum;")
        impl_code.append("        }")
        impl_code.append("    }")
        impl_code.append("}")
        impl_code.append("")

        impl_code.append("} // anonymous namespace")
        impl_code.append("")

        # Constructor
        impl_code.append(f"{self.model_name}::{self.model_name}() {{")
        impl_code.append("    load_weights();")
        impl_code.append("}")
        impl_code.append("")

        # Load weights
        impl_code.append(f"void {self.model_name}::load_weights() {{")
        for i, layer in enumerate(self.layers):
            if layer.weights:
                for weight_name, weight_data in layer.weights.items():
                    if weight_data is not None:
                        impl_code.append(f"    // {layer.name} - {weight_name}")
                        flat_weights = weight_data.flatten().tolist()
                        impl_code.append(f"    {layer.name}_{weight_name} = {{{', '.join(f'{w}f' for w in flat_weights[:10])}")
                        if len(flat_weights) > 10:
                            impl_code.append(f"        /* ... {len(flat_weights)} values total ... */")
                        impl_code.append("    };")
        impl_code.append("}")
        impl_code.append("")

        # Forward pass
        impl_code.append(f"std::vector<float> {self.model_name}::forward(const std::vector<float>& input) {{")
        impl_code.append("    std::vector<float> x = input;")
        impl_code.append("    std::vector<float> tmp;")
        impl_code.append("")

        # Generate forward pass for each layer
        for i, layer in enumerate(self.layers):
            impl_code.append(f"    // Layer {i}: {layer.name} ({layer.layer_type})")

            if layer.layer_type == 'Linear':
                in_features = layer.params['in_features']
                out_features = layer.params['out_features']
                has_bias = layer.params['bias']

                impl_code.append(f"    tmp.resize({out_features});")
                impl_code.append(f"    std::fill(tmp.begin(), tmp.end(), 0.0f);")
                impl_code.append(f"    // Matrix multiply")
                impl_code.append(f"    for (size_t i = 0; i < {out_features}; ++i) {{")
                impl_code.append(f"        for (size_t j = 0; j < {in_features}; ++j) {{")
                impl_code.append(f"            tmp[i] += x[j] * {layer.name}_weight[i * {in_features} + j];")
                impl_code.append(f"        }}")
                if has_bias:
                    impl_code.append(f"        tmp[i] += {layer.name}_bias[i];")
                impl_code.append(f"    }}")
                impl_code.append(f"    x = tmp;")

            elif layer.layer_type == 'ReLU':
                impl_code.append(f"    std::transform(x.begin(), x.end(), x.begin(), relu);")

            elif layer.layer_type == 'Sigmoid':
                impl_code.append(f"    std::transform(x.begin(), x.end(), x.begin(), sigmoid);")

            elif layer.layer_type == 'Tanh':
                impl_code.append(f"    std::transform(x.begin(), x.end(), x.begin(), tanh_act);")

            elif layer.layer_type == 'Softmax':
                impl_code.append(f"    softmax(x);")

            elif layer.layer_type == 'Flatten':
                impl_code.append(f"    // Already flattened in vector representation")

            elif layer.layer_type == 'Dropout':
                impl_code.append(f"    // Dropout is identity in inference mode")

            impl_code.append("")

        impl_code.append("    return x;")
        impl_code.append("}")
        impl_code.append("")

        # Get input/output size
        if self.model_info.get('input_shape'):
            input_size = np.prod([s for s in self.model_info['input_shape'] if s != 1])
        else:
            input_size = self.layers[0].params.get('in_features', 0)

        if self.model_info.get('output_shape'):
            output_size = np.prod([s for s in self.model_info['output_shape'] if s != 1])
        else:
            output_size = self.layers[-1].params.get('out_features', 0)

        impl_code.append(f"size_t {self.model_name}::get_input_size() const {{ return {input_size}; }}")
        impl_code.append(f"size_t {self.model_name}::get_output_size() const {{ return {output_size}; }}")

        # Write to file
        with open(os.path.join(self.output_dir, f"{self.model_name}.cpp"), 'w') as f:
            f.write('\n'.join(impl_code))

    def _generate_weights(self) -> None:
        """Generate separate weights file for easier updates"""
        weights_code = []
        weights_code.append(f"// Weights for {self.model_name}")
        weights_code.append(f"// Total parameters: {self.model_info.get('num_parameters', 0):,}")
        weights_code.append("")

        for i, layer in enumerate(self.layers):
            if layer.weights:
                weights_code.append(f"// Layer {i}: {layer.name} ({layer.layer_type})")
                for weight_name, weight_data in layer.weights.items():
                    if weight_data is not None:
                        weights_code.append(f"// {weight_name}: shape {weight_data.shape}")
                        weights_code.append("")

        with open(os.path.join(self.output_dir, "weights_info.txt"), 'w') as f:
            f.write('\n'.join(weights_code))

    def _generate_example(self) -> None:
        """Generate example usage code"""
        example_code = []
        example_code.append("#include <iostream>")
        example_code.append("#include <vector>")
        example_code.append(f"#include \"{self.model_name}.hpp\"")
        example_code.append("")
        example_code.append("int main() {")
        example_code.append(f"    // Create model instance")
        example_code.append(f"    {self.model_name} model;")
        example_code.append("")
        example_code.append(f"    // Prepare input (size: {model.get_input_size() if hasattr(model, 'get_input_size') else 'N'})")
        example_code.append(f"    std::vector<float> input(model.get_input_size(), 0.0f);")
        example_code.append(f"    // TODO: Fill input with your data")
        example_code.append("")
        example_code.append(f"    // Run inference")
        example_code.append(f"    std::vector<float> output = model.forward(input);")
        example_code.append("")
        example_code.append(f"    // Print results")
        example_code.append(f"    std::cout << \"Output (size \" << output.size() << \"):\" << std::endl;")
        example_code.append(f"    for (size_t i = 0; i < output.size(); ++i) {{")
        example_code.append(f"        std::cout << \"  [\" << i << \"] = \" << output[i] << std::endl;")
        example_code.append(f"    }}")
        example_code.append("")
        example_code.append(f"    return 0;")
        example_code.append("}")

        with open(os.path.join(self.output_dir, "example.cpp"), 'w') as f:
            f.write('\n'.join(example_code))

    def _generate_cmake(self) -> None:
        """Generate CMakeLists.txt for easy building"""
        cmake_code = []
        cmake_code.append("cmake_minimum_required(VERSION 3.10)")
        cmake_code.append(f"project({self.model_name})")
        cmake_code.append("")
        cmake_code.append("set(CMAKE_CXX_STANDARD 17)")
        cmake_code.append("set(CMAKE_CXX_STANDARD_REQUIRED ON)")
        cmake_code.append("")
        cmake_code.append("# Optimization flags for embedded")
        cmake_code.append("set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -O3 -march=native\")")
        cmake_code.append("set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -ffast-math -fno-exceptions -fno-rtti\")")
        cmake_code.append("")
        cmake_code.append(f"# Model library")
        cmake_code.append(f"add_library({self.model_name}_lib {self.model_name}.cpp)")
        cmake_code.append("")
        cmake_code.append(f"# Example executable")
        cmake_code.append(f"add_executable({self.model_name}_example example.cpp)")
        cmake_code.append(f"target_link_libraries({self.model_name}_example {self.model_name}_lib)")
        cmake_code.append("")
        cmake_code.append("# For ARM embedded targets, uncomment:")
        cmake_code.append("# set(CMAKE_SYSTEM_NAME Generic)")
        cmake_code.append("# set(CMAKE_SYSTEM_PROCESSOR arm)")
        cmake_code.append("# set(CMAKE_C_COMPILER arm-none-eabi-gcc)")
        cmake_code.append("# set(CMAKE_CXX_COMPILER arm-none-eabi-g++)")

        with open(os.path.join(self.output_dir, "CMakeLists.txt"), 'w') as f:
            f.write('\n'.join(cmake_code))
