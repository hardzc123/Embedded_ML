"""
Rust Code Generator - Generate optimized Rust code from PyTorch models
"""
import numpy as np
from typing import Dict, List, Any
import os


class RustCodeGenerator:
    """Generate Rust inference code from parsed PyTorch models"""

    def __init__(self, model_info: Dict[str, Any], output_dir: str, optimize: bool = True):
        self.model_info = model_info
        self.output_dir = output_dir
        self.optimize = optimize
        self.layers = model_info['layers']
        self.model_name = "model"

    def generate(self, model_name: str = "model") -> None:
        """Generate complete Rust inference code"""
        self.model_name = model_name.lower()

        os.makedirs(self.output_dir, exist_ok=True)

        # Generate Cargo.toml
        self._generate_cargo_toml()

        # Generate lib.rs with model implementation
        self._generate_lib()

        # Generate example
        self._generate_example()

        # Generate README
        self._generate_readme()

    def _generate_cargo_toml(self) -> None:
        """Generate Cargo.toml"""
        cargo_toml = []
        cargo_toml.append("[package]")
        cargo_toml.append(f'name = "{self.model_name}"')
        cargo_toml.append('version = "0.1.0"')
        cargo_toml.append('edition = "2021"')
        cargo_toml.append("")
        cargo_toml.append("[dependencies]")
        cargo_toml.append("# No external dependencies")
        cargo_toml.append("")
        cargo_toml.append("[profile.release]")
        cargo_toml.append("opt-level = 3")
        cargo_toml.append("lto = true")
        cargo_toml.append("codegen-units = 1")
        cargo_toml.append("panic = 'abort'")
        cargo_toml.append("")
        cargo_toml.append("# Optimize for size (embedded)")
        cargo_toml.append("[profile.release-embedded]")
        cargo_toml.append("inherits = 'release'")
        cargo_toml.append("opt-level = 'z'")
        cargo_toml.append("strip = true")

        with open(os.path.join(self.output_dir, "Cargo.toml"), 'w') as f:
            f.write('\n'.join(cargo_toml))

    def _generate_lib(self) -> None:
        """Generate lib.rs with model implementation"""
        lib_code = []
        lib_code.append("//! Auto-generated Rust inference code")
        lib_code.append(f"//! Model: {self.model_name}")
        lib_code.append("//! No external dependencies - uses only Rust std library")
        lib_code.append("")
        lib_code.append("#![allow(clippy::excessive_precision)]")
        lib_code.append("")

        # Tensor structure
        lib_code.append("/// Lightweight tensor for inference")
        lib_code.append("#[derive(Clone)]")
        lib_code.append("pub struct Tensor {")
        lib_code.append("    pub data: Vec<f32>,")
        lib_code.append("    pub shape: Vec<usize>,")
        lib_code.append("}")
        lib_code.append("")

        lib_code.append("impl Tensor {")
        lib_code.append("    pub fn new(shape: Vec<usize>) -> Self {")
        lib_code.append("        let size: usize = shape.iter().product();")
        lib_code.append("        Self {")
        lib_code.append("            data: vec![0.0; size],")
        lib_code.append("            shape,")
        lib_code.append("        }")
        lib_code.append("    }")
        lib_code.append("")
        lib_code.append("    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Self {")
        lib_code.append("        Self { data, shape }")
        lib_code.append("    }")
        lib_code.append("}")
        lib_code.append("")

        # Helper functions
        lib_code.append("// Activation functions")
        lib_code.append("fn relu(x: &mut [f32]) {")
        lib_code.append("    for val in x.iter_mut() {")
        lib_code.append("        *val = val.max(0.0);")
        lib_code.append("    }")
        lib_code.append("}")
        lib_code.append("")

        lib_code.append("fn sigmoid(x: &mut [f32]) {")
        lib_code.append("    for val in x.iter_mut() {")
        lib_code.append("        *val = 1.0 / (1.0 + (-*val).exp());")
        lib_code.append("    }")
        lib_code.append("}")
        lib_code.append("")

        lib_code.append("fn tanh_act(x: &mut [f32]) {")
        lib_code.append("    for val in x.iter_mut() {")
        lib_code.append("        *val = val.tanh();")
        lib_code.append("    }")
        lib_code.append("}")
        lib_code.append("")

        lib_code.append("fn softmax(x: &mut [f32]) {")
        lib_code.append("    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);")
        lib_code.append("    let mut sum = 0.0;")
        lib_code.append("    for val in x.iter_mut() {")
        lib_code.append("        *val = (*val - max_val).exp();")
        lib_code.append("        sum += *val;")
        lib_code.append("    }")
        lib_code.append("    for val in x.iter_mut() {")
        lib_code.append("        *val /= sum;")
        lib_code.append("    }")
        lib_code.append("}")
        lib_code.append("")

        # Matrix multiplication
        lib_code.append("// Matrix multiply: (m, k) @ (k, n) -> (m, n)")
        lib_code.append("fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {")
        lib_code.append("    for i in 0..m {")
        lib_code.append("        for j in 0..n {")
        lib_code.append("            let mut sum = 0.0;")
        lib_code.append("            for p in 0..k {")
        lib_code.append("                sum += a[i * k + p] * b[p * n + j];")
        lib_code.append("            }")
        lib_code.append("            out[i * n + j] = sum;")
        lib_code.append("        }")
        lib_code.append("    }")
        lib_code.append("}")
        lib_code.append("")

        # Model struct
        lib_code.append(f"/// {self.model_name.capitalize()} - Auto-generated inference model")
        lib_code.append(f"pub struct {self.model_name.capitalize()} {{")

        # Add weight fields
        for i, layer in enumerate(self.layers):
            if layer.weights and any(w is not None for w in layer.weights.values()):
                lib_code.append(f"    // Layer {i}: {layer.name} ({layer.layer_type})")
                for weight_name, weight_data in layer.weights.items():
                    if weight_data is not None:
                        sanitized_name = layer.name.replace('.', '_')
                        lib_code.append(f"    {sanitized_name}_{weight_name}: Vec<f32>,")

        lib_code.append("}")
        lib_code.append("")

        # Implementation
        lib_code.append(f"impl {self.model_name.capitalize()} {{")
        lib_code.append("    /// Create a new model instance with pre-loaded weights")
        lib_code.append("    pub fn new() -> Self {")
        lib_code.append("        Self {")

        # Initialize weights
        for i, layer in enumerate(self.layers):
            if layer.weights:
                for weight_name, weight_data in layer.weights.items():
                    if weight_data is not None:
                        sanitized_name = layer.name.replace('.', '_')
                        flat_weights = weight_data.flatten().tolist()

                        # Write weights in chunks to avoid extremely long lines
                        lib_code.append(f"            {sanitized_name}_{weight_name}: vec![")
                        chunk_size = 8
                        for j in range(0, len(flat_weights), chunk_size):
                            chunk = flat_weights[j:j+chunk_size]
                            weights_str = ', '.join(f'{w}' for w in chunk)
                            lib_code.append(f"                {weights_str},")
                        lib_code.append("            ],")

        lib_code.append("        }")
        lib_code.append("    }")
        lib_code.append("")

        # Forward method
        lib_code.append("    /// Run inference on input data")
        lib_code.append("    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {")
        lib_code.append("        let mut x = input;")
        lib_code.append("")

        # Generate forward pass
        for i, layer in enumerate(self.layers):
            lib_code.append(f"        // Layer {i}: {layer.name} ({layer.layer_type})")

            if layer.layer_type == 'Linear':
                in_features = layer.params['in_features']
                out_features = layer.params['out_features']
                has_bias = layer.params['bias']
                sanitized_name = layer.name.replace('.', '_')

                lib_code.append(f"        let mut tmp = vec![0.0; {out_features}];")
                lib_code.append(f"        for i in 0..{out_features} {{")
                lib_code.append(f"            for j in 0..{in_features} {{")
                lib_code.append(f"                tmp[i] += x[j] * self.{sanitized_name}_weight[i * {in_features} + j];")
                lib_code.append(f"            }}")
                if has_bias:
                    lib_code.append(f"            tmp[i] += self.{sanitized_name}_bias[i];")
                lib_code.append(f"        }}")
                lib_code.append(f"        x = tmp;")

            elif layer.layer_type == 'ReLU':
                lib_code.append(f"        relu(&mut x);")

            elif layer.layer_type == 'Sigmoid':
                lib_code.append(f"        sigmoid(&mut x);")

            elif layer.layer_type == 'Tanh':
                lib_code.append(f"        tanh_act(&mut x);")

            elif layer.layer_type == 'Softmax':
                lib_code.append(f"        softmax(&mut x);")

            elif layer.layer_type == 'Flatten':
                lib_code.append(f"        // Already flattened in vector representation")

            elif layer.layer_type == 'Dropout':
                lib_code.append(f"        // Dropout is identity in inference mode")

            lib_code.append("")

        lib_code.append("        x")
        lib_code.append("    }")

        # Get dimensions
        if self.model_info.get('input_shape'):
            input_size = int(np.prod([s for s in self.model_info['input_shape'] if s != 1]))
        else:
            input_size = self.layers[0].params.get('in_features', 0)

        if self.model_info.get('output_shape'):
            output_size = int(np.prod([s for s in self.model_info['output_shape'] if s != 1]))
        else:
            output_size = self.layers[-1].params.get('out_features', 0)

        lib_code.append("")
        lib_code.append("    /// Get input size")
        lib_code.append(f"    pub fn input_size(&self) -> usize {{ {input_size} }}")
        lib_code.append("")
        lib_code.append("    /// Get output size")
        lib_code.append(f"    pub fn output_size(&self) -> usize {{ {output_size} }}")

        lib_code.append("}")
        lib_code.append("")

        # Default implementation
        lib_code.append(f"impl Default for {self.model_name.capitalize()} {{")
        lib_code.append("    fn default() -> Self {")
        lib_code.append("        Self::new()")
        lib_code.append("    }")
        lib_code.append("}")

        with open(os.path.join(self.output_dir, "src", "lib.rs"), 'w') as f:
            f.write('\n'.join(lib_code))

    def _generate_example(self) -> None:
        """Generate example usage"""
        example_code = []
        example_code.append(f"use {self.model_name}::{self.model_name.capitalize()};")
        example_code.append("")
        example_code.append("fn main() {")
        example_code.append("    // Create model instance")
        example_code.append(f"    let model = {self.model_name.capitalize()}::new();")
        example_code.append("")
        example_code.append("    // Prepare input")
        example_code.append("    let input = vec![0.0; model.input_size()];")
        example_code.append("    // TODO: Fill input with your data")
        example_code.append("")
        example_code.append("    // Run inference")
        example_code.append("    let output = model.forward(input);")
        example_code.append("")
        example_code.append("    // Print results")
        example_code.append("    println!(\"Output (size {}): {:?}\", output.len(), output);")
        example_code.append("}")

        os.makedirs(os.path.join(self.output_dir, "examples"), exist_ok=True)
        with open(os.path.join(self.output_dir, "examples", "inference.rs"), 'w') as f:
            f.write('\n'.join(example_code))

    def _generate_readme(self) -> None:
        """Generate README"""
        readme = []
        readme.append(f"# {self.model_name.capitalize()}")
        readme.append("")
        readme.append("Auto-generated Rust inference code for embedded systems.")
        readme.append("")
        readme.append("## Building")
        readme.append("")
        readme.append("```bash")
        readme.append("# Standard build")
        readme.append("cargo build --release")
        readme.append("")
        readme.append("# Optimized for size (embedded)")
        readme.append("cargo build --profile release-embedded")
        readme.append("```")
        readme.append("")
        readme.append("## Running Example")
        readme.append("")
        readme.append("```bash")
        readme.append("cargo run --example inference --release")
        readme.append("```")
        readme.append("")
        readme.append("## Usage")
        readme.append("")
        readme.append("```rust")
        readme.append(f"use {self.model_name}::{self.model_name.capitalize()};")
        readme.append("")
        readme.append(f"let model = {self.model_name.capitalize()}::new();")
        readme.append("let input = vec![0.0; model.input_size()];")
        readme.append("let output = model.forward(input);")
        readme.append("```")

        with open(os.path.join(self.output_dir, "README.md"), 'w') as f:
            f.write('\n'.join(readme))
