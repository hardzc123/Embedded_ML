"""
Pipeline Analyzer - Analyze and convert Python data processing pipelines to C++/Rust

This tool helps researchers convert their entire Python preprocessing/postprocessing
pipelines (not just models) to C++/Rust for production deployment.
"""

import ast
import inspect
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class Operation:
    """Represents a single operation in the pipeline"""
    name: str
    op_type: str  # 'numpy', 'sklearn', 'custom', 'builtin'
    function: str
    params: Dict[str, Any]
    input_vars: List[str]
    output_var: str
    line_number: int


class PipelineAnalyzer:
    """
    Analyze Python code to extract data processing pipeline

    Example:
        analyzer = PipelineAnalyzer()
        pipeline = analyzer.analyze_function(preprocess_data)
        cpp_code = analyzer.to_cpp(pipeline)
        rust_code = analyzer.to_rust(pipeline)
    """

    def __init__(self):
        self.operations = []
        self.numpy_to_cpp = {
            'mean': 'embedded_ml::math::mean',
            'std': 'embedded_ml::math::std_deviation',
            'min': 'tensor.min()',
            'max': 'tensor.max()',
            'sqrt': 'embedded_ml::math::sqrt',
            'exp': 'embedded_ml::math::exp',
            'log': 'embedded_ml::math::log',
            'abs': 'embedded_ml::math::abs',
            'sum': 'tensor.sum()',
            'clip': 'embedded_ml::preprocessing::clip',
            'normalize': 'embedded_ml::math::normalize',
        }

        self.sklearn_to_cpp = {
            'StandardScaler': 'embedded_ml::preprocessing::StandardScaler',
            'MinMaxScaler': 'embedded_ml::preprocessing::MinMaxScaler',
        }

    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """
        Analyze a Python function to extract pipeline operations

        Args:
            func: Python function to analyze

        Returns:
            Dictionary containing pipeline information
        """
        source = inspect.getsource(func)
        tree = ast.parse(source)

        pipeline = {
            'function_name': func.__name__,
            'operations': [],
            'inputs': [],
            'outputs': [],
            'variables': {}
        }

        # Extract function arguments
        func_def = tree.body[0]
        if isinstance(func_def, ast.FunctionDef):
            pipeline['inputs'] = [arg.arg for arg in func_def.args.args]

        # Analyze each statement
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                op = self._analyze_assignment(node)
                if op:
                    pipeline['operations'].append(op)

        return pipeline

    def _analyze_assignment(self, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Analyze an assignment statement"""
        if not isinstance(node.value, ast.Call):
            return None

        operation = {
            'type': 'unknown',
            'function': '',
            'params': {},
            'output': ''
        }

        # Get output variable name
        if node.targets and isinstance(node.targets[0], ast.Name):
            operation['output'] = node.targets[0].id

        # Analyze function call
        call = node.value
        if isinstance(call.func, ast.Attribute):
            # Method call (e.g., x.mean())
            if isinstance(call.func.value, ast.Name):
                operation['type'] = 'method'
                operation['object'] = call.func.value.id
                operation['function'] = call.func.attr
        elif isinstance(call.func, ast.Name):
            # Function call (e.g., np.mean(x))
            operation['type'] = 'function'
            operation['function'] = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # Module function call (e.g., np.mean)
            if isinstance(call.func.value, ast.Name):
                operation['module'] = call.func.value.id
                operation['function'] = call.func.attr
                operation['type'] = 'module_function'

        # Extract parameters
        for arg in call.args:
            if isinstance(arg, ast.Name):
                if 'inputs' not in operation:
                    operation['inputs'] = []
                operation['inputs'].append(arg.id)
            elif isinstance(arg, ast.Constant):
                if 'constants' not in operation:
                    operation['constants'] = []
                operation['constants'].append(arg.value)

        # Extract keyword arguments
        for keyword in call.keywords:
            if isinstance(keyword.value, ast.Constant):
                operation['params'][keyword.arg] = keyword.value.value

        return operation

    def to_cpp(self, pipeline: Dict[str, Any]) -> str:
        """
        Generate C++ code from pipeline

        Args:
            pipeline: Pipeline dictionary from analyze_function

        Returns:
            C++ code as string
        """
        code = []
        code.append("// Auto-generated C++ data processing pipeline")
        code.append("#include \"preprocessing.hpp\"")
        code.append("#include \"postprocessing.hpp\"")
        code.append("#include \"math_utils.hpp\"")
        code.append("#include \"tensor.hpp\"")
        code.append("")
        code.append("using namespace embedded_ml;")
        code.append("")

        # Function signature
        code.append(f"Tensor<float> {pipeline['function_name']}(")
        for i, inp in enumerate(pipeline['inputs']):
            if i > 0:
                code.append(",")
            code.append(f"    const Tensor<float>& {inp}")
        code.append(") {")

        # Generate operations
        for op in pipeline['operations']:
            cpp_line = self._operation_to_cpp(op)
            if cpp_line:
                code.append(f"    {cpp_line}")

        # Return statement (assume last operation is the result)
        if pipeline['operations']:
            last_var = pipeline['operations'][-1].get('output', 'result')
            code.append(f"    return {last_var};")

        code.append("}")

        return '\n'.join(code)

    def _operation_to_cpp(self, op: Dict[str, Any]) -> str:
        """Convert a single operation to C++ code"""
        if op['type'] == 'method':
            obj = op.get('object', '')
            func = op.get('function', '')
            output = op.get('output', 'result')

            # Map NumPy methods to C++
            if func in ['mean', 'std', 'min', 'max', 'sum']:
                return f"float {output} = {obj}.{func}();"
            elif func in ['reshape']:
                # Handle reshape specially
                return f"// TODO: Implement reshape for {output} = {obj}.{func}()"

        elif op['type'] == 'module_function':
            module = op.get('module', '')
            func = op.get('function', '')
            output = op.get('output', 'result')
            inputs = op.get('inputs', [])

            if module == 'np':
                if func in self.numpy_to_cpp:
                    cpp_func = self.numpy_to_cpp[func]
                    if inputs:
                        return f"auto {output} = {cpp_func}({', '.join(inputs)});"

        elif op['type'] == 'function':
            func = op.get('function', '')
            output = op.get('output', 'result')

            # Check if it's a sklearn scaler
            if func in self.sklearn_to_cpp:
                cpp_class = self.sklearn_to_cpp[func]
                return f"{cpp_class}<float> {output};"

        return f"// TODO: Implement {op}"

    def to_rust(self, pipeline: Dict[str, Any]) -> str:
        """
        Generate Rust code from pipeline

        Args:
            pipeline: Pipeline dictionary from analyze_function

        Returns:
            Rust code as string
        """
        code = []
        code.append("// Auto-generated Rust data processing pipeline")
        code.append("")

        # Function signature
        code.append(f"pub fn {pipeline['function_name']}(")
        for i, inp in enumerate(pipeline['inputs']):
            code.append(f"    {inp}: &Tensor<f32>,")
        code.append(") -> Result<Tensor<f32>, &'static str> {")

        # Generate operations
        for op in pipeline['operations']:
            rust_line = self._operation_to_rust(op)
            if rust_line:
                code.append(f"    {rust_line}")

        # Return statement
        if pipeline['operations']:
            last_var = pipeline['operations'][-1].get('output', 'result')
            code.append(f"    Ok({last_var})")

        code.append("}")

        return '\n'.join(code)

    def _operation_to_rust(self, op: Dict[str, Any]) -> str:
        """Convert a single operation to Rust code"""
        if op['type'] == 'method':
            obj = op.get('object', '')
            func = op.get('function', '')
            output = op.get('output', 'result')

            if func in ['mean', 'min', 'max', 'sum']:
                return f"let {output} = {obj}.{func}();"

        elif op['type'] == 'module_function':
            module = op.get('module', '')
            func = op.get('function', '')
            output = op.get('output', 'result')
            inputs = op.get('inputs', [])

            if module == 'np' and func in self.numpy_to_cpp:
                # Map to Rust equivalent
                return f"let {output} = // TODO: Implement {func};"

        return f"// TODO: Implement {op}"

    def generate_mapping_guide(self) -> str:
        """Generate a guide showing Python -> C++/Rust mappings"""
        guide = []
        guide.append("# Python to C++/Rust Mapping Guide\n")

        guide.append("## NumPy Operations\n")
        guide.append("| Python (NumPy) | C++ | Rust |")
        guide.append("|----------------|-----|------|")

        mappings = [
            ("np.mean(x)", "embedded_ml::math::mean(x)", "x.mean()"),
            ("np.std(x)", "embedded_ml::math::std_deviation(x)", "// TODO"),
            ("np.min(x)", "x.min()", "x.min()"),
            ("np.max(x)", "x.max()", "x.max()"),
            ("np.sum(x)", "x.sum()", "x.sum()"),
            ("np.sqrt(x)", "embedded_ml::math::sqrt(x)", "// TODO"),
            ("np.exp(x)", "embedded_ml::math::exp(x)", "// TODO"),
            ("np.log(x)", "embedded_ml::math::log(x)", "// TODO"),
            ("x.reshape(shape)", "x.reshape(shape)", "x.reshape(shape)?"),
            ("np.clip(x, a, b)", "embedded_ml::preprocessing::clip(x, a, b)", "// TODO"),
        ]

        for py, cpp, rust in mappings:
            guide.append(f"| `{py}` | `{cpp}` | `{rust}` |")

        guide.append("\n## Scikit-learn Preprocessing\n")
        guide.append("| Python (Sklearn) | C++ | Rust |")
        guide.append("|------------------|-----|------|")

        sklearn_mappings = [
            ("StandardScaler()", "preprocessing::StandardScaler<float>()", "// TODO"),
            ("MinMaxScaler()", "preprocessing::MinMaxScaler<float>()", "// TODO"),
            ("scaler.fit(X)", "scaler.fit(X)", "scaler.fit(&X)"),
            ("scaler.transform(X)", "scaler.transform(X)", "scaler.transform(&X)?"),
            ("scaler.fit_transform(X)", "scaler.fit_transform(X)", "// TODO"),
        ]

        for py, cpp, rust in sklearn_mappings:
            guide.append(f"| `{py}` | `{cpp}` | `{rust}` |")

        guide.append("\n## Common Patterns\n")
        guide.append("### Normalization\n")
        guide.append("**Python:**")
        guide.append("```python")
        guide.append("normalized = (x - x.mean()) / x.std()")
        guide.append("```\n")

        guide.append("**C++:**")
        guide.append("```cpp")
        guide.append("auto mean = x.mean();")
        guide.append("auto std = math::std_deviation(x);")
        guide.append("auto normalized = (x - mean) * (1.0f / std);")
        guide.append("```\n")

        guide.append("**Rust:**")
        guide.append("```rust")
        guide.append("let mean = x.mean();")
        guide.append("let std = // calculate std;")
        guide.append("let normalized = // normalize;")
        guide.append("```\n")

        return '\n'.join(guide)


def extract_preprocessing_pipeline(python_file: str) -> Dict[str, Any]:
    """
    Extract preprocessing pipeline from Python file

    Args:
        python_file: Path to Python file

    Returns:
        Dictionary with pipeline information
    """
    with open(python_file, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    pipeline = {
        'imports': [],
        'functions': [],
        'classes': [],
        'constants': {}
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pipeline['imports'].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            pipeline['imports'].append(f"{node.module}")
        elif isinstance(node, ast.FunctionDef):
            pipeline['functions'].append(node.name)
        elif isinstance(node, ast.ClassDef):
            pipeline['classes'].append(node.name)

    return pipeline


def main():
    """Example usage"""
    # Example preprocessing function
    def preprocess_data(data):
        """Example preprocessing pipeline"""
        # Normalization
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / std

        # Clipping
        clipped = np.clip(normalized, -3, 3)

        # Scaling
        scaled = clipped * 0.5 + 0.5

        return scaled

    analyzer = PipelineAnalyzer()
    pipeline = analyzer.analyze_function(preprocess_data)

    print("Pipeline Analysis:")
    print(json.dumps(pipeline, indent=2))

    print("\n" + "="*60)
    print("C++ Code:")
    print("="*60)
    print(analyzer.to_cpp(pipeline))

    print("\n" + "="*60)
    print("Rust Code:")
    print("="*60)
    print(analyzer.to_rust(pipeline))

    print("\n" + "="*60)
    print("Mapping Guide:")
    print("="*60)
    print(analyzer.generate_mapping_guide())


if __name__ == '__main__':
    main()
