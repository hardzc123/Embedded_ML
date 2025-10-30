"""
Benchmark tool to compare PyTorch vs C++ vs Rust inference performance
"""

import time
import subprocess
import os
import sys
import json
import numpy as np
import torch
from typing import Dict, List, Optional


class InferenceBenchmark:
    """Benchmark inference performance across different implementations"""

    def __init__(self, model_path: str, sample_inputs: List[np.ndarray]):
        self.model_path = model_path
        self.sample_inputs = sample_inputs
        self.results = {}

    def benchmark_pytorch(self, num_runs: int = 100) -> Dict:
        """Benchmark PyTorch inference"""
        print("Benchmarking PyTorch...")

        # Load model
        model = torch.load(self.model_path, map_location='cpu')
        model.eval()

        # Convert inputs to tensors
        torch_inputs = [torch.FloatTensor(inp) for inp in self.sample_inputs]

        # Warmup
        with torch.no_grad():
            for inp in torch_inputs[:5]:
                _ = model(inp)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                for inp in torch_inputs:
                    start = time.perf_counter()
                    output = model(inp)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
        }

    def benchmark_cpp(self, executable_path: str, num_runs: int = 100) -> Optional[Dict]:
        """Benchmark C++ inference"""
        if not os.path.exists(executable_path):
            print(f"C++ executable not found: {executable_path}")
            return None

        print("Benchmarking C++...")

        # Create a temporary input file
        input_file = '/tmp/benchmark_input.txt'
        with open(input_file, 'w') as f:
            for inp in self.sample_inputs:
                f.write(' '.join(map(str, inp.flatten())) + '\n')

        # Run benchmark
        try:
            result = subprocess.run(
                [executable_path, input_file, str(num_runs)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split('\n')
                times = [float(line) for line in lines if line.strip()]

                return {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'median_ms': np.median(times),
                }
            else:
                print(f"C++ benchmark failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("C++ benchmark timed out")
            return None
        except Exception as e:
            print(f"Error running C++ benchmark: {e}")
            return None

    def benchmark_rust(self, executable_path: str, num_runs: int = 100) -> Optional[Dict]:
        """Benchmark Rust inference"""
        if not os.path.exists(executable_path):
            print(f"Rust executable not found: {executable_path}")
            return None

        print("Benchmarking Rust...")

        # Create a temporary input file
        input_file = '/tmp/benchmark_input.txt'
        with open(input_file, 'w') as f:
            for inp in self.sample_inputs:
                f.write(' '.join(map(str, inp.flatten())) + '\n')

        # Run benchmark
        try:
            result = subprocess.run(
                [executable_path, input_file, str(num_runs)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split('\n')
                times = [float(line) for line in lines if line.strip()]

                return {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'median_ms': np.median(times),
                }
            else:
                print(f"Rust benchmark failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("Rust benchmark timed out")
            return None
        except Exception as e:
            print(f"Error running Rust benchmark: {e}")
            return None

    def run_all(self, cpp_path: Optional[str] = None, rust_path: Optional[str] = None,
                num_runs: int = 100) -> Dict:
        """Run all benchmarks"""
        results = {}

        # PyTorch
        results['pytorch'] = self.benchmark_pytorch(num_runs)

        # C++
        if cpp_path:
            cpp_result = self.benchmark_cpp(cpp_path, num_runs)
            if cpp_result:
                results['cpp'] = cpp_result

        # Rust
        if rust_path:
            rust_result = self.benchmark_rust(rust_path, num_runs)
            if rust_result:
                results['rust'] = rust_result

        return results

    def print_comparison(self, results: Dict):
        """Print formatted comparison table"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"\nNumber of samples: {len(self.sample_inputs)}")
        print()

        # Table header
        print(f"{'Implementation':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 80)

        # PyTorch baseline
        if 'pytorch' in results:
            pt = results['pytorch']
            print(f"{'PyTorch':<15} {pt['mean_ms']:<12.4f} {pt['std_ms']:<12.4f} {pt['min_ms']:<12.4f} {pt['max_ms']:<12.4f}")

        # C++
        if 'cpp' in results:
            cpp = results['cpp']
            speedup = results['pytorch']['mean_ms'] / cpp['mean_ms'] if 'pytorch' in results else 1.0
            print(f"{'C++':<15} {cpp['mean_ms']:<12.4f} {cpp['std_ms']:<12.4f} {cpp['min_ms']:<12.4f} {cpp['max_ms']:<12.4f} ({speedup:.1f}x)")

        # Rust
        if 'rust' in results:
            rust = results['rust']
            speedup = results['pytorch']['mean_ms'] / rust['mean_ms'] if 'pytorch' in results else 1.0
            print(f"{'Rust':<15} {rust['mean_ms']:<12.4f} {rust['std_ms']:<12.4f} {rust['min_ms']:<12.4f} {rust['max_ms']:<12.4f} ({speedup:.1f}x)")

        print("\n" + "="*80)

    def export_results(self, results: Dict, output_file: str):
        """Export results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {output_file}")


def main():
    """Main benchmark script"""
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <model_path> [cpp_executable] [rust_executable]")
        sys.exit(1)

    model_path = sys.argv[1]
    cpp_path = sys.argv[2] if len(sys.argv) > 2 else None
    rust_path = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Generate sample inputs
    print("Generating sample inputs...")
    # Assuming input size from model (adjust as needed)
    sample_inputs = [np.random.randn(20).astype(np.float32) for _ in range(10)]

    # Create benchmark
    benchmark = InferenceBenchmark(model_path, sample_inputs)

    # Run benchmarks
    results = benchmark.run_all(cpp_path, rust_path, num_runs=100)

    # Print comparison
    benchmark.print_comparison(results)

    # Export results
    benchmark.export_results(results, 'benchmark_results.json')


if __name__ == '__main__':
    main()
