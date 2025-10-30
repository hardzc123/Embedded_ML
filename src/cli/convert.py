#!/usr/bin/env python3
"""
CLI Tool for Converting PyTorch Models to C++/Rust for Embedded Platforms
"""

import argparse
import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.model_parser import ModelParser
from codegen.cpp.generator import CppCodeGenerator
from codegen.rust.generator import RustCodeGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to C++/Rust for embedded platforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to C++
  python convert.py --model model.pt --format cpp --output ./output

  # Convert to Rust with optimization
  python convert.py --model model.pt --format rust --output ./output --optimize

  # Convert with quantization
  python convert.py --model model.pt --format cpp --output ./output --quantize int8

  # Provide sample input for shape inference
  python convert.py --model model.pt --format cpp --output ./output --input-shape 1,3,224,224
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch model (.pt or .pth)')
    parser.add_argument('--format', type=str, choices=['cpp', 'rust', 'both'],
                        default='cpp', help='Output format (cpp, rust, or both)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--name', type=str, default='Model',
                        help='Name for the generated model class/struct')
    parser.add_argument('--optimize', action='store_true',
                        help='Enable optimizations for embedded deployment')
    parser.add_argument('--quantize', type=str, choices=['int8', 'int16', 'none'],
                        default='none', help='Quantization mode')
    parser.add_argument('--input-shape', type=str,
                        help='Input shape for the model (comma-separated, e.g., 1,3,224,224)')
    parser.add_argument('--summary', action='store_true',
                        help='Print model summary')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)

    try:
        # Load the model
        print(f"Loading model from {args.model}...")
        model = torch.load(args.model, map_location='cpu')

        # Handle different model formats
        if isinstance(model, dict):
            if 'model_state_dict' in model:
                # Need to reconstruct the model architecture
                print("Error: Model state dict found but architecture is not saved.")
                print("Please save the complete model using torch.save(model, 'model.pt')")
                sys.exit(1)
            elif 'state_dict' in model:
                print("Error: Only state dict found, need complete model")
                sys.exit(1)
        elif not isinstance(model, nn.Module):
            print("Error: Loaded object is not a PyTorch model")
            sys.exit(1)

        model.eval()

        # Create sample input if shape provided
        sample_input = None
        if args.input_shape:
            try:
                shape = [int(x) for x in args.input_shape.split(',')]
                sample_input = torch.randn(*shape)
                print(f"Using input shape: {shape}")
            except ValueError:
                print(f"Error: Invalid input shape format: {args.input_shape}")
                sys.exit(1)

        # Parse the model
        print("Parsing model architecture...")
        parser_obj = ModelParser(model)
        model_info = parser_obj.parse(sample_input)

        if args.verbose or args.summary:
            print("\n" + parser_obj.export_summary())

        # Generate code
        os.makedirs(args.output, exist_ok=True)

        if args.format in ['cpp', 'both']:
            print(f"\nGenerating C++ code...")
            cpp_output = os.path.join(args.output, 'cpp')
            generator = CppCodeGenerator(model_info, cpp_output, optimize=args.optimize)
            generator.generate(args.name)
            print(f"✓ C++ code generated in {cpp_output}")
            print(f"  - {args.name}.hpp")
            print(f"  - {args.name}.cpp")
            print(f"  - example.cpp")
            print(f"  - CMakeLists.txt")

        if args.format in ['rust', 'both']:
            print(f"\nGenerating Rust code...")
            rust_output = os.path.join(args.output, 'rust')
            os.makedirs(os.path.join(rust_output, 'src'), exist_ok=True)
            generator = RustCodeGenerator(model_info, rust_output, optimize=args.optimize)
            generator.generate(args.name)
            print(f"✓ Rust code generated in {rust_output}")
            print(f"  - Cargo.toml")
            print(f"  - src/lib.rs")
            print(f"  - examples/inference.rs")
            print(f"  - README.md")

        print("\n" + "="*60)
        print("✓ Conversion completed successfully!")
        print("="*60)

        # Print next steps
        print("\nNext steps:")
        if args.format in ['cpp', 'both']:
            print("\n  C++:")
            print(f"    cd {os.path.join(args.output, 'cpp')}")
            print("    mkdir build && cd build")
            print("    cmake ..")
            print("    make")
            print(f"    ./{args.name}_example")

        if args.format in ['rust', 'both']:
            print("\n  Rust:")
            print(f"    cd {os.path.join(args.output, 'rust')}")
            print("    cargo build --release")
            print("    cargo run --example inference --release")

        print("\n" + "="*60)

    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
