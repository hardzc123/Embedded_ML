"""
Verification Script - Compare Python, C++, and Rust outputs

This script verifies that all three implementations produce identical results.
"""

import json
import numpy as np
from typing import Dict, Any


def load_json(filename: str) -> Dict[str, Any]:
    """Load JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def compare_results(python_result: Dict, cpp_result: Dict, rust_result: Dict,
                   tolerance: float = 1e-5) -> bool:
    """
    Compare results from all three implementations

    Args:
        python_result: Python output
        cpp_result: C++ output
        rust_result: Rust output
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if all results match within tolerance
    """
    all_match = True

    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)

    # Check if all implementations ran
    implementations = {
        'Python': python_result,
        'C++': cpp_result,
        'Rust': rust_result
    }

    missing = [name for name, result in implementations.items() if result is None]
    if missing:
        print(f"\n⚠ WARNING: Missing results from: {', '.join(missing)}")
        print("Make sure to run all three implementations first.")
        return False

    # Compare predicted classes
    print("\n1. Predicted Class Comparison:")
    print("-" * 70)

    python_class = python_result['python']['predicted_class']
    cpp_class = cpp_result.get('predicted_class')
    rust_class = rust_result.get('predicted_class')

    print(f"  Python: {python_class}")
    print(f"  C++:    {cpp_class}")
    print(f"  Rust:   {rust_class}")

    if python_class == cpp_class == rust_class:
        print("  ✓ All predicted classes match!")
    else:
        print("  ✗ Predicted classes DO NOT match!")
        all_match = False

    # Compare confidence scores
    print("\n2. Confidence Score Comparison:")
    print("-" * 70)

    python_conf = python_result['python']['confidence']
    cpp_conf = cpp_result.get('confidence')
    rust_conf = rust_result.get('confidence')

    print(f"  Python: {python_conf:.8f}")
    print(f"  C++:    {cpp_conf:.8f}")
    print(f"  Rust:   {rust_conf:.8f}")

    # Check numerical tolerance
    cpp_diff = abs(python_conf - cpp_conf) if cpp_conf is not None else float('inf')
    rust_diff = abs(python_conf - rust_conf) if rust_conf is not None else float('inf')

    print(f"\n  Difference (Python vs C++):  {cpp_diff:.10f}")
    print(f"  Difference (Python vs Rust): {rust_diff:.10f}")

    if cpp_diff < tolerance and rust_diff < tolerance:
        print(f"  ✓ All confidence scores match within tolerance ({tolerance})!")
    else:
        print(f"  ✗ Confidence scores exceed tolerance ({tolerance})!")
        all_match = False

    # Compare probabilities
    print("\n3. Probability Distribution Comparison:")
    print("-" * 70)

    python_probs = np.array(python_result['python']['probabilities'])
    cpp_probs = np.array(cpp_result.get('probabilities', []))
    rust_probs = np.array(rust_result.get('probabilities', []))

    print("  Class | Python     | C++        | Rust")
    print("  " + "-" * 60)

    for i in range(len(python_probs)):
        p_val = python_probs[i] if i < len(python_probs) else 0
        c_val = cpp_probs[i] if i < len(cpp_probs) else 0
        r_val = rust_probs[i] if i < len(rust_probs) else 0

        print(f"  {i}     | {p_val:.6f}   | {c_val:.6f}   | {r_val:.6f}")

    # Calculate maximum difference
    if len(cpp_probs) > 0 and len(rust_probs) > 0:
        max_cpp_diff = np.max(np.abs(python_probs - cpp_probs))
        max_rust_diff = np.max(np.abs(python_probs - rust_probs))

        print(f"\n  Max difference (Python vs C++):  {max_cpp_diff:.10f}")
        print(f"  Max difference (Python vs Rust): {max_rust_diff:.10f}")

        if max_cpp_diff < tolerance and max_rust_diff < tolerance:
            print(f"  ✓ All probabilities match within tolerance ({tolerance})!")
        else:
            print(f"  ✗ Probabilities exceed tolerance ({tolerance})!")
            all_match = False

    # Final verdict
    print("\n" + "="*70)
    if all_match:
        print("✓✓✓ SUCCESS: All implementations produce identical results! ✓✓✓")
        print("="*70)
        print("\nThe Python → C++ and Python → Rust conversions are VERIFIED!")
        print("All three implementations produce the same outputs within")
        print(f"numerical tolerance of {tolerance}.")
    else:
        print("✗✗✗ FAILURE: Implementations produce different results! ✗✗✗")
        print("="*70)
        print("\nPlease check the implementations for errors.")

    print("="*70)

    return all_match


def main():
    print("="*70)
    print("CONVERSION VERIFICATION TOOL")
    print("="*70)
    print("\nThis script compares outputs from Python, C++, and Rust")
    print("implementations to verify they produce identical results.")

    # Load results
    print("\nLoading results...")
    python_result = load_json('verification_results.json')
    cpp_result = load_json('cpp_results.json')
    rust_result = load_json('rust_results.json')

    if python_result:
        print("  ✓ Python results loaded")
    else:
        print("  ✗ Python results not found (run python_implementation.py)")

    if cpp_result:
        print("  ✓ C++ results loaded")
    else:
        print("  ✗ C++ results not found (compile and run cpp_implementation.cpp)")

    if rust_result:
        print("  ✓ Rust results loaded")
    else:
        print("  ✗ Rust results not found (compile and run rust_implementation.rs)")

    # Compare
    if all([python_result, cpp_result, rust_result]):
        success = compare_results(python_result, cpp_result, rust_result)
        return 0 if success else 1
    else:
        print("\n⚠ Cannot verify - missing implementation results")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
