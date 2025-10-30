#!/bin/bash

# Run All Implementations and Verify
# This script runs Python, C++, and Rust implementations and verifies outputs match

echo "======================================================================"
echo "Running All Implementations: Python → C++ → Rust → Verify"
echo "======================================================================"

# Clean previous results
echo ""
echo "Cleaning previous results..."
rm -f cpp_mlp rust_implementation cpp_results.json rust_results.json
rm -f verification_results.json model_weights.json test_input.npy verified_mlp.pt

# Step 1: Run Python
echo ""
echo "======================================================================"
echo "Step 1/4: Running Python Implementation"
echo "======================================================================"
python python_implementation.py
if [ $? -ne 0 ]; then
    echo "❌ Python implementation failed!"
    exit 1
fi

# Step 2: Compile and run C++
echo ""
echo "======================================================================"
echo "Step 2/4: Compiling and Running C++ Implementation"
echo "======================================================================"
g++ -std=c++17 -O3 -I../../../runtime/cpp/include cpp_implementation.cpp -o cpp_mlp
if [ $? -ne 0 ]; then
    echo "❌ C++ compilation failed!"
    exit 1
fi

./cpp_mlp
if [ $? -ne 0 ]; then
    echo "❌ C++ execution failed!"
    exit 1
fi

# Step 3: Compile and run Rust
echo ""
echo "======================================================================"
echo "Step 3/4: Compiling and Running Rust Implementation"
echo "======================================================================"
rustc rust_implementation.rs -o rust_implementation
if [ $? -ne 0 ]; then
    echo "❌ Rust compilation failed!"
    exit 1
fi

./rust_implementation
if [ $? -ne 0 ]; then
    echo "❌ Rust execution failed!"
    exit 1
fi

# Step 4: Verify
echo ""
echo "======================================================================"
echo "Step 4/4: Verifying All Outputs Match"
echo "======================================================================"
python verify_conversion.py
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ SUCCESS! All implementations verified!"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "❌ FAILURE! Implementations do not match!"
    echo "======================================================================"
fi

exit $exit_code
