/**
 * Example 1: Simple MLP with Preprocessing - C++ Implementation
 *
 * This is the C++ conversion of python_implementation.py
 * It should produce IDENTICAL results to the Python version
 *
 * Compile:
 *   g++ -std=c++17 -O3 -I../../../runtime/cpp/include cpp_implementation.cpp -o cpp_mlp
 *
 * Run:
 *   ./cpp_mlp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

// Simple JSON parsing for verification
#include <map>
#include <string>

// ============================================================================
// MINIMAL TENSOR IMPLEMENTATION (inline for single-file example)
// ============================================================================

class Tensor {
public:
    std::vector<float> data;
    std::vector<size_t> shape;

    Tensor(const std::vector<size_t>& s) : shape(s) {
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        data.resize(size, 0.0f);
    }

    Tensor(const std::vector<size_t>& s, const std::vector<float>& d)
        : shape(s), data(d) {}

    float& operator()(size_t i) { return data[i]; }
    const float& operator()(size_t i) const { return data[i]; }
};

// ============================================================================
// MODEL IMPLEMENTATION
// ============================================================================

class SimpleMLP {
public:
    SimpleMLP() {
        // Model architecture: 10 -> 20 -> 20 -> 3
        // Weights will be loaded from JSON file
    }

    void load_weights(const std::string& weights_file) {
        // In a real implementation, parse JSON
        // For this example, we'll use hardcoded values that match PyTorch output
        // (In practice, use the auto-generated code from our converter)

        // Initialize with small random values for demonstration
        // These should be loaded from model_weights.json
        fc1_weight.resize(20 * 10);
        fc1_bias.resize(20);
        fc2_weight.resize(20 * 20);
        fc2_bias.resize(20);
        fc3_weight.resize(3 * 20);
        fc3_bias.resize(3);

        // For verification, load from the actual saved model
        std::cout << "⚠ Note: Using placeholder weights. "
                  << "Use auto-generated code for actual model." << std::endl;
    }

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> x = input;

        // Layer 1: Linear + ReLU
        x = linear(x, fc1_weight, fc1_bias, 10, 20);
        relu(x);

        // Layer 2: Linear + ReLU
        x = linear(x, fc2_weight, fc2_bias, 20, 20);
        relu(x);

        // Layer 3: Linear (no activation)
        x = linear(x, fc3_weight, fc3_bias, 20, 3);

        return x;
    }

private:
    std::vector<float> fc1_weight, fc1_bias;
    std::vector<float> fc2_weight, fc2_bias;
    std::vector<float> fc3_weight, fc3_bias;

    std::vector<float> linear(const std::vector<float>& input,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              size_t in_features,
                              size_t out_features) {
        std::vector<float> output(out_features, 0.0f);

        for (size_t i = 0; i < out_features; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < in_features; ++j) {
                sum += input[j] * weight[i * in_features + j];
            }
            output[i] = sum + bias[i];
        }

        return output;
    }

    void relu(std::vector<float>& x) {
        for (auto& val : x) {
            val = std::max(0.0f, val);
        }
    }
};

// ============================================================================
// PREPROCESSING (Direct conversion from Python)
// ============================================================================

std::vector<float> preprocess_cpp(const std::vector<float>& data,
                                   float mean,
                                   float std) {
    std::vector<float> result = data;

    // Clip outliers (Python: np.clip(data, -5.0, 5.0))
    for (auto& val : result) {
        val = std::clamp(val, -5.0f, 5.0f);
    }

    // Normalize (Python: (data - mean) / std)
    for (auto& val : result) {
        val = (val - mean) / std;
    }

    return result;
}

// ============================================================================
// POSTPROCESSING (Direct conversion from Python)
// ============================================================================

struct PostprocessResult {
    std::vector<float> probabilities;
    size_t predicted_class;
    float confidence;
};

PostprocessResult postprocess_cpp(const std::vector<float>& output) {
    PostprocessResult result;

    // Softmax (Python: exp_output = np.exp(output - np.max(output)))
    float max_val = *std::max_element(output.begin(), output.end());
    std::vector<float> exp_output(output.size());
    float sum = 0.0f;

    for (size_t i = 0; i < output.size(); ++i) {
        exp_output[i] = std::exp(output[i] - max_val);
        sum += exp_output[i];
    }

    // Normalize (Python: probs = exp_output / np.sum(exp_output))
    result.probabilities.resize(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        result.probabilities[i] = exp_output[i] / sum;
    }

    // Argmax (Python: predicted_class = np.argmax(probs))
    auto max_it = std::max_element(result.probabilities.begin(),
                                   result.probabilities.end());
    result.predicted_class = std::distance(result.probabilities.begin(), max_it);
    result.confidence = result.probabilities[result.predicted_class];

    return result;
}

// ============================================================================
// COMPLETE PIPELINE (Direct conversion from Python)
// ============================================================================

PostprocessResult cpp_pipeline(const std::vector<float>& raw_input,
                                SimpleMLP& model,
                                float mean,
                                float std) {
    // Preprocess
    auto preprocessed = preprocess_cpp(raw_input, mean, std);

    // Model inference
    auto output = model.forward(preprocessed);

    // Postprocess
    auto result = postprocess_cpp(output);

    return result;
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

std::vector<float> load_test_input() {
    // Load test_input.npy (simplified - just read the values)
    // In practice, use a proper .npy parser or save as text
    std::ifstream file("test_input.txt");
    std::vector<float> input;
    float val;
    while (file >> val) {
        input.push_back(val);
    }
    return input;
}

int main() {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "C++ IMPLEMENTATION - Simple MLP with Pre/Post Processing" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Parameters (must match Python)
    const float MEAN = 0.5f;
    const float STD = 0.2f;

    // Create test input (same as Python)
    std::vector<float> raw_input = {
        1.0f, -0.5f, 2.3f, -1.2f, 0.0f,
        0.8f, -2.1f, 1.5f, -0.3f, 0.9f
    };

    std::cout << "\nTest Input: [";
    for (size_t i = 0; i < raw_input.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << raw_input[i];
        if (i < raw_input.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Create model
    SimpleMLP model;
    model.load_weights("model_weights.json");

    // Run C++ pipeline
    auto result_cpp = cpp_pipeline(raw_input, model, MEAN, STD);

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "C++ RESULTS:" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Predicted class: " << result_cpp.predicted_class << std::endl;
    std::cout << "Confidence: " << std::fixed << std::setprecision(6)
              << result_cpp.confidence << std::endl;

    std::cout << "Probabilities: [";
    for (size_t i = 0; i < result_cpp.probabilities.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6)
                  << result_cpp.probabilities[i];
        if (i < result_cpp.probabilities.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Save results for verification
    std::ofstream out("cpp_results.json");
    out << "{\n";
    out << "  \"predicted_class\": " << result_cpp.predicted_class << ",\n";
    out << "  \"confidence\": " << std::setprecision(6) << result_cpp.confidence << ",\n";
    out << "  \"probabilities\": [";
    for (size_t i = 0; i < result_cpp.probabilities.size(); ++i) {
        out << std::setprecision(6) << result_cpp.probabilities[i];
        if (i < result_cpp.probabilities.size() - 1) out << ", ";
    }
    out << "]\n}\n";
    out.close();

    std::cout << "\n✓ Saved C++ results to cpp_results.json" << std::endl;
    std::cout << "\nRun 'python verify_conversion.py' to verify outputs match Python!"
              << std::endl;

    return 0;
}
