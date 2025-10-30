/**
 * Activations - Common neural network activation functions
 * Uses only standard C++17 features, no external dependencies
 */

#ifndef EMBEDDED_ML_ACTIVATIONS_HPP
#define EMBEDDED_ML_ACTIVATIONS_HPP

#include "tensor.hpp"
#include <cmath>
#include <algorithm>

namespace embedded_ml {
namespace activations {

/**
 * ReLU (Rectified Linear Unit)
 * f(x) = max(0, x)
 */
template<typename T = float>
void relu(Tensor<T>& tensor) {
    auto& data = tensor.data();
    for (auto& val : data) {
        val = std::max(static_cast<T>(0), val);
    }
}

template<typename T = float>
Tensor<T> relu(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    relu(result);
    return result;
}

/**
 * Leaky ReLU
 * f(x) = x if x > 0 else alpha * x
 */
template<typename T = float>
void leaky_relu(Tensor<T>& tensor, T alpha = 0.01) {
    auto& data = tensor.data();
    for (auto& val : data) {
        val = val > 0 ? val : alpha * val;
    }
}

template<typename T = float>
Tensor<T> leaky_relu(const Tensor<T>& tensor, T alpha = 0.01) {
    Tensor<T> result = tensor.clone();
    leaky_relu(result, alpha);
    return result;
}

/**
 * Sigmoid
 * f(x) = 1 / (1 + exp(-x))
 */
template<typename T = float>
void sigmoid(Tensor<T>& tensor) {
    auto& data = tensor.data();
    for (auto& val : data) {
        val = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-val));
    }
}

template<typename T = float>
Tensor<T> sigmoid(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    sigmoid(result);
    return result;
}

/**
 * Tanh (Hyperbolic Tangent)
 * f(x) = tanh(x)
 */
template<typename T = float>
void tanh_activation(Tensor<T>& tensor) {
    auto& data = tensor.data();
    for (auto& val : data) {
        val = std::tanh(val);
    }
}

template<typename T = float>
Tensor<T> tanh_activation(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    tanh_activation(result);
    return result;
}

/**
 * Softmax
 * f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
 * Numerically stable implementation
 */
template<typename T = float>
void softmax(Tensor<T>& tensor, int dim = -1) {
    // For simplicity, apply softmax to the last dimension
    auto& data = tensor.data();
    const auto& shape = tensor.shape();

    if (shape.empty()) return;

    // Handle 1D case
    if (shape.size() == 1) {
        T max_val = tensor.max();
        T sum = 0;
        for (auto& val : data) {
            val = std::exp(val - max_val);  // Subtract max for numerical stability
            sum += val;
        }
        for (auto& val : data) {
            val /= sum;
        }
        return;
    }

    // Handle 2D case (batch, features)
    if (shape.size() == 2) {
        size_t batch_size = shape[0];
        size_t features = shape[1];

        for (size_t b = 0; b < batch_size; ++b) {
            // Find max in this row
            T max_val = data[b * features];
            for (size_t f = 1; f < features; ++f) {
                max_val = std::max(max_val, data[b * features + f]);
            }

            // Compute exp and sum
            T sum = 0;
            for (size_t f = 0; f < features; ++f) {
                data[b * features + f] = std::exp(data[b * features + f] - max_val);
                sum += data[b * features + f];
            }

            // Normalize
            for (size_t f = 0; f < features; ++f) {
                data[b * features + f] /= sum;
            }
        }
    }
}

template<typename T = float>
Tensor<T> softmax(const Tensor<T>& tensor, int dim = -1) {
    Tensor<T> result = tensor.clone();
    softmax(result, dim);
    return result;
}

/**
 * GELU (Gaussian Error Linear Unit)
 * f(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
 * Approximation for faster computation
 */
template<typename T = float>
void gelu(Tensor<T>& tensor) {
    auto& data = tensor.data();
    const T sqrt_2_over_pi = std::sqrt(static_cast<T>(2.0) / static_cast<T>(M_PI));

    for (auto& val : data) {
        // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        T x3 = val * val * val;
        T inner = sqrt_2_over_pi * (val + static_cast<T>(0.044715) * x3);
        val = static_cast<T>(0.5) * val * (static_cast<T>(1.0) + std::tanh(inner));
    }
}

template<typename T = float>
Tensor<T> gelu(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    gelu(result);
    return result;
}

/**
 * SiLU / Swish
 * f(x) = x * sigmoid(x)
 */
template<typename T = float>
void swish(Tensor<T>& tensor) {
    auto& data = tensor.data();
    for (auto& val : data) {
        val = val / (static_cast<T>(1) + std::exp(-val));
    }
}

template<typename T = float>
Tensor<T> swish(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    swish(result);
    return result;
}

/**
 * ELU (Exponential Linear Unit)
 * f(x) = x if x > 0 else alpha * (exp(x) - 1)
 */
template<typename T = float>
void elu(Tensor<T>& tensor, T alpha = 1.0) {
    auto& data = tensor.data();
    for (auto& val : data) {
        val = val > 0 ? val : alpha * (std::exp(val) - static_cast<T>(1));
    }
}

template<typename T = float>
Tensor<T> elu(const Tensor<T>& tensor, T alpha = 1.0) {
    Tensor<T> result = tensor.clone();
    elu(result, alpha);
    return result;
}

} // namespace activations
} // namespace embedded_ml

#endif // EMBEDDED_ML_ACTIVATIONS_HPP
