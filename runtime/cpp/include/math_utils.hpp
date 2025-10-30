/**
 * Math Utilities - NumPy-like operations for embedded systems
 * No external dependencies - uses only standard C++17
 */

#ifndef EMBEDDED_ML_MATH_UTILS_HPP
#define EMBEDDED_ML_MATH_UTILS_HPP

#include "tensor.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <complex>

namespace embedded_ml {
namespace math {

/**
 * Statistical functions
 */
template<typename T = float>
T mean(const Tensor<T>& tensor) {
    return tensor.mean();
}

template<typename T = float>
T median(const Tensor<T>& tensor) {
    std::vector<T> sorted = tensor.data();
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2;
    } else {
        return sorted[n/2];
    }
}

template<typename T = float>
T variance(const Tensor<T>& tensor, bool unbiased = true) {
    T m = tensor.mean();
    T var = 0;
    for (const auto& val : tensor.data()) {
        T diff = val - m;
        var += diff * diff;
    }
    size_t n = tensor.size();
    return unbiased && n > 1 ? var / (n - 1) : var / n;
}

template<typename T = float>
T std_deviation(const Tensor<T>& tensor, bool unbiased = true) {
    return std::sqrt(variance(tensor, unbiased));
}

template<typename T = float>
struct Statistics {
    T min;
    T max;
    T mean;
    T median;
    T std;
    T variance;

    static Statistics compute(const Tensor<T>& tensor) {
        Statistics stats;
        stats.min = tensor.min();
        stats.max = tensor.max();
        stats.mean = tensor.mean();
        stats.median = math::median(tensor);
        stats.variance = math::variance(tensor);
        stats.std = std::sqrt(stats.variance);
        return stats;
    }
};

/**
 * Element-wise mathematical operations
 */
template<typename T = float>
Tensor<T> abs(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::abs(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> sqrt(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::sqrt(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> square(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = val * val;
    }
    return result;
}

template<typename T = float>
Tensor<T> exp(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::exp(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> log(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::log(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> sin(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::sin(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> cos(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::cos(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> tan(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::tan(val);
    }
    return result;
}

/**
 * Rounding operations
 */
template<typename T = float>
Tensor<T> round(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::round(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> floor(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::floor(val);
    }
    return result;
}

template<typename T = float>
Tensor<T> ceil(const Tensor<T>& tensor) {
    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val = std::ceil(val);
    }
    return result;
}

/**
 * Linear algebra operations
 */
template<typename T = float>
T dot(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Tensors must have same size for dot product");
    }
    T result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a.data()[i] * b.data()[i];
    }
    return result;
}

template<typename T = float>
Tensor<T> cross(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::runtime_error("Cross product requires 3D vectors");
    }

    Tensor<T> result({3});
    result.data()[0] = a.data()[1] * b.data()[2] - a.data()[2] * b.data()[1];
    result.data()[1] = a.data()[2] * b.data()[0] - a.data()[0] * b.data()[2];
    result.data()[2] = a.data()[0] * b.data()[1] - a.data()[1] * b.data()[0];

    return result;
}

template<typename T = float>
T norm(const Tensor<T>& tensor, int p = 2) {
    T sum = 0;
    if (p == 1) {
        for (const auto& val : tensor.data()) {
            sum += std::abs(val);
        }
    } else if (p == 2) {
        for (const auto& val : tensor.data()) {
            sum += val * val;
        }
        sum = std::sqrt(sum);
    } else {
        for (const auto& val : tensor.data()) {
            sum += std::pow(std::abs(val), p);
        }
        sum = std::pow(sum, 1.0 / p);
    }
    return sum;
}

template<typename T = float>
Tensor<T> normalize(const Tensor<T>& tensor, int p = 2) {
    T n = norm(tensor, p);
    if (n == 0) return tensor.clone();

    Tensor<T> result = tensor.clone();
    for (auto& val : result.data()) {
        val /= n;
    }
    return result;
}

/**
 * Random number generation
 */
template<typename T = float>
class Random {
public:
    Random(unsigned int seed = std::random_device{}())
        : gen_(seed) {}

    // Uniform distribution [low, high)
    Tensor<T> uniform(const std::vector<size_t>& shape, T low = 0, T high = 1) {
        std::uniform_real_distribution<T> dist(low, high);
        Tensor<T> result(shape);
        for (auto& val : result.data()) {
            val = dist(gen_);
        }
        return result;
    }

    // Normal distribution
    Tensor<T> normal(const std::vector<size_t>& shape, T mean = 0, T std = 1) {
        std::normal_distribution<T> dist(mean, std);
        Tensor<T> result(shape);
        for (auto& val : result.data()) {
            val = dist(gen_);
        }
        return result;
    }

    // Random integers [low, high]
    Tensor<int> randint(const std::vector<size_t>& shape, int low, int high) {
        std::uniform_int_distribution<int> dist(low, high);
        Tensor<int> result(shape);
        for (auto& val : result.data()) {
            val = dist(gen_);
        }
        return result;
    }

    // Random choice from array
    size_t choice(size_t n) {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        return dist(gen_);
    }

    void seed(unsigned int s) {
        gen_.seed(s);
    }

private:
    std::mt19937 gen_;
};

/**
 * Interpolation
 */
template<typename T = float>
T lerp(T a, T b, T t) {
    return a + t * (b - a);
}

template<typename T = float>
class LinearInterpolator {
public:
    LinearInterpolator(const std::vector<T>& x, const std::vector<T>& y)
        : x_(x), y_(y) {
        if (x.size() != y.size()) {
            throw std::runtime_error("x and y must have same size");
        }
        if (x.size() < 2) {
            throw std::runtime_error("Need at least 2 points for interpolation");
        }
    }

    T interpolate(T x) const {
        // Find surrounding points
        size_t i = 0;
        while (i < x_.size() - 1 && x_[i] < x) {
            i++;
        }
        if (i == 0) i = 1;

        // Linear interpolation
        T x0 = x_[i-1], x1 = x_[i];
        T y0 = y_[i-1], y1 = y_[i];
        T t = (x - x0) / (x1 - x0);

        return lerp(y0, y1, t);
    }

private:
    std::vector<T> x_;
    std::vector<T> y_;
};

/**
 * Convolution operations (1D)
 */
template<typename T = float>
Tensor<T> convolve1d(const Tensor<T>& signal, const Tensor<T>& kernel,
                     const std::string& mode = "valid") {
    if (signal.shape().size() != 1 || kernel.shape().size() != 1) {
        throw std::runtime_error("convolve1d requires 1D tensors");
    }

    size_t signal_len = signal.size();
    size_t kernel_len = kernel.size();
    size_t output_len;

    if (mode == "valid") {
        output_len = signal_len - kernel_len + 1;
    } else if (mode == "same") {
        output_len = signal_len;
    } else if (mode == "full") {
        output_len = signal_len + kernel_len - 1;
    } else {
        throw std::runtime_error("Unknown mode");
    }

    Tensor<T> result({output_len}, static_cast<T>(0));

    // Perform convolution
    for (size_t i = 0; i < output_len; ++i) {
        T sum = 0;
        for (size_t j = 0; j < kernel_len; ++j) {
            int signal_idx = static_cast<int>(i + j) - static_cast<int>(kernel_len/2);
            if (mode == "same" && (signal_idx < 0 || signal_idx >= static_cast<int>(signal_len))) {
                continue;
            }
            if (signal_idx >= 0 && signal_idx < static_cast<int>(signal_len)) {
                sum += signal.data()[signal_idx] * kernel.data()[j];
            }
        }
        result.data()[i] = sum;
    }

    return result;
}

/**
 * FFT helpers (simplified for common use cases)
 */
template<typename T = float>
struct ComplexNumber {
    T real;
    T imag;

    ComplexNumber operator+(const ComplexNumber& other) const {
        return {real + other.real, imag + other.imag};
    }

    ComplexNumber operator-(const ComplexNumber& other) const {
        return {real - other.real, imag - other.imag};
    }

    ComplexNumber operator*(const ComplexNumber& other) const {
        return {
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        };
    }

    T magnitude() const {
        return std::sqrt(real * real + imag * imag);
    }

    T phase() const {
        return std::atan2(imag, real);
    }
};

/**
 * Percentile calculation
 */
template<typename T = float>
T percentile(const Tensor<T>& tensor, T p) {
    if (p < 0 || p > 100) {
        throw std::runtime_error("Percentile must be between 0 and 100");
    }

    std::vector<T> sorted = tensor.data();
    std::sort(sorted.begin(), sorted.end());

    T index = p / 100.0 * (sorted.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper) {
        return sorted[lower];
    }

    T fraction = index - lower;
    return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

/**
 * Histogram
 */
template<typename T = float>
struct Histogram {
    std::vector<size_t> counts;
    std::vector<T> bin_edges;
};

template<typename T = float>
Histogram<T> histogram(const Tensor<T>& tensor, size_t num_bins = 10) {
    Histogram<T> hist;

    T min_val = tensor.min();
    T max_val = tensor.max();
    T bin_width = (max_val - min_val) / num_bins;

    hist.counts.resize(num_bins, 0);
    hist.bin_edges.resize(num_bins + 1);

    // Create bin edges
    for (size_t i = 0; i <= num_bins; ++i) {
        hist.bin_edges[i] = min_val + i * bin_width;
    }

    // Count values in each bin
    for (const auto& val : tensor.data()) {
        size_t bin = static_cast<size_t>((val - min_val) / bin_width);
        if (bin >= num_bins) bin = num_bins - 1;
        hist.counts[bin]++;
    }

    return hist;
}

/**
 * Correlation coefficient
 */
template<typename T = float>
T correlation(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Tensors must have same size");
    }

    T mean_a = a.mean();
    T mean_b = b.mean();

    T numerator = 0;
    T sum_sq_a = 0;
    T sum_sq_b = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        T diff_a = a.data()[i] - mean_a;
        T diff_b = b.data()[i] - mean_b;
        numerator += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }

    T denominator = std::sqrt(sum_sq_a * sum_sq_b);
    return denominator > 0 ? numerator / denominator : 0;
}

} // namespace math
} // namespace embedded_ml

#endif // EMBEDDED_ML_MATH_UTILS_HPP
