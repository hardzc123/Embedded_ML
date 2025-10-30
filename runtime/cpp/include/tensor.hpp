/**
 * Tensor - Lightweight tensor implementation for embedded ML inference
 * Uses only standard C++17 features, no external dependencies
 */

#ifndef EMBEDDED_ML_TENSOR_HPP
#define EMBEDDED_ML_TENSOR_HPP

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <cstring>

namespace embedded_ml {

template<typename T = float>
class Tensor {
public:
    // Constructors
    Tensor() : data_(), shape_(), size_(0) {}

    explicit Tensor(const std::vector<size_t>& shape)
        : shape_(shape), size_(compute_size(shape)) {
        data_.resize(size_);
    }

    Tensor(const std::vector<size_t>& shape, T init_value)
        : shape_(shape), size_(compute_size(shape)) {
        data_.resize(size_, init_value);
    }

    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
        : shape_(shape), data_(data), size_(compute_size(shape)) {
        if (data_.size() != size_) {
            throw std::runtime_error("Data size doesn't match shape");
        }
    }

    // Accessors
    const std::vector<T>& data() const { return data_; }
    std::vector<T>& data() { return data_; }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }

    // Element access
    T& operator()(const std::vector<size_t>& indices) {
        return data_[compute_offset(indices)];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return data_[compute_offset(indices)];
    }

    // Convenient accessors for common dimensions
    T& operator()(size_t i) {
        return data_[i];
    }

    const T& operator()(size_t i) const {
        return data_[i];
    }

    T& operator()(size_t i, size_t j) {
        return data_[i * shape_[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return data_[i * shape_[1] + j];
    }

    // Reshape (returns new tensor, doesn't modify in place)
    Tensor<T> reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = compute_size(new_shape);
        if (new_size != size_) {
            throw std::runtime_error("Reshape size mismatch");
        }
        return Tensor<T>(new_shape, data_);
    }

    // Matrix multiplication (for 2D tensors)
    Tensor<T> matmul(const Tensor<T>& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::runtime_error("Matmul only supports 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }

        size_t m = shape_[0];
        size_t n = other.shape_[1];
        size_t k = shape_[1];

        Tensor<T> result({m, n}, static_cast<T>(0));

        // Naive matrix multiplication (can be optimized with blocking, SIMD, etc.)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = 0;
                for (size_t p = 0; p < k; ++p) {
                    sum += (*this)(i, p) * other(p, j);
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

    // Element-wise operations
    Tensor<T> operator+(const Tensor<T>& other) const {
        check_broadcast_compatible(other);
        Tensor<T> result = *this;
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] += other.data_[i % other.size_];
        }
        return result;
    }

    Tensor<T> operator+(T scalar) const {
        Tensor<T> result = *this;
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] += scalar;
        }
        return result;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        check_broadcast_compatible(other);
        Tensor<T> result = *this;
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] -= other.data_[i % other.size_];
        }
        return result;
    }

    Tensor<T> operator*(const Tensor<T>& other) const {
        check_broadcast_compatible(other);
        Tensor<T> result = *this;
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] *= other.data_[i % other.size_];
        }
        return result;
    }

    Tensor<T> operator*(T scalar) const {
        Tensor<T> result = *this;
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] *= scalar;
        }
        return result;
    }

    // In-place operations (for memory efficiency)
    void add_inplace(const Tensor<T>& other) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] += other.data_[i % other.size_];
        }
    }

    void add_inplace(T scalar) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] += scalar;
        }
    }

    void multiply_inplace(T scalar) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] *= scalar;
        }
    }

    // Reduction operations
    T sum() const {
        return std::accumulate(data_.begin(), data_.end(), static_cast<T>(0));
    }

    T mean() const {
        return sum() / static_cast<T>(size_);
    }

    T max() const {
        return *std::max_element(data_.begin(), data_.end());
    }

    T min() const {
        return *std::min_element(data_.begin(), data_.end());
    }

    // Utility functions
    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void zero() {
        fill(static_cast<T>(0));
    }

    // Clone
    Tensor<T> clone() const {
        return Tensor<T>(shape_, data_);
    }

private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    size_t size_;

    static size_t compute_size(const std::vector<size_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    }

    size_t compute_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("Index dimensions don't match tensor dimensions");
        }
        size_t offset = 0;
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            offset += indices[i] * stride;
            stride *= shape_[i];
        }
        return offset;
    }

    void check_broadcast_compatible(const Tensor<T>& other) const {
        // Simplified broadcast check - just check if one is divisible by the other
        if (size_ % other.size_ != 0 && other.size_ % size_ != 0 && size_ != other.size_) {
            throw std::runtime_error("Tensors are not broadcast compatible");
        }
    }
};

// Helper function for creating tensors from initializer lists
template<typename T>
Tensor<T> make_tensor_1d(const std::vector<T>& data) {
    return Tensor<T>({data.size()}, data);
}

template<typename T>
Tensor<T> make_tensor_2d(size_t rows, size_t cols, const std::vector<T>& data) {
    return Tensor<T>({rows, cols}, data);
}

} // namespace embedded_ml

#endif // EMBEDDED_ML_TENSOR_HPP
