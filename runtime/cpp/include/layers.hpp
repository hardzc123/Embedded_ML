/**
 * Layers - Common neural network layers
 * Uses only standard C++17 features, no external dependencies
 */

#ifndef EMBEDDED_ML_LAYERS_HPP
#define EMBEDDED_ML_LAYERS_HPP

#include "tensor.hpp"
#include <memory>

namespace embedded_ml {
namespace layers {

/**
 * Linear (Fully Connected) Layer
 * y = xW^T + b
 */
template<typename T = float>
class Linear {
public:
    Linear(const Tensor<T>& weight, const Tensor<T>* bias = nullptr)
        : weight_(weight), bias_(bias ? std::make_unique<Tensor<T>>(*bias) : nullptr) {

        if (weight.shape().size() != 2) {
            throw std::runtime_error("Linear layer weight must be 2D");
        }
    }

    Tensor<T> forward(const Tensor<T>& input) const {
        // input: (batch, in_features) or (in_features,)
        // weight: (out_features, in_features)
        // output: (batch, out_features) or (out_features,)

        bool is_batched = input.shape().size() == 2;

        Tensor<T> x = input;
        if (!is_batched) {
            // Add batch dimension
            x = input.reshape({1, input.size()});
        }

        // x @ weight.T
        Tensor<T> result = matmul_transpose(x, weight_);

        // Add bias if present
        if (bias_) {
            result.add_inplace(*bias_);
        }

        if (!is_batched) {
            // Remove batch dimension
            result = result.reshape({result.size()});
        }

        return result;
    }

private:
    Tensor<T> weight_;
    std::unique_ptr<Tensor<T>> bias_;

    // Matrix multiply with transposed second matrix
    Tensor<T> matmul_transpose(const Tensor<T>& a, const Tensor<T>& b) const {
        // a: (m, k), b: (n, k) -> output: (m, n)
        size_t m = a.shape()[0];
        size_t k = a.shape()[1];
        size_t n = b.shape()[0];

        if (b.shape()[1] != k) {
            throw std::runtime_error("Matrix dimensions don't match");
        }

        Tensor<T> result({m, n}, static_cast<T>(0));

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = 0;
                for (size_t p = 0; p < k; ++p) {
                    sum += a(i, p) * b(j, p);  // Note: b(j, p) not b(p, j) because we want transpose
                }
                result(i, j) = sum;
            }
        }

        return result;
    }
};

/**
 * Conv2D Layer
 * 2D Convolution
 */
template<typename T = float>
class Conv2d {
public:
    struct Params {
        size_t in_channels;
        size_t out_channels;
        size_t kernel_h;
        size_t kernel_w;
        size_t stride_h = 1;
        size_t stride_w = 1;
        size_t padding_h = 0;
        size_t padding_w = 0;
    };

    Conv2d(const Tensor<T>& weight, const Tensor<T>* bias, const Params& params)
        : weight_(weight),
          bias_(bias ? std::make_unique<Tensor<T>>(*bias) : nullptr),
          params_(params) {

        // Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        if (weight.shape().size() != 4) {
            throw std::runtime_error("Conv2d weight must be 4D");
        }
    }

    Tensor<T> forward(const Tensor<T>& input) const {
        // input: (batch, in_channels, height, width)
        const auto& in_shape = input.shape();

        if (in_shape.size() != 4) {
            throw std::runtime_error("Conv2d input must be 4D (batch, channels, height, width)");
        }

        size_t batch = in_shape[0];
        size_t in_h = in_shape[2];
        size_t in_w = in_shape[3];

        // Calculate output dimensions
        size_t out_h = (in_h + 2 * params_.padding_h - params_.kernel_h) / params_.stride_h + 1;
        size_t out_w = (in_w + 2 * params_.padding_w - params_.kernel_w) / params_.stride_w + 1;

        Tensor<T> output({batch, params_.out_channels, out_h, out_w}, static_cast<T>(0));

        // Perform convolution
        for (size_t b = 0; b < batch; ++b) {
            for (size_t oc = 0; oc < params_.out_channels; ++oc) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        T sum = 0;

                        // Convolve with kernel
                        for (size_t ic = 0; ic < params_.in_channels; ++ic) {
                            for (size_t kh = 0; kh < params_.kernel_h; ++kh) {
                                for (size_t kw = 0; kw < params_.kernel_w; ++kw) {
                                    int ih = oh * params_.stride_h + kh - params_.padding_h;
                                    int iw = ow * params_.stride_w + kw - params_.padding_w;

                                    // Check bounds (padding)
                                    if (ih >= 0 && ih < static_cast<int>(in_h) &&
                                        iw >= 0 && iw < static_cast<int>(in_w)) {

                                        size_t in_idx = ((b * params_.in_channels + ic) * in_h + ih) * in_w + iw;
                                        size_t w_idx = (((oc * params_.in_channels + ic) * params_.kernel_h + kh) * params_.kernel_w + kw);

                                        sum += input.data()[in_idx] * weight_.data()[w_idx];
                                    }
                                }
                            }
                        }

                        // Add bias
                        if (bias_) {
                            sum += bias_->data()[oc];
                        }

                        size_t out_idx = ((b * params_.out_channels + oc) * out_h + oh) * out_w + ow;
                        output.data()[out_idx] = sum;
                    }
                }
            }
        }

        return output;
    }

private:
    Tensor<T> weight_;
    std::unique_ptr<Tensor<T>> bias_;
    Params params_;
};

/**
 * MaxPool2d Layer
 */
template<typename T = float>
class MaxPool2d {
public:
    MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0)
        : kernel_size_(kernel_size),
          stride_(stride > 0 ? stride : kernel_size),
          padding_(padding) {}

    Tensor<T> forward(const Tensor<T>& input) const {
        // input: (batch, channels, height, width)
        const auto& in_shape = input.shape();

        size_t batch = in_shape[0];
        size_t channels = in_shape[1];
        size_t in_h = in_shape[2];
        size_t in_w = in_shape[3];

        size_t out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;

        Tensor<T> output({batch, channels, out_h, out_w});

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        T max_val = std::numeric_limits<T>::lowest();

                        for (size_t kh = 0; kh < kernel_size_; ++kh) {
                            for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;

                                if (ih >= 0 && ih < static_cast<int>(in_h) &&
                                    iw >= 0 && iw < static_cast<int>(in_w)) {
                                    size_t idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                                    max_val = std::max(max_val, input.data()[idx]);
                                }
                            }
                        }

                        size_t out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        output.data()[out_idx] = max_val;
                    }
                }
            }
        }

        return output;
    }

private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
};

/**
 * BatchNorm2d Layer (Inference mode)
 */
template<typename T = float>
class BatchNorm2d {
public:
    BatchNorm2d(const Tensor<T>& running_mean,
                const Tensor<T>& running_var,
                const Tensor<T>* gamma = nullptr,
                const Tensor<T>* beta = nullptr,
                T eps = 1e-5)
        : running_mean_(running_mean),
          running_var_(running_var),
          gamma_(gamma ? std::make_unique<Tensor<T>>(*gamma) : nullptr),
          beta_(beta ? std::make_unique<Tensor<T>>(*beta) : nullptr),
          eps_(eps) {}

    Tensor<T> forward(const Tensor<T>& input) const {
        // input: (batch, channels, height, width) or (batch, channels)
        Tensor<T> output = input.clone();
        const auto& shape = input.shape();

        size_t batch = shape[0];
        size_t channels = shape[1];
        size_t spatial_size = input.size() / (batch * channels);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                T mean = running_mean_.data()[c];
                T var = running_var_.data()[c];
                T std = std::sqrt(var + eps_);

                T scale = gamma_ ? gamma_->data()[c] : static_cast<T>(1);
                T shift = beta_ ? beta_->data()[c] : static_cast<T>(0);

                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = (b * channels + c) * spatial_size + s;
                    output.data()[idx] = scale * (output.data()[idx] - mean) / std + shift;
                }
            }
        }

        return output;
    }

private:
    Tensor<T> running_mean_;
    Tensor<T> running_var_;
    std::unique_ptr<Tensor<T>> gamma_;
    std::unique_ptr<Tensor<T>> beta_;
    T eps_;
};

/**
 * Flatten Layer
 */
template<typename T = float>
class Flatten {
public:
    Flatten(int start_dim = 1, int end_dim = -1)
        : start_dim_(start_dim), end_dim_(end_dim) {}

    Tensor<T> forward(const Tensor<T>& input) const {
        const auto& shape = input.shape();

        // Simple case: flatten all dimensions after start_dim
        if (start_dim_ == 1 && shape.size() > 1) {
            size_t batch = shape[0];
            size_t features = input.size() / batch;
            return input.reshape({batch, features});
        }

        // For other cases, just flatten everything
        return input.reshape({input.size()});
    }

private:
    int start_dim_;
    int end_dim_;
};

} // namespace layers
} // namespace embedded_ml

#endif // EMBEDDED_ML_LAYERS_HPP
