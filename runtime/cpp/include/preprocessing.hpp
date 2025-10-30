/**
 * Data Preprocessing Utilities
 * Replicate common Python/NumPy data preprocessing operations
 * No external dependencies - uses only standard C++17
 */

#ifndef EMBEDDED_ML_PREPROCESSING_HPP
#define EMBEDDED_ML_PREPROCESSING_HPP

#include "tensor.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace embedded_ml {
namespace preprocessing {

/**
 * Normalization - Scale values to [0, 1]
 * new_value = (value - min) / (max - min)
 */
template<typename T = float>
class MinMaxScaler {
public:
    MinMaxScaler(T feature_min = 0, T feature_max = 1)
        : feature_min_(feature_min), feature_max_(feature_max),
          data_min_(0), data_max_(0), fitted_(false) {}

    // Fit the scaler to data
    void fit(const Tensor<T>& data) {
        data_min_ = data.min();
        data_max_ = data.max();
        fitted_ = true;
    }

    // Transform data
    Tensor<T> transform(const Tensor<T>& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }

        Tensor<T> result = data.clone();
        T data_range = data_max_ - data_min_;
        T feature_range = feature_max_ - feature_min_;

        if (data_range == 0) {
            result.fill(feature_min_);
            return result;
        }

        for (size_t i = 0; i < result.size(); ++i) {
            result.data()[i] = feature_min_ + (result.data()[i] - data_min_) * feature_range / data_range;
        }

        return result;
    }

    // Fit and transform in one step
    Tensor<T> fit_transform(const Tensor<T>& data) {
        fit(data);
        return transform(data);
    }

    // Transform single value
    T transform_value(T value) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }
        T data_range = data_max_ - data_min_;
        T feature_range = feature_max_ - feature_min_;
        if (data_range == 0) return feature_min_;
        return feature_min_ + (value - data_min_) * feature_range / data_range;
    }

    // Inverse transform
    Tensor<T> inverse_transform(const Tensor<T>& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }

        Tensor<T> result = data.clone();
        T data_range = data_max_ - data_min_;
        T feature_range = feature_max_ - feature_min_;

        for (size_t i = 0; i < result.size(); ++i) {
            result.data()[i] = data_min_ + (result.data()[i] - feature_min_) * data_range / feature_range;
        }

        return result;
    }

private:
    T feature_min_;
    T feature_max_;
    T data_min_;
    T data_max_;
    bool fitted_;
};

/**
 * Standardization - Scale to zero mean and unit variance
 * new_value = (value - mean) / std
 */
template<typename T = float>
class StandardScaler {
public:
    StandardScaler() : mean_(0), std_(0), fitted_(false) {}

    void fit(const Tensor<T>& data) {
        mean_ = data.mean();

        // Calculate standard deviation
        T variance = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            T diff = data.data()[i] - mean_;
            variance += diff * diff;
        }
        variance /= data.size();
        std_ = std::sqrt(variance);

        fitted_ = true;
    }

    Tensor<T> transform(const Tensor<T>& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }

        Tensor<T> result = data.clone();

        if (std_ == 0) {
            result.fill(0);
            return result;
        }

        for (size_t i = 0; i < result.size(); ++i) {
            result.data()[i] = (result.data()[i] - mean_) / std_;
        }

        return result;
    }

    Tensor<T> fit_transform(const Tensor<T>& data) {
        fit(data);
        return transform(data);
    }

    // Transform with known mean and std (for deployment)
    static Tensor<T> transform_with_params(const Tensor<T>& data, T mean, T std) {
        Tensor<T> result = data.clone();
        for (size_t i = 0; i < result.size(); ++i) {
            result.data()[i] = (result.data()[i] - mean) / std;
        }
        return result;
    }

    T transform_value(T value) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }
        if (std_ == 0) return 0;
        return (value - mean_) / std_;
    }

    Tensor<T> inverse_transform(const Tensor<T>& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }

        Tensor<T> result = data.clone();
        for (size_t i = 0; i < result.size(); ++i) {
            result.data()[i] = result.data()[i] * std_ + mean_;
        }

        return result;
    }

    T get_mean() const { return mean_; }
    T get_std() const { return std_; }

private:
    T mean_;
    T std_;
    bool fitted_;
};

/**
 * Image normalization (common in computer vision)
 * Normalize each channel with its mean and std
 */
template<typename T = float>
class ImageNormalizer {
public:
    ImageNormalizer(const std::vector<T>& mean, const std::vector<T>& std)
        : mean_(mean), std_(std) {
        if (mean.size() != std.size()) {
            throw std::runtime_error("Mean and std must have same size");
        }
    }

    // Normalize image tensor (C, H, W) or (N, C, H, W)
    Tensor<T> transform(const Tensor<T>& image) const {
        Tensor<T> result = image.clone();
        const auto& shape = image.shape();

        if (shape.size() == 3) {
            // Single image (C, H, W)
            size_t channels = shape[0];
            size_t spatial_size = shape[1] * shape[2];

            for (size_t c = 0; c < channels; ++c) {
                T mean = c < mean_.size() ? mean_[c] : 0;
                T std = c < std_.size() ? std_[c] : 1;

                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = c * spatial_size + s;
                    result.data()[idx] = (result.data()[idx] - mean) / std;
                }
            }
        } else if (shape.size() == 4) {
            // Batch of images (N, C, H, W)
            size_t batch = shape[0];
            size_t channels = shape[1];
            size_t spatial_size = shape[2] * shape[3];

            for (size_t b = 0; b < batch; ++b) {
                for (size_t c = 0; c < channels; ++c) {
                    T mean = c < mean_.size() ? mean_[c] : 0;
                    T std = c < std_.size() ? std_[c] : 1;

                    for (size_t s = 0; s < spatial_size; ++s) {
                        size_t idx = ((b * channels + c) * spatial_size) + s;
                        result.data()[idx] = (result.data()[idx] - mean) / std;
                    }
                }
            }
        }

        return result;
    }

private:
    std::vector<T> mean_;
    std::vector<T> std_;
};

/**
 * One-hot encoding for categorical data
 */
template<typename T = float>
class OneHotEncoder {
public:
    explicit OneHotEncoder(size_t num_classes) : num_classes_(num_classes) {}

    // Encode single class index to one-hot vector
    Tensor<T> encode(size_t class_idx) const {
        if (class_idx >= num_classes_) {
            throw std::out_of_range("Class index out of range");
        }

        Tensor<T> result({num_classes_}, static_cast<T>(0));
        result.data()[class_idx] = static_cast<T>(1);
        return result;
    }

    // Encode batch of class indices
    Tensor<T> encode_batch(const std::vector<size_t>& class_indices) const {
        Tensor<T> result({class_indices.size(), num_classes_}, static_cast<T>(0));

        for (size_t i = 0; i < class_indices.size(); ++i) {
            if (class_indices[i] >= num_classes_) {
                throw std::out_of_range("Class index out of range");
            }
            result.data()[i * num_classes_ + class_indices[i]] = static_cast<T>(1);
        }

        return result;
    }

    // Decode one-hot back to class index
    size_t decode(const Tensor<T>& one_hot) const {
        return std::distance(one_hot.data().begin(),
                           std::max_element(one_hot.data().begin(), one_hot.data().end()));
    }

private:
    size_t num_classes_;
};

/**
 * Clip values to range [min, max]
 */
template<typename T = float>
void clip(Tensor<T>& tensor, T min_val, T max_val) {
    for (auto& val : tensor.data()) {
        val = std::clamp(val, min_val, max_val);
    }
}

/**
 * Apply log transformation: log(1 + x)
 */
template<typename T = float>
void log1p(Tensor<T>& tensor) {
    for (auto& val : tensor.data()) {
        val = std::log(static_cast<T>(1) + val);
    }
}

/**
 * Apply exponential transformation: exp(x) - 1
 */
template<typename T = float>
void expm1(Tensor<T>& tensor) {
    for (auto& val : tensor.data()) {
        val = std::exp(val) - static_cast<T>(1);
    }
}

/**
 * Apply power transformation: x^power
 */
template<typename T = float>
void power(Tensor<T>& tensor, T power) {
    for (auto& val : tensor.data()) {
        val = std::pow(val, power);
    }
}

/**
 * Missing value imputation (simple mean strategy)
 */
template<typename T = float>
class SimpleImputer {
public:
    SimpleImputer(T missing_value = std::numeric_limits<T>::quiet_NaN())
        : missing_value_(missing_value), fill_value_(0), fitted_(false) {}

    void fit(const Tensor<T>& data) {
        // Calculate mean of non-missing values
        T sum = 0;
        size_t count = 0;

        for (size_t i = 0; i < data.size(); ++i) {
            if (!is_missing(data.data()[i])) {
                sum += data.data()[i];
                count++;
            }
        }

        fill_value_ = count > 0 ? sum / count : 0;
        fitted_ = true;
    }

    Tensor<T> transform(const Tensor<T>& data) const {
        if (!fitted_) {
            throw std::runtime_error("Imputer not fitted");
        }

        Tensor<T> result = data.clone();
        for (size_t i = 0; i < result.size(); ++i) {
            if (is_missing(result.data()[i])) {
                result.data()[i] = fill_value_;
            }
        }

        return result;
    }

    Tensor<T> fit_transform(const Tensor<T>& data) {
        fit(data);
        return transform(data);
    }

private:
    T missing_value_;
    T fill_value_;
    bool fitted_;

    bool is_missing(T value) const {
        if (std::isnan(missing_value_)) {
            return std::isnan(value);
        }
        return value == missing_value_;
    }
};

} // namespace preprocessing
} // namespace embedded_ml

#endif // EMBEDDED_ML_PREPROCESSING_HPP
