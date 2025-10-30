/**
 * Postprocessing Utilities
 * Common operations for model output processing
 * No external dependencies - uses only standard C++17
 */

#ifndef EMBEDDED_ML_POSTPROCESSING_HPP
#define EMBEDDED_ML_POSTPROCESSING_HPP

#include "tensor.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

namespace embedded_ml {
namespace postprocessing {

/**
 * Argmax - Find index of maximum value
 */
template<typename T = float>
size_t argmax(const Tensor<T>& tensor) {
    const auto& data = tensor.data();
    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

/**
 * Argmax along specific dimension for 2D tensor
 */
template<typename T = float>
std::vector<size_t> argmax_axis(const Tensor<T>& tensor, int axis = 1) {
    if (tensor.shape().size() != 2) {
        throw std::runtime_error("argmax_axis currently only supports 2D tensors");
    }

    size_t rows = tensor.shape()[0];
    size_t cols = tensor.shape()[1];
    std::vector<size_t> result;

    if (axis == 1) {
        // Find max along each row
        result.resize(rows);
        for (size_t i = 0; i < rows; ++i) {
            size_t max_idx = 0;
            T max_val = tensor.data()[i * cols];
            for (size_t j = 1; j < cols; ++j) {
                if (tensor.data()[i * cols + j] > max_val) {
                    max_val = tensor.data()[i * cols + j];
                    max_idx = j;
                }
            }
            result[i] = max_idx;
        }
    } else if (axis == 0) {
        // Find max along each column
        result.resize(cols);
        for (size_t j = 0; j < cols; ++j) {
            size_t max_idx = 0;
            T max_val = tensor.data()[j];
            for (size_t i = 1; i < rows; ++i) {
                if (tensor.data()[i * cols + j] > max_val) {
                    max_val = tensor.data()[i * cols + j];
                    max_idx = i;
                }
            }
            result[j] = max_idx;
        }
    }

    return result;
}

/**
 * Top-K values and indices
 */
template<typename T = float>
struct TopKResult {
    std::vector<T> values;
    std::vector<size_t> indices;
};

template<typename T = float>
TopKResult<T> topk(const Tensor<T>& tensor, size_t k) {
    if (k > tensor.size()) {
        k = tensor.size();
    }

    // Create pairs of (value, index)
    std::vector<std::pair<T, size_t>> pairs;
    pairs.reserve(tensor.size());
    for (size_t i = 0; i < tensor.size(); ++i) {
        pairs.emplace_back(tensor.data()[i], i);
    }

    // Partial sort to get top k
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    TopKResult<T> result;
    result.values.reserve(k);
    result.indices.reserve(k);

    for (size_t i = 0; i < k; ++i) {
        result.values.push_back(pairs[i].first);
        result.indices.push_back(pairs[i].second);
    }

    return result;
}

/**
 * Bounding box structure
 */
template<typename T = float>
struct BoundingBox {
    T x1, y1, x2, y2;  // Coordinates
    T score;            // Confidence score
    size_t class_id;    // Class label

    T area() const {
        return (x2 - x1) * (y2 - y1);
    }

    T intersection(const BoundingBox& other) const {
        T x_left = std::max(x1, other.x1);
        T y_top = std::max(y1, other.y1);
        T x_right = std::min(x2, other.x2);
        T y_bottom = std::min(y2, other.y2);

        if (x_right < x_left || y_bottom < y_top) {
            return 0;
        }

        return (x_right - x_left) * (y_bottom - y_top);
    }

    T iou(const BoundingBox& other) const {
        T inter = intersection(other);
        if (inter == 0) return 0;

        T union_area = area() + other.area() - inter;
        return inter / union_area;
    }
};

/**
 * Non-Maximum Suppression (NMS)
 * Used in object detection to remove duplicate detections
 */
template<typename T = float>
std::vector<BoundingBox<T>> nms(std::vector<BoundingBox<T>>& boxes, T iou_threshold = 0.5) {
    if (boxes.empty()) return {};

    // Sort boxes by score (descending)
    std::sort(boxes.begin(), boxes.end(),
             [](const BoundingBox<T>& a, const BoundingBox<T>& b) {
                 return a.score > b.score;
             });

    std::vector<BoundingBox<T>> result;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(boxes[i]);

        // Suppress overlapping boxes
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;

            if (boxes[i].iou(boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

/**
 * Class-aware NMS (apply NMS per class)
 */
template<typename T = float>
std::vector<BoundingBox<T>> class_aware_nms(std::vector<BoundingBox<T>>& boxes,
                                             T iou_threshold = 0.5) {
    if (boxes.empty()) return {};

    // Group boxes by class
    std::vector<std::vector<BoundingBox<T>>> boxes_per_class;
    size_t max_class = 0;

    for (const auto& box : boxes) {
        max_class = std::max(max_class, box.class_id);
    }

    boxes_per_class.resize(max_class + 1);

    for (auto& box : boxes) {
        boxes_per_class[box.class_id].push_back(box);
    }

    // Apply NMS per class
    std::vector<BoundingBox<T>> result;

    for (auto& class_boxes : boxes_per_class) {
        if (class_boxes.empty()) continue;
        auto nms_boxes = nms(class_boxes, iou_threshold);
        result.insert(result.end(), nms_boxes.begin(), nms_boxes.end());
    }

    return result;
}

/**
 * Threshold filtering
 * Keep only predictions above confidence threshold
 */
template<typename T = float>
Tensor<T> threshold_filter(const Tensor<T>& scores, T threshold) {
    std::vector<T> filtered;
    for (const auto& score : scores.data()) {
        if (score >= threshold) {
            filtered.push_back(score);
        }
    }
    return Tensor<T>({filtered.size()}, filtered);
}

/**
 * Temperature scaling for calibration
 * output = softmax(logits / temperature)
 */
template<typename T = float>
Tensor<T> temperature_scaling(const Tensor<T>& logits, T temperature = 1.0) {
    Tensor<T> scaled = logits.clone();

    // Scale by temperature
    for (auto& val : scaled.data()) {
        val /= temperature;
    }

    // Apply softmax (simplified for 1D)
    T max_val = scaled.max();
    T sum = 0;

    for (auto& val : scaled.data()) {
        val = std::exp(val - max_val);
        sum += val;
    }

    for (auto& val : scaled.data()) {
        val /= sum;
    }

    return scaled;
}

/**
 * Label smoothing
 * Smooth hard labels to prevent overconfidence
 */
template<typename T = float>
Tensor<T> label_smoothing(const Tensor<T>& labels, T alpha = 0.1) {
    Tensor<T> smoothed = labels.clone();
    size_t num_classes = labels.size();

    for (auto& val : smoothed.data()) {
        if (val == 1.0) {
            val = 1.0 - alpha;
        } else {
            val = alpha / (num_classes - 1);
        }
    }

    return smoothed;
}

/**
 * Moving average filter for time series smoothing
 */
template<typename T = float>
class MovingAverageFilter {
public:
    explicit MovingAverageFilter(size_t window_size)
        : window_size_(window_size) {
        buffer_.reserve(window_size);
    }

    T update(T value) {
        buffer_.push_back(value);

        if (buffer_.size() > window_size_) {
            buffer_.erase(buffer_.begin());
        }

        T sum = std::accumulate(buffer_.begin(), buffer_.end(), static_cast<T>(0));
        return sum / buffer_.size();
    }

    void reset() {
        buffer_.clear();
    }

private:
    size_t window_size_;
    std::vector<T> buffer_;
};

/**
 * Exponential moving average
 */
template<typename T = float>
class ExponentialMovingAverage {
public:
    explicit ExponentialMovingAverage(T alpha = 0.2)
        : alpha_(alpha), value_(0), initialized_(false) {}

    T update(T new_value) {
        if (!initialized_) {
            value_ = new_value;
            initialized_ = true;
        } else {
            value_ = alpha_ * new_value + (1 - alpha_) * value_;
        }
        return value_;
    }

    T get() const { return value_; }

    void reset() {
        initialized_ = false;
        value_ = 0;
    }

private:
    T alpha_;
    T value_;
    bool initialized_;
};

/**
 * Ensemble voting (for multiple model outputs)
 */
template<typename T = float>
class EnsembleVoting {
public:
    // Hard voting - majority vote
    static size_t hard_voting(const std::vector<size_t>& predictions) {
        if (predictions.empty()) {
            throw std::runtime_error("Empty predictions");
        }

        // Count votes
        std::vector<size_t> counts;
        for (size_t pred : predictions) {
            if (pred >= counts.size()) {
                counts.resize(pred + 1, 0);
            }
            counts[pred]++;
        }

        // Find class with most votes
        return std::distance(counts.begin(),
                           std::max_element(counts.begin(), counts.end()));
    }

    // Soft voting - average probabilities
    static Tensor<T> soft_voting(const std::vector<Tensor<T>>& probability_vectors) {
        if (probability_vectors.empty()) {
            throw std::runtime_error("Empty probability vectors");
        }

        size_t num_classes = probability_vectors[0].size();
        Tensor<T> result({num_classes}, static_cast<T>(0));

        // Average probabilities
        for (const auto& probs : probability_vectors) {
            if (probs.size() != num_classes) {
                throw std::runtime_error("Inconsistent probability vector sizes");
            }
            for (size_t i = 0; i < num_classes; ++i) {
                result.data()[i] += probs.data()[i];
            }
        }

        // Normalize
        T num_models = static_cast<T>(probability_vectors.size());
        for (auto& val : result.data()) {
            val /= num_models;
        }

        return result;
    }
};

/**
 * Confusion Matrix (for evaluation)
 */
template<typename T = size_t>
class ConfusionMatrix {
public:
    explicit ConfusionMatrix(size_t num_classes)
        : num_classes_(num_classes) {
        matrix_.resize(num_classes * num_classes, 0);
    }

    void add(size_t true_label, size_t predicted_label) {
        if (true_label >= num_classes_ || predicted_label >= num_classes_) {
            throw std::out_of_range("Label out of range");
        }
        matrix_[true_label * num_classes_ + predicted_label]++;
    }

    T get(size_t true_label, size_t predicted_label) const {
        return matrix_[true_label * num_classes_ + predicted_label];
    }

    double accuracy() const {
        T correct = 0;
        T total = 0;
        for (size_t i = 0; i < num_classes_; ++i) {
            for (size_t j = 0; j < num_classes_; ++j) {
                T count = get(i, j);
                total += count;
                if (i == j) {
                    correct += count;
                }
            }
        }
        return total > 0 ? static_cast<double>(correct) / total : 0.0;
    }

    double precision(size_t class_idx) const {
        T true_positive = get(class_idx, class_idx);
        T false_positive = 0;
        for (size_t i = 0; i < num_classes_; ++i) {
            if (i != class_idx) {
                false_positive += get(i, class_idx);
            }
        }
        T total = true_positive + false_positive;
        return total > 0 ? static_cast<double>(true_positive) / total : 0.0;
    }

    double recall(size_t class_idx) const {
        T true_positive = get(class_idx, class_idx);
        T false_negative = 0;
        for (size_t j = 0; j < num_classes_; ++j) {
            if (j != class_idx) {
                false_negative += get(class_idx, j);
            }
        }
        T total = true_positive + false_negative;
        return total > 0 ? static_cast<double>(true_positive) / total : 0.0;
    }

    double f1_score(size_t class_idx) const {
        double prec = precision(class_idx);
        double rec = recall(class_idx);
        return (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
    }

    void reset() {
        std::fill(matrix_.begin(), matrix_.end(), 0);
    }

private:
    size_t num_classes_;
    std::vector<T> matrix_;
};

} // namespace postprocessing
} // namespace embedded_ml

#endif // EMBEDDED_ML_POSTPROCESSING_HPP
