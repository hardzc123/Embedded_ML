/**
 * End-to-End Production Pipeline in C++
 *
 * This shows how to convert a complete Python production pipeline to C++
 * Including preprocessing, model inference, and postprocessing
 *
 * Compile with:
 *   g++ -std=c++17 -O3 -I../../runtime/cpp/include main.cpp -o pipeline
 */

#ifndef PRODUCTION_PIPELINE_HPP
#define PRODUCTION_PIPELINE_HPP

#include "tensor.hpp"
#include "preprocessing.hpp"
#include "postprocessing.hpp"
#include "math_utils.hpp"
#include "activations.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>

using namespace embedded_ml;

namespace production {

/**
 * Configuration structure
 * Loaded from deployment_config.json at compile time or runtime
 */
struct PipelineConfig {
    // Preprocessing parameters
    float preprocess_mean = 0.5f;
    float preprocess_std = 0.2f;
    float min_val = -5.0f;
    float max_val = 5.0f;

    // Postprocessing parameters
    float confidence_threshold = 0.7f;
    size_t top_k = 5;
    bool use_temperature_scaling = true;
    float temperature = 1.5f;

    // Class labels
    std::vector<std::string> labels = {
        "cat", "dog", "bird", "fish", "hamster",
        "rabbit", "turtle", "snake", "lizard", "frog"
    };
};

/**
 * Data Preprocessor
 * Converts raw input to model-ready format
 */
class DataPreprocessor {
public:
    explicit DataPreprocessor(const PipelineConfig& config)
        : config_(config) {}

    /**
     * Handle missing values (replace NaN/Inf with mean)
     */
    Tensor<float> handle_missing_values(const Tensor<float>& data) const {
        Tensor<float> result = data.clone();

        for (auto& val : result.data()) {
            if (!std::isfinite(val)) {
                val = config_.preprocess_mean;
            }
        }

        return result;
    }

    /**
     * Clip outliers
     */
    Tensor<float> clip_outliers(const Tensor<float>& data) const {
        Tensor<float> result = data.clone();
        preprocessing::clip(result, config_.min_val, config_.max_val);
        return result;
    }

    /**
     * Z-score normalization
     */
    Tensor<float> normalize(const Tensor<float>& data) const {
        return preprocessing::StandardScaler<float>::transform_with_params(
            data, config_.preprocess_mean, config_.preprocess_std);
    }

    /**
     * Apply mathematical transformations
     * Python: np.log1p(np.abs(data)) * np.sign(data)
     */
    Tensor<float> apply_transforms(const Tensor<float>& data) const {
        Tensor<float> result = data.clone();

        for (auto& val : result.data()) {
            float sign = val >= 0 ? 1.0f : -1.0f;
            val = std::log1p(std::abs(val)) * sign;
        }

        return result;
    }

    /**
     * Complete preprocessing pipeline
     * This mirrors the Python preprocess() method exactly
     */
    Tensor<float> preprocess(const Tensor<float>& raw_data) const {
        // Step 1: Handle missing values
        auto data = handle_missing_values(raw_data);

        // Step 2: Clip outliers
        data = clip_outliers(data);

        // Step 3: Normalize
        data = normalize(data);

        // Step 4: Apply transforms
        data = apply_transforms(data);

        return data;
    }

private:
    const PipelineConfig& config_;
};

/**
 * Prediction result structure
 */
struct Prediction {
    std::string label;
    float probability;
    size_t index;
};

/**
 * Result Postprocessor
 * Processes model output into human-readable results
 */
class ResultPostprocessor {
public:
    explicit ResultPostprocessor(const PipelineConfig& config)
        : config_(config) {}

    /**
     * Apply softmax with temperature scaling
     * Python equivalent:
     *   logits = logits / temperature
     *   exp_logits = np.exp(logits - np.max(logits))
     *   return exp_logits / np.sum(exp_logits)
     */
    Tensor<float> apply_softmax(const Tensor<float>& logits) const {
        Tensor<float> result = logits.clone();

        // Temperature scaling
        if (config_.use_temperature_scaling) {
            result.multiply_inplace(1.0f / config_.temperature);
        }

        // Softmax with numerical stability
        float max_val = result.max();
        float sum = 0.0f;

        for (auto& val : result.data()) {
            val = std::exp(val - max_val);
            sum += val;
        }

        for (auto& val : result.data()) {
            val /= sum;
        }

        return result;
    }

    /**
     * Get top-k predictions
     * Python equivalent: np.argsort(probabilities)[-k:][::-1]
     */
    std::vector<Prediction> get_top_k_predictions(
        const Tensor<float>& probabilities) const
    {
        auto topk_result = postprocessing::topk(probabilities, config_.top_k);

        std::vector<Prediction> predictions;
        predictions.reserve(topk_result.indices.size());

        for (size_t i = 0; i < topk_result.indices.size(); ++i) {
            size_t idx = topk_result.indices[i];
            predictions.push_back({
                idx < config_.labels.size() ? config_.labels[idx] : "unknown",
                topk_result.values[i],
                idx
            });
        }

        return predictions;
    }

    /**
     * Filter by confidence threshold
     */
    std::vector<Prediction> filter_by_confidence(
        const std::vector<Prediction>& predictions) const
    {
        std::vector<Prediction> filtered;

        for (const auto& pred : predictions) {
            if (pred.probability >= config_.confidence_threshold) {
                filtered.push_back(pred);
            }
        }

        return filtered;
    }

    /**
     * Complete postprocessing pipeline
     */
    struct PostprocessResult {
        Prediction prediction;
        std::vector<Prediction> top_k;
        std::vector<float> probabilities;
        float confidence;
    };

    PostprocessResult postprocess(const Tensor<float>& model_output) const {
        PostprocessResult result;

        // Step 1: Apply softmax
        auto probabilities = apply_softmax(model_output);
        result.probabilities = probabilities.data();

        // Step 2: Get top-k predictions
        auto top_predictions = get_top_k_predictions(probabilities);

        // Step 3: Filter by confidence
        auto confident_predictions = filter_by_confidence(top_predictions);

        // Step 4: Determine final prediction
        if (!confident_predictions.empty()) {
            result.prediction = confident_predictions[0];
        } else if (!top_predictions.empty()) {
            result.prediction = top_predictions[0];
        } else {
            result.prediction = {"unknown", 0.0f, 0};
        }

        // Return top 3 for user
        result.top_k = std::vector<Prediction>(
            top_predictions.begin(),
            top_predictions.begin() + std::min(size_t(3), top_predictions.size())
        );

        result.confidence = result.prediction.probability;

        return result;
    }

private:
    const PipelineConfig& config_;
};

/**
 * Complete Production Pipeline
 * Combines preprocessing, model inference, and postprocessing
 *
 * Template parameter Model should be the auto-generated model class
 */
template<typename Model>
class ProductionPipeline {
public:
    ProductionPipeline(const PipelineConfig& config)
        : config_(config),
          preprocessor_(config),
          postprocessor_(config),
          model_() {}

    /**
     * Single prediction
     * This is the main entry point for inference
     */
    typename ResultPostprocessor::PostprocessResult predict(const Tensor<float>& raw_input) {
        // STEP 1: Preprocessing
        auto preprocessed = preprocessor_.preprocess(raw_input);

        // STEP 2: Model Inference
        auto model_output = model_.forward(preprocessed.data());

        // Convert to tensor for postprocessing
        Tensor<float> output_tensor({model_output.size()}, model_output);

        // STEP 3: Postprocessing
        auto result = postprocessor_.postprocess(output_tensor);

        return result;
    }

    /**
     * Batch prediction
     */
    std::vector<typename ResultPostprocessor::PostprocessResult>
    batch_predict(const std::vector<Tensor<float>>& raw_inputs) {
        std::vector<typename ResultPostprocessor::PostprocessResult> results;
        results.reserve(raw_inputs.size());

        for (const auto& input : raw_inputs) {
            results.push_back(predict(input));
        }

        return results;
    }

private:
    PipelineConfig config_;
    DataPreprocessor preprocessor_;
    ResultPostprocessor postprocessor_;
    Model model_;
};

} // namespace production

#endif // PRODUCTION_PIPELINE_HPP
