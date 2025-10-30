/**
 * Example 2: Sensor Data Preprocessing - C++ Implementation
 *
 * Direct conversion from Python showing data processing operations
 * No ML model - pure data preprocessing for embedded sensors
 *
 * Compile: g++ -std=c++17 -O3 cpp_version.cpp -o cpp_sensor
 * Run: ./cpp_sensor
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <limits>

// ============================================================================
// SENSOR PREPROCESSING PIPELINE (Direct conversion from Python)
// ============================================================================

struct ProcessingResult {
    std::vector<float> original;
    std::vector<float> processed;
    std::vector<size_t> anomaly_indices;
    size_t num_anomalies;
    struct {
        float mean;
        float std;
        float min;
        float max;
    } statistics;
};

ProcessingResult sensor_preprocessing_pipeline(const std::vector<float>& raw_readings) {
    ProcessingResult result;
    result.original = raw_readings;

    // Step 1: Handle missing values (NaN/Inf)
    // Python: mask = ~np.isfinite(data)
    std::vector<float> data = raw_readings;

    // Find median of valid values
    std::vector<float> valid_values;
    for (float val : data) {
        if (std::isfinite(val)) {
            valid_values.push_back(val);
        }
    }

    // Calculate median
    std::sort(valid_values.begin(), valid_values.end());
    float median_val = valid_values[valid_values.size() / 2];

    // Replace invalid values with median
    for (float& val : data) {
        if (!std::isfinite(val)) {
            val = median_val;
        }
    }

    // Step 2: Remove outliers (clip to ±3 standard deviations)
    // Python: mean = np.mean(data)
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();

    // Python: std = np.std(data)
    float variance = 0.0f;
    for (float val : data) {
        float diff = val - mean;
        variance += diff * diff;
    }
    float std = std::sqrt(variance / data.size());

    // Python: data = np.clip(data, mean - 3*std, mean + 3*std)
    float lower_bound = mean - 3 * std;
    float upper_bound = mean + 3 * std;
    for (float& val : data) {
        val = std::clamp(val, lower_bound, upper_bound);
    }

    // Step 3: Moving average filter (window=5)
    // Python: smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    const size_t window_size = 5;
    std::vector<float> smoothed(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        float sum = 0.0f;
        size_t count = 0;

        // Centered window
        int half_window = window_size / 2;
        for (int j = -half_window; j <= half_window; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx >= 0 && idx < static_cast<int>(data.size())) {
                sum += data[idx];
                count++;
            }
        }

        smoothed[i] = sum / count;
    }

    // Step 4: Normalize (z-score normalization)
    // Python: normalized_mean = np.mean(smoothed)
    float normalized_mean = std::accumulate(smoothed.begin(), smoothed.end(), 0.0f)
                          / smoothed.size();

    // Python: normalized_std = np.std(smoothed)
    float norm_variance = 0.0f;
    for (float val : smoothed) {
        float diff = val - normalized_mean;
        norm_variance += diff * diff;
    }
    float normalized_std = std::sqrt(norm_variance / smoothed.size());

    // Python: normalized = (smoothed - normalized_mean) / (normalized_std + 1e-8)
    std::vector<float> normalized(smoothed.size());
    for (size_t i = 0; i < smoothed.size(); ++i) {
        normalized[i] = (smoothed[i] - normalized_mean) / (normalized_std + 1e-8f);
    }

    result.processed = normalized;

    // Step 5: Detect anomalies (values beyond ±2.5 std)
    // Python: anomalies = np.abs(normalized) > anomaly_threshold
    const float anomaly_threshold = 2.5f;
    for (size_t i = 0; i < normalized.size(); ++i) {
        if (std::abs(normalized[i]) > anomaly_threshold) {
            result.anomaly_indices.push_back(i);
        }
    }
    result.num_anomalies = result.anomaly_indices.size();

    // Compute statistics
    result.statistics.mean = normalized_mean;
    result.statistics.std = normalized_std;
    result.statistics.min = *std::min_element(normalized.begin(), normalized.end());
    result.statistics.max = *std::max_element(normalized.begin(), normalized.end());

    return result;
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

std::vector<float> load_sensor_data(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);

    float val;
    while (file >> val) {
        data.push_back(val);
    }

    return data;
}

void save_results(const ProcessingResult& result, const std::string& filename) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(6);

    file << "{\n";
    file << "  \"num_anomalies\": " << result.num_anomalies << ",\n";
    file << "  \"anomaly_indices\": [";
    for (size_t i = 0; i < result.anomaly_indices.size(); ++i) {
        file << result.anomaly_indices[i];
        if (i < result.anomaly_indices.size() - 1) file << ", ";
    }
    file << "],\n";

    file << "  \"statistics\": {\n";
    file << "    \"mean\": " << result.statistics.mean << ",\n";
    file << "    \"std\": " << result.statistics.std << ",\n";
    file << "    \"min\": " << result.statistics.min << ",\n";
    file << "    \"max\": " << result.statistics.max << "\n";
    file << "  },\n";

    file << "  \"processed\": [";
    for (size_t i = 0; i < result.processed.size(); ++i) {
        file << result.processed[i];
        if (i < result.processed.size() - 1) file << ", ";
    }
    file << "]\n";

    file << "}\n";
}

int main() {
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "C++ IMPLEMENTATION - Sensor Data Preprocessing" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Load sensor data
    auto raw_readings = load_sensor_data("sensor_data.txt");

    if (raw_readings.empty()) {
        std::cerr << "Error: Could not load sensor_data.txt" << std::endl;
        std::cerr << "Run python_version.py first to generate test data" << std::endl;
        return 1;
    }

    std::cout << "\nLoaded " << raw_readings.size() << " sensor readings" << std::endl;

    // Run preprocessing pipeline
    auto result = sensor_preprocessing_pipeline(raw_readings);

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "C++ PROCESSING RESULTS:" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Processed " << result.processed.size() << " samples" << std::endl;
    std::cout << "Detected " << result.num_anomalies << " anomalies at indices: [";
    for (size_t i = 0; i < result.anomaly_indices.size(); ++i) {
        std::cout << result.anomaly_indices[i];
        if (i < result.anomaly_indices.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\nProcessed data statistics:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Mean: " << result.statistics.mean << std::endl;
    std::cout << "  Std:  " << result.statistics.std << std::endl;
    std::cout << "  Min:  " << result.statistics.min << std::endl;
    std::cout << "  Max:  " << result.statistics.max << std::endl;

    std::cout << "\nFirst 5 processed values: [";
    for (size_t i = 0; i < std::min(size_t(5), result.processed.size()); ++i) {
        std::cout << std::setprecision(4) << result.processed[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Last 5 processed values:  [";
    for (size_t i = result.processed.size() - 5; i < result.processed.size(); ++i) {
        std::cout << std::setprecision(4) << result.processed[i];
        if (i < result.processed.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Save results
    save_results(result, "cpp_results.json");
    std::cout << "\n✓ Saved results to cpp_results.json" << std::endl;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Run 'python verify_preprocessing.py' to verify outputs match!" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return 0;
}
