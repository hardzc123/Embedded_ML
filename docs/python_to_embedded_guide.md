## Python to Embedded: Complete Conversion Guide

A comprehensive guide for researchers and engineers converting entire Python ML pipelines to C++/Rust for embedded deployment.

---

## 🎯 What This Guide Covers

Most conversion tools only handle the **neural network model**. But in production, you also need:

- ✅ Data preprocessing (normalization, scaling, transformations)
- ✅ Data postprocessing (softmax, argmax, confidence filtering)
- ✅ Mathematical operations (NumPy operations)
- ✅ Statistical calculations (mean, std, percentiles)
- ✅ Image processing (resizing, normalization)
- ✅ Time series operations (moving averages, smoothing)
- ✅ Result formatting and filtering

**This guide shows you how to convert ALL of it!**

---

## 📚 Table of Contents

1. [Quick Reference: Python → C++/Rust](#quick-reference)
2. [Data Preprocessing](#data-preprocessing)
3. [Mathematical Operations](#mathematical-operations)
4. [Postprocessing Operations](#postprocessing-operations)
5. [Complete Pipeline Example](#complete-pipeline-example)
6. [Advanced Patterns](#advanced-patterns)
7. [Performance Optimization](#performance-optimization)

---

## <a name="quick-reference"></a>📋 Quick Reference: Python → C++/Rust

### NumPy Operations

| Python (NumPy) | C++ | Rust | Header/Module |
|----------------|-----|------|---------------|
| `x.mean()` | `x.mean()` | `x.mean()` | tensor.hpp |
| `x.std()` | `math::std_deviation(x)` | - | math_utils.hpp |
| `x.min()` | `x.min()` | `x.min()` | tensor.hpp |
| `x.max()` | `x.max()` | `x.max()` | tensor.hpp |
| `x.sum()` | `x.sum()` | `x.sum()` | tensor.hpp |
| `np.sqrt(x)` | `math::sqrt(x)` | - | math_utils.hpp |
| `np.exp(x)` | `math::exp(x)` | - | math_utils.hpp |
| `np.log(x)` | `math::log(x)` | - | math_utils.hpp |
| `np.abs(x)` | `math::abs(x)` | - | math_utils.hpp |
| `np.clip(x, a, b)` | `preprocessing::clip(x, a, b)` | `clip(&mut x, a, b)` | preprocessing.hpp |
| `np.argmax(x)` | `postprocessing::argmax(x)` | - | postprocessing.hpp |
| `np.argsort(x)[-k:]` | `postprocessing::topk(x, k)` | - | postprocessing.hpp |
| `np.dot(a, b)` | `math::dot(a, b)` | - | math_utils.hpp |
| `np.matmul(a, b)` | `a.matmul(b)` | `a.matmul(&b)?` | tensor.hpp |
| `x.reshape(shape)` | `x.reshape(shape)` | `x.reshape(shape)?` | tensor.hpp |

### Scikit-learn Preprocessing

| Python (Sklearn) | C++ | Rust |
|------------------|-----|------|
| `StandardScaler()` | `preprocessing::StandardScaler<float>()` | `StandardScaler::new()` |
| `MinMaxScaler()` | `preprocessing::MinMaxScaler<float>()` | `MinMaxScaler::new()` |
| `scaler.fit(X)` | `scaler.fit(X)` | `scaler.fit(&X)` |
| `scaler.transform(X)` | `scaler.transform(X)` | `scaler.transform(&X)?` |
| `scaler.fit_transform(X)` | `scaler.fit_transform(X)` | `scaler.fit_transform(&X)?` |

---

## <a name="data-preprocessing"></a>🔄 Data Preprocessing

### 1. Normalization (Z-score)

**Python:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_normalized = scaler.transform(X_test)

# Or manually:
mean = X.mean()
std = X.std()
X_normalized = (X - mean) / std
```

**C++:**
```cpp
#include "preprocessing.hpp"

// Using StandardScaler
preprocessing::StandardScaler<float> scaler;
scaler.fit(X_train);
auto X_normalized = scaler.transform(X_test);

// Or manually:
float mean = X.mean();
float std = math::std_deviation(X);
auto X_normalized = (X - mean) * (1.0f / std);

// For deployment with known parameters:
auto X_normalized = preprocessing::StandardScaler<float>::transform_with_params(
    X, MEAN, STD);  // MEAN, STD are constants from training
```

**Rust:**
```rust
use preprocessing::StandardScaler;

// Using StandardScaler
let mut scaler = StandardScaler::new();
scaler.fit(&X_train);
let X_normalized = scaler.transform(&X_test)?;

// For deployment with known parameters:
let X_normalized = StandardScaler::transform_with_params(&X, MEAN, STD);
```

---

### 2. Min-Max Scaling

**Python:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
```

**C++:**
```cpp
#include "preprocessing.hpp"

preprocessing::MinMaxScaler<float> scaler(0.0f, 1.0f);
auto X_scaled = scaler.fit_transform(X);
```

**Rust:**
```rust
let mut scaler = MinMaxScaler::new(0.0, 1.0);
let X_scaled = scaler.fit_transform(&X)?;
```

---

### 3. Clipping / Outlier Removal

**Python:**
```python
import numpy as np

X_clipped = np.clip(X, -3, 3)
```

**C++:**
```cpp
#include "preprocessing.hpp"

Tensor<float> X_clipped = X.clone();
preprocessing::clip(X_clipped, -3.0f, 3.0f);
```

**Rust:**
```rust
use preprocessing::clip;

let mut X_clipped = X.clone();
clip(&mut X_clipped, -3.0, 3.0);
```

---

### 4. Handling Missing Values

**Python:**
```python
import numpy as np

# Replace NaN with mean
mask = np.isnan(X)
X[mask] = np.nanmean(X)

# Using sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**C++:**
```cpp
#include "preprocessing.hpp"

// Manual replacement
Tensor<float> X_clean = X.clone();
float mean = // calculate mean of non-NaN values
for (auto& val : X_clean.data()) {
    if (!std::isfinite(val)) {
        val = mean;
    }
}

// Using SimpleImputer
preprocessing::SimpleImputer<float> imputer;
auto X_imputed = imputer.fit_transform(X);
```

---

### 5. Log Transformation

**Python:**
```python
import numpy as np

X_log = np.log1p(X)  # log(1 + x), stable for small values
```

**C++:**
```cpp
#include "preprocessing.hpp"

Tensor<float> X_log = X.clone();
preprocessing::log1p(X_log);
```

**Rust:**
```rust
use preprocessing::log1p;

let mut X_log = X.clone();
log1p(&mut X_log);
```

---

### 6. Image Normalization

**Python:**
```python
import torchvision.transforms as transforms

# ImageNet normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
img_normalized = normalize(img)
```

**C++:**
```cpp
#include "preprocessing.hpp"

// ImageNet normalization
std::vector<float> mean = {0.485f, 0.456f, 0.406f};
std::vector<float> std = {0.229f, 0.224f, 0.225f};

preprocessing::ImageNormalizer<float> normalizer(mean, std);
auto img_normalized = normalizer.transform(img);
```

---

## <a name="mathematical-operations"></a>🔢 Mathematical Operations

### Statistical Functions

**Python:**
```python
import numpy as np

mean = np.mean(X)
median = np.median(X)
std = np.std(X)
var = np.var(X)
min_val = np.min(X)
max_val = np.max(X)
sum_val = np.sum(X)
```

**C++:**
```cpp
#include "math_utils.hpp"

float mean = math::mean(X);
float median = math::median(X);
float std = math::std_deviation(X);
float var = math::variance(X);
float min_val = X.min();
float max_val = X.max();
float sum_val = X.sum();
```

---

### Element-wise Operations

**Python:**
```python
import numpy as np

# Element-wise operations
Y = np.sqrt(X)
Y = np.exp(X)
Y = np.log(X)
Y = np.abs(X)
Y = np.square(X)
Y = X ** 2.5
```

**C++:**
```cpp
#include "math_utils.hpp"

auto Y = math::sqrt(X);
auto Y = math::exp(X);
auto Y = math::log(X);
auto Y = math::abs(X);
auto Y = math::square(X);

Tensor<float> Y = X.clone();
math::power(Y, 2.5f);
```

---

### Linear Algebra

**Python:**
```python
import numpy as np

# Dot product
dot_prod = np.dot(a, b)

# Matrix multiplication
C = np.matmul(A, B)

# Norm
l2_norm = np.linalg.norm(x, 2)
l1_norm = np.linalg.norm(x, 1)

# Normalize vector
x_normalized = x / np.linalg.norm(x)
```

**C++:**
```cpp
#include "math_utils.hpp"

// Dot product
float dot_prod = math::dot(a, b);

// Matrix multiplication
auto C = A.matmul(B);

// Norm
float l2_norm = math::norm(x, 2);
float l1_norm = math::norm(x, 1);

// Normalize vector
auto x_normalized = math::normalize(x, 2);
```

---

## <a name="postprocessing-operations"></a>📊 Postprocessing Operations

### 1. Softmax

**Python:**
```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

probs = softmax(logits)
```

**C++:**
```cpp
#include "activations.hpp"

Tensor<float> probs = logits.clone();
activations::softmax(probs);
```

**Rust:**
```rust
use activations::softmax;

let mut probs = logits.clone();
softmax(&mut probs);
```

---

### 2. Argmax / Argmin

**Python:**
```python
import numpy as np

predicted_class = np.argmax(probabilities)
```

**C++:**
```cpp
#include "postprocessing.hpp"

size_t predicted_class = postprocessing::argmax(probabilities);
```

---

### 3. Top-K Predictions

**Python:**
```python
import numpy as np

# Get indices of top 5 values
top_k_indices = np.argsort(scores)[-5:][::-1]
top_k_values = scores[top_k_indices]
```

**C++:**
```cpp
#include "postprocessing.hpp"

auto topk = postprocessing::topk(scores, 5);
// topk.indices contains top 5 indices
// topk.values contains top 5 values

for (size_t i = 0; i < topk.indices.size(); ++i) {
    std::cout << "Class " << topk.indices[i]
              << ": " << topk.values[i] << std::endl;
}
```

---

### 4. Confidence Thresholding

**Python:**
```python
# Filter predictions by confidence
confident_predictions = [
    pred for pred in predictions
    if pred['confidence'] >= threshold
]
```

**C++:**
```cpp
#include "postprocessing.hpp"

auto filtered = postprocessing::threshold_filter(scores, threshold);
```

---

### 5. Non-Maximum Suppression (Object Detection)

**Python:**
```python
def nms(boxes, scores, iou_threshold=0.5):
    # Sort by score
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        # Remove overlapping boxes
        # ... (NMS implementation)

    return keep
```

**C++:**
```cpp
#include "postprocessing.hpp"

// Create bounding boxes
std::vector<postprocessing::BoundingBox<float>> boxes;
for (size_t i = 0; i < detections.size(); ++i) {
    boxes.push_back({
        x1, y1, x2, y2,  // coordinates
        score,           // confidence
        class_id         // class label
    });
}

// Apply NMS
auto filtered_boxes = postprocessing::nms(boxes, 0.5f);
```

---

### 6. Temperature Scaling (Calibration)

**Python:**
```python
def temperature_scaling(logits, temperature=1.5):
    return softmax(logits / temperature)
```

**C++:**
```cpp
#include "postprocessing.hpp"

auto calibrated_probs = postprocessing::temperature_scaling(logits, 1.5f);
```

---

### 7. Moving Average (Temporal Smoothing)

**Python:**
```python
class MovingAverage:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = []

    def update(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer)
```

**C++:**
```cpp
#include "postprocessing.hpp"

postprocessing::MovingAverageFilter<float> filter(5);

for (float new_value : stream) {
    float smoothed = filter.update(new_value);
    // Use smoothed value
}
```

---

## <a name="complete-pipeline-example"></a>🔄 Complete Pipeline Example

### Python Research Code

```python
class MLPipeline:
    def __init__(self, model, config):
        self.model = model
        self.mean = config['mean']
        self.std = config['std']
        self.threshold = config['threshold']

    def predict(self, raw_input):
        # Preprocessing
        x = self.preprocess(raw_input)

        # Model inference
        logits = self.model(x)

        # Postprocessing
        result = self.postprocess(logits)

        return result

    def preprocess(self, x):
        # Handle missing values
        x = np.nan_to_num(x, nan=self.mean)

        # Clip outliers
        x = np.clip(x, -5, 5)

        # Normalize
        x = (x - self.mean) / self.std

        # Log transform
        x = np.log1p(np.abs(x)) * np.sign(x)

        return x

    def postprocess(self, logits):
        # Softmax
        probs = softmax(logits)

        # Top-k
        top_k_idx = np.argsort(probs)[-5:][::-1]

        # Filter by confidence
        results = []
        for idx in top_k_idx:
            if probs[idx] >= self.threshold:
                results.append({
                    'class': idx,
                    'probability': probs[idx]
                })

        return results
```

### Embedded C++ Code

```cpp
#include "preprocessing.hpp"
#include "postprocessing.hpp"
#include "activations.hpp"
#include "MyModel.hpp"  // Auto-generated

class MLPipeline {
public:
    MLPipeline(float mean, float std, float threshold)
        : mean_(mean), std_(std), threshold_(threshold) {}

    std::vector<Result> predict(const Tensor<float>& raw_input) {
        // Preprocessing
        auto x = preprocess(raw_input);

        // Model inference
        auto logits_vec = model_.forward(x.data());
        Tensor<float> logits({logits_vec.size()}, logits_vec);

        // Postprocessing
        auto results = postprocess(logits);

        return results;
    }

private:
    Tensor<float> preprocess(const Tensor<float>& x) {
        Tensor<float> result = x.clone();

        // Handle missing values
        for (auto& val : result.data()) {
            if (!std::isfinite(val)) {
                val = mean_;
            }
        }

        // Clip outliers
        preprocessing::clip(result, -5.0f, 5.0f);

        // Normalize
        for (auto& val : result.data()) {
            val = (val - mean_) / std_;
        }

        // Log transform
        for (auto& val : result.data()) {
            float sign = val >= 0 ? 1.0f : -1.0f;
            val = std::log1p(std::abs(val)) * sign;
        }

        return result;
    }

    std::vector<Result> postprocess(const Tensor<float>& logits) {
        // Softmax
        Tensor<float> probs = logits.clone();
        activations::softmax(probs);

        // Top-k
        auto topk = postprocessing::topk(probs, 5);

        // Filter by confidence
        std::vector<Result> results;
        for (size_t i = 0; i < topk.indices.size(); ++i) {
            if (topk.values[i] >= threshold_) {
                results.push_back({
                    topk.indices[i],  // class
                    topk.values[i]    // probability
                });
            }
        }

        return results;
    }

    float mean_;
    float std_;
    float threshold_;
    MyModel model_;
};
```

---

## <a name="advanced-patterns"></a>🚀 Advanced Patterns

### 1. Custom Transformations

**Python:**
```python
def custom_transform(x):
    # Box-Cox-like transformation
    return np.sign(x) * (np.abs(x) ** 0.5)
```

**C++:**
```cpp
Tensor<float> custom_transform(const Tensor<float>& x) {
    Tensor<float> result = x.clone();
    for (auto& val : result.data()) {
        float sign = val >= 0 ? 1.0f : -1.0f;
        val = sign * std::pow(std::abs(val), 0.5f);
    }
    return result;
}
```

---

### 2. Feature Engineering

**Python:**
```python
def engineer_features(x):
    features = []

    # Original features
    features.append(x)

    # Squared features
    features.append(x ** 2)

    # Log features
    features.append(np.log1p(np.abs(x)))

    # Rolling statistics
    features.append(rolling_mean(x, window=5))

    return np.concatenate(features)
```

**C++:**
```cpp
Tensor<float> engineer_features(const Tensor<float>& x) {
    std::vector<float> all_features;

    // Original features
    all_features.insert(all_features.end(), x.data().begin(), x.data().end());

    // Squared features
    auto squared = math::square(x);
    all_features.insert(all_features.end(), squared.data().begin(), squared.data().end());

    // Log features
    Tensor<float> log_feat = x.clone();
    preprocessing::log1p(log_feat);
    all_features.insert(all_features.end(), log_feat.data().begin(), log_feat.data().end());

    return Tensor<float>({all_features.size()}, all_features);
}
```

---

### 3. Ensemble Predictions

**Python:**
```python
def ensemble_predict(models, x):
    predictions = [model(x) for model in models]

    # Soft voting (average probabilities)
    avg_probs = np.mean(predictions, axis=0)

    return avg_probs
```

**C++:**
```cpp
#include "postprocessing.hpp"

Tensor<float> ensemble_predict(
    const std::vector<Model>& models,
    const Tensor<float>& x)
{
    std::vector<Tensor<float>> predictions;

    for (const auto& model : models) {
        auto output = model.forward(x.data());
        predictions.push_back(Tensor<float>({output.size()}, output));
    }

    // Soft voting
    auto avg_probs = postprocessing::EnsembleVoting<float>::soft_voting(predictions);

    return avg_probs;
}
```

---

## <a name="performance-optimization"></a>⚡ Performance Optimization

### 1. In-Place Operations

**Python:**
```python
# Creates new array
x_normalized = (x - mean) / std
```

**C++ (Non-optimized):**
```cpp
// Creates temporary objects
auto x_normalized = (x - mean) * (1.0f / std);
```

**C++ (Optimized):**
```cpp
// In-place modification
Tensor<float> x_normalized = x.clone();
x_normalized.add_inplace(-mean);
x_normalized.multiply_inplace(1.0f / std);
```

---

### 2. Avoid Unnecessary Copies

**C++ (Non-optimized):**
```cpp
Tensor<float> preprocess(Tensor<float> x) {  // Copy!
    x = normalize(x);  // Another copy!
    return x;          // Move, but still...
}
```

**C++ (Optimized):**
```cpp
void preprocess_inplace(Tensor<float>& x) {  // Reference, no copy
    normalize_inplace(x);
}

// Or use move semantics
Tensor<float> preprocess(Tensor<float>&& x) {  // Move
    normalize_inplace(x);
    return std::move(x);
}
```

---

### 3. Static Allocation

For ultra-constrained embedded systems:

**C++ (Dynamic):**
```cpp
Tensor<float> x({100, 100});  // Allocates on heap
```

**C++ (Static):**
```cpp
// For fixed sizes known at compile time
std::array<float, 10000> buffer;
Tensor<float> x({100, 100}, buffer.data(), buffer.size());
```

---

## 🎯 Summary

This guide provides a **complete mapping** from Python research code to production-ready C++/Rust code for embedded systems.

**Key Takeaways:**

1. **Not just models** - Convert entire pipelines including preprocessing, postprocessing, and data transformations

2. **No external dependencies** - All utilities implemented using only C++17 std or Rust std

3. **Direct mapping** - Clear correspondence between Python and C++/Rust code

4. **Production-ready** - Handles edge cases, optimized for embedded, memory-efficient

5. **Extensive library** - Comprehensive utilities for preprocessing, math, postprocessing

**Next Steps:**

- See [End-to-End Production Example](../examples/end_to_end_production/)
- Use [Pipeline Analyzer](../src/pipeline/pipeline_analyzer.py) to automatically convert
- Check [API Reference](./api_reference.md) for complete documentation

---

**Convert your entire Python ML workflow to embedded C++/Rust - not just the model!** 🚀
