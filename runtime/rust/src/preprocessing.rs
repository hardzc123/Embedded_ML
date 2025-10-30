/// Data preprocessing utilities for Rust
/// Replicates common Python/NumPy preprocessing operations

use crate::tensor::Tensor;

/// MinMax Scaler - Scale values to [0, 1]
pub struct MinMaxScaler {
    feature_min: f32,
    feature_max: f32,
    data_min: f32,
    data_max: f32,
    fitted: bool,
}

impl MinMaxScaler {
    pub fn new(feature_min: f32, feature_max: f32) -> Self {
        Self {
            feature_min,
            feature_max,
            data_min: 0.0,
            data_max: 0.0,
            fitted: false,
        }
    }

    pub fn fit(&mut self, data: &Tensor<f32>) {
        self.data_min = data.min();
        self.data_max = data.max();
        self.fitted = true;
    }

    pub fn transform(&self, data: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        if !self.fitted {
            return Err("Scaler not fitted");
        }

        let mut result = data.clone();
        let data_range = self.data_max - self.data_min;
        let feature_range = self.feature_max - self.feature_min;

        if data_range == 0.0 {
            result.data_mut().fill(self.feature_min);
            return Ok(result);
        }

        for val in result.data_mut() {
            *val = self.feature_min + (*val - self.data_min) * feature_range / data_range;
        }

        Ok(result)
    }

    pub fn fit_transform(&mut self, data: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        self.fit(data);
        self.transform(data)
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

/// Standard Scaler - Standardize features by removing mean and scaling to unit variance
pub struct StandardScaler {
    mean: f32,
    std: f32,
    fitted: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            fitted: false,
        }
    }

    pub fn fit(&mut self, data: &Tensor<f32>) {
        self.mean = data.mean();

        // Calculate standard deviation
        let mut variance = 0.0;
        for val in data.data() {
            let diff = val - self.mean;
            variance += diff * diff;
        }
        variance /= data.size() as f32;
        self.std = variance.sqrt();

        self.fitted = true;
    }

    pub fn transform(&self, data: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        if !self.fitted {
            return Err("Scaler not fitted");
        }

        let mut result = data.clone();

        if self.std == 0.0 {
            result.data_mut().fill(0.0);
            return Ok(result);
        }

        for val in result.data_mut() {
            *val = (*val - self.mean) / self.std;
        }

        Ok(result)
    }

    pub fn fit_transform(&mut self, data: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        self.fit(data);
        self.transform(data)
    }

    /// Transform with known mean and std (for deployment)
    pub fn transform_with_params(data: &Tensor<f32>, mean: f32, std: f32) -> Tensor<f32> {
        let mut result = data.clone();
        for val in result.data_mut() {
            *val = (*val - mean) / std;
        }
        result
    }

    pub fn get_mean(&self) -> f32 {
        self.mean
    }

    pub fn get_std(&self) -> f32 {
        self.std
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Image Normalizer - Normalize image channels
pub struct ImageNormalizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl ImageNormalizer {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Result<Self, &'static str> {
        if mean.len() != std.len() {
            return Err("Mean and std must have same length");
        }
        Ok(Self { mean, std })
    }

    pub fn transform(&self, image: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        let mut result = image.clone();
        let shape = image.shape();

        if shape.len() == 3 {
            // Single image (C, H, W)
            let channels = shape[0];
            let spatial_size = shape[1] * shape[2];

            for c in 0..channels {
                let mean = if c < self.mean.len() { self.mean[c] } else { 0.0 };
                let std = if c < self.std.len() { self.std[c] } else { 1.0 };

                for s in 0..spatial_size {
                    let idx = c * spatial_size + s;
                    result.data_mut()[idx] = (result.data()[idx] - mean) / std;
                }
            }
        } else if shape.len() == 4 {
            // Batch of images (N, C, H, W)
            let batch = shape[0];
            let channels = shape[1];
            let spatial_size = shape[2] * shape[3];

            for b in 0..batch {
                for c in 0..channels {
                    let mean = if c < self.mean.len() { self.mean[c] } else { 0.0 };
                    let std = if c < self.std.len() { self.std[c] } else { 1.0 };

                    for s in 0..spatial_size {
                        let idx = (b * channels + c) * spatial_size + s;
                        result.data_mut()[idx] = (result.data()[idx] - mean) / std;
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Clip values to range
pub fn clip(tensor: &mut Tensor<f32>, min_val: f32, max_val: f32) {
    for val in tensor.data_mut() {
        *val = val.clamp(min_val, max_val);
    }
}

/// Log transformation: log(1 + x)
pub fn log1p(tensor: &mut Tensor<f32>) {
    for val in tensor.data_mut() {
        *val = (1.0 + *val).ln();
    }
}

/// Exponential transformation: exp(x) - 1
pub fn expm1(tensor: &mut Tensor<f32>) {
    for val in tensor.data_mut() {
        *val = val.exp() - 1.0;
    }
}

/// Power transformation
pub fn power(tensor: &mut Tensor<f32>, p: f32) {
    for val in tensor.data_mut() {
        *val = val.powf(p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let data = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut scaler = StandardScaler::new();

        let normalized = scaler.fit_transform(&data).unwrap();

        // Mean should be approximately 0
        assert!((normalized.mean()).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_scaler() {
        let data = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut scaler = MinMaxScaler::default();

        let scaled = scaler.fit_transform(&data).unwrap();

        // Min should be 0, max should be 1
        assert!((scaled.min() - 0.0).abs() < 1e-6);
        assert!((scaled.max() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip() {
        let mut data = Tensor::from_vec(vec![5], vec![-5.0, -1.0, 0.0, 1.0, 5.0]).unwrap();
        clip(&mut data, -2.0, 2.0);

        assert_eq!(data.data(), &[-2.0, -1.0, 0.0, 1.0, 2.0]);
    }
}
