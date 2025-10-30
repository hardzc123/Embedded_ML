/// Activation functions for neural networks
/// All functions operate in-place for memory efficiency

use crate::tensor::Tensor;

/// ReLU (Rectified Linear Unit): f(x) = max(0, x)
pub fn relu(tensor: &mut Tensor<f32>) {
    for val in tensor.data_mut() {
        *val = val.max(0.0);
    }
}

/// Leaky ReLU: f(x) = x if x > 0 else alpha * x
pub fn leaky_relu(tensor: &mut Tensor<f32>, alpha: f32) {
    for val in tensor.data_mut() {
        if *val < 0.0 {
            *val *= alpha;
        }
    }
}

/// Sigmoid: f(x) = 1 / (1 + exp(-x))
pub fn sigmoid(tensor: &mut Tensor<f32>) {
    for val in tensor.data_mut() {
        *val = 1.0 / (1.0 + (-*val).exp());
    }
}

/// Tanh: f(x) = tanh(x)
pub fn tanh(tensor: &mut Tensor<f32>) {
    for val in tensor.data_mut() {
        *val = val.tanh();
    }
}

/// Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
/// Numerically stable implementation
pub fn softmax(tensor: &mut Tensor<f32>) {
    let shape = tensor.shape().to_vec();

    // Handle 1D case
    if shape.len() == 1 {
        let max_val = tensor.max();
        let mut sum = 0.0;

        for val in tensor.data_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        for val in tensor.data_mut() {
            *val /= sum;
        }
        return;
    }

    // Handle 2D case (batch, features)
    if shape.len() == 2 {
        let batch_size = shape[0];
        let features = shape[1];

        for b in 0..batch_size {
            // Find max in this row
            let mut max_val = f32::NEG_INFINITY;
            for f in 0..features {
                let idx = b * features + f;
                max_val = max_val.max(tensor.data()[idx]);
            }

            // Compute exp and sum
            let mut sum = 0.0;
            for f in 0..features {
                let idx = b * features + f;
                let val = (tensor.data()[idx] - max_val).exp();
                tensor.data_mut()[idx] = val;
                sum += val;
            }

            // Normalize
            for f in 0..features {
                let idx = b * features + f;
                tensor.data_mut()[idx] /= sum;
            }
        }
    }
}

/// GELU (Gaussian Error Linear Unit)
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(tensor: &mut Tensor<f32>) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/pi)

    for val in tensor.data_mut() {
        let x3 = val.powi(3);
        let inner = SQRT_2_OVER_PI * (*val + 0.044715 * x3);
        *val = 0.5 * *val * (1.0 + inner.tanh());
    }
}

/// SiLU / Swish: f(x) = x * sigmoid(x)
pub fn swish(tensor: &mut Tensor<f32>) {
    for val in tensor.data_mut() {
        *val = *val / (1.0 + (-*val).exp());
    }
}

/// ELU (Exponential Linear Unit)
/// f(x) = x if x > 0 else alpha * (exp(x) - 1)
pub fn elu(tensor: &mut Tensor<f32>, alpha: f32) {
    for val in tensor.data_mut() {
        if *val <= 0.0 {
            *val = alpha * (val.exp() - 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let mut t = Tensor::from_vec(vec![4], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
        relu(&mut t);
        assert_eq!(t.data(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let mut t = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
        sigmoid(&mut t);
        assert!((t.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let mut t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        softmax(&mut t);
        let sum: f32 = t.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
