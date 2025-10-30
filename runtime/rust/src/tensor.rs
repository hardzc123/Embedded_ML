/// Tensor - Lightweight tensor implementation for embedded ML inference
/// Uses only Rust standard library, supports no_std for bare metal

#[cfg(not(feature = "no_std"))]
use std::vec::Vec;

#[cfg(feature = "no_std")]
use alloc::vec::Vec;

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Clone + Default> Tensor<T> {
    /// Create a new tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![T::default(); size],
            shape,
        }
    }

    /// Create a tensor from data and shape
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Result<Self, &'static str> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err("Data size doesn't match shape");
        }
        Ok(Self { data, shape })
    }

    /// Get reference to data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get mutable reference to data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape tensor (returns new tensor)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, &'static str> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err("Reshape size mismatch");
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    /// Get element at index (for 1D)
    pub fn get(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    /// Get mutable element at index (for 1D)
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.data.get_mut(i)
    }

    /// Get element at 2D index
    pub fn get_2d(&self, i: usize, j: usize) -> Option<&T> {
        if self.shape.len() != 2 {
            return None;
        }
        let idx = i * self.shape[1] + j;
        self.data.get(idx)
    }

    /// Set element at 2D index
    pub fn set_2d(&mut self, i: usize, j: usize, value: T) -> Result<(), &'static str> {
        if self.shape.len() != 2 {
            return Err("Not a 2D tensor");
        }
        let idx = i * self.shape[1] + j;
        if idx < self.data.len() {
            self.data[idx] = value;
            Ok(())
        } else {
            Err("Index out of bounds")
        }
    }
}

// Floating point specific operations
impl Tensor<f32> {
    /// Matrix multiplication for 2D tensors
    pub fn matmul(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matmul only supports 2D tensors");
        }
        if self.shape[1] != other.shape[0] {
            return Err("Matrix dimensions don't match");
        }

        let m = self.shape[0];
        let n = other.shape[1];
        let k = self.shape[1];

        let mut result = Tensor::new(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result.data[i * n + j] = sum;
            }
        }

        Ok(result)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        if self.size() != other.size() && other.size() != 1 {
            return Err("Tensor sizes don't match for addition");
        }

        let mut result = self.clone();
        if other.size() == 1 {
            let scalar = other.data[0];
            for val in result.data_mut() {
                *val += scalar;
            }
        } else {
            for (i, val) in result.data_mut().iter_mut().enumerate() {
                *val += other.data[i];
            }
        }
        Ok(result)
    }

    /// Add scalar in-place
    pub fn add_scalar_inplace(&mut self, scalar: f32) {
        for val in self.data_mut() {
            *val += scalar;
        }
    }

    /// Add tensor in-place
    pub fn add_inplace(&mut self, other: &Tensor<f32>) {
        for (i, val) in self.data_mut().iter_mut().enumerate() {
            *val += other.data[i % other.size()];
        }
    }

    /// Element-wise multiplication
    pub fn multiply(&self, scalar: f32) -> Tensor<f32> {
        let mut result = self.clone();
        for val in result.data_mut() {
            *val *= scalar;
        }
        result
    }

    /// Multiply in-place
    pub fn multiply_inplace(&mut self, scalar: f32) {
        for val in self.data_mut() {
            *val *= scalar;
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.sum() / self.size() as f32
    }

    /// Max element
    pub fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Min element
    pub fn min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Fill with value
    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    /// Zero all elements
    pub fn zero(&mut self) {
        self.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::<f32>::new(vec![2, 3]);
        assert_eq!(t.size(), 6);
        assert_eq!(t.shape(), &[2, 3]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // First element should be 1*1 + 2*3 + 3*5 = 22
        assert_eq!(c.get(0).unwrap(), &22.0);
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }
}
