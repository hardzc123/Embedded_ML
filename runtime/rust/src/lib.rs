/*!
# Embedded ML Runtime

Lightweight ML inference runtime for embedded systems.
Zero external dependencies - uses only Rust standard library.

## Features

- **No Dependencies**: Uses only Rust std library
- **no_std Support**: Can run on bare metal embedded systems
- **Memory Efficient**: Optimized for resource-constrained devices
- **Type Safe**: Leverages Rust's type system for safety

## Example

```rust
use embedded_ml_runtime::{Tensor, activations, layers};

// Create a simple linear layer
let weight = Tensor::from_vec(vec![2, 3], vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
]).unwrap();

let bias = Tensor::from_vec(vec![2], vec![0.1, 0.2]).unwrap();

let linear = layers::Linear::new(weight, Some(bias)).unwrap();

// Create input
let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

// Forward pass
let mut output = linear.forward(&input).unwrap();

// Apply activation
activations::relu(&mut output);

println!("Output: {:?}", output.data());
```
*/

#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

pub mod tensor;
pub mod activations;
pub mod layers;

// Re-export main types
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let weight = Tensor::from_vec(vec![2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]).unwrap();

        let bias = Tensor::from_vec(vec![2], vec![0.1, 0.2]).unwrap();
        let linear = layers::Linear::new(weight, Some(bias)).unwrap();

        let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2]);
        // Expected: [1*1 + 2*2 + 3*3 + 0.1, 1*4 + 2*5 + 3*6 + 0.2] = [14.1, 28.2]
        assert!((output.data()[0] - 14.1).abs() < 1e-5);
        assert!((output.data()[1] - 28.2).abs() < 1e-5);
    }

    #[test]
    fn test_relu_activation() {
        let mut tensor = Tensor::from_vec(vec![4], vec![-2.0, -1.0, 0.0, 1.0]).unwrap();
        activations::relu(&mut tensor);
        assert_eq!(tensor.data(), &[0.0, 0.0, 0.0, 1.0]);
    }
}
