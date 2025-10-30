/**
 * Example 1: Simple MLP with Preprocessing - Rust Implementation
 *
 * This is the Rust conversion of python_implementation.py
 * It should produce IDENTICAL results to the Python and C++ versions
 *
 * Build and run:
 *   rustc rust_implementation.rs && ./rust_implementation
 */

use std::fs::File;
use std::io::Write;

// ============================================================================
// MINIMAL TENSOR IMPLEMENTATION (inline for single-file example)
// ============================================================================

#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }
}

// ============================================================================
// MODEL IMPLEMENTATION
// ============================================================================

struct SimpleMLP {
    fc1_weight: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<f32>,
    fc2_bias: Vec<f32>,
    fc3_weight: Vec<f32>,
    fc3_bias: Vec<f32>,
}

impl SimpleMLP {
    fn new() -> Self {
        Self {
            fc1_weight: vec![0.0; 20 * 10],
            fc1_bias: vec![0.0; 20],
            fc2_weight: vec![0.0; 20 * 20],
            fc2_bias: vec![0.0; 20],
            fc3_weight: vec![0.0; 3 * 20],
            fc3_bias: vec![0.0; 3],
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Layer 1: Linear + ReLU
        let mut x = self.linear(input, &self.fc1_weight, &self.fc1_bias, 10, 20);
        Self::relu(&mut x);

        // Layer 2: Linear + ReLU
        x = self.linear(&x, &self.fc2_weight, &self.fc2_bias, 20, 20);
        Self::relu(&mut x);

        // Layer 3: Linear (no activation)
        x = self.linear(&x, &self.fc3_weight, &self.fc3_bias, 20, 3);

        x
    }

    fn linear(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; out_features];

        for i in 0..out_features {
            let mut sum = 0.0;
            for j in 0..in_features {
                sum += input[j] * weight[i * in_features + j];
            }
            output[i] = sum + bias[i];
        }

        output
    }

    fn relu(x: &mut [f32]) {
        for val in x.iter_mut() {
            *val = val.max(0.0);
        }
    }
}

// ============================================================================
// PREPROCESSING (Direct conversion from Python)
// ============================================================================

fn preprocess_rust(data: &[f32], mean: f32, std: f32) -> Vec<f32> {
    let mut result = data.to_vec();

    // Clip outliers (Python: np.clip(data, -5.0, 5.0))
    for val in result.iter_mut() {
        *val = val.clamp(-5.0, 5.0);
    }

    // Normalize (Python: (data - mean) / std)
    for val in result.iter_mut() {
        *val = (*val - mean) / std;
    }

    result
}

// ============================================================================
// POSTPROCESSING (Direct conversion from Python)
// ============================================================================

struct PostprocessResult {
    probabilities: Vec<f32>,
    predicted_class: usize,
    confidence: f32,
}

fn postprocess_rust(output: &[f32]) -> PostprocessResult {
    // Softmax (Python: exp_output = np.exp(output - np.max(output)))
    let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut exp_output: Vec<f32> = output.iter().map(|&x| (x - max_val).exp()).collect();

    let sum: f32 = exp_output.iter().sum();

    // Normalize (Python: probs = exp_output / np.sum(exp_output))
    let probabilities: Vec<f32> = exp_output.iter().map(|&x| x / sum).collect();

    // Argmax (Python: predicted_class = np.argmax(probs))
    let (predicted_class, &max_prob) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    PostprocessResult {
        probabilities,
        predicted_class,
        confidence: max_prob,
    }
}

// ============================================================================
// COMPLETE PIPELINE (Direct conversion from Python)
// ============================================================================

fn rust_pipeline(
    raw_input: &[f32],
    model: &SimpleMLP,
    mean: f32,
    std: f32,
) -> PostprocessResult {
    // Preprocess
    let preprocessed = preprocess_rust(raw_input, mean, std);

    // Model inference
    let output = model.forward(&preprocessed);

    // Postprocess
    postprocess_rust(&output)
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

fn main() {
    println!("{}", "=".repeat(70));
    println!("RUST IMPLEMENTATION - Simple MLP with Pre/Post Processing");
    println!("{}", "=".repeat(70));

    // Parameters (must match Python)
    const MEAN: f32 = 0.5;
    const STD: f32 = 0.2;

    // Create test input (same as Python and C++)
    let raw_input: Vec<f32> = vec![
        1.0, -0.5, 2.3, -1.2, 0.0,
        0.8, -2.1, 1.5, -0.3, 0.9,
    ];

    print!("\nTest Input: [");
    for (i, &val) in raw_input.iter().enumerate() {
        print!("{:.4}", val);
        if i < raw_input.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    // Create model
    let model = SimpleMLP::new();

    // Run Rust pipeline
    let result = rust_pipeline(&raw_input, &model, MEAN, STD);

    println!("\n{}", "=".repeat(70));
    println!("RUST RESULTS:");
    println!("{}", "=".repeat(70));
    println!("Predicted class: {}", result.predicted_class);
    println!("Confidence: {:.6}", result.confidence);

    print!("Probabilities: [");
    for (i, &prob) in result.probabilities.iter().enumerate() {
        print!("{:.6}", prob);
        if i < result.probabilities.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    // Save results for verification
    let json_output = format!(
        "{{\n  \"predicted_class\": {},\n  \"confidence\": {:.6},\n  \"probabilities\": [{}]\n}}",
        result.predicted_class,
        result.confidence,
        result
            .probabilities
            .iter()
            .map(|p| format!("{:.6}", p))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut file = File::create("rust_results.json").unwrap();
    file.write_all(json_output.as_bytes()).unwrap();

    println!("\n✓ Saved Rust results to rust_results.json");
    println!("\nRun 'python verify_conversion.py' to verify outputs match Python!");
}
