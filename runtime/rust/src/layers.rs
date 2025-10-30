/// Neural network layers for inference
/// All layers are designed for embedded systems with minimal memory footprint

use crate::tensor::Tensor;

/// Linear (Fully Connected) Layer
pub struct Linear {
    weight: Tensor<f32>,  // (out_features, in_features)
    bias: Option<Tensor<f32>>,  // (out_features,)
}

impl Linear {
    pub fn new(weight: Tensor<f32>, bias: Option<Tensor<f32>>) -> Result<Self, &'static str> {
        if weight.shape().len() != 2 {
            return Err("Weight must be 2D");
        }
        if let Some(ref b) = bias {
            if b.shape().len() != 1 {
                return Err("Bias must be 1D");
            }
            if b.size() != weight.shape()[0] {
                return Err("Bias size must match output features");
            }
        }
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        // input: (batch, in_features) or (in_features,)
        // weight: (out_features, in_features)
        // output: (batch, out_features) or (out_features,)

        let is_batched = input.shape().len() == 2;

        let x = if !is_batched {
            // Add batch dimension
            input.reshape(vec![1, input.size()])?
        } else {
            input.clone()
        };

        // Matrix multiply with transposed weight: x @ weight.T
        let mut result = self.matmul_transpose(&x)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            result.add_inplace(bias);
        }

        if !is_batched {
            // Remove batch dimension
            result = result.reshape(vec![result.size()])?;
        }

        Ok(result)
    }

    fn matmul_transpose(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        // a: (m, k), weight: (n, k) -> output: (m, n)
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = self.weight.shape()[0];

        if self.weight.shape()[1] != k {
            return Err("Matrix dimensions don't match");
        }

        let mut result = Tensor::new(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a.data()[i * k + p] * self.weight.data()[j * k + p];
                }
                result.data_mut()[i * n + j] = sum;
            }
        }

        Ok(result)
    }
}

/// Conv2D Layer
pub struct Conv2d {
    weight: Tensor<f32>,  // (out_channels, in_channels, kernel_h, kernel_w)
    bias: Option<Tensor<f32>>,  // (out_channels,)
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<Self, &'static str> {
        if weight.shape().len() != 4 {
            return Err("Conv2d weight must be 4D");
        }

        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        })
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        // input: (batch, in_channels, height, width)
        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err("Conv2d input must be 4D");
        }

        let batch = in_shape[0];
        let in_h = in_shape[2];
        let in_w = in_shape[3];

        // Calculate output dimensions
        let out_h = (in_h + 2 * self.padding_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w + 2 * self.padding_w - self.kernel_w) / self.stride_w + 1;

        let mut output = Tensor::new(vec![batch, self.out_channels, out_h, out_w]);

        // Perform convolution
        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        // Convolve with kernel
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_h {
                                for kw in 0..self.kernel_w {
                                    let ih = oh * self.stride_h + kh;
                                    let iw = ow * self.stride_w + kw;

                                    // Handle padding
                                    if ih >= self.padding_h
                                        && ih < in_h + self.padding_h
                                        && iw >= self.padding_w
                                        && iw < in_w + self.padding_w
                                    {
                                        let ih = ih - self.padding_h;
                                        let iw = iw - self.padding_w;

                                        let in_idx = ((b * self.in_channels + ic) * in_h + ih) * in_w + iw;
                                        let w_idx = (((oc * self.in_channels + ic) * self.kernel_h + kh)
                                            * self.kernel_w
                                            + kw);

                                        sum += input.data()[in_idx] * self.weight.data()[w_idx];
                                    }
                                }
                            }
                        }

                        // Add bias
                        if let Some(ref bias) = self.bias {
                            sum += bias.data()[oc];
                        }

                        let out_idx = ((b * self.out_channels + oc) * out_h + oh) * out_w + ow;
                        output.data_mut()[out_idx] = sum;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// MaxPool2d Layer
pub struct MaxPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        let stride = if stride > 0 { stride } else { kernel_size };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        // input: (batch, channels, height, width)
        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err("MaxPool2d input must be 4D");
        }

        let batch = in_shape[0];
        let channels = in_shape[1];
        let in_h = in_shape[2];
        let in_w = in_shape[3];

        let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = Tensor::new(vec![batch, channels, out_h, out_w]);

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;

                                if ih >= self.padding
                                    && ih < in_h + self.padding
                                    && iw >= self.padding
                                    && iw < in_w + self.padding
                                {
                                    let ih = ih - self.padding;
                                    let iw = iw - self.padding;
                                    let idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                                    max_val = max_val.max(input.data()[idx]);
                                }
                            }
                        }

                        let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        output.data_mut()[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// BatchNorm2d Layer (inference mode)
pub struct BatchNorm2d {
    running_mean: Tensor<f32>,
    running_var: Tensor<f32>,
    gamma: Option<Tensor<f32>>,
    beta: Option<Tensor<f32>>,
    eps: f32,
}

impl BatchNorm2d {
    pub fn new(
        running_mean: Tensor<f32>,
        running_var: Tensor<f32>,
        gamma: Option<Tensor<f32>>,
        beta: Option<Tensor<f32>>,
        eps: f32,
    ) -> Self {
        Self {
            running_mean,
            running_var,
            gamma,
            beta,
            eps,
        }
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        let mut output = input.clone();
        let shape = input.shape();

        let batch = shape[0];
        let channels = shape[1];
        let spatial_size = input.size() / (batch * channels);

        for b in 0..batch {
            for c in 0..channels {
                let mean = self.running_mean.data()[c];
                let var = self.running_var.data()[c];
                let std = (var + self.eps).sqrt();

                let scale = self.gamma.as_ref().map_or(1.0, |g| g.data()[c]);
                let shift = self.beta.as_ref().map_or(0.0, |b| b.data()[c]);

                for s in 0..spatial_size {
                    let idx = (b * channels + c) * spatial_size + s;
                    output.data_mut()[idx] = scale * (output.data()[idx] - mean) / std + shift;
                }
            }
        }

        Ok(output)
    }
}

/// Flatten Layer
pub struct Flatten {
    start_dim: usize,
}

impl Flatten {
    pub fn new(start_dim: usize) -> Self {
        Self { start_dim }
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, &'static str> {
        let shape = input.shape();

        if self.start_dim == 1 && shape.len() > 1 {
            let batch = shape[0];
            let features = input.size() / batch;
            return input.reshape(vec![batch, features]);
        }

        // Default: flatten everything
        input.reshape(vec![input.size()])
    }
}
