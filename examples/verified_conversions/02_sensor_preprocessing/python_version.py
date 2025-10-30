"""
Example 2: Sensor Data Preprocessing Pipeline
Pure data processing conversion (no ML model)

This example shows converting data preprocessing operations that researchers
commonly write in Python using NumPy/Pandas to C++/Rust for embedded devices.

Common use case: Processing sensor readings before sending to model
"""

import numpy as np
import json


def sensor_preprocessing_pipeline(raw_readings):
    """
    Complete sensor data preprocessing pipeline

    Steps:
    1. Handle missing values (NaN/Inf)
    2. Remove outliers (clipping)
    3. Apply moving average filter (smoothing)
    4. Normalize (z-score)
    5. Detect anomalies

    Args:
        raw_readings: Array of raw sensor values

    Returns:
        Dictionary with processed data and detected anomalies
    """

    # Step 1: Handle missing values
    data = np.copy(raw_readings)
    mask = ~np.isfinite(data)
    if np.any(mask):
        # Replace with median of valid values
        median_val = np.median(data[~mask])
        data[mask] = median_val

    # Step 2: Remove outliers (clip to ±3 standard deviations)
    mean = np.mean(data)
    std = np.std(data)
    data = np.clip(data, mean - 3*std, mean + 3*std)

    # Step 3: Moving average filter (window=5)
    window_size = 5
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')

    # Step 4: Normalize (z-score normalization)
    normalized_mean = np.mean(smoothed)
    normalized_std = np.std(smoothed)
    normalized = (smoothed - normalized_mean) / (normalized_std + 1e-8)

    # Step 5: Detect anomalies (values beyond ±2.5 std)
    anomaly_threshold = 2.5
    anomalies = np.abs(normalized) > anomaly_threshold

    return {
        'original': raw_readings.tolist(),
        'processed': normalized.tolist(),
        'anomaly_indices': np.where(anomalies)[0].tolist(),
        'num_anomalies': int(np.sum(anomalies)),
        'statistics': {
            'mean': float(normalized_mean),
            'std': float(normalized_std),
            'min': float(np.min(normalized)),
            'max': float(np.max(normalized)),
        }
    }


def main():
    print("="*70)
    print("PYTHON IMPLEMENTATION - Sensor Data Preprocessing")
    print("="*70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create synthetic sensor data with some issues
    num_samples = 20
    raw_readings = np.random.randn(num_samples) * 10 + 50  # Mean ~50, std ~10

    # Add some outliers
    raw_readings[5] = 150  # Outlier
    raw_readings[15] = -20  # Outlier

    # Add some missing values
    raw_readings[10] = np.nan
    raw_readings[12] = np.inf

    print(f"\nRaw sensor readings ({num_samples} samples):")
    print(f"Min: {np.nanmin(raw_readings):.2f}, Max: {np.nanmax(raw_readings):.2f}")
    print(f"Mean: {np.nanmean(raw_readings):.2f}, Std: {np.nanstd(raw_readings):.2f}")
    print(f"Missing values: {np.sum(~np.isfinite(raw_readings))}")

    # Run preprocessing pipeline
    result = sensor_preprocessing_pipeline(raw_readings)

    print("\n" + "="*70)
    print("PYTHON PROCESSING RESULTS:")
    print("="*70)
    print(f"Processed {len(result['processed'])} samples")
    print(f"Detected {result['num_anomalies']} anomalies at indices: {result['anomaly_indices']}")
    print(f"\nProcessed data statistics:")
    print(f"  Mean: {result['statistics']['mean']:.6f}")
    print(f"  Std:  {result['statistics']['std']:.6f}")
    print(f"  Min:  {result['statistics']['min']:.6f}")
    print(f"  Max:  {result['statistics']['max']:.6f}")

    print(f"\nFirst 5 processed values: {[f'{v:.4f}' for v in result['processed'][:5]]}")
    print(f"Last 5 processed values:  {[f'{v:.4f}' for v in result['processed'][-5:]]}")

    # Save results for verification
    with open('python_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Save raw input for C++/Rust
    np.save('sensor_data.npy', raw_readings)

    with open('sensor_data.txt', 'w') as f:
        for val in raw_readings:
            f.write(f"{val}\n")

    print("\n✓ Saved results to python_results.json")
    print("✓ Saved input data to sensor_data.npy and sensor_data.txt")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Compile C++:  g++ -std=c++17 -O3 cpp_version.cpp -o cpp_sensor")
    print("2. Run C++:      ./cpp_sensor")
    print("3. Compile Rust: rustc rust_version.rs")
    print("4. Run Rust:     ./rust_version")
    print("5. Verify:       python verify_preprocessing.py")
    print("="*70)


if __name__ == '__main__':
    main()
