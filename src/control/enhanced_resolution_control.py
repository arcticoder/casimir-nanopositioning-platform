"""
Enhanced Resolution Control System
==================================

This module implements adaptive bandwidth multi-sensor fusion for achieving
≤0.05 nm resolution with enhanced signal processing and noise mitigation.

Mathematical Formulation:
σ_resolution ≤ 5×10⁻¹¹ m (0.05 nm requirement)

Multi-sensor fusion estimate:
x̂(t) = ∑ᵢ₌₁ᴺ wᵢ(t) × xᵢ(t)

Optimal weights (Kalman filter):
wᵢ(t) = σᵢ⁻²(t) / ∑ⱼ₌₁ᴺ σⱼ⁻²(t)

Adaptive bandwidth control:
BW(t) = BW₀ × [1 + α × SNR⁻¹(t)]

Enhanced Allan variance for stability:
σ²(τ) = ∫₀^∞ |H(f)|² × Sₓ(f) × sin⁴(πfτ)/(πfτ)² df
"""

import numpy as np
from scipy import signal
from scipy.optimize import minimize, minimize_scalar
from typing import Dict, List, Tuple, Optional, Callable, NamedTuple
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json
from collections import deque

# Resolution requirements
RESOLUTION_REQUIREMENT = 5e-11  # 0.05 nm
RESOLUTION_MARGIN_FACTOR = 2.0  # Safety margin

class SensorType(Enum):
    """Enhanced sensor types with specific characteristics."""
    CAPACITIVE = "capacitive"
    INTERFEROMETRIC = "interferometric"
    PIEZO_RESISTIVE = "piezo_resistive"
    OPTICAL_ENCODER = "optical_encoder"
    INDUCTIVE = "inductive"

@dataclass
class SensorCharacteristics:
    """Enhanced sensor characteristics with noise models."""
    
    sensor_type: SensorType
    
    # Basic specifications
    resolution: float           # Fundamental resolution (m)
    bandwidth: float           # -3dB bandwidth (Hz)
    range_full_scale: float    # Full-scale range (m)
    linearity_error: float     # Linearity error (% of full scale)
    
    # Noise characteristics
    thermal_noise_density: float    # V/√Hz at 1 Hz
    shot_noise_current: float       # A (for photodiodes)
    flicker_noise_corner: float     # 1/f noise corner frequency (Hz)
    quantization_noise: float       # ADC quantization noise (LSB)
    
    # Environmental sensitivity
    temperature_coefficient: float  # Drift per temperature (1/K)
    pressure_sensitivity: float     # Sensitivity to pressure (1/Pa)
    humidity_sensitivity: float     # Sensitivity to humidity (1/%RH)
    
    # Dynamic characteristics
    response_time: float        # Step response time (s)
    settling_time: float        # Settling time to 0.1% (s)
    overshoot_percent: float    # Step response overshoot (%)
    
    @classmethod
    def get_enhanced_sensors(cls) -> Dict[SensorType, 'SensorCharacteristics']:
        """
        Get enhanced sensor characteristics based on high-precision specifications.
        """
        return {
            SensorType.CAPACITIVE: cls(
                sensor_type=SensorType.CAPACITIVE,
                resolution=1e-11,           # 0.01 nm
                bandwidth=10000,            # 10 kHz
                range_full_scale=100e-6,    # 100 µm
                linearity_error=0.01,       # 0.01%
                thermal_noise_density=5e-9, # 5 nV/√Hz
                shot_noise_current=0,       # No shot noise
                flicker_noise_corner=100,   # 100 Hz
                quantization_noise=1e-12,   # 1 pm ADC noise
                temperature_coefficient=1e-5, # 10 ppm/K
                pressure_sensitivity=1e-8,   # 10 nm/Pa
                humidity_sensitivity=1e-7,   # 100 nm/%RH
                response_time=0.0001,       # 0.1 ms
                settling_time=0.0005,       # 0.5 ms
                overshoot_percent=5.0       # 5%
            ),
            
            SensorType.INTERFEROMETRIC: cls(
                sensor_type=SensorType.INTERFEROMETRIC,
                resolution=5e-12,           # 0.005 nm (λ/100,000)
                bandwidth=1000,             # 1 kHz
                range_full_scale=1e-3,      # 1 mm
                linearity_error=0.001,      # 0.001%
                thermal_noise_density=1e-9, # 1 nV/√Hz
                shot_noise_current=1e-12,   # 1 pA shot noise
                flicker_noise_corner=10,    # 10 Hz
                quantization_noise=5e-13,   # 0.5 pm
                temperature_coefficient=5e-6, # 5 ppm/K
                pressure_sensitivity=1e-9,   # 1 nm/Pa
                humidity_sensitivity=5e-8,   # 50 nm/%RH
                response_time=0.001,        # 1 ms
                settling_time=0.005,        # 5 ms
                overshoot_percent=2.0       # 2%
            ),
            
            SensorType.PIEZO_RESISTIVE: cls(
                sensor_type=SensorType.PIEZO_RESISTIVE,
                resolution=2e-11,           # 0.02 nm
                bandwidth=5000,             # 5 kHz
                range_full_scale=50e-6,     # 50 µm
                linearity_error=0.05,       # 0.05%
                thermal_noise_density=10e-9, # 10 nV/√Hz
                shot_noise_current=0,       # No shot noise
                flicker_noise_corner=1000,  # 1 kHz
                quantization_noise=2e-12,   # 2 pm
                temperature_coefficient=2e-4, # 200 ppm/K
                pressure_sensitivity=5e-8,   # 50 nm/Pa
                humidity_sensitivity=2e-7,   # 200 nm/%RH
                response_time=0.0002,       # 0.2 ms
                settling_time=0.001,        # 1 ms
                overshoot_percent=10.0      # 10%
            ),
            
            SensorType.OPTICAL_ENCODER: cls(
                sensor_type=SensorType.OPTICAL_ENCODER,
                resolution=1e-11,           # 0.01 nm interpolated
                bandwidth=50000,            # 50 kHz
                range_full_scale=10e-3,     # 10 mm
                linearity_error=0.1,        # 0.1%
                thermal_noise_density=2e-9, # 2 nV/√Hz
                shot_noise_current=10e-12,  # 10 pA
                flicker_noise_corner=1,     # 1 Hz
                quantization_noise=1e-12,   # 1 pm
                temperature_coefficient=1e-5, # 10 ppm/K
                pressure_sensitivity=0,      # Insensitive to pressure
                humidity_sensitivity=1e-8,   # 10 nm/%RH
                response_time=0.00002,      # 0.02 ms
                settling_time=0.0001,       # 0.1 ms
                overshoot_percent=1.0       # 1%
            ),
            
            SensorType.INDUCTIVE: cls(
                sensor_type=SensorType.INDUCTIVE,
                resolution=5e-11,           # 0.05 nm
                bandwidth=2000,             # 2 kHz
                range_full_scale=200e-6,    # 200 µm
                linearity_error=0.02,       # 0.02%
                thermal_noise_density=15e-9, # 15 nV/√Hz
                shot_noise_current=0,       # No shot noise
                flicker_noise_corner=500,   # 500 Hz
                quantization_noise=5e-12,   # 5 pm
                temperature_coefficient=5e-5, # 50 ppm/K
                pressure_sensitivity=1e-7,   # 100 nm/Pa
                humidity_sensitivity=1e-6,   # 1 µm/%RH
                response_time=0.0005,       # 0.5 ms
                settling_time=0.0025,       # 2.5 ms
                overshoot_percent=15.0      # 15%
            )
        }

@dataclass
class FilterParameters:
    """Enhanced filter parameters for resolution optimization."""
    
    # Kalman filter parameters
    process_noise_variance: float = 1e-24    # m²
    measurement_noise_variance: float = 1e-22 # m²
    initial_error_covariance: float = 1e-20   # m²
    
    # Adaptive bandwidth parameters
    base_bandwidth: float = 1000.0           # Hz
    adaptation_factor: float = 0.1           # Adaptation rate
    min_bandwidth: float = 10.0              # Hz
    max_bandwidth: float = 10000.0           # Hz
    
    # Allan variance parameters
    tau_min: float = 0.001                   # s (minimum averaging time)
    tau_max: float = 1000.0                  # s (maximum averaging time)
    tau_points: int = 100                    # Number of tau points
    
    # Signal processing parameters
    decimation_factor: int = 4               # Decimation for low-pass
    anti_alias_order: int = 8                # Anti-aliasing filter order
    notch_frequencies: List[float] = None    # Notch filter frequencies (Hz)
    
    def __post_init__(self):
        if self.notch_frequencies is None:
            self.notch_frequencies = [50.0, 60.0, 100.0]  # Power line harmonics

class SensorReading(NamedTuple):
    """Individual sensor reading with metadata."""
    timestamp: float
    value: float
    sensor_type: SensorType
    noise_estimate: float
    quality_factor: float

class FusedEstimate(NamedTuple):
    """Fused multi-sensor estimate with uncertainty."""
    timestamp: float
    position: float
    position_uncertainty: float
    resolution_estimate: float
    sensor_weights: Dict[SensorType, float]
    snr_estimate: float

class EnhancedResolutionControl:
    """
    Enhanced resolution control system for achieving ≤0.05 nm resolution.
    
    LaTeX Formulations Implemented:
    
    1. Resolution Requirement:
    σ_resolution ≤ 5×10⁻¹¹ m
    
    2. Multi-Sensor Fusion:
    x̂(t) = ∑ᵢ₌₁ᴺ wᵢ(t) × xᵢ(t)
    
    3. Optimal Kalman Weights:
    wᵢ(t) = σᵢ⁻²(t) / ∑ⱼ₌₁ᴺ σⱼ⁻²(t)
    
    4. Adaptive Bandwidth:
    BW(t) = BW₀ × [1 + α × SNR⁻¹(t)]
    
    5. Allan Variance:
    σ²(τ) = ∫₀^∞ |H(f)|² × Sₓ(f) × sin⁴(πfτ)/(πfτ)² df
    
    6. Enhanced SNR Calculation:
    SNR(t) = 10×log₁₀[P_signal(t) / P_noise(t)]
    
    7. Resolution Enhancement Factor:
    η = √N × ∏ᵢ₌₁ᴺ wᵢ²
    """
    
    def __init__(self, sensor_configs: List[Dict], 
                 filter_params: Optional[FilterParameters] = None):
        """
        Initialize enhanced resolution control system.
        
        Args:
            sensor_configs: List of sensor configuration dictionaries
            filter_params: Filter parameters, uses defaults if None
        """
        self.sensor_configs = sensor_configs
        self.sensors = SensorCharacteristics.get_enhanced_sensors()
        self.filter_params = filter_params or FilterParameters()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kalman filter states for each sensor
        self.kalman_states = {}
        self.kalman_covariances = {}
        self.adaptive_bandwidths = {}
        
        for i, config in enumerate(sensor_configs):
            sensor_type = config['sensor_type']
            key = f'sensor_{i}_{sensor_type.value}'
            
            # Initialize Kalman filter
            self.kalman_states[key] = 0.0
            self.kalman_covariances[key] = self.filter_params.initial_error_covariance
            self.adaptive_bandwidths[key] = self.filter_params.base_bandwidth
        
        # Data storage
        self.sensor_readings_history = deque(maxlen=10000)
        self.fused_estimates_history = deque(maxlen=10000)
        self.resolution_history = deque(maxlen=1000)
        
        # Performance tracking
        self.resolution_violations = 0
        self.snr_history = deque(maxlen=1000)
        
        # Allan variance computation
        self.allan_variance_data = deque(maxlen=50000)
        
        self.logger.info(f"Enhanced resolution control initialized for {len(sensor_configs)} sensors")
    
    def update_sensor_reading(self, sensor_reading: SensorReading) -> Dict:
        """
        Update individual sensor reading with Kalman filtering.
        
        Args:
            sensor_reading: New sensor reading
            
        Returns:
            Dictionary with filtered estimate and metadata
        """
        # Find sensor configuration
        sensor_config = None
        sensor_key = None
        
        for i, config in enumerate(self.sensor_configs):
            if config['sensor_type'] == sensor_reading.sensor_type:
                sensor_key = f'sensor_{i}_{sensor_reading.sensor_type.value}'
                sensor_config = config
                break
        
        if sensor_config is None:
            raise ValueError(f"Unknown sensor type: {sensor_reading.sensor_type}")
        
        # Get sensor characteristics
        sensor_char = self.sensors[sensor_reading.sensor_type]
        
        # Kalman filter prediction step
        # x_k|k-1 = F * x_k-1|k-1  (simple position model, F = 1)
        predicted_state = self.kalman_states[sensor_key]
        
        # P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        predicted_covariance = (self.kalman_covariances[sensor_key] + 
                              self.filter_params.process_noise_variance)
        
        # Measurement noise variance (adaptive based on SNR)
        measurement_noise = self._calculate_measurement_noise(sensor_reading, sensor_char)
        
        # Kalman gain: K = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^-1
        # For position measurement, H = 1
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_noise)
        
        # State update: x_k|k = x_k|k-1 + K * (z_k - H * x_k|k-1)
        innovation = sensor_reading.value - predicted_state
        updated_state = predicted_state + kalman_gain * innovation
        
        # Covariance update: P_k|k = (I - K * H) * P_k|k-1
        updated_covariance = (1 - kalman_gain) * predicted_covariance
        
        # Store updated estimates
        self.kalman_states[sensor_key] = updated_state
        self.kalman_covariances[sensor_key] = updated_covariance
        
        # Adaptive bandwidth update
        self._update_adaptive_bandwidth(sensor_key, sensor_reading, sensor_char)
        
        # Store reading in history
        self.sensor_readings_history.append(sensor_reading)
        
        filtered_result = {
            'sensor_key': sensor_key,
            'filtered_position': updated_state,
            'position_uncertainty': np.sqrt(updated_covariance),
            'kalman_gain': kalman_gain,
            'innovation': innovation,
            'measurement_noise': measurement_noise,
            'adaptive_bandwidth': self.adaptive_bandwidths[sensor_key]
        }
        
        return filtered_result
    
    def fuse_multi_sensor_readings(self, timestamp: float) -> FusedEstimate:
        """
        Fuse multiple sensor readings using optimal weighting.
        
        LaTeX: x̂(t) = ∑ᵢ₌₁ᴺ wᵢ(t) × xᵢ(t)
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Fused estimate with uncertainty quantification
        """
        if not self.kalman_states:
            raise ValueError("No sensor states available for fusion")
        
        # Calculate optimal weights (inverse variance weighting)
        weights = {}
        total_inverse_variance = 0.0
        
        for sensor_key, covariance in self.kalman_covariances.items():
            if covariance > 0:
                inverse_variance = 1.0 / covariance
                weights[sensor_key] = inverse_variance
                total_inverse_variance += inverse_variance
            else:
                weights[sensor_key] = 0.0
        
        # Normalize weights
        if total_inverse_variance > 0:
            for sensor_key in weights:
                weights[sensor_key] /= total_inverse_variance
        else:
            # Equal weights fallback
            equal_weight = 1.0 / len(self.kalman_states)
            for sensor_key in weights:
                weights[sensor_key] = equal_weight
        
        # Calculate fused position estimate
        fused_position = 0.0
        for sensor_key, weight in weights.items():
            fused_position += weight * self.kalman_states[sensor_key]
        
        # Calculate fused uncertainty (optimal fusion formula)
        fused_variance = 1.0 / total_inverse_variance if total_inverse_variance > 0 else np.inf
        fused_uncertainty = np.sqrt(fused_variance)
        
        # Calculate resolution estimate
        resolution_estimate = self._calculate_resolution_estimate(weights, timestamp)
        
        # Calculate SNR estimate
        snr_estimate = self._calculate_snr_estimate(fused_position, fused_uncertainty)
        
        # Create sensor type mapping for weights
        sensor_type_weights = {}
        for i, config in enumerate(self.sensor_configs):
            sensor_type = config['sensor_type']
            sensor_key = f'sensor_{i}_{sensor_type.value}'
            if sensor_key in weights:
                sensor_type_weights[sensor_type] = weights[sensor_key]
        
        # Create fused estimate
        fused_estimate = FusedEstimate(
            timestamp=timestamp,
            position=fused_position,
            position_uncertainty=fused_uncertainty,
            resolution_estimate=resolution_estimate,
            sensor_weights=sensor_type_weights,
            snr_estimate=snr_estimate
        )
        
        # Store in history
        self.fused_estimates_history.append(fused_estimate)
        self.resolution_history.append(resolution_estimate)
        self.snr_history.append(snr_estimate)
        
        # Check resolution requirement
        if resolution_estimate > RESOLUTION_REQUIREMENT:
            self.resolution_violations += 1
            self.logger.warning(f"Resolution requirement violated: "
                              f"{resolution_estimate*1e9:.3f} nm > {RESOLUTION_REQUIREMENT*1e9:.2f} nm")
        
        return fused_estimate
    
    def _calculate_measurement_noise(self, reading: SensorReading, 
                                   characteristics: SensorCharacteristics) -> float:
        """
        Calculate adaptive measurement noise based on sensor characteristics and conditions.
        
        Args:
            reading: Sensor reading
            characteristics: Sensor characteristics
            
        Returns:
            Measurement noise variance
        """
        # Base thermal noise
        thermal_noise = characteristics.thermal_noise_density**2
        
        # Shot noise (for optical sensors)
        shot_noise = characteristics.shot_noise_current * 1.6e-19  # q*I
        
        # Flicker noise (1/f)
        flicker_noise = thermal_noise * characteristics.flicker_noise_corner / max(1.0, reading.timestamp)
        
        # Quantization noise
        quantization_noise = characteristics.quantization_noise**2
        
        # Environmental noise (temperature, pressure, humidity effects)
        environmental_noise = (characteristics.temperature_coefficient * 0.1)**2  # Assume 0.1K variation
        
        # Quality factor adjustment
        quality_adjustment = 1.0 / max(0.1, reading.quality_factor)
        
        total_noise_variance = (thermal_noise + shot_noise + flicker_noise + 
                              quantization_noise + environmental_noise) * quality_adjustment
        
        return max(self.filter_params.measurement_noise_variance, total_noise_variance)
    
    def _update_adaptive_bandwidth(self, sensor_key: str, reading: SensorReading,
                                 characteristics: SensorCharacteristics):
        """
        Update adaptive bandwidth based on signal conditions.
        
        LaTeX: BW(t) = BW₀ × [1 + α × SNR⁻¹(t)]
        
        Args:
            sensor_key: Sensor identifier
            reading: Current sensor reading
            characteristics: Sensor characteristics
        """
        # Estimate SNR from reading quality
        snr_linear = max(0.1, reading.quality_factor)
        snr_db = 10 * np.log10(snr_linear)
        
        # Calculate adaptive bandwidth
        base_bw = self.filter_params.base_bandwidth
        adaptation_factor = self.filter_params.adaptation_factor
        
        # Inverse relationship: lower SNR -> higher bandwidth (more filtering)
        snr_factor = 1.0 + adaptation_factor / max(0.1, snr_linear)
        adaptive_bw = base_bw / snr_factor
        
        # Apply constraints
        adaptive_bw = np.clip(adaptive_bw, 
                            self.filter_params.min_bandwidth,
                            self.filter_params.max_bandwidth)
        
        # Low-pass filter the bandwidth update to avoid rapid changes
        alpha = 0.1  # Bandwidth update rate
        current_bw = self.adaptive_bandwidths.get(sensor_key, base_bw)
        self.adaptive_bandwidths[sensor_key] = alpha * adaptive_bw + (1 - alpha) * current_bw
    
    def _calculate_resolution_estimate(self, weights: Dict[str, float], 
                                     timestamp: float) -> float:
        """
        Calculate system resolution estimate based on sensor fusion.
        
        LaTeX: η = √N × ∏ᵢ₌₁ᴺ wᵢ²
        
        Args:
            weights: Sensor weights dictionary
            timestamp: Current timestamp
            
        Returns:
            Resolution estimate (m)
        """
        if not weights:
            return np.inf
        
        # Basic resolution calculation from individual sensor resolutions
        individual_resolutions = []
        weight_values = []
        
        for i, config in enumerate(self.sensor_configs):
            sensor_type = config['sensor_type']
            sensor_key = f'sensor_{i}_{sensor_type.value}'
            
            if sensor_key in weights and weights[sensor_key] > 0:
                sensor_char = self.sensors[sensor_type]
                individual_resolutions.append(sensor_char.resolution)
                weight_values.append(weights[sensor_key])
        
        if not individual_resolutions:
            return np.inf
        
        # Weighted RMS combination
        weighted_resolution_squared = sum(w * r**2 for w, r in zip(weight_values, individual_resolutions))
        total_weight = sum(weight_values)
        
        if total_weight > 0:
            weighted_resolution = np.sqrt(weighted_resolution_squared / total_weight)
        else:
            weighted_resolution = min(individual_resolutions)
        
        # Enhancement factor from fusion
        N = len(individual_resolutions)
        weight_product = np.prod([w**2 for w in weight_values]) if weight_values else 1.0
        enhancement_factor = np.sqrt(N) * np.sqrt(weight_product)
        
        # Avoid over-optimistic estimates
        enhancement_factor = max(1.0, enhancement_factor)
        
        resolution_estimate = weighted_resolution / enhancement_factor
        
        return resolution_estimate
    
    def _calculate_snr_estimate(self, signal: float, noise: float) -> float:
        """
        Calculate signal-to-noise ratio estimate.
        
        LaTeX: SNR(t) = 10×log₁₀[P_signal(t) / P_noise(t)]
        
        Args:
            signal: Signal amplitude
            noise: Noise amplitude (standard deviation)
            
        Returns:
            SNR in dB
        """
        if noise <= 0:
            return 100.0  # Very high SNR
        
        # Power-based SNR calculation
        signal_power = signal**2
        noise_power = noise**2
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(max(1e-12, snr_linear))
        else:
            snr_db = 100.0
        
        return snr_db
    
    def calculate_allan_variance(self, data_window: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Allan variance for stability analysis.
        
        LaTeX: σ²(τ) = ∫₀^∞ |H(f)|² × Sₓ(f) × sin⁴(πfτ)/(πfτ)² df
        
        Args:
            data_window: Number of recent samples to use
            
        Returns:
            Tuple of (tau_values, allan_variance_values)
        """
        if len(self.fused_estimates_history) < 100:
            self.logger.warning("Insufficient data for Allan variance calculation")
            return np.array([]), np.array([])
        
        # Extract position data from recent estimates
        recent_estimates = list(self.fused_estimates_history)[-data_window:]
        positions = [est.position for est in recent_estimates]
        timestamps = [est.timestamp for est in recent_estimates]
        
        if len(set(timestamps)) < len(timestamps):
            # Handle duplicate timestamps
            unique_data = {}
            for t, p in zip(timestamps, positions):
                unique_data[t] = p
            timestamps = sorted(unique_data.keys())
            positions = [unique_data[t] for t in timestamps]
        
        positions = np.array(positions)
        
        # Estimate sampling rate
        if len(timestamps) > 1:
            dt = np.median(np.diff(timestamps))
        else:
            dt = 1.0
        
        # Generate tau values (logarithmic spacing)
        tau_min = max(2 * dt, self.filter_params.tau_min)
        tau_max = min(len(positions) * dt / 10, self.filter_params.tau_max)
        
        tau_values = np.logspace(np.log10(tau_min), np.log10(tau_max), 
                               self.filter_params.tau_points)
        
        allan_variances = []
        
        for tau in tau_values:
            # Calculate Allan variance for this tau
            m = int(tau / dt)  # Number of samples per averaging interval
            
            if m < 2 or m > len(positions) // 3:
                allan_variances.append(np.nan)
                continue
            
            # Create overlapping averages
            averages = []
            for i in range(len(positions) - m + 1):
                avg = np.mean(positions[i:i+m])
                averages.append(avg)
            
            if len(averages) < 2:
                allan_variances.append(np.nan)
                continue
            
            # Calculate Allan variance: σ²(τ) = <(y_{i+1} - y_i)²> / 2
            differences = np.diff(averages)
            allan_var = np.mean(differences**2) / 2.0
            allan_variances.append(allan_var)
        
        allan_variances = np.array(allan_variances)
        
        # Remove NaN values
        valid_indices = ~np.isnan(allan_variances)
        tau_values = tau_values[valid_indices]
        allan_variances = allan_variances[valid_indices]
        
        return tau_values, allan_variances
    
    def optimize_sensor_weights(self, target_resolution: float = None) -> Dict[str, float]:
        """
        Optimize sensor weights for target resolution.
        
        Args:
            target_resolution: Target resolution (m), uses requirement if None
            
        Returns:
            Optimized weights dictionary
        """
        if target_resolution is None:
            target_resolution = RESOLUTION_REQUIREMENT
        
        if not self.sensor_configs:
            return {}
        
        def objective(weights):
            """Objective function for weight optimization."""
            # Normalize weights
            w_norm = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Create weight dictionary
            weight_dict = {}
            for i, config in enumerate(self.sensor_configs):
                sensor_type = config['sensor_type']
                sensor_key = f'sensor_{i}_{sensor_type.value}'
                weight_dict[sensor_key] = w_norm[i]
            
            # Calculate resolution estimate
            resolution_est = self._calculate_resolution_estimate(weight_dict, time.time())
            
            # Objective: minimize resolution while penalizing extreme weights
            resolution_penalty = (resolution_est / target_resolution - 1)**2
            weight_penalty = 0.01 * np.sum((w_norm - 1/len(weights))**2)  # Prefer balanced weights
            
            return resolution_penalty + weight_penalty
        
        # Initial guess (equal weights)
        n_sensors = len(self.sensor_configs)
        initial_weights = np.ones(n_sensors) / n_sensors
        
        # Constraints: weights must be positive and sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        bounds = [(0.001, 1.0) for _ in range(n_sensors)]  # Positive weights
        
        try:
            result = minimize(objective, initial_weights, 
                            bounds=bounds, constraints=constraints,
                            method='SLSQP')
            
            if result.success:
                optimized_weights = {}
                for i, config in enumerate(self.sensor_configs):
                    sensor_type = config['sensor_type']
                    sensor_key = f'sensor_{i}_{sensor_type.value}'
                    optimized_weights[sensor_key] = result.x[i]
                
                self.logger.info("Sensor weight optimization successful")
                return optimized_weights
            else:
                self.logger.warning("Weight optimization failed, using equal weights")
                return self._get_equal_weights()
                
        except Exception as e:
            self.logger.error(f"Weight optimization error: {e}")
            return self._get_equal_weights()
    
    def _get_equal_weights(self) -> Dict[str, float]:
        """Get equal weights for all sensors."""
        n_sensors = len(self.sensor_configs)
        equal_weight = 1.0 / n_sensors if n_sensors > 0 else 0.0
        
        weights = {}
        for i, config in enumerate(self.sensor_configs):
            sensor_type = config['sensor_type']
            sensor_key = f'sensor_{i}_{sensor_type.value}'
            weights[sensor_key] = equal_weight
        
        return weights
    
    def check_resolution_requirement(self) -> Dict[str, any]:
        """
        Check if resolution requirement is satisfied.
        
        Returns:
            Dictionary with requirement satisfaction results
        """
        if not self.resolution_history:
            return {'status': 'no_data'}
        
        current_resolution = self.resolution_history[-1]
        recent_resolutions = list(self.resolution_history)[-100:]  # Last 100 measurements
        
        # Check current requirement
        requirement_satisfied = current_resolution <= RESOLUTION_REQUIREMENT
        
        # Calculate statistics
        mean_resolution = np.mean(recent_resolutions)
        rms_resolution = np.sqrt(np.mean(np.array(recent_resolutions)**2))
        max_resolution = np.max(recent_resolutions)
        
        # Success rate
        success_count = sum(1 for r in recent_resolutions if r <= RESOLUTION_REQUIREMENT)
        success_rate = success_count / len(recent_resolutions) * 100
        
        # Margin analysis
        margin_factor = RESOLUTION_REQUIREMENT / current_resolution if current_resolution > 0 else np.inf
        
        results = {
            'requirement_satisfied': requirement_satisfied,
            'current_resolution_nm': current_resolution * 1e9,
            'requirement_nm': RESOLUTION_REQUIREMENT * 1e9,
            'mean_resolution_nm': mean_resolution * 1e9,
            'rms_resolution_nm': rms_resolution * 1e9,
            'max_resolution_nm': max_resolution * 1e9,
            'success_rate_percent': success_rate,
            'margin_factor': margin_factor,
            'total_violations': self.resolution_violations,
            'measurement_count': len(recent_resolutions)
        }
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary for resolution control.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.fused_estimates_history:
            return {'status': 'no_data'}
        
        # Resolution analysis
        resolution_results = self.check_resolution_requirement()
        
        # SNR analysis
        snr_data = list(self.snr_history)[-100:] if self.snr_history else []
        
        # Allan variance analysis
        tau_values, allan_variances = self.calculate_allan_variance()
        
        performance = {
            'resolution_performance': resolution_results,
            'sensor_fusion': {
                'active_sensors': len(self.sensor_configs),
                'fusion_estimates': len(self.fused_estimates_history),
                'kalman_states': len(self.kalman_states)
            },
            'adaptive_control': {
                'adaptive_bandwidths': dict(self.adaptive_bandwidths),
                'base_bandwidth_hz': self.filter_params.base_bandwidth,
                'bandwidth_range_hz': [self.filter_params.min_bandwidth, 
                                     self.filter_params.max_bandwidth]
            }
        }
        
        if snr_data:
            performance['signal_quality'] = {
                'current_snr_db': snr_data[-1],
                'mean_snr_db': np.mean(snr_data),
                'min_snr_db': np.min(snr_data),
                'max_snr_db': np.max(snr_data)
            }
        
        if len(tau_values) > 0 and len(allan_variances) > 0:
            performance['stability_analysis'] = {
                'allan_variance_available': True,
                'tau_range_s': [float(np.min(tau_values)), float(np.max(tau_values))],
                'min_allan_variance_m2': float(np.min(allan_variances)),
                'stability_time_constant_s': float(tau_values[np.argmin(allan_variances)]) if len(allan_variances) > 0 else 0
            }
        
        return performance


if __name__ == "__main__":
    """Example usage of enhanced resolution control."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== ENHANCED RESOLUTION CONTROL ===")
    print("Target: ≤0.05 nm resolution")
    
    # Define sensor configurations
    sensor_configs = [
        {
            'sensor_type': SensorType.INTERFEROMETRIC,
            'position': [0, 0, 0],
            'orientation': [0, 0, 1]
        },
        {
            'sensor_type': SensorType.CAPACITIVE,
            'position': [1e-3, 0, 0],
            'orientation': [0, 0, 1]
        },
        {
            'sensor_type': SensorType.OPTICAL_ENCODER,
            'position': [0, 1e-3, 0],
            'orientation': [1, 0, 0]
        }
    ]
    
    # Initialize controller
    controller = EnhancedResolutionControl(sensor_configs)
    
    print(f"\nInitialized with {len(sensor_configs)} sensors:")
    for i, config in enumerate(sensor_configs):
        sensor_char = controller.sensors[config['sensor_type']]
        print(f"  {i+1}. {config['sensor_type'].value}: "
              f"resolution={sensor_char.resolution*1e9:.3f} nm, "
              f"bandwidth={sensor_char.bandwidth:.0f} Hz")
    
    # Simulate sensor readings
    timestamp = time.time()
    
    for i, config in enumerate(sensor_configs):
        # Simulate realistic sensor reading
        true_position = 1e-6 * np.sin(2 * np.pi * 0.1 * timestamp)  # Slow drift
        sensor_char = controller.sensors[config['sensor_type']]
        
        # Add realistic noise
        noise = np.random.normal(0, sensor_char.resolution / 3)
        measured_position = true_position + noise
        
        # Quality factor based on sensor characteristics
        quality_factor = 1.0 / (1.0 + noise**2 / sensor_char.resolution**2)
        
        sensor_reading = SensorReading(
            timestamp=timestamp,
            value=measured_position,
            sensor_type=config['sensor_type'],
            noise_estimate=abs(noise),
            quality_factor=quality_factor
        )
        
        # Update sensor
        filtered_result = controller.update_sensor_reading(sensor_reading)
        print(f"\nSensor {config['sensor_type'].value}:")
        print(f"  Raw reading: {measured_position*1e9:.3f} nm")
        print(f"  Filtered: {filtered_result['filtered_position']*1e9:.3f} nm")
        print(f"  Uncertainty: ±{filtered_result['position_uncertainty']*1e9:.3f} nm")
        print(f"  Adaptive BW: {filtered_result['adaptive_bandwidth']:.1f} Hz")
    
    # Perform sensor fusion
    fused_estimate = controller.fuse_multi_sensor_readings(timestamp)
    
    print(f"\nFused Estimate:")
    print(f"  Position: {fused_estimate.position*1e9:.3f} nm")
    print(f"  Uncertainty: ±{fused_estimate.position_uncertainty*1e9:.3f} nm")
    print(f"  Resolution estimate: {fused_estimate.resolution_estimate*1e9:.3f} nm")
    print(f"  SNR: {fused_estimate.snr_estimate:.1f} dB")
    
    print(f"\nSensor Weights:")
    for sensor_type, weight in fused_estimate.sensor_weights.items():
        print(f"  {sensor_type.value}: {weight:.3f}")
    
    # Check resolution requirement
    resolution_results = controller.check_resolution_requirement()
    
    print(f"\nResolution Performance:")
    print(f"  Requirement satisfied: {'✓' if resolution_results['requirement_satisfied'] else '✗'}")
    print(f"  Current: {resolution_results['current_resolution_nm']:.3f} nm")
    print(f"  Requirement: {resolution_results['requirement_nm']:.2f} nm")
    print(f"  Margin: {resolution_results['margin_factor']:.2f}x")
    
    # Performance summary
    performance = controller.get_performance_summary()
    
    print(f"\nSystem Performance:")
    rf = performance['resolution_performance']
    print(f"  Success rate: {rf['success_rate_percent']:.1f}%")
    sf = performance['sensor_fusion']
    print(f"  Active sensors: {sf['active_sensors']}")
    print(f"  Fusion estimates: {sf['fusion_estimates']}")
    
    if 'signal_quality' in performance:
        sq = performance['signal_quality']
        print(f"  Mean SNR: {sq['mean_snr_db']:.1f} dB")
