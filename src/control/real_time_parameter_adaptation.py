"""
Real-Time Parameter Adaptation System
===================================

This module implements real-time parameter adaptation with feedback-controlled
optimization for maintaining performance under varying conditions.

Mathematical Formulation:
Adaptive Parameter Update:
θ(k+1) = θ(k) + α × ∇J(θ(k)) × e(k)

Recursive Least Squares (RLS):
P(k) = P(k-1) - P(k-1)φ(k)φᵀ(k)P(k-1) / [λ + φᵀ(k)P(k-1)φ(k)]
θ(k) = θ(k-1) + P(k)φ(k)[y(k) - φᵀ(k)θ(k-1)]

Model Reference Adaptive Control (MRAC):
u(k) = θ₁ᵀ(k)r(k) + θ₂ᵀ(k)y(k) + θ₃ᵀ(k)u(k-1)

Lyapunov Stability Condition:
V̇ = -e(k)ᵀQe(k) ≤ 0 for stable adaptation

Enhanced Forgetting Factor:
λ(k) = λ₀ + (1-λ₀) × exp(-|e(k)|/σ)
"""

import numpy as np
from scipy import signal, linalg, optimize
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional, Callable, NamedTuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import threading
from collections import deque
from abc import ABC, abstractmethod

# Adaptation parameters
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_FORGETTING_FACTOR = 0.95
MIN_FORGETTING_FACTOR = 0.90
MAX_FORGETTING_FACTOR = 0.999
ADAPTATION_THRESHOLD = 1e-6
PARAMETER_BOUNDS_FACTOR = 10.0

class AdaptationType(Enum):
    """Types of parameter adaptation algorithms."""
    GRADIENT_DESCENT = "gradient_descent"
    RECURSIVE_LEAST_SQUARES = "rls"
    MODEL_REFERENCE = "mrac"
    KALMAN_FILTER = "kalman"
    NEURAL_NETWORK = "neural"

class ParameterType(Enum):
    """Types of parameters to adapt."""
    CONTROLLER_GAINS = "controller_gains"
    FILTER_COEFFICIENTS = "filter_coefficients"
    MODEL_PARAMETERS = "model_parameters"
    REFERENCE_TRAJECTORY = "reference_trajectory"
    NOISE_STATISTICS = "noise_statistics"

@dataclass
class AdaptationParameters:
    """Parameters for adaptive algorithms."""
    
    learning_rate: float = DEFAULT_LEARNING_RATE
    forgetting_factor: float = DEFAULT_FORGETTING_FACTOR
    adaptation_threshold: float = ADAPTATION_THRESHOLD
    parameter_bounds: Tuple[float, float] = (-10.0, 10.0)
    
    # RLS specific parameters
    initial_covariance: float = 1000.0
    regularization: float = 1e-6
    
    # MRAC specific parameters
    reference_model_poles: List[float] = field(default_factory=lambda: [-2.0, -3.0])
    adaptation_gain: float = 1.0
    
    # Stability monitoring
    stability_margin: float = 0.1
    max_parameter_change: float = 0.5
    
    # Performance monitoring
    performance_window: int = 100
    performance_threshold: float = 0.01

class ParameterEstimate(NamedTuple):
    """Parameter estimate with uncertainty."""
    value: np.ndarray
    covariance: np.ndarray
    timestamp: float
    confidence: float

class AdaptationStatus(NamedTuple):
    """Status of adaptation process."""
    is_adapting: bool
    convergence_rate: float
    parameter_change_rate: float
    stability_margin: float
    performance_index: float

class BaseAdaptiveAlgorithm(ABC):
    """Base class for adaptive algorithms."""
    
    def __init__(self, n_parameters: int, adaptation_params: AdaptationParameters):
        self.n_parameters = n_parameters
        self.params = adaptation_params
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize parameters
        self.theta = np.zeros(n_parameters)
        self.theta_covariance = np.eye(n_parameters) * adaptation_params.initial_covariance
        
        # History storage
        self.parameter_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
        # Adaptation state
        self.adaptation_active = True
        self.last_update_time = time.time()
        
    @abstractmethod
    def update(self, measurement: float, reference: float, 
               regressor: np.ndarray, timestamp: float) -> ParameterEstimate:
        """Update parameter estimates."""
        pass
    
    @abstractmethod
    def predict(self, regressor: np.ndarray) -> float:
        """Predict output using current parameters."""
        pass
    
    def get_adaptation_status(self) -> AdaptationStatus:
        """Get current adaptation status."""
        if len(self.parameter_history) < 2:
            return AdaptationStatus(
                is_adapting=self.adaptation_active,
                convergence_rate=0.0,
                parameter_change_rate=0.0,
                stability_margin=1.0,
                performance_index=0.0
            )
        
        # Calculate convergence rate
        recent_errors = list(self.error_history)[-self.params.performance_window:]
        if len(recent_errors) > 1:
            error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
            convergence_rate = -error_trend  # Negative slope means convergence
        else:
            convergence_rate = 0.0
        
        # Calculate parameter change rate
        recent_params = list(self.parameter_history)[-10:]
        if len(recent_params) > 1:
            param_changes = [np.linalg.norm(p.value - recent_params[0].value) 
                           for p in recent_params[1:]]
            parameter_change_rate = np.mean(param_changes) if param_changes else 0.0
        else:
            parameter_change_rate = 0.0
        
        # Estimate stability margin (simplified)
        stability_margin = max(0.0, self.params.stability_margin - parameter_change_rate)
        
        # Performance index
        recent_performance = list(self.performance_history)[-self.params.performance_window:]
        performance_index = np.mean(recent_performance) if recent_performance else 0.0
        
        return AdaptationStatus(
            is_adapting=self.adaptation_active,
            convergence_rate=convergence_rate,
            parameter_change_rate=parameter_change_rate,
            stability_margin=stability_margin,
            performance_index=performance_index
        )

class RecursiveLeastSquares(BaseAdaptiveAlgorithm):
    """
    Recursive Least Squares parameter estimation.
    
    LaTeX Equations:
    P(k) = P(k-1) - P(k-1)φ(k)φᵀ(k)P(k-1) / [λ + φᵀ(k)P(k-1)φ(k)]
    θ(k) = θ(k-1) + P(k)φ(k)[y(k) - φᵀ(k)θ(k-1)]
    """
    
    def __init__(self, n_parameters: int, adaptation_params: AdaptationParameters):
        super().__init__(n_parameters, adaptation_params)
        
        # RLS-specific initialization
        self.P = np.eye(n_parameters) * adaptation_params.initial_covariance
        self.lambda_adaptive = adaptation_params.forgetting_factor
        
    def update(self, measurement: float, reference: float, 
               regressor: np.ndarray, timestamp: float) -> ParameterEstimate:
        """
        Update parameters using RLS algorithm.
        
        Args:
            measurement: Current system output
            reference: Reference/desired output
            regressor: Regressor vector φ(k)
            timestamp: Current timestamp
            
        Returns:
            Updated parameter estimate
        """
        if len(regressor) != self.n_parameters:
            raise ValueError(f"Regressor size {len(regressor)} != n_parameters {self.n_parameters}")
        
        # Prediction error
        y_pred = np.dot(regressor, self.theta)
        error = measurement - y_pred
        
        # Adaptive forgetting factor based on error magnitude
        error_normalized = abs(error) / (1 + abs(measurement))
        self.lambda_adaptive = self._compute_adaptive_forgetting_factor(error_normalized)
        
        # RLS update equations
        # Compute gain vector: K = P*φ / (λ + φᵀ*P*φ)
        Pphi = np.dot(self.P, regressor)
        denominator = self.lambda_adaptive + np.dot(regressor, Pphi)
        
        if abs(denominator) < self.params.regularization:
            denominator = self.params.regularization
        
        K = Pphi / denominator
        
        # Update parameter estimates
        self.theta += K * error
        
        # Apply parameter bounds
        self.theta = np.clip(self.theta, 
                           self.params.parameter_bounds[0],
                           self.params.parameter_bounds[1])
        
        # Update covariance matrix: P = (P - K*φᵀ*P) / λ
        self.P = (self.P - np.outer(K, np.dot(regressor, self.P))) / self.lambda_adaptive
        
        # Ensure P remains positive definite
        eigenvals = np.linalg.eigvals(self.P)
        if np.min(eigenvals) < self.params.regularization:
            self.P += np.eye(self.n_parameters) * self.params.regularization
        
        # Store in history
        param_estimate = ParameterEstimate(
            value=self.theta.copy(),
            covariance=self.P.copy(),
            timestamp=timestamp,
            confidence=self._compute_confidence()
        )
        
        self.parameter_history.append(param_estimate)
        self.error_history.append(abs(error))
        self.performance_history.append(error**2)
        
        return param_estimate
    
    def predict(self, regressor: np.ndarray) -> float:
        """Predict output using current parameters."""
        return np.dot(regressor, self.theta)
    
    def _compute_adaptive_forgetting_factor(self, error_normalized: float) -> float:
        """
        Compute adaptive forgetting factor.
        
        LaTeX: λ(k) = λ₀ + (1-λ₀) × exp(-|e(k)|/σ)
        """
        sigma = 0.1  # Error scaling factor
        
        # Exponential adaptation
        lambda_new = (self.params.forgetting_factor + 
                     (1 - self.params.forgetting_factor) * np.exp(-error_normalized / sigma))
        
        # Apply bounds
        lambda_new = np.clip(lambda_new, MIN_FORGETTING_FACTOR, MAX_FORGETTING_FACTOR)
        
        return lambda_new
    
    def _compute_confidence(self) -> float:
        """Compute confidence measure based on covariance trace."""
        trace_P = np.trace(self.P)
        confidence = 1.0 / (1.0 + trace_P / self.n_parameters)
        return np.clip(confidence, 0.0, 1.0)

class ModelReferenceAdaptiveControl(BaseAdaptiveAlgorithm):
    """
    Model Reference Adaptive Control.
    
    LaTeX: u(k) = θ₁ᵀ(k)r(k) + θ₂ᵀ(k)y(k) + θ₃ᵀ(k)u(k-1)
    """
    
    def __init__(self, n_parameters: int, adaptation_params: AdaptationParameters):
        super().__init__(n_parameters, adaptation_params)
        
        # Reference model
        poles = adaptation_params.reference_model_poles
        self.reference_model = self._create_reference_model(poles)
        
        # MRAC state
        self.reference_output_history = deque(maxlen=10)
        self.control_history = deque(maxlen=10)
        
    def _create_reference_model(self, poles: List[float]) -> signal.TransferFunction:
        """Create reference model from desired poles."""
        # Create characteristic polynomial from poles
        char_poly = np.poly(poles)
        
        # Unit gain reference model
        num = [char_poly[-1]]  # DC gain = 1
        den = char_poly
        
        return signal.TransferFunction(num, den)
    
    def update(self, measurement: float, reference: float, 
               regressor: np.ndarray, timestamp: float) -> ParameterEstimate:
        """
        Update parameters using MRAC adaptation law.
        
        Args:
            measurement: Current system output
            reference: Reference input
            regressor: [r(k), y(k), u(k-1), ...]
            timestamp: Current timestamp
            
        Returns:
            Updated parameter estimate
        """
        # Get reference model output
        reference_output = self._compute_reference_output(reference)
        
        # Model following error
        error = measurement - reference_output
        
        # MIT rule adaptation (simplified Lyapunov approach)
        # θ̇ = -Γ × e × ∂e/∂θ
        # For discrete time: θ(k+1) = θ(k) - γ × e(k) × φ(k)
        
        adaptation_gain = self.params.adaptation_gain
        learning_rate = self.params.learning_rate
        
        # Gradient of error with respect to parameters (approximated by regressor)
        gradient = regressor * error
        
        # Parameter update with Lyapunov stability consideration
        parameter_update = -adaptation_gain * learning_rate * gradient
        
        # Apply update with bounds
        self.theta += parameter_update
        self.theta = np.clip(self.theta,
                           self.params.parameter_bounds[0],
                           self.params.parameter_bounds[1])
        
        # Update covariance (simplified)
        parameter_change = np.linalg.norm(parameter_update)
        self.theta_covariance *= (1 - learning_rate)
        self.theta_covariance += learning_rate * parameter_change * np.eye(self.n_parameters)
        
        # Store histories
        self.reference_output_history.append(reference_output)
        
        param_estimate = ParameterEstimate(
            value=self.theta.copy(),
            covariance=self.theta_covariance.copy(),
            timestamp=timestamp,
            confidence=self._compute_mrac_confidence(error)
        )
        
        self.parameter_history.append(param_estimate)
        self.error_history.append(abs(error))
        self.performance_history.append(error**2)
        
        return param_estimate
    
    def predict(self, regressor: np.ndarray) -> float:
        """Predict control output using current parameters."""
        return np.dot(regressor, self.theta)
    
    def _compute_reference_output(self, reference: float) -> float:
        """Compute reference model output (simplified first-order)."""
        if not self.reference_output_history:
            return reference
        
        # Simple first-order reference model: y_m(k) = a*y_m(k-1) + b*r(k)
        a = 0.9  # Pole location
        b = 0.1  # Gain
        
        y_m_prev = self.reference_output_history[-1] if self.reference_output_history else 0
        y_m = a * y_m_prev + b * reference
        
        return y_m
    
    def _compute_mrac_confidence(self, error: float) -> float:
        """Compute MRAC confidence based on tracking error."""
        # Confidence decreases with larger tracking errors
        confidence = np.exp(-abs(error) / 0.1)
        return np.clip(confidence, 0.0, 1.0)

class GradientDescentAdaptation(BaseAdaptiveAlgorithm):
    """
    Gradient descent parameter adaptation.
    
    LaTeX: θ(k+1) = θ(k) + α × ∇J(θ(k)) × e(k)
    """
    
    def __init__(self, n_parameters: int, adaptation_params: AdaptationParameters):
        super().__init__(n_parameters, adaptation_params)
        
        # Gradient descent specific
        self.momentum = np.zeros(n_parameters)
        self.momentum_factor = 0.9
        
        # Adam optimizer parameters
        self.m = np.zeros(n_parameters)  # First moment
        self.v = np.zeros(n_parameters)  # Second moment
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step
    
    def update(self, measurement: float, reference: float, 
               regressor: np.ndarray, timestamp: float) -> ParameterEstimate:
        """
        Update parameters using gradient descent with Adam optimization.
        
        Args:
            measurement: Current system output
            reference: Reference/desired output
            regressor: Gradient vector or regressor
            timestamp: Current timestamp
            
        Returns:
            Updated parameter estimate
        """
        # Prediction error
        y_pred = self.predict(regressor)
        error = measurement - y_pred
        
        # Gradient of cost function J = e²/2
        # ∇J = ∂J/∂θ = e × ∂e/∂θ = e × (-∂ŷ/∂θ) = -e × φ
        gradient = -error * regressor
        
        self.t += 1
        
        # Adam optimization
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Parameter update
        parameter_update = self.params.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.theta += parameter_update
        
        # Apply parameter bounds
        self.theta = np.clip(self.theta,
                           self.params.parameter_bounds[0],
                           self.params.parameter_bounds[1])
        
        # Update covariance estimate (simplified)
        gradient_norm = np.linalg.norm(gradient)
        self.theta_covariance = (0.99 * self.theta_covariance + 
                               0.01 * gradient_norm * np.eye(self.n_parameters))
        
        param_estimate = ParameterEstimate(
            value=self.theta.copy(),
            covariance=self.theta_covariance.copy(),
            timestamp=timestamp,
            confidence=self._compute_gradient_confidence(gradient_norm)
        )
        
        self.parameter_history.append(param_estimate)
        self.error_history.append(abs(error))
        self.performance_history.append(error**2)
        
        return param_estimate
    
    def predict(self, regressor: np.ndarray) -> float:
        """Predict output using current parameters."""
        return np.dot(regressor, self.theta)
    
    def _compute_gradient_confidence(self, gradient_norm: float) -> float:
        """Compute confidence based on gradient magnitude."""
        # High confidence when gradient is small (near convergence)
        confidence = 1.0 / (1.0 + gradient_norm)
        return np.clip(confidence, 0.0, 1.0)

class RealTimeParameterAdaptationSystem:
    """
    Real-time parameter adaptation system with multiple algorithms and safety monitoring.
    
    Features:
    1. Multiple adaptation algorithms (RLS, MRAC, Gradient Descent)
    2. Real-time parameter updates
    3. Stability monitoring and safeguards
    4. Performance tracking and optimization
    5. Automatic algorithm switching
    6. Parameter bounds enforcement
    """
    
    def __init__(self, parameter_configs: List[Dict],
                 adaptation_params: Optional[AdaptationParameters] = None):
        """
        Initialize real-time parameter adaptation system.
        
        Args:
            parameter_configs: List of parameter configuration dictionaries
            adaptation_params: Adaptation parameters, uses defaults if None
        """
        self.parameter_configs = parameter_configs
        self.adaptation_params = adaptation_params or AdaptationParameters()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize adaptive algorithms
        self.adaptive_algorithms = {}
        self.active_algorithms = {}
        
        for config in parameter_configs:
            param_name = config['name']
            param_type = config['type']
            n_params = config['size']
            algorithm_type = config.get('algorithm', AdaptationType.RECURSIVE_LEAST_SQUARES)
            
            # Create appropriate algorithm
            if algorithm_type == AdaptationType.RECURSIVE_LEAST_SQUARES:
                algorithm = RecursiveLeastSquares(n_params, self.adaptation_params)
            elif algorithm_type == AdaptationType.MODEL_REFERENCE:
                algorithm = ModelReferenceAdaptiveControl(n_params, self.adaptation_params)
            elif algorithm_type == AdaptationType.GRADIENT_DESCENT:
                algorithm = GradientDescentAdaptation(n_params, self.adaptation_params)
            else:
                # Default to RLS
                algorithm = RecursiveLeastSquares(n_params, self.adaptation_params)
            
            self.adaptive_algorithms[param_name] = algorithm
            self.active_algorithms[param_name] = True
        
        # System state
        self.is_running = False
        self.adaptation_thread = None
        self.system_measurements = deque(maxlen=1000)
        self.reference_signals = deque(maxlen=1000)
        
        # Performance monitoring
        self.performance_metrics = {}
        self.stability_violations = 0
        self.adaptation_events = deque(maxlen=500)
        
        # Safety monitoring
        self.safety_monitor_active = True
        self.emergency_stop_triggered = False
        
        self.logger.info(f"Parameter adaptation system initialized with {len(parameter_configs)} parameter sets")
    
    def start_adaptation(self, update_rate_hz: float = 1000.0):
        """
        Start real-time parameter adaptation.
        
        Args:
            update_rate_hz: Adaptation update rate in Hz
        """
        if self.is_running:
            self.logger.warning("Adaptation already running")
            return
        
        self.is_running = True
        self.update_period = 1.0 / update_rate_hz
        
        # Start adaptation thread
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        self.logger.info(f"Real-time adaptation started at {update_rate_hz} Hz")
    
    def stop_adaptation(self):
        """Stop real-time parameter adaptation."""
        self.is_running = False
        
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=1.0)
        
        self.logger.info("Real-time adaptation stopped")
    
    def update_measurement(self, measurement: Dict[str, float], 
                         reference: Dict[str, float], 
                         regressor: Dict[str, np.ndarray]):
        """
        Update system with new measurement and reference data.
        
        Args:
            measurement: Dictionary of measurement values
            reference: Dictionary of reference values
            regressor: Dictionary of regressor vectors
        """
        timestamp = time.time()
        
        # Store measurements
        self.system_measurements.append({
            'timestamp': timestamp,
            'measurement': measurement.copy(),
            'reference': reference.copy(),
            'regressor': regressor.copy()
        })
        
        # Update adaptive algorithms
        for param_name, algorithm in self.adaptive_algorithms.items():
            if (param_name in measurement and param_name in reference and 
                param_name in regressor and self.active_algorithms.get(param_name, False)):
                
                try:
                    # Update algorithm
                    param_estimate = algorithm.update(
                        measurement[param_name],
                        reference[param_name],
                        regressor[param_name],
                        timestamp
                    )
                    
                    # Check safety constraints
                    if self.safety_monitor_active:
                        self._check_safety_constraints(param_name, param_estimate)
                    
                    # Record adaptation event
                    self.adaptation_events.append({
                        'timestamp': timestamp,
                        'parameter': param_name,
                        'estimate': param_estimate,
                        'algorithm_status': algorithm.get_adaptation_status()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Adaptation update failed for {param_name}: {e}")
                    self.active_algorithms[param_name] = False
    
    def _adaptation_loop(self):
        """Main adaptation loop running in separate thread."""
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Perform adaptation updates if new data available
                if self.system_measurements:
                    self._process_adaptation_step()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check stability and safety
                if self.safety_monitor_active:
                    self._monitor_system_stability()
                
                # Sleep to maintain update rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.update_period - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Adaptation loop error: {e}")
                if not self.emergency_stop_triggered:
                    self._trigger_emergency_stop()
    
    def _process_adaptation_step(self):
        """Process one adaptation step."""
        if not self.system_measurements:
            return
        
        # Get latest measurement
        latest_data = self.system_measurements[-1]
        
        # Process each parameter
        for param_name, algorithm in self.adaptive_algorithms.items():
            if not self.active_algorithms.get(param_name, False):
                continue
            
            measurement_data = latest_data['measurement']
            reference_data = latest_data['reference']
            regressor_data = latest_data['regressor']
            
            if (param_name in measurement_data and param_name in reference_data and 
                param_name in regressor_data):
                
                # Get adaptation status
                status = algorithm.get_adaptation_status()
                
                # Adaptive algorithm switching based on performance
                if status.performance_index > self.adaptation_params.performance_threshold:
                    self._consider_algorithm_switch(param_name, status)
                
                # Adjust adaptation rate based on convergence
                if status.convergence_rate > 0:
                    # Converging - can increase learning rate slightly
                    current_lr = algorithm.params.learning_rate
                    new_lr = min(current_lr * 1.01, 0.1)
                    algorithm.params.learning_rate = new_lr
                elif status.convergence_rate < -0.01:
                    # Diverging - reduce learning rate
                    current_lr = algorithm.params.learning_rate
                    new_lr = max(current_lr * 0.95, 0.001)
                    algorithm.params.learning_rate = new_lr
    
    def _check_safety_constraints(self, param_name: str, param_estimate: ParameterEstimate):
        """
        Check safety constraints for parameter estimates.
        
        Args:
            param_name: Parameter name
            param_estimate: Parameter estimate to check
        """
        # Check parameter bounds
        param_values = param_estimate.value
        bounds = self.adaptation_params.parameter_bounds
        
        if np.any(param_values < bounds[0]) or np.any(param_values > bounds[1]):
            self.logger.warning(f"Parameter {param_name} out of bounds: {param_values}")
            self.stability_violations += 1
        
        # Check parameter change rate
        algorithm = self.adaptive_algorithms[param_name]
        if len(algorithm.parameter_history) > 1:
            prev_estimate = algorithm.parameter_history[-2]
            param_change = np.linalg.norm(param_estimate.value - prev_estimate.value)
            
            if param_change > self.adaptation_params.max_parameter_change:
                self.logger.warning(f"Large parameter change in {param_name}: {param_change}")
                self.stability_violations += 1
                
                # Reduce learning rate temporarily
                algorithm.params.learning_rate *= 0.5
        
        # Check confidence level
        if param_estimate.confidence < 0.1:
            self.logger.warning(f"Low confidence in {param_name}: {param_estimate.confidence}")
    
    def _monitor_system_stability(self):
        """Monitor overall system stability."""
        # Check recent stability violations
        if self.stability_violations > 10:  # Threshold
            self.logger.error("Multiple stability violations detected")
            self._trigger_emergency_stop()
        
        # Check algorithm health
        unhealthy_algorithms = 0
        for param_name, algorithm in self.adaptive_algorithms.items():
            if not self.active_algorithms.get(param_name, False):
                unhealthy_algorithms += 1
        
        if unhealthy_algorithms > len(self.adaptive_algorithms) / 2:
            self.logger.error("Too many unhealthy algorithms")
            self._trigger_emergency_stop()
        
        # Reset violation counter periodically
        if len(self.adaptation_events) > 100:  # Every 100 events
            self.stability_violations = max(0, self.stability_violations - 1)
    
    def _consider_algorithm_switch(self, param_name: str, status: AdaptationStatus):
        """
        Consider switching adaptation algorithm based on performance.
        
        Args:
            param_name: Parameter name
            status: Current algorithm status
        """
        current_algorithm = self.adaptive_algorithms[param_name]
        current_type = type(current_algorithm).__name__
        
        # Simple switching logic based on performance
        if status.performance_index > 0.1 and status.convergence_rate < 0:
            # Poor performance and diverging - try different algorithm
            
            config = next(c for c in self.parameter_configs if c['name'] == param_name)
            n_params = config['size']
            
            if isinstance(current_algorithm, RecursiveLeastSquares):
                # Switch to gradient descent
                new_algorithm = GradientDescentAdaptation(n_params, self.adaptation_params)
                self.logger.info(f"Switching {param_name} from RLS to Gradient Descent")
            elif isinstance(current_algorithm, GradientDescentAdaptation):
                # Switch to MRAC
                new_algorithm = ModelReferenceAdaptiveControl(n_params, self.adaptation_params)
                self.logger.info(f"Switching {param_name} from Gradient Descent to MRAC")
            else:
                # Switch back to RLS
                new_algorithm = RecursiveLeastSquares(n_params, self.adaptation_params)
                self.logger.info(f"Switching {param_name} to RLS")
            
            # Transfer state (simplified)
            new_algorithm.theta = current_algorithm.theta.copy()
            
            # Replace algorithm
            self.adaptive_algorithms[param_name] = new_algorithm
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop of adaptation."""
        if self.emergency_stop_triggered:
            return
        
        self.emergency_stop_triggered = True
        self.logger.critical("EMERGENCY STOP: Adaptation system halted due to safety violation")
        
        # Disable all adaptation
        for param_name in self.active_algorithms:
            self.active_algorithms[param_name] = False
        
        # Stop adaptation thread
        self.is_running = False
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        if not self.adaptation_events:
            return
        
        # Calculate metrics for each parameter
        for param_name in self.adaptive_algorithms:
            # Get recent events for this parameter
            param_events = [e for e in list(self.adaptation_events)[-100:] 
                          if e['parameter'] == param_name]
            
            if not param_events:
                continue
            
            # Performance metrics
            recent_errors = [abs(e['algorithm_status'].convergence_rate) for e in param_events]
            recent_changes = [e['algorithm_status'].parameter_change_rate for e in param_events]
            recent_confidence = [e['estimate'].confidence for e in param_events]
            
            self.performance_metrics[param_name] = {
                'mean_error': np.mean(recent_errors) if recent_errors else 0.0,
                'mean_parameter_change': np.mean(recent_changes) if recent_changes else 0.0,
                'mean_confidence': np.mean(recent_confidence) if recent_confidence else 0.0,
                'algorithm_type': type(self.adaptive_algorithms[param_name]).__name__,
                'is_active': self.active_algorithms.get(param_name, False)
            }
    
    def get_current_parameters(self) -> Dict[str, ParameterEstimate]:
        """
        Get current parameter estimates for all parameters.
        
        Returns:
            Dictionary of current parameter estimates
        """
        current_params = {}
        
        for param_name, algorithm in self.adaptive_algorithms.items():
            if algorithm.parameter_history:
                current_params[param_name] = algorithm.parameter_history[-1]
            else:
                # Return initial estimate
                current_params[param_name] = ParameterEstimate(
                    value=algorithm.theta.copy(),
                    covariance=algorithm.theta_covariance.copy(),
                    timestamp=time.time(),
                    confidence=0.0
                )
        
        return current_params
    
    def get_adaptation_summary(self) -> Dict:
        """
        Get comprehensive adaptation system summary.
        
        Returns:
            Dictionary with adaptation system status and performance
        """
        current_params = self.get_current_parameters()
        
        summary = {
            'system_status': {
                'is_running': self.is_running,
                'emergency_stop': self.emergency_stop_triggered,
                'safety_monitor_active': self.safety_monitor_active,
                'stability_violations': self.stability_violations
            },
            'algorithm_status': {},
            'performance_metrics': self.performance_metrics.copy(),
            'parameter_estimates': {},
            'adaptation_events_count': len(self.adaptation_events),
            'measurement_history_length': len(self.system_measurements)
        }
        
        # Algorithm status
        for param_name, algorithm in self.adaptive_algorithms.items():
            status = algorithm.get_adaptation_status()
            summary['algorithm_status'][param_name] = {
                'is_adapting': status.is_adapting,
                'convergence_rate': status.convergence_rate,
                'parameter_change_rate': status.parameter_change_rate,
                'stability_margin': status.stability_margin,
                'performance_index': status.performance_index,
                'algorithm_type': type(algorithm).__name__,
                'is_active': self.active_algorithms.get(param_name, False)
            }
        
        # Parameter estimates
        for param_name, estimate in current_params.items():
            summary['parameter_estimates'][param_name] = {
                'value': estimate.value.tolist(),
                'confidence': estimate.confidence,
                'timestamp': estimate.timestamp,
                'covariance_trace': np.trace(estimate.covariance)
            }
        
        return summary
    
    def reset_adaptation(self, param_name: Optional[str] = None):
        """
        Reset adaptation for specific parameter or all parameters.
        
        Args:
            param_name: Parameter name to reset, or None for all
        """
        if param_name is None:
            # Reset all
            for name, algorithm in self.adaptive_algorithms.items():
                algorithm.theta = np.zeros_like(algorithm.theta)
                algorithm.theta_covariance = (np.eye(len(algorithm.theta)) * 
                                            self.adaptation_params.initial_covariance)
                algorithm.parameter_history.clear()
                algorithm.error_history.clear()
                algorithm.performance_history.clear()
                self.active_algorithms[name] = True
            
            self.stability_violations = 0
            self.emergency_stop_triggered = False
            
            self.logger.info("All adaptation algorithms reset")
        
        elif param_name in self.adaptive_algorithms:
            # Reset specific parameter
            algorithm = self.adaptive_algorithms[param_name]
            algorithm.theta = np.zeros_like(algorithm.theta)
            algorithm.theta_covariance = (np.eye(len(algorithm.theta)) * 
                                        self.adaptation_params.initial_covariance)
            algorithm.parameter_history.clear()
            algorithm.error_history.clear()
            algorithm.performance_history.clear()
            self.active_algorithms[param_name] = True
            
            self.logger.info(f"Adaptation algorithm for {param_name} reset")
        else:
            self.logger.warning(f"Unknown parameter: {param_name}")


if __name__ == "__main__":
    """Example usage of real-time parameter adaptation system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== REAL-TIME PARAMETER ADAPTATION SYSTEM ===")
    
    # Define parameter configurations
    parameter_configs = [
        {
            'name': 'controller_gains',
            'type': ParameterType.CONTROLLER_GAINS,
            'size': 3,  # PID gains
            'algorithm': AdaptationType.RECURSIVE_LEAST_SQUARES
        },
        {
            'name': 'filter_coefficients',
            'type': ParameterType.FILTER_COEFFICIENTS,
            'size': 2,  # Filter coefficients
            'algorithm': AdaptationType.GRADIENT_DESCENT
        },
        {
            'name': 'model_parameters',
            'type': ParameterType.MODEL_PARAMETERS,
            'size': 4,  # Model parameters
            'algorithm': AdaptationType.MODEL_REFERENCE
        }
    ]
    
    # Create adaptation system
    adaptation_params = AdaptationParameters(
        learning_rate=0.05,
        forgetting_factor=0.98,
        adaptation_threshold=1e-4
    )
    
    adaptation_system = RealTimeParameterAdaptationSystem(
        parameter_configs=parameter_configs,
        adaptation_params=adaptation_params
    )
    
    print(f"\nInitialized adaptation system with {len(parameter_configs)} parameter sets:")
    for config in parameter_configs:
        print(f"  - {config['name']}: {config['size']} parameters, {config['algorithm'].value}")
    
    # Start adaptation
    print(f"\nStarting real-time adaptation...")
    adaptation_system.start_adaptation(update_rate_hz=100.0)
    
    # Simulate system operation
    print(f"\nSimulating system operation for 5 seconds...")
    
    for i in range(500):  # 5 seconds at 100 Hz
        # Simulate measurements
        measurement = {
            'controller_gains': 1.0 + 0.1 * np.sin(2 * np.pi * 0.1 * i * 0.01),
            'filter_coefficients': 0.5 + 0.05 * np.random.randn(),
            'model_parameters': 2.0 + 0.1 * np.cos(2 * np.pi * 0.05 * i * 0.01)
        }
        
        # Simulate references
        reference = {
            'controller_gains': 1.0,
            'filter_coefficients': 0.5,
            'model_parameters': 2.0
        }
        
        # Simulate regressors (feature vectors)
        regressor = {
            'controller_gains': np.array([1.0, measurement['controller_gains'], 
                                        measurement['controller_gains']**2]),
            'filter_coefficients': np.array([1.0, measurement['filter_coefficients']]),
            'model_parameters': np.array([1.0, measurement['model_parameters'], 
                                        np.sin(measurement['model_parameters']), 
                                        np.cos(measurement['model_parameters'])])
        }
        
        # Update adaptation system
        adaptation_system.update_measurement(measurement, reference, regressor)
        
        # Print progress every second
        if i % 100 == 0:
            current_params = adaptation_system.get_current_parameters()
            print(f"  Time: {i*0.01:.1f}s")
            for param_name, estimate in current_params.items():
                print(f"    {param_name}: {estimate.value[:2]} (confidence: {estimate.confidence:.3f})")
    
    # Stop adaptation
    print(f"\nStopping adaptation...")
    adaptation_system.stop_adaptation()
    
    # Get final summary
    summary = adaptation_system.get_adaptation_summary()
    
    print(f"\nFinal Adaptation Summary:")
    print(f"  System Status: {'Running' if summary['system_status']['is_running'] else 'Stopped'}")
    print(f"  Emergency Stop: {'Yes' if summary['system_status']['emergency_stop'] else 'No'}")
    print(f"  Stability Violations: {summary['system_status']['stability_violations']}")
    print(f"  Total Adaptation Events: {summary['adaptation_events_count']}")
    
    print(f"\nAlgorithm Performance:")
    for param_name, status in summary['algorithm_status'].items():
        print(f"  {param_name}:")
        print(f"    Algorithm: {status['algorithm_type']}")
        print(f"    Convergence Rate: {status['convergence_rate']:.6f}")
        print(f"    Performance Index: {status['performance_index']:.6f}")
        print(f"    Stability Margin: {status['stability_margin']:.3f}")
    
    print(f"\nFinal Parameter Estimates:")
    for param_name, estimate in summary['parameter_estimates'].items():
        print(f"  {param_name}: {estimate['value'][:3]} (confidence: {estimate['confidence']:.3f})")
    
    print(f"\nReal-time parameter adaptation demonstration complete!")
