"""
Bayesian State Estimation Module
===============================

This module implements advanced Bayesian state estimation techniques for the
digital twin, including Kalman filtering, particle filtering, and ensemble methods.

Mathematical Formulation:

Posterior Update:
P(X_true | Y_measurements) ∝ P(Y_measurements | X_true) × P(X_true)

Extended Kalman Filter:
dX̂/dt = f(X̂, U) + K(t) × [Y_measured - h(X̂)]
dP/dt = F×P + P×F^T + Q - P×H^T×R^(-1)×H×P

Unscented Kalman Filter:
X_sigma = [X̂, X̂ + √((n+λ)P), X̂ - √((n+λ)P)]
Y_pred = h(X_sigma)
X̂ = Σᵢ W_i × X_sigma,i

Particle Filter:
w_i ∝ p(y_k | x_k^i) × w_{k-1}^i
x_k^i ~ p(x_k | x_{k-1}^i, u_k)
"""

import numpy as np
from scipy import linalg, stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable, NamedTuple
import logging
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
from abc import ABC, abstractmethod

# Estimation performance targets
ESTIMATION_ACCURACY_TARGET = 0.95
CONVERGENCE_TIME_TARGET = 0.1  # seconds
COMPUTATIONAL_EFFICIENCY_TARGET = 0.001  # seconds per update

class EstimationType(Enum):
    """Types of Bayesian estimation algorithms."""
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    PARTICLE_FILTER = "particle_filter"
    ENSEMBLE_KALMAN = "ensemble_kalman"
    VARIATIONAL_BAYES = "variational_bayes"

class MeasurementType(Enum):
    """Types of measurements."""
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    TEMPERATURE = "temperature"
    ELECTROMAGNETIC = "electromagnetic"
    QUANTUM_STATE = "quantum_state"

@dataclass
class EstimationParameters:
    """Parameters for Bayesian estimation."""
    
    # Kalman filter parameters
    process_noise_std: float = 1e-9
    measurement_noise_std: float = 1e-11
    initial_state_uncertainty: float = 1e-6
    
    # Unscented Kalman parameters
    alpha: float = 1e-3  # Spread parameter
    beta: float = 2.0    # Distribution parameter (2 for Gaussian)
    kappa: float = 0.0   # Secondary scaling parameter
    
    # Particle filter parameters
    n_particles: int = 1000
    resampling_threshold: float = 0.5  # Effective sample size threshold
    proposal_std: float = 1e-9
    
    # Ensemble Kalman parameters
    n_ensemble: int = 100
    inflation_factor: float = 1.05
    
    # Convergence criteria
    convergence_tolerance: float = 1e-12
    max_iterations: int = 100

class EstimationResult(NamedTuple):
    """Result of Bayesian estimation."""
    state_estimate: np.ndarray
    covariance: np.ndarray
    likelihood: float
    innovation: np.ndarray
    timestamp: float
    convergence_iterations: int

class BayesianEstimator(ABC):
    """Abstract base class for Bayesian estimators."""
    
    def __init__(self, state_size: int, measurement_size: int, 
                 params: EstimationParameters):
        self.state_size = state_size
        self.measurement_size = measurement_size
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # State estimation
        self.state_estimate = np.zeros(state_size)
        self.covariance = np.eye(state_size) * params.initial_state_uncertainty**2
        
        # Performance tracking
        self.estimation_history = deque(maxlen=1000)
        self.computation_times = deque(maxlen=100)
        self.likelihood_history = deque(maxlen=1000)
        
    @abstractmethod
    def predict(self, dynamics_function: Callable, control_input: np.ndarray, 
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step of Bayesian estimation."""
        pass
    
    @abstractmethod
    def update(self, measurement: np.ndarray, 
               measurement_function: Callable) -> EstimationResult:
        """Update step of Bayesian estimation."""
        pass
    
    def get_estimation_quality(self) -> Dict[str, float]:
        """Get estimation quality metrics."""
        if len(self.estimation_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate metrics
        recent_results = list(self.estimation_history)[-50:]
        
        # State estimation accuracy (consistency of estimates)
        state_consistency = 0.0
        if len(recent_results) > 1:
            states = [result.state_estimate for result in recent_results]
            state_diffs = [np.linalg.norm(states[i] - states[i-1]) 
                          for i in range(1, len(states))]
            state_consistency = 1.0 / (1.0 + np.mean(state_diffs))
        
        # Convergence time
        convergence_times = [result.convergence_iterations for result in recent_results]
        avg_convergence_iterations = np.mean(convergence_times)
        
        # Computational efficiency
        avg_computation_time = np.mean(list(self.computation_times)) if self.computation_times else 0.0
        
        # Likelihood trend
        likelihood_trend = 0.0
        if len(self.likelihood_history) > 10:
            recent_likelihoods = list(self.likelihood_history)[-20:]
            likelihood_trend = np.polyfit(range(len(recent_likelihoods)), recent_likelihoods, 1)[0]
        
        return {
            'state_consistency': state_consistency,
            'avg_convergence_iterations': avg_convergence_iterations,
            'avg_computation_time_s': avg_computation_time,
            'likelihood_trend': likelihood_trend,
            'estimation_accuracy_satisfied': state_consistency >= ESTIMATION_ACCURACY_TARGET,
            'convergence_time_satisfied': avg_computation_time <= CONVERGENCE_TIME_TARGET,
            'efficiency_satisfied': avg_computation_time <= COMPUTATIONAL_EFFICIENCY_TARGET
        }

class ExtendedKalmanFilter(BayesianEstimator):
    """
    Extended Kalman Filter for nonlinear state estimation.
    
    Mathematical Implementation:
    
    Prediction:
    X̂⁻ = f(X̂⁺, U, dt)
    P⁻ = F × P⁺ × F^T + Q
    
    Update:
    K = P⁻ × H^T × (H × P⁻ × H^T + R)^(-1)
    X̂⁺ = X̂⁻ + K × (Y - h(X̂⁻))
    P⁺ = (I - K × H) × P⁻
    
    where F = ∂f/∂X, H = ∂h/∂X (Jacobians)
    """
    
    def __init__(self, state_size: int, measurement_size: int, 
                 params: EstimationParameters):
        super().__init__(state_size, measurement_size, params)
        
        # EKF-specific initialization
        self.process_noise = np.eye(state_size) * params.process_noise_std**2
        self.measurement_noise = np.eye(measurement_size) * params.measurement_noise_std**2
        
    def predict(self, dynamics_function: Callable, control_input: np.ndarray, 
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extended Kalman Filter prediction step.
        
        Args:
            dynamics_function: Nonlinear dynamics f(x, u, dt)
            control_input: Control input vector
            dt: Time step
            
        Returns:
            Predicted state and covariance
        """
        # Predict state using nonlinear dynamics
        self.state_estimate = dynamics_function(self.state_estimate, control_input, dt)
        
        # Compute Jacobian F = ∂f/∂x
        jacobian_F = self._compute_dynamics_jacobian(
            dynamics_function, self.state_estimate, control_input, dt
        )
        
        # Predict covariance
        self.covariance = (jacobian_F @ self.covariance @ jacobian_F.T + 
                          self.process_noise)
        
        return self.state_estimate, self.covariance
    
    def update(self, measurement: np.ndarray, 
               measurement_function: Callable) -> EstimationResult:
        """
        Extended Kalman Filter update step.
        
        Args:
            measurement: Measurement vector
            measurement_function: Measurement model h(x)
            
        Returns:
            Estimation result with updated state and covariance
        """
        start_time = time.time()
        
        # Predicted measurement
        predicted_measurement = measurement_function(self.state_estimate)
        
        # Innovation (measurement residual)
        innovation = measurement - predicted_measurement
        
        # Compute measurement Jacobian H = ∂h/∂x
        jacobian_H = self._compute_measurement_jacobian(
            measurement_function, self.state_estimate
        )
        
        # Innovation covariance
        S = jacobian_H @ self.covariance @ jacobian_H.T + self.measurement_noise
        
        # Kalman gain
        try:
            K = self.covariance @ jacobian_H.T @ linalg.inv(S)
        except linalg.LinAlgError:
            # Handle singular matrix
            K = self.covariance @ jacobian_H.T @ linalg.pinv(S)
            self.logger.warning("Singular innovation covariance, using pseudo-inverse")
        
        # State update
        self.state_estimate += K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_size) - K @ jacobian_H
        self.covariance = (I_KH @ self.covariance @ I_KH.T + 
                          K @ self.measurement_noise @ K.T)
        
        # Compute likelihood
        likelihood = self._compute_likelihood(innovation, S)
        
        # Computation time
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        # Create result
        result = EstimationResult(
            state_estimate=self.state_estimate.copy(),
            covariance=self.covariance.copy(),
            likelihood=likelihood,
            innovation=innovation,
            timestamp=time.time(),
            convergence_iterations=1  # EKF converges in one iteration
        )
        
        # Store in history
        self.estimation_history.append(result)
        self.likelihood_history.append(likelihood)
        
        return result
    
    def _compute_dynamics_jacobian(self, dynamics_function: Callable, 
                                  state: np.ndarray, control: np.ndarray, 
                                  dt: float) -> np.ndarray:
        """Compute Jacobian of dynamics function using finite differences."""
        epsilon = 1e-8
        jacobian = np.zeros((self.state_size, self.state_size))
        
        f_nominal = dynamics_function(state, control, dt)
        
        for i in range(self.state_size):
            state_pert = state.copy()
            state_pert[i] += epsilon
            f_pert = dynamics_function(state_pert, control, dt)
            jacobian[:, i] = (f_pert - f_nominal) / epsilon
        
        return jacobian
    
    def _compute_measurement_jacobian(self, measurement_function: Callable,
                                    state: np.ndarray) -> np.ndarray:
        """Compute Jacobian of measurement function using finite differences."""
        epsilon = 1e-8
        jacobian = np.zeros((self.measurement_size, self.state_size))
        
        h_nominal = measurement_function(state)
        
        for i in range(self.state_size):
            state_pert = state.copy()
            state_pert[i] += epsilon
            h_pert = measurement_function(state_pert)
            jacobian[:, i] = (h_pert - h_nominal) / epsilon
        
        return jacobian
    
    def _compute_likelihood(self, innovation: np.ndarray, 
                           innovation_covariance: np.ndarray) -> float:
        """Compute measurement likelihood."""
        try:
            # Multivariate normal likelihood
            inv_S = linalg.inv(innovation_covariance)
            log_likelihood = (-0.5 * innovation.T @ inv_S @ innovation - 
                             0.5 * np.log(linalg.det(2 * np.pi * innovation_covariance)))
            return float(log_likelihood)
        except:
            return -np.inf

class UnscentedKalmanFilter(BayesianEstimator):
    """
    Unscented Kalman Filter for better handling of nonlinearities.
    
    Mathematical Implementation:
    
    Sigma Points:
    X_sigma = [X̂, X̂ + √((n+λ)P), X̂ - √((n+λ)P)]
    
    Prediction:
    X_sigma⁻ = f(X_sigma⁺, U, dt)
    X̂⁻ = Σᵢ W_m,i × X_sigma,i⁻
    P⁻ = Σᵢ W_c,i × (X_sigma,i⁻ - X̂⁻)(X_sigma,i⁻ - X̂⁻)^T + Q
    
    Update:
    Y_sigma = h(X_sigma⁻)
    Ŷ = Σᵢ W_m,i × Y_sigma,i
    """
    
    def __init__(self, state_size: int, measurement_size: int, 
                 params: EstimationParameters):
        super().__init__(state_size, measurement_size, params)
        
        # UKF parameters
        self.alpha = params.alpha
        self.beta = params.beta
        self.kappa = params.kappa
        
        # Compute sigma point parameters
        self.n = state_size
        self.lambda_param = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Weights
        self.n_sigma = 2 * self.n + 1
        self.W_m = np.zeros(self.n_sigma)  # Mean weights
        self.W_c = np.zeros(self.n_sigma)  # Covariance weights
        
        self.W_m[0] = self.lambda_param / (self.n + self.lambda_param)
        self.W_c[0] = self.W_m[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, self.n_sigma):
            self.W_m[i] = 1 / (2 * (self.n + self.lambda_param))
            self.W_c[i] = self.W_m[i]
        
        # Noise covariances
        self.process_noise = np.eye(state_size) * params.process_noise_std**2
        self.measurement_noise = np.eye(measurement_size) * params.measurement_noise_std**2
    
    def predict(self, dynamics_function: Callable, control_input: np.ndarray, 
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """UKF prediction step using sigma points."""
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.state_estimate, self.covariance)
        
        # Propagate sigma points through dynamics
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[i, :] = dynamics_function(sigma_points[i, :], control_input, dt)
        
        # Predict mean
        self.state_estimate = np.sum(self.W_m[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Predict covariance
        self.covariance = self.process_noise.copy()
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i, :] - self.state_estimate
            self.covariance += self.W_c[i] * np.outer(diff, diff)
        
        return self.state_estimate, self.covariance
    
    def update(self, measurement: np.ndarray, 
               measurement_function: Callable) -> EstimationResult:
        """UKF update step using sigma points."""
        start_time = time.time()
        
        # Generate sigma points from predicted state
        sigma_points = self._generate_sigma_points(self.state_estimate, self.covariance)
        
        # Propagate sigma points through measurement function
        measurement_sigma = np.zeros((self.n_sigma, self.measurement_size))
        for i in range(self.n_sigma):
            measurement_sigma[i, :] = measurement_function(sigma_points[i, :])
        
        # Predicted measurement mean
        measurement_pred = np.sum(self.W_m[:, np.newaxis] * measurement_sigma, axis=0)
        
        # Innovation
        innovation = measurement - measurement_pred
        
        # Innovation covariance
        S = self.measurement_noise.copy()
        for i in range(self.n_sigma):
            diff = measurement_sigma[i, :] - measurement_pred
            S += self.W_c[i] * np.outer(diff, diff)
        
        # Cross-covariance
        P_xy = np.zeros((self.state_size, self.measurement_size))
        for i in range(self.n_sigma):
            state_diff = sigma_points[i, :] - self.state_estimate
            measurement_diff = measurement_sigma[i, :] - measurement_pred
            P_xy += self.W_c[i] * np.outer(state_diff, measurement_diff)
        
        # Kalman gain
        try:
            K = P_xy @ linalg.inv(S)
        except linalg.LinAlgError:
            K = P_xy @ linalg.pinv(S)
            self.logger.warning("Singular innovation covariance in UKF")
        
        # State update
        self.state_estimate += K @ innovation
        
        # Covariance update
        self.covariance -= K @ S @ K.T
        
        # Compute likelihood
        likelihood = self._compute_likelihood(innovation, S)
        
        # Computation time
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        # Create result
        result = EstimationResult(
            state_estimate=self.state_estimate.copy(),
            covariance=self.covariance.copy(),
            likelihood=likelihood,
            innovation=innovation,
            timestamp=time.time(),
            convergence_iterations=1
        )
        
        # Store in history
        self.estimation_history.append(result)
        self.likelihood_history.append(likelihood)
        
        return result
    
    def _generate_sigma_points(self, mean: np.ndarray, 
                              covariance: np.ndarray) -> np.ndarray:
        """Generate sigma points for UKF."""
        sigma_points = np.zeros((self.n_sigma, self.n))
        
        # Compute matrix square root
        try:
            sqrt_matrix = linalg.cholesky((self.n + self.lambda_param) * covariance).T
        except linalg.LinAlgError:
            # Use eigenvalue decomposition if Cholesky fails
            eigenvals, eigenvecs = linalg.eigh(covariance)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive
            sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals * (self.n + self.lambda_param)))
        
        # Central sigma point
        sigma_points[0, :] = mean
        
        # Positive sigma points
        for i in range(self.n):
            sigma_points[i + 1, :] = mean + sqrt_matrix[i, :]
        
        # Negative sigma points
        for i in range(self.n):
            sigma_points[i + self.n + 1, :] = mean - sqrt_matrix[i, :]
        
        return sigma_points
    
    def _compute_likelihood(self, innovation: np.ndarray, 
                           innovation_covariance: np.ndarray) -> float:
        """Compute measurement likelihood."""
        try:
            inv_S = linalg.inv(innovation_covariance)
            log_likelihood = (-0.5 * innovation.T @ inv_S @ innovation - 
                             0.5 * np.log(linalg.det(2 * np.pi * innovation_covariance)))
            return float(log_likelihood)
        except:
            return -np.inf

class ParticleFilter(BayesianEstimator):
    """
    Particle Filter for highly nonlinear and non-Gaussian estimation.
    
    Mathematical Implementation:
    
    Prediction:
    x_k^i ~ p(x_k | x_{k-1}^i, u_k)  [Sample from transition model]
    
    Update:
    w_k^i ∝ p(y_k | x_k^i) × w_{k-1}^i  [Importance weights]
    w_k^i = w_k^i / Σⱼ w_k^j  [Normalize weights]
    
    Resampling:
    if N_eff < threshold: resample particles
    """
    
    def __init__(self, state_size: int, measurement_size: int, 
                 params: EstimationParameters):
        super().__init__(state_size, measurement_size, params)
        
        # Particle filter parameters
        self.n_particles = params.n_particles
        self.resampling_threshold = params.resampling_threshold
        self.proposal_std = params.proposal_std
        
        # Initialize particles
        self.particles = np.random.multivariate_normal(
            self.state_estimate, 
            self.covariance, 
            self.n_particles
        )
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Noise parameters
        self.process_noise_std = params.process_noise_std
        self.measurement_noise_std = params.measurement_noise_std
    
    def predict(self, dynamics_function: Callable, control_input: np.ndarray, 
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Particle filter prediction step."""
        
        # Propagate each particle through dynamics with noise
        for i in range(self.n_particles):
            # Add process noise
            noise = np.random.normal(0, self.process_noise_std, self.state_size)
            
            # Propagate particle
            self.particles[i, :] = dynamics_function(
                self.particles[i, :], control_input, dt
            ) + noise
        
        # Compute mean and covariance
        self.state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        
        # Weighted covariance
        self.covariance = np.zeros((self.state_size, self.state_size))
        for i in range(self.n_particles):
            diff = self.particles[i, :] - self.state_estimate
            self.covariance += self.weights[i] * np.outer(diff, diff)
        
        return self.state_estimate, self.covariance
    
    def update(self, measurement: np.ndarray, 
               measurement_function: Callable) -> EstimationResult:
        """Particle filter update step."""
        start_time = time.time()
        
        # Update weights based on measurement likelihood
        for i in range(self.n_particles):
            predicted_measurement = measurement_function(self.particles[i, :])
            
            # Measurement likelihood (assuming Gaussian)
            residual = measurement - predicted_measurement
            likelihood = stats.multivariate_normal.pdf(
                residual, 
                mean=np.zeros_like(residual), 
                cov=np.eye(len(residual)) * self.measurement_noise_std**2
            )
            
            self.weights[i] *= likelihood
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # Reset weights if all are zero
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Compute effective sample size
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        
        # Resample if needed
        if effective_sample_size < self.resampling_threshold * self.n_particles:
            self._resample()
        
        # Update state estimate
        self.state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        
        # Update covariance
        self.covariance = np.zeros((self.state_size, self.state_size))
        for i in range(self.n_particles):
            diff = self.particles[i, :] - self.state_estimate
            self.covariance += self.weights[i] * np.outer(diff, diff)
        
        # Compute innovation (using weighted mean)
        predicted_measurements = np.array([
            measurement_function(self.particles[i, :]) for i in range(self.n_particles)
        ])
        predicted_measurement_mean = np.average(predicted_measurements, weights=self.weights, axis=0)
        innovation = measurement - predicted_measurement_mean
        
        # Compute likelihood
        likelihood = np.sum(self.weights)  # Total likelihood
        
        # Computation time
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        # Create result
        result = EstimationResult(
            state_estimate=self.state_estimate.copy(),
            covariance=self.covariance.copy(),
            likelihood=likelihood,
            innovation=innovation,
            timestamp=time.time(),
            convergence_iterations=1
        )
        
        # Store in history
        self.estimation_history.append(result)
        self.likelihood_history.append(likelihood)
        
        return result
    
    def _resample(self):
        """Systematic resampling of particles."""
        cumulative_weights = np.cumsum(self.weights)
        
        # Generate random numbers
        random_nums = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        
        # Resample
        new_particles = np.zeros_like(self.particles)
        new_weights = np.ones(self.n_particles) / self.n_particles
        
        i, j = 0, 0
        while i < self.n_particles and j < self.n_particles:
            if random_nums[i] < cumulative_weights[j]:
                new_particles[i, :] = self.particles[j, :]
                i += 1
            else:
                j += 1
        
        self.particles = new_particles
        self.weights = new_weights

class BayesianStateEstimationSystem:
    """
    Comprehensive Bayesian state estimation system with multiple algorithms.
    
    Features:
    1. Multiple estimation algorithms (EKF, UKF, PF)
    2. Automatic algorithm selection based on performance
    3. Adaptive parameter tuning
    4. Multi-sensor fusion
    5. Real-time performance monitoring
    """
    
    def __init__(self, state_size: int, measurement_size: int,
                 estimation_params: Optional[EstimationParameters] = None):
        """
        Initialize Bayesian state estimation system.
        
        Args:
            state_size: Dimension of state vector
            measurement_size: Dimension of measurement vector
            estimation_params: Estimation parameters
        """
        self.state_size = state_size
        self.measurement_size = measurement_size
        self.params = estimation_params or EstimationParameters()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize estimators
        self.estimators = {
            EstimationType.EXTENDED_KALMAN: ExtendedKalmanFilter(
                state_size, measurement_size, self.params
            ),
            EstimationType.UNSCENTED_KALMAN: UnscentedKalmanFilter(
                state_size, measurement_size, self.params
            ),
            EstimationType.PARTICLE_FILTER: ParticleFilter(
                state_size, measurement_size, self.params
            )
        }
        
        # Active estimator
        self.active_estimator_type = EstimationType.EXTENDED_KALMAN
        self.active_estimator = self.estimators[self.active_estimator_type]
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.algorithm_switches = 0
        self.estimation_results = deque(maxlen=1000)
        
        # Multi-sensor fusion
        self.sensor_weights = {}
        self.sensor_reliability = {}
        
        self.logger.info(f"Bayesian estimation system initialized with {len(self.estimators)} algorithms")
    
    def estimate(self, measurement: np.ndarray, measurement_function: Callable,
                dynamics_function: Callable, control_input: np.ndarray,
                dt: float, measurement_type: MeasurementType = MeasurementType.POSITION) -> EstimationResult:
        """
        Perform Bayesian state estimation.
        
        Args:
            measurement: Measurement vector
            measurement_function: Measurement model h(x)
            dynamics_function: System dynamics f(x, u, dt)
            control_input: Control input
            dt: Time step
            measurement_type: Type of measurement
            
        Returns:
            Estimation result
        """
        # Prediction step
        self.active_estimator.predict(dynamics_function, control_input, dt)
        
        # Update step
        result = self.active_estimator.update(measurement, measurement_function)
        
        # Store result
        self.estimation_results.append(result)
        
        # Update performance tracking
        self._update_performance_tracking()
        
        # Check if algorithm switching is needed
        self._check_algorithm_switching()
        
        return result
    
    def multi_sensor_fusion(self, measurements: Dict[MeasurementType, np.ndarray],
                           measurement_functions: Dict[MeasurementType, Callable],
                           sensor_reliabilities: Optional[Dict[MeasurementType, float]] = None) -> EstimationResult:
        """
        Fuse measurements from multiple sensors.
        
        Args:
            measurements: Dictionary of measurements by type
            measurement_functions: Dictionary of measurement functions by type
            sensor_reliabilities: Optional sensor reliability weights
            
        Returns:
            Fused estimation result
        """
        if sensor_reliabilities is None:
            sensor_reliabilities = {sensor_type: 1.0 for sensor_type in measurements}
        
        # Normalize reliabilities
        total_reliability = sum(sensor_reliabilities.values())
        normalized_reliabilities = {k: v/total_reliability for k, v in sensor_reliabilities.items()}
        
        # Initialize fused measurement and covariance
        fused_measurement = None
        fused_measurement_cov = None
        
        # Sequential update approach
        for sensor_type, measurement in measurements.items():
            if sensor_type in measurement_functions:
                reliability = normalized_reliabilities.get(sensor_type, 1.0)
                
                # Scale measurement noise by reliability
                original_noise = self.active_estimator.measurement_noise
                scaled_noise = original_noise / reliability
                
                # Temporarily update measurement noise
                self.active_estimator.measurement_noise = scaled_noise
                
                # Perform update
                result = self.active_estimator.update(measurement, measurement_functions[sensor_type])
                
                # Restore original noise
                self.active_estimator.measurement_noise = original_noise
        
        return result
    
    def adaptive_parameter_tuning(self, performance_target: float = 0.95):
        """
        Adaptively tune estimation parameters based on performance.
        
        Args:
            performance_target: Target estimation performance
        """
        if len(self.performance_history) < 10:
            return
        
        # Get recent performance
        recent_performance = list(self.performance_history)[-10:]
        avg_performance = np.mean([p['state_consistency'] for p in recent_performance])
        
        if avg_performance < performance_target:
            # Adjust parameters to improve performance
            
            if self.active_estimator_type == EstimationType.EXTENDED_KALMAN:
                # Reduce process noise for EKF
                estimator = self.estimators[EstimationType.EXTENDED_KALMAN]
                estimator.process_noise *= 0.9
                
            elif self.active_estimator_type == EstimationType.UNSCENTED_KALMAN:
                # Adjust UKF parameters
                estimator = self.estimators[EstimationType.UNSCENTED_KALMAN]
                estimator.alpha = min(estimator.alpha * 1.1, 1.0)
                
            elif self.active_estimator_type == EstimationType.PARTICLE_FILTER:
                # Increase particle count
                estimator = self.estimators[EstimationType.PARTICLE_FILTER]
                if estimator.n_particles < 5000:
                    new_n_particles = int(estimator.n_particles * 1.2)
                    # Re-initialize with more particles
                    estimator.n_particles = new_n_particles
                    estimator.particles = np.random.multivariate_normal(
                        estimator.state_estimate, estimator.covariance, new_n_particles
                    )
                    estimator.weights = np.ones(new_n_particles) / new_n_particles
            
            self.logger.info(f"Adapted parameters for {self.active_estimator_type.value}")
    
    def _update_performance_tracking(self):
        """Update performance tracking metrics."""
        if len(self.estimators) == 0:
            return
        
        # Get performance metrics from active estimator
        performance = self.active_estimator.get_estimation_quality()
        performance['algorithm'] = self.active_estimator_type.value
        performance['timestamp'] = time.time()
        
        self.performance_history.append(performance)
    
    def _check_algorithm_switching(self):
        """Check if algorithm switching is beneficial."""
        if len(self.performance_history) < 20:
            return
        
        # Get recent performance of active algorithm
        recent_performance = [p for p in list(self.performance_history)[-10:] 
                            if p['algorithm'] == self.active_estimator_type.value]
        
        if not recent_performance:
            return
        
        avg_performance = np.mean([p['state_consistency'] for p in recent_performance])
        
        # If performance is poor, try switching algorithms
        if avg_performance < 0.7:  # Threshold for switching
            # Find best performing algorithm
            best_algorithm = None
            best_performance = 0
            
            for est_type, estimator in self.estimators.items():
                if est_type != self.active_estimator_type:
                    est_performance = estimator.get_estimation_quality()
                    if est_performance.get('state_consistency', 0) > best_performance:
                        best_performance = est_performance['state_consistency']
                        best_algorithm = est_type
            
            # Switch if better algorithm found
            if best_algorithm and best_performance > avg_performance * 1.1:
                self._switch_algorithm(best_algorithm)
    
    def _switch_algorithm(self, new_algorithm_type: EstimationType):
        """Switch to a different estimation algorithm."""
        old_algorithm = self.active_estimator_type
        
        # Transfer state from old to new estimator
        new_estimator = self.estimators[new_algorithm_type]
        new_estimator.state_estimate = self.active_estimator.state_estimate.copy()
        new_estimator.covariance = self.active_estimator.covariance.copy()
        
        # Switch active estimator
        self.active_estimator_type = new_algorithm_type
        self.active_estimator = new_estimator
        self.algorithm_switches += 1
        
        self.logger.info(f"Switched from {old_algorithm.value} to {new_algorithm_type.value}")
    
    def get_estimation_summary(self) -> Dict:
        """
        Get comprehensive estimation system summary.
        
        Returns:
            Dictionary with estimation performance and status
        """
        # Overall performance
        if self.performance_history:
            recent_performance = list(self.performance_history)[-20:]
            avg_consistency = np.mean([p['state_consistency'] for p in recent_performance])
            avg_computation_time = np.mean([p['avg_computation_time_s'] for p in recent_performance])
            avg_convergence = np.mean([p['avg_convergence_iterations'] for p in recent_performance])
        else:
            avg_consistency = 0.0
            avg_computation_time = 0.0
            avg_convergence = 0.0
        
        # Algorithm performance comparison
        algorithm_performance = {}
        for est_type, estimator in self.estimators.items():
            performance = estimator.get_estimation_quality()
            algorithm_performance[est_type.value] = performance
        
        # Current state
        current_state = {
            'state_estimate': self.active_estimator.state_estimate.tolist(),
            'covariance_trace': np.trace(self.active_estimator.covariance),
            'covariance_determinant': linalg.det(self.active_estimator.covariance)
        }
        
        summary = {
            'system_performance': {
                'average_consistency': avg_consistency,
                'average_computation_time_s': avg_computation_time,
                'average_convergence_iterations': avg_convergence,
                'accuracy_target_met': avg_consistency >= ESTIMATION_ACCURACY_TARGET,
                'efficiency_target_met': avg_computation_time <= COMPUTATIONAL_EFFICIENCY_TARGET
            },
            'active_algorithm': {
                'type': self.active_estimator_type.value,
                'algorithm_switches': self.algorithm_switches,
                'performance': algorithm_performance.get(self.active_estimator_type.value, {})
            },
            'algorithm_comparison': algorithm_performance,
            'current_state': current_state,
            'system_status': {
                'estimation_results_count': len(self.estimation_results),
                'performance_history_length': len(self.performance_history),
                'available_algorithms': list(self.estimators.keys())
            }
        }
        
        return summary


if __name__ == "__main__":
    """Example usage of Bayesian state estimation system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== BAYESIAN STATE ESTIMATION SYSTEM ===")
    print("Advanced multi-algorithm estimation with adaptive switching")
    
    # System parameters
    state_size = 6  # [x, y, z, vx, vy, vz]
    measurement_size = 3  # [x, y, z] position measurements
    
    # Initialize estimation system
    params = EstimationParameters(
        process_noise_std=1e-9,
        measurement_noise_std=1e-11,
        n_particles=500
    )
    
    estimation_system = BayesianStateEstimationSystem(
        state_size, measurement_size, params
    )
    
    print(f"\nInitialized estimation system:")
    print(f"  State size: {state_size}")
    print(f"  Measurement size: {measurement_size}")
    print(f"  Available algorithms: {len(estimation_system.estimators)}")
    print(f"  Active algorithm: {estimation_system.active_estimator_type.value}")
    
    # Define system dynamics (simple double integrator)
    def dynamics_function(state, control, dt):
        """Simple dynamics: x' = Ax + Bu"""
        A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        B = np.array([
            [dt**2/2, 0, 0],
            [0, dt**2/2, 0],
            [0, 0, dt**2/2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt]
        ])
        
        control_padded = np.zeros(3)
        if len(control) > 0:
            control_padded[:min(len(control), 3)] = control[:min(len(control), 3)]
        
        return A @ state + B @ control_padded
    
    # Define measurement function
    def measurement_function(state):
        """Position measurement: y = Cx"""
        C = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        return C @ state
    
    # Simulation loop
    print(f"\nRunning estimation simulation...")
    
    true_state = np.array([0, 0, 0, 0.1e-6, 0, 0])  # Initial true state
    dt = 1e-3  # 1 ms time step
    
    estimation_results = []
    
    for step in range(200):  # 200 ms simulation
        # Control input (small acceleration)
        control = np.array([1e-12, 0, 1e-12])  # pN forces
        
        # Update true state
        true_state = dynamics_function(true_state, control, dt)
        
        # Generate noisy measurement
        true_measurement = measurement_function(true_state)
        measurement_noise = np.random.normal(0, params.measurement_noise_std, measurement_size)
        measurement = true_measurement + measurement_noise
        
        # Perform estimation
        result = estimation_system.estimate(
            measurement, measurement_function, dynamics_function, control, dt
        )
        
        estimation_results.append({
            'step': step,
            'true_state': true_state.copy(),
            'measurement': measurement.copy(),
            'estimated_state': result.state_estimate.copy(),
            'estimation_error': np.linalg.norm(true_state - result.state_estimate),
            'likelihood': result.likelihood
        })
        
        # Print progress
        if step % 50 == 0:
            pos_error = np.linalg.norm(true_state[:3] - result.state_estimate[:3])
            print(f"  Step {step}: position error = {pos_error:.2e} m")
    
    # Multi-sensor fusion example
    print(f"\nTesting multi-sensor fusion...")
    
    # Simulate multiple sensors
    measurements = {
        MeasurementType.POSITION: true_measurement + np.random.normal(0, 1e-11, 3),
        MeasurementType.VELOCITY: true_state[3:6] + np.random.normal(0, 1e-9, 3)
    }
    
    measurement_functions = {
        MeasurementType.POSITION: lambda x: x[:3],
        MeasurementType.VELOCITY: lambda x: x[3:6]
    }
    
    sensor_reliabilities = {
        MeasurementType.POSITION: 0.9,
        MeasurementType.VELOCITY: 0.7
    }
    
    # Note: This is a conceptual example - in practice, we'd need to handle different measurement sizes
    # For demonstration, we'll just use the position measurement
    fusion_result = estimation_system.multi_sensor_fusion(
        {MeasurementType.POSITION: measurements[MeasurementType.POSITION]},
        {MeasurementType.POSITION: measurement_functions[MeasurementType.POSITION]},
        {MeasurementType.POSITION: sensor_reliabilities[MeasurementType.POSITION]}
    )
    
    print(f"Multi-sensor fusion completed")
    
    # Adaptive parameter tuning
    print(f"\nPerforming adaptive parameter tuning...")
    estimation_system.adaptive_parameter_tuning()
    
    # Performance summary
    summary = estimation_system.get_estimation_summary()
    
    print(f"\nEstimation System Performance Summary:")
    
    if 'system_performance' in summary:
        sp = summary['system_performance']
        print(f"  Average consistency: {sp['average_consistency']:.4f}")
        print(f"  Average computation time: {sp['average_computation_time_s']*1000:.2f} ms")
        print(f"  Accuracy target met: {'✓' if sp['accuracy_target_met'] else '✗'}")
        print(f"  Efficiency target met: {'✓' if sp['efficiency_target_met'] else '✗'}")
    
    if 'active_algorithm' in summary:
        aa = summary['active_algorithm']
        print(f"  Active algorithm: {aa['type']}")
        print(f"  Algorithm switches: {aa['algorithm_switches']}")
    
    # Calculate final estimation accuracy
    final_errors = [result['estimation_error'] for result in estimation_results[-20:]]
    avg_final_error = np.mean(final_errors)
    
    print(f"\nFinal Estimation Performance:")
    print(f"  Average estimation error: {avg_final_error:.2e} m")
    print(f"  Final position error: {estimation_results[-1]['estimation_error']:.2e} m")
    print(f"  Estimation target: {ESTIMATION_ACCURACY_TARGET:.2f}")
    
    success = avg_final_error < 1e-9  # 1 nm accuracy target
    print(f"  Estimation success: {'✓' if success else '✗'}")
    
    print(f"\nBayesian state estimation demonstration complete!")
