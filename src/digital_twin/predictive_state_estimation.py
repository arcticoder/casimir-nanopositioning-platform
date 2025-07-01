"""
Predictive State Estimation with Multi-Physics Kalman Filtering
Advanced Digital Twin Framework for Casimir Nanopositioning Platform

Implements adaptive Kalman filtering with:
1. Multi-physics state estimation across mechanical, thermal, electromagnetic, quantum domains
2. Innovation-based adaptation for dynamic model correction
3. Adaptive process noise with Fisher information regularization
4. Real-time parameter learning and model updating

Mathematical Foundation:
Enhanced Kalman Update:
x̂(k+1|k) = A_adaptive(k)x̂(k|k) + B(k)u(k) + Γ(k)w(k)

Innovation-Based Adaptation:
ν(k) = y(k) - C(k)x̂(k|k-1)
S(k) = C(k)P(k|k-1)C(k)ᵀ + R_adaptive(k)

Adaptive Process Noise:
Q_adaptive(k) = Q₀ + α∇_Q[L(θ,Q)] + β∑ᵢ₌₁ᴺ K(k-i)ν(k-i)ν(k-i)ᵀK(k-i)ᵀ

Model Correction:
θ̂(k+1) = θ̂(k) + μ∇_θ||ν(k)||²₂ + λ∇_θ[H(θ,P)]

Where: H(θ,P) = tr(P⁻¹∇²_θL)  [Fisher information regularization]

Author: Predictive State Estimation Team
Version: 1.0.0 (Multi-Physics Adaptive Kalman)
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from collections import deque
import threading
from abc import ABC, abstractmethod

@dataclass
class MultiPhysicsState:
    """Multi-physics state vector representing all domains."""
    # Mechanical state [position, velocity, acceleration, force]
    mechanical_state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # Thermal state [temperature, heat_flux, thermal_expansion]
    thermal_state: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Electromagnetic state [E_field, B_field, voltage, current]
    electromagnetic_state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # Quantum state [coherence, squeezing, entanglement]
    quantum_state: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Combined state vector
    @property
    def combined_state(self) -> np.ndarray:
        return np.concatenate([
            self.mechanical_state,
            self.thermal_state,
            self.electromagnetic_state,
            self.quantum_state
        ])
    
    @combined_state.setter
    def combined_state(self, state: np.ndarray):
        """Set combined state and update individual domain states."""
        self.mechanical_state = state[:4]
        self.thermal_state = state[4:7]
        self.electromagnetic_state = state[7:11]
        self.quantum_state = state[11:14]

@dataclass
class AdaptiveKalmanParams:
    """Parameters for adaptive Kalman filtering."""
    # State dimensions
    n_mechanical: int = 4
    n_thermal: int = 3
    n_electromagnetic: int = 4
    n_quantum: int = 3
    
    @property
    def total_state_dim(self) -> int:
        return self.n_mechanical + self.n_thermal + self.n_electromagnetic + self.n_quantum
    
    # Adaptation parameters
    innovation_memory_length: int = 50  # Number of innovations to remember
    adaptation_rate_alpha: float = 0.01  # Process noise adaptation rate
    adaptation_rate_beta: float = 0.005  # Innovation-based adaptation rate
    parameter_learning_rate_mu: float = 0.001  # Parameter learning rate
    fisher_regularization_lambda: float = 0.1   # Fisher information weight
    
    # Initial uncertainties
    initial_mechanical_variance: float = 1e-12
    initial_thermal_variance: float = 1e-6
    initial_electromagnetic_variance: float = 1e-9
    initial_quantum_variance: float = 1e-15
    
    # Process noise base levels
    base_process_noise_mechanical: float = 1e-15
    base_process_noise_thermal: float = 1e-9
    base_process_noise_electromagnetic: float = 1e-12
    base_process_noise_quantum: float = 1e-18
    
    # Measurement noise levels
    measurement_noise_mechanical: float = 1e-12
    measurement_noise_thermal: float = 1e-6
    measurement_noise_electromagnetic: float = 1e-9
    measurement_noise_quantum: float = 1e-15

class AdaptiveSystemModel:
    """Adaptive system model for multi-physics dynamics."""
    
    def __init__(self, params: AdaptiveKalmanParams):
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Model parameters (learnable)
        self.model_parameters = {
            'mechanical_damping': 0.1,
            'mechanical_stiffness': 1e6,
            'thermal_conductivity': 100.0,
            'electromagnetic_coupling': 0.5,
            'quantum_decoherence_rate': 1e6
        }
        
        # Parameter gradients (for learning)
        self.parameter_gradients = {key: 0.0 for key in self.model_parameters.keys()}
        
    def get_adaptive_state_matrix(self, dt: float, current_params: Dict[str, float]) -> np.ndarray:
        """
        Compute adaptive state transition matrix A_adaptive(k).
        
        Args:
            dt: Time step
            current_params: Current model parameters
            
        Returns:
            Adaptive state transition matrix
        """
        try:
            n = self.params.total_state_dim
            A = np.eye(n)
            
            # Mechanical dynamics block [position, velocity, acceleration, force]
            damping = current_params.get('mechanical_damping', 0.1)
            stiffness = current_params.get('mechanical_stiffness', 1e6)
            
            # Mechanical state transition (position, velocity, acceleration, force)
            A[0, 1] = dt  # position += velocity * dt
            A[1, 2] = dt  # velocity += acceleration * dt
            A[2, 2] = 1 - damping * dt  # damped acceleration
            A[2, 3] = dt / 1e-12  # force to acceleration (assuming mass = 1e-12 kg)
            
            # Thermal dynamics block [temperature, heat_flux, expansion]
            thermal_cond = current_params.get('thermal_conductivity', 100.0)
            thermal_idx = self.params.n_mechanical
            
            A[thermal_idx, thermal_idx] = 1 - thermal_cond * dt * 1e-3  # temperature decay
            A[thermal_idx, thermal_idx + 1] = dt * 1e-6  # heat flux contribution
            A[thermal_idx + 2, thermal_idx] = dt * 1e-8  # thermal expansion
            
            # Electromagnetic dynamics block [E, B, V, I]
            em_coupling = current_params.get('electromagnetic_coupling', 0.5)
            em_idx = self.params.n_mechanical + self.params.n_thermal
            
            # Simple electromagnetic coupling
            A[em_idx, em_idx + 2] = dt * em_coupling  # E field from voltage
            A[em_idx + 1, em_idx + 3] = dt * em_coupling  # B field from current
            
            # Quantum dynamics block [coherence, squeezing, entanglement]
            decoherence_rate = current_params.get('quantum_decoherence_rate', 1e6)
            quantum_idx = (self.params.n_mechanical + self.params.n_thermal + 
                          self.params.n_electromagnetic)
            
            # Quantum decoherence
            A[quantum_idx, quantum_idx] = np.exp(-decoherence_rate * dt)
            A[quantum_idx + 1, quantum_idx + 1] = np.exp(-decoherence_rate * dt * 0.5)
            A[quantum_idx + 2, quantum_idx + 2] = np.exp(-decoherence_rate * dt * 0.1)
            
            return A
            
        except Exception as e:
            self.logger.error(f"Failed to compute adaptive state matrix: {e}")
            return np.eye(n)
    
    def compute_parameter_gradients(self, innovation: np.ndarray, 
                                  state: np.ndarray, dt: float) -> Dict[str, float]:
        """
        Compute gradients of model parameters with respect to innovation.
        
        ∇_θ||ν(k)||²₂
        """
        try:
            gradients = {}
            
            # Mechanical parameter gradients
            gradients['mechanical_damping'] = -2 * dt * state[2] * innovation[2] if len(innovation) > 2 else 0.0
            gradients['mechanical_stiffness'] = -2 * dt * state[0] * innovation[2] if len(innovation) > 2 else 0.0
            
            # Thermal parameter gradients
            thermal_idx = self.params.n_mechanical
            if len(innovation) > thermal_idx:
                gradients['thermal_conductivity'] = (-2 * dt * 1e-3 * state[thermal_idx] * 
                                                   innovation[thermal_idx])
            else:
                gradients['thermal_conductivity'] = 0.0
            
            # Electromagnetic parameter gradients
            em_idx = self.params.n_mechanical + self.params.n_thermal
            if len(innovation) > em_idx:
                gradients['electromagnetic_coupling'] = (-2 * dt * state[em_idx + 2] * 
                                                       innovation[em_idx])
            else:
                gradients['electromagnetic_coupling'] = 0.0
            
            # Quantum parameter gradients
            quantum_idx = (self.params.n_mechanical + self.params.n_thermal + 
                          self.params.n_electromagnetic)
            if len(innovation) > quantum_idx:
                gradients['quantum_decoherence_rate'] = (-2 * dt * state[quantum_idx] * 
                                                       innovation[quantum_idx])
            else:
                gradients['quantum_decoherence_rate'] = 0.0
            
            return gradients
            
        except Exception as e:
            self.logger.error(f"Failed to compute parameter gradients: {e}")
            return {key: 0.0 for key in self.model_parameters.keys()}

class PredictiveStateEstimator:
    """Predictive state estimator with multi-physics adaptive Kalman filtering."""
    
    def __init__(self, params: AdaptiveKalmanParams):
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Initialize system model
        self.system_model = AdaptiveSystemModel(params)
        
        # State estimation
        self.state = MultiPhysicsState()
        self.covariance = self._initialize_covariance()
        
        # Adaptive components
        self.innovation_history = deque(maxlen=params.innovation_memory_length)
        self.kalman_gain_history = deque(maxlen=params.innovation_memory_length)
        
        # Process and measurement noise (adaptive)
        self.process_noise = self._initialize_process_noise()
        self.measurement_noise = self._initialize_measurement_noise()
        
        # Performance tracking
        self.estimation_errors = []
        self.innovation_norms = []
        self.computation_times = []
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _initialize_covariance(self) -> np.ndarray:
        """Initialize state covariance matrix."""
        n = self.params.total_state_dim
        P = np.zeros((n, n))
        
        # Mechanical block
        mech_vars = [self.params.initial_mechanical_variance] * self.params.n_mechanical
        P[:self.params.n_mechanical, :self.params.n_mechanical] = np.diag(mech_vars)
        
        # Thermal block
        start_idx = self.params.n_mechanical
        end_idx = start_idx + self.params.n_thermal
        therm_vars = [self.params.initial_thermal_variance] * self.params.n_thermal
        P[start_idx:end_idx, start_idx:end_idx] = np.diag(therm_vars)
        
        # Electromagnetic block
        start_idx = end_idx
        end_idx = start_idx + self.params.n_electromagnetic
        em_vars = [self.params.initial_electromagnetic_variance] * self.params.n_electromagnetic
        P[start_idx:end_idx, start_idx:end_idx] = np.diag(em_vars)
        
        # Quantum block
        start_idx = end_idx
        quantum_vars = [self.params.initial_quantum_variance] * self.params.n_quantum
        P[start_idx:, start_idx:] = np.diag(quantum_vars)
        
        return P
    
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize process noise covariance matrix."""
        n = self.params.total_state_dim
        Q = np.zeros((n, n))
        
        # Base process noise levels for each domain
        noise_levels = (
            [self.params.base_process_noise_mechanical] * self.params.n_mechanical +
            [self.params.base_process_noise_thermal] * self.params.n_thermal +
            [self.params.base_process_noise_electromagnetic] * self.params.n_electromagnetic +
            [self.params.base_process_noise_quantum] * self.params.n_quantum
        )
        
        Q = np.diag(noise_levels)
        return Q
    
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize measurement noise covariance matrix."""
        n = self.params.total_state_dim
        R = np.zeros((n, n))
        
        # Measurement noise levels for each domain
        noise_levels = (
            [self.params.measurement_noise_mechanical] * self.params.n_mechanical +
            [self.params.measurement_noise_thermal] * self.params.n_thermal +
            [self.params.measurement_noise_electromagnetic] * self.params.n_electromagnetic +
            [self.params.measurement_noise_quantum] * self.params.n_quantum
        )
        
        R = np.diag(noise_levels)
        return R
    
    def predict(self, control_input: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of adaptive Kalman filter.
        
        x̂(k+1|k) = A_adaptive(k)x̂(k|k) + B(k)u(k) + Γ(k)w(k)
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Get adaptive state transition matrix
                A_adaptive = self.system_model.get_adaptive_state_matrix(
                    dt, self.system_model.model_parameters)
                
                # Control input matrix (simplified)
                B = self._get_control_matrix()
                
                # Predict state
                x_pred = A_adaptive @ self.state.combined_state
                if control_input is not None and len(control_input) > 0:
                    x_pred += B @ control_input
                
                # Predict covariance with adaptive process noise
                P_pred = A_adaptive @ self.covariance @ A_adaptive.T + self.process_noise
                
                # Ensure positive definiteness
                P_pred = self._ensure_positive_definite(P_pred)
                
                computation_time = time.time() - start_time
                self.computation_times.append(computation_time)
                
                self.logger.debug(f"Prediction completed in {computation_time:.6f}s")
                return x_pred, P_pred
                
            except Exception as e:
                self.logger.error(f"Prediction step failed: {e}")
                return self.state.combined_state, self.covariance
    
    def update(self, measurement: np.ndarray, predicted_state: np.ndarray,
              predicted_covariance: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of adaptive Kalman filter with innovation-based adaptation.
        
        Innovation: ν(k) = y(k) - C(k)x̂(k|k-1)
        """
        with self._lock:
            try:
                # Measurement matrix (assuming direct measurement of all states)
                C = np.eye(len(measurement), self.params.total_state_dim)
                
                # Innovation sequence
                innovation = measurement - C @ predicted_state
                
                # Innovation covariance with adaptive measurement noise
                S = C @ predicted_covariance @ C.T + self._get_adaptive_measurement_noise()
                
                # Kalman gain
                try:
                    S_inv = la.inv(S)
                    K = predicted_covariance @ C.T @ S_inv
                except la.LinAlgError:
                    # Use pseudo-inverse if S is singular
                    K = predicted_covariance @ C.T @ la.pinv(S)
                
                # Updated state estimate
                x_updated = predicted_state + K @ innovation
                
                # Updated covariance (Joseph form for numerical stability)
                I_KC = np.eye(self.params.total_state_dim) - K @ C
                P_updated = I_KC @ predicted_covariance @ I_KC.T + K @ self.measurement_noise @ K.T
                
                # Store innovation and Kalman gain for adaptation
                self.innovation_history.append(innovation.copy())
                self.kalman_gain_history.append(K.copy())
                
                # Adaptive updates
                self._update_adaptive_process_noise(innovation, K, dt)
                self._update_model_parameters(innovation, predicted_state, dt)
                
                # Update internal state
                self.state.combined_state = x_updated
                self.covariance = P_updated
                
                # Track performance
                innovation_norm = la.norm(innovation)
                self.innovation_norms.append(innovation_norm)
                
                self.logger.debug(f"Update completed, innovation norm: {innovation_norm:.6e}")
                return x_updated, P_updated
                
            except Exception as e:
                self.logger.error(f"Update step failed: {e}")
                return predicted_state, predicted_covariance
    
    def _get_control_matrix(self) -> np.ndarray:
        """Get control input matrix B(k)."""
        # Simplified control matrix (force input affects mechanical acceleration)
        B = np.zeros((self.params.total_state_dim, 1))
        B[2, 0] = 1.0 / 1e-12  # Force to acceleration conversion (mass = 1e-12 kg)
        return B
    
    def _get_adaptive_measurement_noise(self) -> np.ndarray:
        """Get adaptive measurement noise covariance R_adaptive(k)."""
        # Start with base measurement noise
        R_adaptive = self.measurement_noise.copy()
        
        # Adapt based on recent innovation statistics
        if len(self.innovation_history) > 5:
            recent_innovations = np.array(list(self.innovation_history)[-5:])
            innovation_variance = np.var(recent_innovations, axis=0)
            
            # Increase measurement noise if innovations are large
            adaptation_factor = 1.0 + 0.1 * np.sqrt(innovation_variance / np.diag(self.measurement_noise))
            R_adaptive *= np.diag(adaptation_factor)
        
        return R_adaptive
    
    def _update_adaptive_process_noise(self, innovation: np.ndarray, 
                                     kalman_gain: np.ndarray, dt: float) -> None:
        """
        Update adaptive process noise covariance.
        
        Q_adaptive(k) = Q₀ + α∇_Q[L(θ,Q)] + β∑ᵢ₌₁ᴺ K(k-i)ν(k-i)ν(k-i)ᵀK(k-i)ᵀ
        """
        try:
            alpha = self.params.adaptation_rate_alpha
            beta = self.params.adaptation_rate_beta
            
            # Innovation-based adaptation term
            innovation_contribution = kalman_gain @ np.outer(innovation, innovation) @ kalman_gain.T
            
            # Historical innovation contribution
            historical_contribution = np.zeros_like(self.process_noise)
            if len(self.innovation_history) > 1 and len(self.kalman_gain_history) > 1:
                for i, (past_innovation, past_gain) in enumerate(zip(
                    list(self.innovation_history)[:-1], 
                    list(self.kalman_gain_history)[:-1]
                )):
                    weight = np.exp(-0.1 * i)  # Exponential decay for older innovations
                    historical_contribution += (weight * past_gain @ 
                                              np.outer(past_innovation, past_innovation) @ past_gain.T)
            
            # Update process noise
            self.process_noise += alpha * innovation_contribution + beta * historical_contribution
            
            # Ensure minimum noise levels
            min_noise = 1e-18
            self.process_noise = np.maximum(self.process_noise, min_noise * np.eye(self.process_noise.shape[0]))
            
        except Exception as e:
            self.logger.debug(f"Process noise adaptation failed: {e}")
    
    def _update_model_parameters(self, innovation: np.ndarray, 
                               state: np.ndarray, dt: float) -> None:
        """
        Update model parameters using innovation-based learning.
        
        θ̂(k+1) = θ̂(k) + μ∇_θ||ν(k)||²₂ + λ∇_θ[H(θ,P)]
        """
        try:
            mu = self.params.parameter_learning_rate_mu
            lambda_reg = self.params.fisher_regularization_lambda
            
            # Compute parameter gradients
            gradients = self.system_model.compute_parameter_gradients(innovation, state, dt)
            
            # Fisher information regularization (simplified)
            fisher_term = self._compute_fisher_information_gradient()
            
            # Update parameters
            for param_name, gradient in gradients.items():
                if param_name in self.system_model.model_parameters:
                    fisher_contrib = fisher_term.get(param_name, 0.0)
                    update = mu * gradient + lambda_reg * fisher_contrib
                    
                    self.system_model.model_parameters[param_name] -= update
                    
                    # Ensure parameter bounds
                    self._enforce_parameter_bounds(param_name)
            
        except Exception as e:
            self.logger.debug(f"Parameter update failed: {e}")
    
    def _compute_fisher_information_gradient(self) -> Dict[str, float]:
        """
        Compute Fisher information gradient: ∇_θ[H(θ,P)] = ∇_θ[tr(P⁻¹∇²_θL)]
        """
        # Simplified Fisher information gradient (would need full Hessian in practice)
        fisher_gradients = {}
        
        try:
            # Use diagonal approximation of Fisher information
            P_diag = np.diag(self.covariance)
            
            # Mechanical parameters
            fisher_gradients['mechanical_damping'] = 1.0 / (P_diag[2] + 1e-12)
            fisher_gradients['mechanical_stiffness'] = 1.0 / (P_diag[0] + 1e-12)
            
            # Other parameters (simplified)
            fisher_gradients['thermal_conductivity'] = 1.0 / (P_diag[self.params.n_mechanical] + 1e-12)
            fisher_gradients['electromagnetic_coupling'] = 1e-6
            fisher_gradients['quantum_decoherence_rate'] = 1e-12
            
        except Exception as e:
            self.logger.debug(f"Fisher information computation failed: {e}")
            fisher_gradients = {key: 0.0 for key in self.system_model.model_parameters.keys()}
        
        return fisher_gradients
    
    def _enforce_parameter_bounds(self, param_name: str) -> None:
        """Enforce physical bounds on parameters."""
        bounds = {
            'mechanical_damping': (0.001, 10.0),
            'mechanical_stiffness': (1e3, 1e9),
            'thermal_conductivity': (0.1, 1000.0),
            'electromagnetic_coupling': (0.01, 10.0),
            'quantum_decoherence_rate': (1e3, 1e9)
        }
        
        if param_name in bounds:
            min_val, max_val = bounds[param_name]
            current_val = self.system_model.model_parameters[param_name]
            self.system_model.model_parameters[param_name] = np.clip(current_val, min_val, max_val)
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite."""
        try:
            eigenvals, eigenvecs = la.eigh(matrix)
            eigenvals_clipped = np.maximum(eigenvals, 1e-15)
            return eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
        except Exception:
            return np.eye(matrix.shape[0]) * 1e-12
    
    def get_estimation_performance(self) -> Dict[str, Any]:
        """Get estimation performance metrics."""
        return {
            'average_computation_time': np.mean(self.computation_times) if self.computation_times else 0.0,
            'recent_innovation_norm': self.innovation_norms[-1] if self.innovation_norms else 0.0,
            'average_innovation_norm': np.mean(self.innovation_norms) if self.innovation_norms else 0.0,
            'current_model_parameters': self.system_model.model_parameters.copy(),
            'state_uncertainty': np.sqrt(np.diag(self.covariance)),
            'effective_rank': np.trace(self.covariance) / np.max(np.diag(self.covariance))
        }

def main():
    """Demonstration of predictive state estimation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Predictive State Estimation Demonstration")
    
    # Initialize estimator
    params = AdaptiveKalmanParams()
    estimator = PredictiveStateEstimator(params)
    
    # Simulation parameters
    dt = 1e-6  # 1 microsecond time step
    n_steps = 1000
    
    print(f"\n📊 ADAPTIVE KALMAN FILTER STATUS:")
    print(f"   Total State Dimension:   {params.total_state_dim}")
    print(f"   Time Step:              {dt*1e6:.1f} μs")
    print(f"   Simulation Steps:       {n_steps}")
    
    # Simulate estimation process
    for k in range(n_steps):
        # Generate synthetic control input
        control_input = np.array([1e-12 * np.sin(2 * np.pi * k * dt * 1000)])  # 1 kHz force
        
        # Prediction step
        x_pred, P_pred = estimator.predict(control_input, dt)
        
        # Generate synthetic measurement (with noise)
        true_state = x_pred + np.random.multivariate_normal(
            np.zeros(params.total_state_dim), estimator.process_noise)
        measurement = true_state + np.random.multivariate_normal(
            np.zeros(params.total_state_dim), estimator.measurement_noise)
        
        # Update step
        x_updated, P_updated = estimator.update(measurement, x_pred, P_pred, dt)
        
        # Log progress
        if k % 100 == 0:
            performance = estimator.get_estimation_performance()
            logger.info(f"Step {k}: Innovation norm = {performance['recent_innovation_norm']:.2e}")
    
    # Final performance analysis
    final_performance = estimator.get_estimation_performance()
    
    print(f"\n📈 ESTIMATION PERFORMANCE:")
    print(f"   Average Computation Time: {final_performance['average_computation_time']*1e6:.1f} μs")
    print(f"   Average Innovation Norm:  {final_performance['average_innovation_norm']:.2e}")
    print(f"   Effective Rank:          {final_performance['effective_rank']:.2f}")
    
    print(f"\n🔧 LEARNED MODEL PARAMETERS:")
    for param, value in final_performance['current_model_parameters'].items():
        print(f"   {param:25s}: {value:.3e}")
    
    print(f"\n📊 STATE UNCERTAINTIES:")
    uncertainties = final_performance['state_uncertainty']
    state_names = (['mech_pos', 'mech_vel', 'mech_acc', 'mech_force'] +
                  ['temp', 'heat_flux', 'expansion'] +
                  ['E_field', 'B_field', 'voltage', 'current'] +
                  ['coherence', 'squeezing', 'entanglement'])
    
    for name, uncertainty in zip(state_names, uncertainties):
        print(f"   {name:15s}: ±{uncertainty:.2e}")
    
    print(f"\n✅ Predictive State Estimation Successfully Demonstrated")

if __name__ == "__main__":
    main()
