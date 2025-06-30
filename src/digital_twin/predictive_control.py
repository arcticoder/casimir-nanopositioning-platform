"""
Predictive Control Module
========================

This module implements advanced predictive control techniques for the digital twin,
including model predictive control (MPC), receding horizon control, and failure
prediction with optimal control under uncertainty.

Mathematical Formulation:

Model Predictive Control:
min J = Σₖ₌₀ᴺ⁻¹ [‖y(k) - r(k)‖²Q + ‖u(k)‖²R] + ‖x(N)‖²P

Subject to:
x(k+1) = f(x(k), u(k), w(k))
y(k) = h(x(k), v(k))
u_min ≤ u(k) ≤ u_max
x_min ≤ x(k) ≤ x_max

Stochastic MPC with Uncertainty:
min E[J] subject to Pr[constraints violated] ≤ ε

Robust MPC:
min max J subject to constraints ∀ w ∈ W
"""

import numpy as np
from scipy import optimize, linalg
from scipy.stats import multivariate_normal
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Callable, Union, NamedTuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
from abc import ABC, abstractmethod
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Performance targets for predictive control
MPC_SOLVE_TIME_TARGET = 0.001  # seconds
MPC_TRACKING_ERROR_TARGET = 1e-9  # meters
MPC_CONSTRAINT_VIOLATION_PROB = 0.05  # 5% max violation probability

class MPCType(Enum):
    """Types of model predictive control."""
    LINEAR_MPC = "linear_mpc"
    NONLINEAR_MPC = "nonlinear_mpc"
    STOCHASTIC_MPC = "stochastic_mpc"
    ROBUST_MPC = "robust_mpc"
    ADAPTIVE_MPC = "adaptive_mpc"
    DISTRIBUTED_MPC = "distributed_mpc"

class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"
    SOFT = "soft"
    CHANCE = "chance"
    ROBUST = "robust"

@dataclass
class MPCParameters:
    """Parameters for model predictive control."""
    
    # Horizon parameters
    prediction_horizon: int = 20
    control_horizon: int = 10
    sample_time: float = 1e-3  # 1 ms
    
    # Weighting matrices
    Q_weight: np.ndarray = field(default_factory=lambda: np.eye(6))  # State weights
    R_weight: np.ndarray = field(default_factory=lambda: np.eye(3))  # Control weights
    P_weight: Optional[np.ndarray] = None  # Terminal state weights
    
    # Constraints
    u_min: np.ndarray = field(default_factory=lambda: np.array([-1e-6, -1e-6, -1e-6]))  # Control limits
    u_max: np.ndarray = field(default_factory=lambda: np.array([1e-6, 1e-6, 1e-6]))
    x_min: Optional[np.ndarray] = None  # State limits
    x_max: Optional[np.ndarray] = None
    
    # Stochastic MPC parameters
    chance_constraint_probability: float = 0.95
    uncertainty_samples: int = 1000
    
    # Robust MPC parameters
    uncertainty_set_type: str = "ellipsoidal"  # ellipsoidal, polytopic
    robustness_level: float = 0.1
    
    # Solver parameters
    solver: str = "OSQP"  # OSQP, ECOS, SCS
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Failure prediction parameters
    failure_prediction_horizon: int = 100
    failure_threshold: float = 1e-8  # Failure threshold for tracking error
    
    def __post_init__(self):
        if self.P_weight is None:
            # Terminal weight as solution to discrete algebraic Riccati equation
            self.P_weight = self.Q_weight.copy()

class MPCResult(NamedTuple):
    """Result of MPC optimization."""
    optimal_control: np.ndarray
    predicted_trajectory: np.ndarray
    cost: float
    solve_time: float
    solver_status: str
    constraint_violations: int
    feasible: bool

class PredictiveController(ABC):
    """Abstract base class for predictive controllers."""
    
    def __init__(self, state_size: int, control_size: int, 
                 params: MPCParameters):
        self.state_size = state_size
        self.control_size = control_size
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.solve_times = deque(maxlen=1000)
        self.tracking_errors = deque(maxlen=1000)
        self.constraint_violations = deque(maxlen=1000)
        
        # Prediction history
        self.prediction_history = deque(maxlen=100)
        
    @abstractmethod
    def solve(self, current_state: np.ndarray, reference: np.ndarray,
              dynamics_function: Callable) -> MPCResult:
        """Solve MPC optimization problem."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get controller performance metrics."""
        if not self.solve_times:
            return {'status': 'no_data'}
        
        avg_solve_time = np.mean(list(self.solve_times))
        max_solve_time = np.max(list(self.solve_times))
        
        avg_tracking_error = 0.0
        if self.tracking_errors:
            avg_tracking_error = np.mean(list(self.tracking_errors))
        
        constraint_violation_rate = 0.0
        if self.constraint_violations:
            constraint_violation_rate = np.mean(list(self.constraint_violations))
        
        return {
            'avg_solve_time_s': avg_solve_time,
            'max_solve_time_s': max_solve_time,
            'avg_tracking_error_m': avg_tracking_error,
            'constraint_violation_rate': constraint_violation_rate,
            'solve_time_satisfied': avg_solve_time <= MPC_SOLVE_TIME_TARGET,
            'tracking_satisfied': avg_tracking_error <= MPC_TRACKING_ERROR_TARGET,
            'constraint_satisfied': constraint_violation_rate <= MPC_CONSTRAINT_VIOLATION_PROB,
            'n_solutions': len(self.solve_times)
        }

class LinearMPCController(PredictiveController):
    """
    Linear Model Predictive Controller.
    
    Mathematical Implementation:
    
    Discrete-time linear system:
    x(k+1) = A×x(k) + B×u(k) + w(k)
    y(k) = C×x(k) + v(k)
    
    Quadratic cost function:
    J = Σₖ₌₀ᴺ⁻¹ [‖y(k) - r(k)‖²Q + ‖u(k)‖²R] + ‖x(N)‖²P
    
    QP formulation:
    min 0.5×z^T×H×z + f^T×z
    s.t. A_ineq×z ≤ b_ineq
         A_eq×z = b_eq
    """
    
    def __init__(self, state_size: int, control_size: int, 
                 params: MPCParameters, A: np.ndarray, B: np.ndarray, 
                 C: Optional[np.ndarray] = None):
        """
        Initialize Linear MPC controller.
        
        Args:
            state_size: State vector dimension
            control_size: Control vector dimension
            params: MPC parameters
            A: State transition matrix
            B: Control input matrix
            C: Output matrix (if None, assume full state feedback)
        """
        super().__init__(state_size, control_size, params)
        
        self.A = A
        self.B = B
        self.C = C if C is not None else np.eye(state_size)
        self.output_size = self.C.shape[0]
        
        # Pre-compute matrices for efficiency
        self._setup_qp_matrices()
        
    def _setup_qp_matrices(self):
        """Pre-compute matrices for QP formulation."""
        N = self.params.prediction_horizon
        M = self.params.control_horizon
        
        # Prediction matrices
        self.Phi = np.zeros((N * self.output_size, self.state_size))
        self.Gamma = np.zeros((N * self.output_size, M * self.control_size))
        
        # Build prediction matrices
        A_power = np.eye(self.state_size)
        
        for i in range(N):
            # Phi matrix (state influence)
            self.Phi[i*self.output_size:(i+1)*self.output_size, :] = self.C @ A_power
            
            # Gamma matrix (control influence)
            AB_sum = np.zeros((self.state_size, self.control_size))
            A_temp = np.eye(self.state_size)
            
            for j in range(min(i+1, M)):
                AB_sum += A_temp @ self.B
                A_temp = A_temp @ self.A
            
            self.Gamma[i*self.output_size:(i+1)*self.output_size, 
                      :self.control_size] = self.C @ AB_sum
            
            # For control horizon
            for j in range(1, min(i+1, M)):
                A_temp = np.eye(self.state_size)
                for k in range(i-j):
                    A_temp = A_temp @ self.A
                
                self.Gamma[i*self.output_size:(i+1)*self.output_size,
                          j*self.control_size:(j+1)*self.control_size] = self.C @ A_temp @ self.B
            
            A_power = A_power @ self.A
        
        # Build cost matrices
        Q_tilde = np.kron(np.eye(N), self.params.Q_weight[:self.output_size, :self.output_size])
        R_tilde = np.kron(np.eye(M), self.params.R_weight)
        
        # QP matrices
        self.H = 2 * (self.Gamma.T @ Q_tilde @ self.Gamma + R_tilde)
        
        # Ensure H is positive definite
        eigvals = np.linalg.eigvals(self.H)
        if np.min(eigvals) <= 0:
            self.H += np.eye(self.H.shape[0]) * (1e-6 - np.min(eigvals))
    
    def solve(self, current_state: np.ndarray, reference: np.ndarray,
              dynamics_function: Optional[Callable] = None) -> MPCResult:
        """
        Solve linear MPC optimization problem.
        
        Args:
            current_state: Current state vector
            reference: Reference trajectory
            dynamics_function: Not used for linear MPC
            
        Returns:
            MPC optimization result
        """
        start_time = time.time()
        
        N = self.params.prediction_horizon
        M = self.params.control_horizon
        
        # Reference vector
        if reference.ndim == 1:
            # Constant reference
            r_vec = np.tile(reference, N)
        else:
            # Time-varying reference
            r_vec = reference[:N * self.output_size].flatten()
        
        # Linear term in QP
        Q_tilde = np.kron(np.eye(N), self.params.Q_weight[:self.output_size, :self.output_size])
        f = -2 * self.Gamma.T @ Q_tilde @ (r_vec - self.Phi @ current_state)
        
        # Setup constraints
        A_ineq, b_ineq = self._setup_constraints(current_state)
        
        # Solve QP using CVXPY
        try:
            u_var = cp.Variable(M * self.control_size)
            
            # Objective
            objective = cp.Minimize(0.5 * cp.quad_form(u_var, self.H) + f.T @ u_var)
            
            # Constraints
            constraints = []
            if A_ineq.size > 0:
                constraints.append(A_ineq @ u_var <= b_ineq)
            
            # Create and solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=self.params.solver, 
                         max_iters=self.params.max_iterations,
                         eps=self.params.tolerance)
            
            solve_time = time.time() - start_time
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                optimal_control = u_var.value
                predicted_trajectory = self._predict_trajectory(current_state, optimal_control)
                
                # Calculate tracking error
                if reference.ndim == 1:
                    tracking_error = np.linalg.norm(predicted_trajectory[0, :self.output_size] - reference)
                else:
                    tracking_error = np.linalg.norm(predicted_trajectory[0, :self.output_size] - reference[0])
                
                result = MPCResult(
                    optimal_control=optimal_control[:self.control_size],  # Only first control
                    predicted_trajectory=predicted_trajectory,
                    cost=problem.value,
                    solve_time=solve_time,
                    solver_status=problem.status,
                    constraint_violations=0,  # Would need to check explicitly
                    feasible=True
                )
                
                # Update performance tracking
                self.solve_times.append(solve_time)
                self.tracking_errors.append(tracking_error)
                self.constraint_violations.append(0)
                
                return result
            
            else:
                # Infeasible or error
                self.logger.warning(f"MPC solver failed with status: {problem.status}")
                
                result = MPCResult(
                    optimal_control=np.zeros(self.control_size),
                    predicted_trajectory=np.zeros((N, self.state_size)),
                    cost=float('inf'),
                    solve_time=solve_time,
                    solver_status=problem.status,
                    constraint_violations=0,
                    feasible=False
                )
                
                self.solve_times.append(solve_time)
                self.tracking_errors.append(float('inf'))
                self.constraint_violations.append(1)
                
                return result
                
        except Exception as e:
            solve_time = time.time() - start_time
            self.logger.error(f"MPC solver exception: {e}")
            
            result = MPCResult(
                optimal_control=np.zeros(self.control_size),
                predicted_trajectory=np.zeros((N, self.state_size)),
                cost=float('inf'),
                solve_time=solve_time,
                solver_status="ERROR",
                constraint_violations=0,
                feasible=False
            )
            
            self.solve_times.append(solve_time)
            return result
    
    def _setup_constraints(self, current_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Setup inequality constraints for QP."""
        M = self.params.control_horizon
        
        # Control constraints: u_min ≤ u(k) ≤ u_max
        A_u = np.vstack([
            np.kron(np.eye(M), np.eye(self.control_size)),      # u ≤ u_max
            np.kron(np.eye(M), -np.eye(self.control_size))      # -u ≤ -u_min
        ])
        
        b_u = np.hstack([
            np.tile(self.params.u_max, M),
            np.tile(-self.params.u_min, M)
        ])
        
        # State constraints (if specified)
        if self.params.x_min is not None and self.params.x_max is not None:
            # Would need to add state constraints here
            # This requires additional matrices relating control to state evolution
            pass
        
        return A_u, b_u
    
    def _predict_trajectory(self, initial_state: np.ndarray, 
                           control_sequence: np.ndarray) -> np.ndarray:
        """Predict state trajectory given control sequence."""
        N = self.params.prediction_horizon
        M = self.params.control_horizon
        
        trajectory = np.zeros((N, self.state_size))
        state = initial_state.copy()
        
        for k in range(N):
            trajectory[k, :] = state
            
            # Apply control (repeat last control if beyond control horizon)
            if k < M:
                control = control_sequence[k*self.control_size:(k+1)*self.control_size]
            else:
                control = control_sequence[(M-1)*self.control_size:M*self.control_size]
            
            # Update state
            state = self.A @ state + self.B @ control
        
        return trajectory

class StochasticMPCController(PredictiveController):
    """
    Stochastic Model Predictive Controller with chance constraints.
    
    Mathematical Implementation:
    
    Stochastic system:
    x(k+1) = f(x(k), u(k)) + w(k)
    
    Chance constraints:
    Pr[g(x(k), u(k)) ≤ 0] ≥ 1 - ε
    
    Scenario-based approach:
    Sample N scenarios {w₁, w₂, ..., wₙ}
    Solve robust optimization over all scenarios
    """
    
    def __init__(self, state_size: int, control_size: int, 
                 params: MPCParameters):
        super().__init__(state_size, control_size, params)
        
        # Uncertainty model
        self.uncertainty_mean = np.zeros(state_size)
        self.uncertainty_cov = np.eye(state_size) * 1e-12
        
    def solve(self, current_state: np.ndarray, reference: np.ndarray,
              dynamics_function: Callable) -> MPCResult:
        """
        Solve stochastic MPC with chance constraints.
        
        Args:
            current_state: Current state vector
            reference: Reference trajectory
            dynamics_function: Nonlinear dynamics function
            
        Returns:
            Stochastic MPC result
        """
        start_time = time.time()
        
        N = self.params.prediction_horizon
        M = self.params.control_horizon
        
        # Generate uncertainty scenarios
        uncertainty_samples = np.random.multivariate_normal(
            self.uncertainty_mean, 
            self.uncertainty_cov, 
            self.params.uncertainty_samples
        )
        
        # Setup optimization variables
        try:
            u_vars = [cp.Variable(self.control_size) for _ in range(M)]
            
            # Scenario variables
            x_scenarios = {}
            for i in range(self.params.uncertainty_samples):
                x_scenarios[i] = [cp.Variable(self.state_size) for _ in range(N+1)]
                x_scenarios[i][0].value = current_state
            
            # Objective function
            cost = 0
            for i in range(self.params.uncertainty_samples):
                scenario_cost = 0
                for k in range(N):
                    if reference.ndim == 1:
                        ref_k = reference
                    else:
                        ref_k = reference[k] if k < len(reference) else reference[-1]
                    
                    # State tracking cost
                    state_error = x_scenarios[i][k+1] - ref_k
                    scenario_cost += cp.quad_form(state_error, self.params.Q_weight)
                    
                    # Control cost
                    if k < M:
                        scenario_cost += cp.quad_form(u_vars[k], self.params.R_weight)
                
                cost += scenario_cost
            
            cost = cost / self.params.uncertainty_samples
            
            # Constraints
            constraints = []
            
            # Dynamics constraints for each scenario
            for i in range(self.params.uncertainty_samples):
                for k in range(N):
                    if k < M:
                        control_k = u_vars[k]
                    else:
                        control_k = u_vars[M-1]  # Hold last control
                    
                    # Linearized dynamics around current state
                    x_nom = x_scenarios[i][k]
                    u_nom = control_k
                    
                    # Simple linearization (in practice, would be more sophisticated)
                    A_lin = np.eye(self.state_size)  # Placeholder
                    B_lin = np.eye(self.state_size, self.control_size)  # Placeholder
                    
                    # Dynamics constraint with uncertainty
                    constraints.append(
                        x_scenarios[i][k+1] == A_lin @ x_scenarios[i][k] + 
                        B_lin @ control_k + uncertainty_samples[i]
                    )
            
            # Control constraints
            for k in range(M):
                constraints.append(u_vars[k] >= self.params.u_min)
                constraints.append(u_vars[k] <= self.params.u_max)
            
            # Chance constraints (approximated by scenario constraints)
            violation_threshold = int(self.params.uncertainty_samples * 
                                    (1 - self.params.chance_constraint_probability))
            
            # This is a simplified implementation
            # In practice, chance constraints would be handled more rigorously
            
            # Solve problem
            objective = cp.Minimize(cost)
            problem = cp.Problem(objective, constraints)
            
            problem.solve(solver=self.params.solver,
                         max_iters=self.params.max_iterations)
            
            solve_time = time.time() - start_time
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                optimal_control = u_vars[0].value
                
                # Predict nominal trajectory
                predicted_trajectory = self._predict_nominal_trajectory(
                    current_state, [u.value for u in u_vars], dynamics_function
                )
                
                result = MPCResult(
                    optimal_control=optimal_control,
                    predicted_trajectory=predicted_trajectory,
                    cost=problem.value,
                    solve_time=solve_time,
                    solver_status=problem.status,
                    constraint_violations=0,
                    feasible=True
                )
                
                self.solve_times.append(solve_time)
                return result
            
            else:
                result = MPCResult(
                    optimal_control=np.zeros(self.control_size),
                    predicted_trajectory=np.zeros((N, self.state_size)),
                    cost=float('inf'),
                    solve_time=solve_time,
                    solver_status=problem.status,
                    constraint_violations=0,
                    feasible=False
                )
                
                self.solve_times.append(solve_time)
                return result
                
        except Exception as e:
            solve_time = time.time() - start_time
            self.logger.error(f"Stochastic MPC solver exception: {e}")
            
            result = MPCResult(
                optimal_control=np.zeros(self.control_size),
                predicted_trajectory=np.zeros((N, self.state_size)),
                cost=float('inf'),
                solve_time=solve_time,
                solver_status="ERROR",
                constraint_violations=0,
                feasible=False
            )
            
            self.solve_times.append(solve_time)
            return result
    
    def _predict_nominal_trajectory(self, initial_state: np.ndarray,
                                  control_sequence: List[np.ndarray],
                                  dynamics_function: Callable) -> np.ndarray:
        """Predict nominal trajectory without uncertainty."""
        N = self.params.prediction_horizon
        M = len(control_sequence)
        
        trajectory = np.zeros((N, self.state_size))
        state = initial_state.copy()
        
        for k in range(N):
            trajectory[k, :] = state
            
            # Apply control
            if k < M:
                control = control_sequence[k]
            else:
                control = control_sequence[M-1]
            
            # Update state using dynamics function
            state = dynamics_function(state, control, self.params.sample_time)
        
        return trajectory

class FailurePredictionSystem:
    """
    Failure prediction system for predictive maintenance and safety.
    
    Features:
    1. Multi-step ahead prediction
    2. Failure mode identification
    3. Risk assessment
    4. Preventive action recommendations
    """
    
    def __init__(self, state_size: int, failure_threshold: float = 1e-8):
        """
        Initialize failure prediction system.
        
        Args:
            state_size: State vector dimension
            failure_threshold: Threshold for failure detection
        """
        self.state_size = state_size
        self.failure_threshold = failure_threshold
        self.logger = logging.getLogger(__name__)
        
        # Failure models
        self.failure_models = {}
        
        # Prediction history
        self.prediction_history = deque(maxlen=1000)
        self.failure_events = deque(maxlen=100)
        
    def predict_failure_probability(self, current_state: np.ndarray,
                                  predicted_trajectory: np.ndarray,
                                  uncertainty_bounds: Optional[np.ndarray] = None) -> Dict:
        """
        Predict probability of failure over prediction horizon.
        
        Args:
            current_state: Current system state
            predicted_trajectory: Predicted state trajectory
            uncertainty_bounds: Uncertainty bounds for each prediction step
            
        Returns:
            Failure prediction results
        """
        N = len(predicted_trajectory)
        
        # Initialize failure probabilities
        failure_probs = np.zeros(N)
        failure_modes = []
        
        for k in range(N):
            state_k = predicted_trajectory[k]
            
            # Position-based failure detection
            position_error = np.linalg.norm(state_k[:3])  # Assume first 3 are positions
            position_failure_prob = self._sigmoid_failure_model(
                position_error, self.failure_threshold, sharpness=1e8
            )
            
            # Velocity-based failure detection
            if self.state_size >= 6:
                velocity_magnitude = np.linalg.norm(state_k[3:6])
                velocity_failure_prob = self._sigmoid_failure_model(
                    velocity_magnitude, self.failure_threshold * 1e3, sharpness=1e5
                )
            else:
                velocity_failure_prob = 0.0
            
            # Combined failure probability
            failure_probs[k] = 1 - (1 - position_failure_prob) * (1 - velocity_failure_prob)
            
            # Identify failure modes
            if failure_probs[k] > 0.1:  # 10% threshold
                mode = self._identify_failure_mode(state_k)
                failure_modes.append({
                    'timestep': k,
                    'probability': failure_probs[k],
                    'mode': mode,
                    'state': state_k.copy()
                })
        
        # Time to failure estimate
        time_to_failure = self._estimate_time_to_failure(failure_probs)
        
        # Risk assessment
        risk_level = self._assess_risk_level(failure_probs)
        
        # Preventive actions
        preventive_actions = self._recommend_preventive_actions(
            failure_modes, risk_level
        )
        
        prediction_result = {
            'failure_probabilities': failure_probs,
            'failure_modes': failure_modes,
            'time_to_failure_steps': time_to_failure,
            'time_to_failure_seconds': time_to_failure * 1e-3,  # Assuming 1ms timestep
            'risk_level': risk_level,
            'max_failure_probability': np.max(failure_probs),
            'preventive_actions': preventive_actions,
            'prediction_confidence': self._calculate_prediction_confidence(
                predicted_trajectory, uncertainty_bounds
            )
        }
        
        # Store prediction
        self.prediction_history.append({
            'timestamp': time.time(),
            'current_state': current_state.copy(),
            'prediction': prediction_result
        })
        
        return prediction_result
    
    def _sigmoid_failure_model(self, value: float, threshold: float, 
                              sharpness: float = 1e6) -> float:
        """Sigmoid failure probability model."""
        x = sharpness * (value - threshold)
        return 1 / (1 + np.exp(-x))
    
    def _identify_failure_mode(self, state: np.ndarray) -> str:
        """Identify the type of failure mode."""
        position_error = np.linalg.norm(state[:3])
        
        if self.state_size >= 6:
            velocity_magnitude = np.linalg.norm(state[3:6])
        else:
            velocity_magnitude = 0.0
        
        if position_error > self.failure_threshold:
            if velocity_magnitude > self.failure_threshold * 1e3:
                return "position_velocity_divergence"
            else:
                return "position_drift"
        elif velocity_magnitude > self.failure_threshold * 1e3:
            return "velocity_instability"
        else:
            return "unknown"
    
    def _estimate_time_to_failure(self, failure_probs: np.ndarray) -> int:
        """Estimate time steps until failure occurs."""
        failure_threshold_prob = 0.5  # 50% probability threshold
        
        for k, prob in enumerate(failure_probs):
            if prob >= failure_threshold_prob:
                return k
        
        return len(failure_probs)  # No failure predicted within horizon
    
    def _assess_risk_level(self, failure_probs: np.ndarray) -> str:
        """Assess overall risk level."""
        max_prob = np.max(failure_probs)
        avg_prob = np.mean(failure_probs)
        
        if max_prob > 0.8 or avg_prob > 0.3:
            return "critical"
        elif max_prob > 0.5 or avg_prob > 0.1:
            return "high"
        elif max_prob > 0.2 or avg_prob > 0.05:
            return "medium"
        else:
            return "low"
    
    def _recommend_preventive_actions(self, failure_modes: List[Dict], 
                                    risk_level: str) -> List[str]:
        """Recommend preventive actions based on failure prediction."""
        actions = []
        
        if risk_level == "critical":
            actions.append("immediate_shutdown")
            actions.append("emergency_diagnostic")
        elif risk_level == "high":
            actions.append("reduce_operation_speed")
            actions.append("increase_monitoring_frequency")
            actions.append("schedule_maintenance")
        elif risk_level == "medium":
            actions.append("enhanced_monitoring")
            actions.append("parameter_adjustment")
        
        # Mode-specific actions
        for mode_info in failure_modes:
            mode = mode_info['mode']
            if mode == "position_drift":
                actions.append("recalibrate_position_sensors")
                actions.append("check_thermal_compensation")
            elif mode == "velocity_instability":
                actions.append("adjust_control_gains")
                actions.append("check_actuator_response")
            elif mode == "position_velocity_divergence":
                actions.append("full_system_diagnostic")
                actions.append("controller_reinitialization")
        
        return list(set(actions))  # Remove duplicates
    
    def _calculate_prediction_confidence(self, predicted_trajectory: np.ndarray,
                                       uncertainty_bounds: Optional[np.ndarray]) -> float:
        """Calculate confidence in failure prediction."""
        if uncertainty_bounds is None:
            return 0.5  # Medium confidence without uncertainty information
        
        # Calculate prediction confidence based on uncertainty propagation
        relative_uncertainty = np.mean(uncertainty_bounds / np.abs(predicted_trajectory))
        
        # High uncertainty -> low confidence
        confidence = 1 / (1 + 10 * relative_uncertainty)
        
        return float(np.clip(confidence, 0.0, 1.0))

class PredictiveControlSystem:
    """
    Comprehensive predictive control system with multiple MPC algorithms
    and failure prediction capabilities.
    """
    
    def __init__(self, state_size: int, control_size: int,
                 params: Optional[MPCParameters] = None):
        """
        Initialize predictive control system.
        
        Args:
            state_size: State vector dimension
            control_size: Control vector dimension
            params: MPC parameters
        """
        self.state_size = state_size
        self.control_size = control_size
        self.params = params or MPCParameters()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize controllers (requires system matrices for linear MPC)
        self.controllers = {}
        
        # Initialize failure prediction
        self.failure_predictor = FailurePredictionSystem(
            state_size, self.params.failure_threshold
        )
        
        # Performance tracking
        self.control_history = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=100)
        
        # Active controller
        self.active_controller_type = MPCType.LINEAR_MPC
        self.active_controller = None
        
        self.logger.info("Predictive control system initialized")
    
    def add_linear_controller(self, A: np.ndarray, B: np.ndarray, 
                             C: Optional[np.ndarray] = None):
        """Add linear MPC controller with system matrices."""
        self.controllers[MPCType.LINEAR_MPC] = LinearMPCController(
            self.state_size, self.control_size, self.params, A, B, C
        )
        
        if self.active_controller is None:
            self.active_controller = self.controllers[MPCType.LINEAR_MPC]
            self.active_controller_type = MPCType.LINEAR_MPC
        
        self.logger.info("Linear MPC controller added")
    
    def add_stochastic_controller(self):
        """Add stochastic MPC controller."""
        self.controllers[MPCType.STOCHASTIC_MPC] = StochasticMPCController(
            self.state_size, self.control_size, self.params
        )
        
        self.logger.info("Stochastic MPC controller added")
    
    def control_step(self, current_state: np.ndarray, reference: np.ndarray,
                    dynamics_function: Optional[Callable] = None) -> Dict:
        """
        Execute one control step with predictive control.
        
        Args:
            current_state: Current system state
            reference: Reference trajectory or setpoint
            dynamics_function: System dynamics function
            
        Returns:
            Control step results
        """
        if self.active_controller is None:
            raise RuntimeError("No active controller available")
        
        # Solve MPC
        mpc_result = self.active_controller.solve(
            current_state, reference, dynamics_function
        )
        
        # Failure prediction
        failure_prediction = self.failure_predictor.predict_failure_probability(
            current_state, mpc_result.predicted_trajectory
        )
        
        # Control step result
        control_result = {
            'control_signal': mpc_result.optimal_control,
            'predicted_trajectory': mpc_result.predicted_trajectory,
            'mpc_cost': mpc_result.cost,
            'solve_time': mpc_result.solve_time,
            'feasible': mpc_result.feasible,
            'failure_prediction': failure_prediction,
            'controller_type': self.active_controller_type.value,
            'timestamp': time.time()
        }
        
        # Store control history
        self.control_history.append(control_result)
        
        # Update performance metrics
        self._update_performance_metrics(mpc_result, failure_prediction)
        
        return control_result
    
    def _update_performance_metrics(self, mpc_result: MPCResult, 
                                  failure_prediction: Dict):
        """Update system performance metrics."""
        controller_performance = self.active_controller.get_performance_metrics()
        
        system_performance = {
            'timestamp': time.time(),
            'controller_performance': controller_performance,
            'failure_risk_level': failure_prediction['risk_level'],
            'max_failure_probability': failure_prediction['max_failure_probability'],
            'time_to_failure_s': failure_prediction['time_to_failure_seconds'],
            'mpc_feasible': mpc_result.feasible,
            'solve_time_s': mpc_result.solve_time
        }
        
        self.performance_metrics.append(system_performance)
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary."""
        # Recent performance
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-20:]
            
            avg_solve_time = np.mean([m['solve_time_s'] for m in recent_metrics])
            feasibility_rate = np.mean([m['mpc_feasible'] for m in recent_metrics])
            avg_failure_prob = np.mean([m['max_failure_probability'] for m in recent_metrics])
            
            risk_levels = [m['failure_risk_level'] for m in recent_metrics]
            risk_distribution = {
                'low': risk_levels.count('low') / len(risk_levels),
                'medium': risk_levels.count('medium') / len(risk_levels),
                'high': risk_levels.count('high') / len(risk_levels),
                'critical': risk_levels.count('critical') / len(risk_levels)
            }
        else:
            avg_solve_time = 0.0
            feasibility_rate = 0.0
            avg_failure_prob = 0.0
            risk_distribution = {}
        
        # Controller performance
        controller_performance = {}
        for controller_type, controller in self.controllers.items():
            controller_performance[controller_type.value] = controller.get_performance_metrics()
        
        summary = {
            'system_status': {
                'active_controller': self.active_controller_type.value,
                'available_controllers': list(self.controllers.keys()),
                'control_history_length': len(self.control_history),
                'state_size': self.state_size,
                'control_size': self.control_size
            },
            'performance_summary': {
                'avg_solve_time_s': avg_solve_time,
                'feasibility_rate': feasibility_rate,
                'avg_failure_probability': avg_failure_prob,
                'risk_distribution': risk_distribution,
                'solve_time_target_met': avg_solve_time <= MPC_SOLVE_TIME_TARGET,
                'feasibility_acceptable': feasibility_rate >= 0.95
            },
            'controller_performance': controller_performance,
            'failure_prediction_status': {
                'prediction_history_length': len(self.failure_predictor.prediction_history),
                'failure_events_count': len(self.failure_predictor.failure_events)
            }
        }
        
        return summary


if __name__ == "__main__":
    """Example usage of predictive control system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== PREDICTIVE CONTROL SYSTEM ===")
    print("Advanced MPC with failure prediction")
    
    # System parameters
    state_size = 6  # [x, y, z, vx, vy, vz]
    control_size = 3  # [fx, fy, fz]
    
    # Initialize predictive control system
    params = MPCParameters(
        prediction_horizon=20,
        control_horizon=10,
        sample_time=1e-3
    )
    
    pc_system = PredictiveControlSystem(state_size, control_size, params)
    
    print(f"\nInitialized predictive control system:")
    print(f"  State size: {state_size}")
    print(f"  Control size: {control_size}")
    print(f"  Prediction horizon: {params.prediction_horizon}")
    print(f"  Control horizon: {params.control_horizon}")
    
    # Add linear controller
    # Simple double integrator system
    A = np.array([
        [1, 0, 0, 1e-3, 0, 0],
        [0, 1, 0, 0, 1e-3, 0],
        [0, 0, 1, 0, 0, 1e-3],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    B = np.array([
        [5e-7, 0, 0],
        [0, 5e-7, 0],
        [0, 0, 5e-7],
        [1e-3, 0, 0],
        [0, 1e-3, 0],
        [0, 0, 1e-3]
    ])
    
    pc_system.add_linear_controller(A, B)
    print(f"Added linear MPC controller")
    
    # Add stochastic controller
    pc_system.add_stochastic_controller()
    print(f"Added stochastic MPC controller")
    
    # Simulation
    print(f"\nRunning predictive control simulation...")
    
    current_state = np.array([10e-9, 5e-9, 0, 0, 0, 0])  # Initial state
    reference = np.array([0, 0, 0, 0, 0, 0])  # Target state
    
    # Control loop
    for step in range(50):
        # Control step
        control_result = pc_system.control_step(current_state, reference)
        
        # Apply control (simulate system response)
        control_signal = control_result['control_signal']
        current_state = A @ current_state + B @ control_signal
        
        # Print progress
        if step % 10 == 0:
            position_error = np.linalg.norm(current_state[:3])
            failure_risk = control_result['failure_prediction']['risk_level']
            solve_time = control_result['solve_time']
            
            print(f"  Step {step:2d}: pos_error={position_error:.2e}m, "
                  f"risk={failure_risk}, solve_time={solve_time:.1f}ms")
    
    print(f"\nFinal state:")
    print(f"  Position: [{current_state[0]:.2e}, {current_state[1]:.2e}, {current_state[2]:.2e}] m")
    print(f"  Velocity: [{current_state[3]:.2e}, {current_state[4]:.2e}, {current_state[5]:.2e}] m/s")
    
    # System summary
    summary = pc_system.get_system_summary()
    
    print(f"\nSystem Performance Summary:")
    
    perf_summary = summary.get('performance_summary', {})
    print(f"  Average solve time: {perf_summary.get('avg_solve_time_s', 0)*1000:.2f} ms")
    print(f"  Feasibility rate: {perf_summary.get('feasibility_rate', 0)*100:.1f}%")
    print(f"  Average failure probability: {perf_summary.get('avg_failure_probability', 0)*100:.2f}%")
    print(f"  Solve time target met: {'✓' if perf_summary.get('solve_time_target_met', False) else '✗'}")
    
    risk_dist = perf_summary.get('risk_distribution', {})
    if risk_dist:
        print(f"  Risk distribution:")
        for risk_level, percentage in risk_dist.items():
            print(f"    {risk_level}: {percentage*100:.1f}%")
    
    # Controller comparison
    ctrl_perf = summary.get('controller_performance', {})
    print(f"\nController Performance:")
    for ctrl_type, metrics in ctrl_perf.items():
        if metrics.get('status') != 'no_data':
            print(f"  {ctrl_type}:")
            print(f"    Avg solve time: {metrics.get('avg_solve_time_s', 0)*1000:.2f} ms")
            print(f"    Tracking error: {metrics.get('avg_tracking_error_m', 0):.2e} m")
            print(f"    Solutions: {metrics.get('n_solutions', 0)}")
    
    print(f"\nPredictive control demonstration complete!")
