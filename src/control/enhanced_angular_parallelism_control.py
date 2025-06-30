"""
Enhanced Angular Parallelism Control System
===========================================

This module implements multi-rate cascaded control architecture for achieving
≤1 µrad parallelism across 100 µm span using advanced control formulations
derived from workspace mathematical analysis.

Mathematical Formulation:
ε_parallel(t) = M_angular × [∑ᵢ₌₁ⁿ Fᵢ(rᵢ,θᵢ) - F_target] ≤ 1×10⁻⁶ rad

Multi-rate control architecture with:
- Fast loop: >1 kHz for fine angular correction
- Slow loop: ~10 Hz for structural compensation  
- Thermal loop: ~0.1 Hz for thermal drift compensation
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize_scalar
import control as ct
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time

# Physical constants
PI = np.pi
MICRO_RAD_LIMIT = 1e-6  # 1 µrad parallelism requirement

class ControlLoopType(Enum):
    """Control loop types for multi-rate architecture."""
    FAST = "fast"      # >1 kHz
    SLOW = "slow"      # ~10 Hz  
    THERMAL = "thermal" # ~0.1 Hz

@dataclass
class ParallelismControllerParams:
    """Parameters for multi-rate parallelism controller."""
    
    # Fast loop parameters (>1 kHz)
    Kp_fast: float = 2.5
    Ki_fast: float = 1500.0
    Kd_fast: float = 0.001
    tau_fast: float = 0.0001  # Filter time constant
    
    # Slow loop parameters (~10 Hz)
    Kp_slow: float = 1.0
    Ki_slow: float = 50.0
    Kd_slow: float = 0.01
    tau_slow: float = 0.01
    
    # Thermal loop parameters (~0.1 Hz)
    thermal_numerator: float = 2.5
    thermal_den_coeffs: List[float] = None  # [1, 6, 100] for s² + 6s + 100
    
    # H∞ compensation parameters
    h_inf_gamma: float = 1.5  # Robustness parameter
    gain_margin_db: float = 19.24
    phase_margin_deg: float = 91.7
    
    def __post_init__(self):
        if self.thermal_den_coeffs is None:
            self.thermal_den_coeffs = [1, 6, 100]

class EnhancedAngularParallelismControl:
    """
    Enhanced angular parallelism control system implementing multi-rate 
    cascaded control architecture for sub-microrad performance.
    
    LaTeX Formulations Implemented:
    
    1. Multi-Rate Control Matrix:
    M_angular = [
        K_fast(s) × G_fast(s)     0                   0
        0            K_slow(s) × G_slow(s)    0  
        0            0                        K_thermal(s)
    ]
    
    2. Fast Loop Controller:
    K_fast(s) = Kp_f + Ki_f/s + Kd_f×s/(τf×s + 1)
    
    3. Slow Loop Controller:
    K_slow(s) = Kp_s + Ki_s/s + Kd_s×s/(τs×s + 1)
    
    4. Thermal Loop Controller:
    K_thermal(s) = 2.5/(s² + 6s + 100) × H∞_comp(s)
    
    5. Angular Error Constraint:
    ε_parallel(t) ≤ 1×10⁻⁶ rad
    """
    
    def __init__(self, params: Optional[ParallelismControllerParams] = None,
                 n_actuators: int = 5):
        """
        Initialize enhanced angular parallelism control system.
        
        Args:
            params: Controller parameters, uses defaults if None
            n_actuators: Number of actuators for distributed control
        """
        self.params = params or ParallelismControllerParams()
        self.n_actuators = n_actuators
        self.logger = logging.getLogger(__name__)
        
        # Control system state
        self.is_initialized = False
        self.control_loops = {}
        self.actuator_positions = np.zeros(n_actuators)
        self.angular_errors = np.zeros(3)  # [θx, θy, θz]
        
        # Multi-rate control state
        self.fast_loop_state = np.zeros(3)    # PID integrator states
        self.slow_loop_state = np.zeros(3)
        self.thermal_loop_state = np.zeros(2) # Second-order system state
        
        # Performance monitoring
        self.angular_error_history = []
        self.control_signal_history = []
        
        # Initialize control loops
        self._initialize_control_loops()
        
    def _initialize_control_loops(self):
        """Initialize all control loops with specified transfer functions."""
        
        # Fast loop controller: K_fast(s) = Kp + Ki/s + Kd*s/(τs + 1)
        fast_num = [
            self.params.Kd_fast * self.params.tau_fast,
            self.params.Kp_fast * self.params.tau_fast + self.params.Kd_fast,
            self.params.Ki_fast * self.params.tau_fast
        ]
        fast_den = [self.params.tau_fast, 1, 0]
        self.control_loops[ControlLoopType.FAST] = ct.TransferFunction(fast_num, fast_den)
        
        # Slow loop controller: K_slow(s) = Kp + Ki/s + Kd*s/(τs + 1)
        slow_num = [
            self.params.Kd_slow * self.params.tau_slow,
            self.params.Kp_slow * self.params.tau_slow + self.params.Kd_slow,
            self.params.Ki_slow * self.params.tau_slow
        ]
        slow_den = [self.params.tau_slow, 1, 0]
        self.control_loops[ControlLoopType.SLOW] = ct.TransferFunction(slow_num, slow_den)
        
        # Thermal loop controller: K_thermal(s) = 2.5/(s² + 6s + 100)
        thermal_num = [self.params.thermal_numerator]
        thermal_den = self.params.thermal_den_coeffs
        self.control_loops[ControlLoopType.THERMAL] = ct.TransferFunction(thermal_num, thermal_den)
        
        self.is_initialized = True
        self.logger.info("Multi-rate control loops initialized successfully")
    
    def calculate_angular_error(self, actuator_forces: np.ndarray, 
                              target_force: float,
                              actuator_positions: np.ndarray) -> np.ndarray:
        """
        Calculate angular error from actuator force distribution.
        
        LaTeX: ε_parallel(t) = M_angular × [∑ᵢ₌₁ⁿ Fᵢ(rᵢ,θᵢ) - F_target]
        
        Args:
            actuator_forces: Forces from each actuator (N)
            target_force: Target total force (N)
            actuator_positions: Positions of actuators (m)
            
        Returns:
            Angular errors [θx, θy, θz] in radians
        """
        if len(actuator_forces) != self.n_actuators:
            raise ValueError(f"Expected {self.n_actuators} actuator forces")
        
        # Calculate moment arms and torques
        # Assume actuators are positioned in a grid pattern
        x_positions = actuator_positions[:self.n_actuators//2] if len(actuator_positions) >= self.n_actuators else np.linspace(-50e-6, 50e-6, self.n_actuators)
        y_positions = np.zeros_like(x_positions)
        
        # Calculate torques about each axis
        torque_x = np.sum(actuator_forces * y_positions)  # Rotation about x-axis
        torque_y = np.sum(actuator_forces * x_positions)  # Rotation about y-axis
        torque_z = 0.0  # Assume no z-axis rotation for parallel plates
        
        # Convert torques to angular errors (simplified model)
        # In practice, this would involve the system's moment of inertia
        moment_of_inertia = 1e-12  # kg⋅m² (estimated for small system)
        
        angular_errors = np.array([
            torque_x / (target_force * 100e-6),  # Normalize by force and span
            torque_y / (target_force * 100e-6),
            torque_z / (target_force * 100e-6)
        ])
        
        self.angular_errors = angular_errors
        self.logger.debug(f"Angular errors: θx={angular_errors[0]*1e6:.2f} µrad, "
                         f"θy={angular_errors[1]*1e6:.2f} µrad")
        
        return angular_errors
    
    def multi_rate_control_update(self, angular_errors: np.ndarray, 
                                dt_fast: float = 1e-4,
                                dt_slow: float = 0.1,
                                dt_thermal: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Perform multi-rate control update for all three loops.
        
        Args:
            angular_errors: Current angular errors [θx, θy, θz] (rad)
            dt_fast: Fast loop time step (s)
            dt_slow: Slow loop time step (s) 
            dt_thermal: Thermal loop time step (s)
            
        Returns:
            Dictionary with control signals from each loop
        """
        if not self.is_initialized:
            raise RuntimeError("Control loops not initialized")
        
        control_signals = {}
        
        # Fast loop control (>1 kHz) - High-frequency disturbance rejection
        fast_control = self._pid_control_update(
            angular_errors, 
            self.fast_loop_state,
            self.params.Kp_fast,
            self.params.Ki_fast, 
            self.params.Kd_fast,
            dt_fast
        )
        control_signals['fast'] = fast_control
        
        # Slow loop control (~10 Hz) - Structural compensation
        slow_control = self._pid_control_update(
            angular_errors,
            self.slow_loop_state, 
            self.params.Kp_slow,
            self.params.Ki_slow,
            self.params.Kd_slow,
            dt_slow
        )
        control_signals['slow'] = slow_control
        
        # Thermal loop control (~0.1 Hz) - Long-term drift compensation
        thermal_control = self._thermal_control_update(angular_errors, dt_thermal)
        control_signals['thermal'] = thermal_control
        
        # Combined control signal
        total_control = fast_control + slow_control + thermal_control
        control_signals['total'] = total_control
        
        # Store for monitoring
        self.control_signal_history.append({
            'timestamp': time.time(),
            'fast': fast_control.copy(),
            'slow': slow_control.copy(), 
            'thermal': thermal_control.copy(),
            'total': total_control.copy()
        })
        
        self.logger.debug(f"Multi-rate control: fast_rms={np.linalg.norm(fast_control):.2e}, "
                         f"slow_rms={np.linalg.norm(slow_control):.2e}, "
                         f"thermal_rms={np.linalg.norm(thermal_control):.2e}")
        
        return control_signals
    
    def _pid_control_update(self, errors: np.ndarray, integrator_state: np.ndarray,
                           kp: float, ki: float, kd: float, dt: float) -> np.ndarray:
        """
        PID control update for a given loop.
        
        Args:
            errors: Current errors
            integrator_state: Integrator state (modified in-place)
            kp, ki, kd: PID gains
            dt: Time step
            
        Returns:
            Control signal
        """
        # Proportional term
        proportional = kp * errors
        
        # Integral term with anti-windup
        integrator_state += errors * dt
        
        # Anti-windup: limit integrator state
        max_integral = 1e-3  # Maximum integral contribution
        integrator_state = np.clip(integrator_state, -max_integral/ki, max_integral/ki)
        integral = ki * integrator_state
        
        # Derivative term (with filtering to reduce noise)
        if not hasattr(self, '_previous_errors'):
            self._previous_errors = errors.copy()
        
        derivative = kd * (errors - self._previous_errors) / dt
        self._previous_errors = errors.copy()
        
        # Apply derivative filtering
        derivative_filtered = derivative / (1 + dt/0.001)  # 1ms filter
        
        control_signal = proportional + integral + derivative_filtered
        
        return control_signal
    
    def _thermal_control_update(self, angular_errors: np.ndarray, dt: float) -> np.ndarray:
        """
        Thermal loop control update using second-order system.
        
        LaTeX: K_thermal(s) = 2.5/(s² + 6s + 100)
        
        Args:
            angular_errors: Current angular errors
            dt: Time step
            
        Returns:
            Thermal control signal
        """
        # State-space representation of K_thermal(s) = 2.5/(s² + 6s + 100)
        # ẋ₁ = x₂
        # ẋ₂ = -100x₁ - 6x₂ + u
        # y = 2.5x₁
        
        A = np.array([[0, 1], [-100, -6]])
        B = np.array([[0], [1]])
        C = np.array([[2.5, 0]])
        
        # Input is the angular error (use magnitude for SISO system)
        input_signal = np.linalg.norm(angular_errors)
        
        # Discrete-time update using Euler method
        state_dot = A @ self.thermal_loop_state + B.flatten() * input_signal
        self.thermal_loop_state += state_dot * dt
        
        # Output
        thermal_output = C @ self.thermal_loop_state
        
        # Distribute thermal control across all angular axes
        thermal_control = np.ones(3) * thermal_output[0] * np.sign(angular_errors)
        
        return thermal_control
    
    def h_infinity_compensation(self, plant_tf: ct.TransferFunction,
                              weight_funcs: Optional[Dict[str, ct.TransferFunction]] = None) -> ct.TransferFunction:
        """
        Design H∞ controller for robust performance.
        
        LaTeX: min_K ||Tzw||∞ subject to stability and performance constraints
        
        Args:
            plant_tf: Plant transfer function
            weight_funcs: Weighting functions for mixed-sensitivity design
            
        Returns:
            H∞ controller transfer function
        """
        if weight_funcs is None:
            # Default weighting functions
            weight_funcs = {
                'W1': ct.TransferFunction([1, 0.1], [1, 100]),    # Performance weight
                'W2': ct.TransferFunction([0.1, 1], [1, 0.001]),  # Control effort weight
                'W3': ct.TransferFunction([1], [1, 10])           # Robustness weight
            }
        
        try:
            # Simplified H∞ synthesis (would use hinfsyn in full implementation)
            # For now, use LQG with robustness constraints
            
            # Convert to state-space
            plant_ss = ct.tf2ss(plant_tf)
            A, B, C, D = ct.ssdata(plant_ss)
            
            # LQR design with robustness margins
            Q = np.eye(A.shape[0]) * 10  # State weighting
            R = np.array([[1]])          # Control weighting
            
            K_lqr, S, E = ct.lqr(A, B, Q, R)
            
            # Kalman filter design
            G = B  # Process noise input
            W = np.eye(A.shape[0]) * 0.1  # Process noise covariance
            V = np.array([[0.01]])        # Measurement noise covariance
            
            L, P, E_kf = ct.lqe(A, G, C, W, V)
            
            # H∞ controller (LQG with robustness)
            A_ctrl = A - B @ K_lqr - L @ C
            B_ctrl = L
            C_ctrl = -K_lqr
            D_ctrl = np.zeros((K_lqr.shape[0], L.shape[1]))
            
            h_inf_controller = ct.StateSpace(A_ctrl, B_ctrl, C_ctrl, D_ctrl)
            
            # Verify robustness margins
            loop_tf = ct.series(h_inf_controller, plant_tf)
            gm, pm, wg, wp = ct.margin(loop_tf)
            
            gm_db = 20 * np.log10(gm) if gm > 0 else -100
            pm_deg = pm * 180 / PI if pm > 0 else 0
            
            if gm_db >= self.params.gain_margin_db and pm_deg >= self.params.phase_margin_deg:
                self.logger.info(f"H∞ controller meets margins: GM={gm_db:.1f}dB, PM={pm_deg:.1f}°")
            else:
                self.logger.warning(f"H∞ controller margins insufficient: GM={gm_db:.1f}dB, PM={pm_deg:.1f}°")
            
            return h_inf_controller
            
        except Exception as e:
            self.logger.error(f"H∞ controller design failed: {e}")
            # Fall back to simple PID
            return ct.TransferFunction([self.params.Kp_fast, self.params.Ki_fast], [1, 0])
    
    def check_parallelism_constraint(self, angular_errors: np.ndarray) -> Dict[str, bool]:
        """
        Check if parallelism constraints are satisfied.
        
        Args:
            angular_errors: Angular errors [θx, θy, θz] (rad)
            
        Returns:
            Dictionary with constraint satisfaction results
        """
        results = {}
        
        # Individual axis constraints
        results['theta_x_ok'] = abs(angular_errors[0]) <= MICRO_RAD_LIMIT
        results['theta_y_ok'] = abs(angular_errors[1]) <= MICRO_RAD_LIMIT
        results['theta_z_ok'] = abs(angular_errors[2]) <= MICRO_RAD_LIMIT
        
        # Overall constraint
        max_error = np.max(np.abs(angular_errors))
        results['overall_ok'] = max_error <= MICRO_RAD_LIMIT
        results['max_error_urad'] = max_error * 1e6
        results['margin_factor'] = MICRO_RAD_LIMIT / max_error if max_error > 0 else float('inf')
        
        # Store in history
        self.angular_error_history.append({
            'timestamp': time.time(),
            'errors_urad': angular_errors * 1e6,
            'constraint_satisfied': results['overall_ok']
        })
        
        return results
    
    def optimize_controller_gains(self, target_bandwidth: float = 1000.0) -> ParallelismControllerParams:
        """
        Optimize controller gains for target bandwidth and stability margins.
        
        Args:
            target_bandwidth: Target closed-loop bandwidth (Hz)
            
        Returns:
            Optimized controller parameters
        """
        def objective(gains):
            """Optimization objective function."""
            kp_f, ki_f, kd_f, kp_s, ki_s, kd_s = gains
            
            try:
                # Create test controller
                test_params = ParallelismControllerParams(
                    Kp_fast=kp_f, Ki_fast=ki_f, Kd_fast=kd_f,
                    Kp_slow=kp_s, Ki_slow=ki_s, Kd_slow=kd_s
                )
                
                # Build closed-loop system
                fast_num = [kd_f * test_params.tau_fast,
                           kp_f * test_params.tau_fast + kd_f,
                           ki_f * test_params.tau_fast]
                fast_den = [test_params.tau_fast, 1, 0]
                controller_tf = ct.TransferFunction(fast_num, fast_den)
                
                # Simple plant model
                plant_tf = ct.TransferFunction([1], [1e-6, 2e-3, 1])
                
                # Closed-loop system
                loop_tf = ct.series(controller_tf, plant_tf)
                cl_tf = ct.feedback(loop_tf, 1)
                
                # Calculate performance metrics
                gm, pm, wg, wp = ct.margin(loop_tf)
                
                # Bandwidth (3dB frequency)
                w, mag, phase = ct.bode(cl_tf, plot=False)
                bandwidth = None
                for i, m in enumerate(mag):
                    if 20*np.log10(m) <= -3:
                        bandwidth = w[i] / (2*PI)
                        break
                
                if bandwidth is None:
                    bandwidth = wp / (2*PI) if wp > 0 else 100
                
                # Objective: minimize deviation from target
                gm_db = 20 * np.log10(gm) if gm > 0 else -100
                pm_deg = pm * 180 / PI if pm > 0 else 0
                
                bandwidth_error = (bandwidth - target_bandwidth)**2 / target_bandwidth**2
                margin_penalty = max(0, self.params.gain_margin_db - gm_db)**2 + \
                               max(0, self.params.phase_margin_deg - pm_deg)**2
                
                return bandwidth_error + 0.1 * margin_penalty
                
            except:
                return 1e6  # Large penalty for unstable/invalid controllers
        
        # Initial guess
        initial_gains = [
            self.params.Kp_fast, self.params.Ki_fast, self.params.Kd_fast,
            self.params.Kp_slow, self.params.Ki_slow, self.params.Kd_slow
        ]
        
        # Bounds for gains
        bounds = [
            (0.1, 10),    # Kp_fast
            (10, 5000),   # Ki_fast
            (1e-4, 0.1),  # Kd_fast
            (0.1, 5),     # Kp_slow
            (1, 200),     # Ki_slow
            (1e-3, 0.1)   # Kd_slow
        ]
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, initial_gains, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimized_params = ParallelismControllerParams(
                    Kp_fast=result.x[0], Ki_fast=result.x[1], Kd_fast=result.x[2],
                    Kp_slow=result.x[3], Ki_slow=result.x[4], Kd_slow=result.x[5]
                )
                
                self.logger.info(f"Controller optimization successful: objective={result.fun:.4f}")
                return optimized_params
            else:
                self.logger.warning("Controller optimization failed, using default parameters")
                return self.params
                
        except Exception as e:
            self.logger.error(f"Controller optimization error: {e}")
            return self.params
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.angular_error_history:
            return {'status': 'no_data'}
        
        # Extract recent performance data
        recent_errors = [entry['errors_urad'] for entry in self.angular_error_history[-100:]]
        recent_constraints = [entry['constraint_satisfied'] for entry in self.angular_error_history[-100:]]
        
        if not recent_errors:
            return {'status': 'insufficient_data'}
        
        recent_errors_array = np.array(recent_errors)
        
        performance = {
            'parallelism_constraint_satisfaction': {
                'success_rate_percent': np.mean(recent_constraints) * 100,
                'current_max_error_urad': np.max(np.abs(self.angular_errors)) * 1e6,
                'requirement_urad': MICRO_RAD_LIMIT * 1e6
            },
            'angular_error_statistics': {
                'rms_error_urad': np.sqrt(np.mean(recent_errors_array**2, axis=0)).tolist(),
                'max_error_urad': np.max(np.abs(recent_errors_array), axis=0).tolist(),
                'mean_error_urad': np.mean(recent_errors_array, axis=0).tolist()
            },
            'controller_performance': {
                'control_loops_active': len(self.control_loops),
                'n_actuators': self.n_actuators,
                'history_length': len(self.angular_error_history)
            }
        }
        
        return performance


if __name__ == "__main__":
    """Example usage of enhanced angular parallelism control."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== ENHANCED ANGULAR PARALLELISM CONTROL ===")
    print("Target: ≤1 µrad across 100 µm span")
    
    # Initialize controller
    controller = EnhancedAngularParallelismControl(n_actuators=5)
    
    # Simulate actuator forces and positions
    actuator_forces = np.array([1e-9, 1.1e-9, 0.9e-9, 1.05e-9, 0.95e-9])  # N
    target_force = 1e-9  # N
    actuator_positions = np.linspace(-50e-6, 50e-6, 5)  # 100 µm span
    
    # Calculate angular errors
    angular_errors = controller.calculate_angular_error(
        actuator_forces, target_force, actuator_positions
    )
    
    print(f"\nAngular Errors:")
    print(f"  θx: {angular_errors[0]*1e6:.3f} µrad")
    print(f"  θy: {angular_errors[1]*1e6:.3f} µrad") 
    print(f"  θz: {angular_errors[2]*1e6:.3f} µrad")
    
    # Multi-rate control update
    control_signals = controller.multi_rate_control_update(angular_errors)
    
    print(f"\nControl Signals:")
    for loop_type, signal in control_signals.items():
        if isinstance(signal, np.ndarray):
            print(f"  {loop_type}: RMS = {np.linalg.norm(signal):.2e}")
    
    # Check constraints
    constraint_results = controller.check_parallelism_constraint(angular_errors)
    
    print(f"\nConstraint Satisfaction:")
    print(f"  Overall OK: {'✓' if constraint_results['overall_ok'] else '✗'}")
    print(f"  Max error: {constraint_results['max_error_urad']:.3f} µrad")
    print(f"  Margin: {constraint_results['margin_factor']:.2f}x")
    
    # Controller optimization
    print(f"\nOptimizing controller gains...")
    optimized_params = controller.optimize_controller_gains(target_bandwidth=1000)
    
    print(f"Optimized Parameters:")
    print(f"  Fast loop: Kp={optimized_params.Kp_fast:.2f}, Ki={optimized_params.Ki_fast:.1f}")
    print(f"  Slow loop: Kp={optimized_params.Kp_slow:.2f}, Ki={optimized_params.Ki_slow:.1f}")
    
    # Performance summary
    performance = controller.get_performance_summary()
    if 'parallelism_constraint_satisfaction' in performance:
        print(f"\nPerformance Summary:")
        pcs = performance['parallelism_constraint_satisfaction']
        print(f"  Success rate: {pcs['success_rate_percent']:.1f}%")
        print(f"  Current max error: {pcs['current_max_error_urad']:.3f} µrad")
        print(f"  Requirement: {pcs['requirement_urad']:.1f} µrad")
