"""
Advanced Interferometric Feedback Control System
===============================================

This module implements comprehensive interferometric feedback control including:
- Complete phase shift calculations with electro-optic effects
- Advanced PID control with gain and phase margins
- Real-time feedback optimization
- Multi-loop control architecture

Based on formulations found in workspace survey from:
- lqg-anec-framework/docs/technical_implementation_specs.tex
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize
from typing import Dict, Tuple, List, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import control as ct

# Physical constants
C = 299792458.0  # m/s
PI = np.pi

class ControllerType(Enum):
    """Control system types."""
    PID = "pid"
    LQG = "lqg"
    H_INFINITY = "h_infinity"
    ADAPTIVE = "adaptive"

@dataclass
class OpticalProperties:
    """Optical properties for interferometric control."""
    wavelength: float           # m
    refractive_index: float     # n₀
    electro_optic_coeff: float  # r (m/V)
    path_length: float          # L (m)
    
@dataclass
class ControllerSpecs:
    """Controller performance specifications."""
    gain_margin_db: float = 19.24    # dB
    phase_margin_deg: float = 91.7   # degrees
    bandwidth_hz: float = 1000       # Hz
    settling_time_ms: float = 2.0    # ms
    overshoot_percent: float = 5.0   # %

class AdvancedInterferometricControl:
    """
    Advanced interferometric feedback control system.
    
    LaTeX Formulations Implemented:
    
    1. Phase Shift Calculation:
    Δφ = (2π/λ) Δn L
    
    2. Refractive Index Change:
    Δn = (1/2) n₀³ r E
    
    3. Control Transfer Function:
    H(s) = (K_p s² + K_i s + K_d s³)/(s³ + a₂s² + a₁s + a₀)
    
    4. Stability Margins:
    Gain Margin = 19.24 dB
    Phase Margin = 91.7°
    """
    
    def __init__(self, optical_props: OpticalProperties, 
                 controller_specs: Optional[ControllerSpecs] = None):
        """
        Initialize interferometric control system.
        
        Args:
            optical_props: Optical system properties
            controller_specs: Controller specifications
        """
        self.optical = optical_props
        self.specs = controller_specs or ControllerSpecs()
        self.logger = logging.getLogger(__name__)
        
        # Control system state
        self.controller_type = ControllerType.PID
        self.controller_params = {}
        self.is_designed = False
        
    def calculate_phase_shift(self, delta_n: float) -> float:
        """
        Calculate optical phase shift from refractive index change.
        
        LaTeX: Δφ = (2π/λ) Δn L
        
        Args:
            delta_n: Refractive index change
            
        Returns:
            Phase shift (radians)
        """
        phase_shift = (2 * PI / self.optical.wavelength) * delta_n * self.optical.path_length
        
        self.logger.debug(f"Phase shift: Δφ = {phase_shift:.4f} rad")
        return phase_shift
    
    def calculate_refractive_index_change(self, electric_field: float) -> float:
        """
        Calculate refractive index change from electric field.
        
        LaTeX: Δn = (1/2) n₀³ r E
        
        Args:
            electric_field: Applied electric field (V/m)
            
        Returns:
            Refractive index change
        """
        delta_n = 0.5 * (self.optical.refractive_index**3) * \
                 self.optical.electro_optic_coeff * electric_field
        
        self.logger.debug(f"Refractive index change: Δn = {delta_n:.6e}")
        return delta_n
    
    def electric_field_to_phase_shift(self, electric_field: float) -> float:
        """
        Direct calculation of phase shift from electric field.
        
        Combines the two previous calculations for efficiency.
        
        Args:
            electric_field: Applied electric field (V/m)
            
        Returns:
            Phase shift (radians)
        """
        delta_n = self.calculate_refractive_index_change(electric_field)
        phase_shift = self.calculate_phase_shift(delta_n)
        
        return phase_shift
    
    def design_pid_controller(self, plant_tf: Optional[ct.TransferFunction] = None) -> Dict[str, float]:
        """
        Design PID controller with specified margins.
        
        LaTeX: H(s) = (K_p s² + K_i s + K_d s³)/(s³ + a₂s² + a₁s + a₀)
        
        Args:
            plant_tf: Plant transfer function, uses default if None
            
        Returns:
            Dictionary with PID parameters
        """
        if plant_tf is None:
            # Default plant model (second-order with delay)
            num = [1]
            den = [1, 2*0.1*10, 10**2]  # ωₙ = 10 rad/s, ζ = 0.1
            plant_tf = ct.TransferFunction(num, den)
        
        # PID controller design using root locus or optimization
        def objective(params):
            """Optimization objective for PID tuning."""
            kp, ki, kd = params
            
            # PID transfer function
            pid_num = [kd, kp, ki]
            pid_den = [1, 0]
            pid_tf = ct.TransferFunction(pid_num, pid_den)
            
            # Closed-loop system
            try:
                loop_tf = ct.series(pid_tf, plant_tf)
                cl_tf = ct.feedback(loop_tf, 1)
                
                # Calculate margins
                gm, pm, wg, wp = ct.margin(loop_tf)
                
                # Convert to dB and degrees
                gm_db = 20 * np.log10(gm) if gm > 0 else -100
                pm_deg = pm * 180 / PI if pm > 0 else 0
                
                # Objective: minimize deviation from target margins
                gm_error = (gm_db - self.specs.gain_margin_db)**2
                pm_error = (pm_deg - self.specs.phase_margin_deg)**2
                
                return gm_error + pm_error
                
            except:
                return 1e6  # Penalty for unstable systems
        
        # Initial guess
        initial_params = [1.0, 0.1, 0.01]  # [Kp, Ki, Kd]
        
        # Optimize PID parameters
        result = minimize(objective, initial_params, 
                         bounds=[(0.01, 100), (0.001, 10), (0.0001, 1)],
                         method='L-BFGS-B')
        
        if result.success:
            kp, ki, kd = result.x
            
            # Verify final performance
            pid_num = [kd, kp, ki]
            pid_den = [1, 0]
            pid_tf = ct.TransferFunction(pid_num, pid_den)
            loop_tf = ct.series(pid_tf, plant_tf)
            
            gm, pm, wg, wp = ct.margin(loop_tf)
            gm_db = 20 * np.log10(gm) if gm > 0 else -100
            pm_deg = pm * 180 / PI if pm > 0 else 0
            
            self.controller_params = {
                'kp': kp,
                'ki': ki, 
                'kd': kd,
                'type': 'pid',
                'gain_margin_db': gm_db,
                'phase_margin_deg': pm_deg,
                'crossover_freq': wp
            }
            
            self.is_designed = True
            self.logger.info(f"PID controller designed: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.4f}")
            self.logger.info(f"Margins: GM={gm_db:.1f}dB, PM={pm_deg:.1f}°")
            
        else:
            self.logger.error("PID controller design failed")
            self.controller_params = {'kp': 1.0, 'ki': 0.1, 'kd': 0.01, 'type': 'pid'}
        
        return self.controller_params
    
    def design_advanced_controller(self, controller_type: ControllerType,
                                 plant_tf: Optional[ct.TransferFunction] = None) -> Dict[str, float]:
        """
        Design advanced controller (LQG, H∞, etc.).
        
        Args:
            controller_type: Type of controller to design
            plant_tf: Plant transfer function
            
        Returns:
            Controller parameters
        """
        if plant_tf is None:
            # Default plant model
            A = np.array([[0, 1], [-100, -20]])  # State matrix
            B = np.array([[0], [1]])             # Input matrix  
            C = np.array([[1, 0]])               # Output matrix
            D = np.array([[0]])                  # Feedthrough
            
            plant_ss = ct.StateSpace(A, B, C, D)
            plant_tf = ct.ss2tf(plant_ss)
        
        if controller_type == ControllerType.LQG:
            return self._design_lqg_controller(plant_tf)
        elif controller_type == ControllerType.H_INFINITY:
            return self._design_hinf_controller(plant_tf)
        elif controller_type == ControllerType.ADAPTIVE:
            return self._design_adaptive_controller(plant_tf)
        else:
            # Fall back to PID
            return self.design_pid_controller(plant_tf)
    
    def _design_lqg_controller(self, plant_tf: ct.TransferFunction) -> Dict[str, float]:
        """Design LQG controller."""
        try:
            # Convert to state-space
            plant_ss = ct.tf2ss(plant_tf)
            A, B, C, D = ct.ssdata(plant_ss)
            
            # LQR design
            Q = np.eye(A.shape[0])  # State weighting
            R = np.array([[1]])     # Control weighting
            
            K, S, E = ct.lqr(A, B, Q, R)
            
            # Kalman filter design
            G = B  # Process noise input
            W = np.eye(A.shape[0])  # Process noise covariance
            V = np.array([[0.01]])  # Measurement noise covariance
            
            L, P, E_kf = ct.lqe(A, G, C, W, V)
            
            # LQG controller
            A_ctrl = A - B @ K - L @ C
            B_ctrl = L
            C_ctrl = -K
            D_ctrl = np.zeros((K.shape[0], L.shape[1]))
            
            controller_ss = ct.StateSpace(A_ctrl, B_ctrl, C_ctrl, D_ctrl)
            
            self.controller_params = {
                'type': 'lqg',
                'K_matrix': K.tolist(),
                'L_matrix': L.tolist(),
                'controller_ss': controller_ss
            }
            
            self.logger.info("LQG controller designed successfully")
            return self.controller_params
            
        except Exception as e:
            self.logger.error(f"LQG design failed: {e}")
            return self.design_pid_controller(plant_tf)
    
    def _design_hinf_controller(self, plant_tf: ct.TransferFunction) -> Dict[str, float]:
        """Design H-infinity controller."""
        # Simplified H-infinity design (would need robust control toolbox for full implementation)
        self.logger.warning("H-infinity design not fully implemented, using PID")
        return self.design_pid_controller(plant_tf)
    
    def _design_adaptive_controller(self, plant_tf: ct.TransferFunction) -> Dict[str, float]:
        """Design adaptive controller."""
        # Adaptive control would require online parameter estimation
        self.logger.warning("Adaptive design not fully implemented, using PID")
        return self.design_pid_controller(plant_tf)
    
    def simulate_closed_loop_response(self, reference_signal: np.ndarray,
                                    time_vector: np.ndarray,
                                    disturbance: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Simulate closed-loop system response.
        
        Args:
            reference_signal: Reference input
            time_vector: Time vector
            disturbance: External disturbance (optional)
            
        Returns:
            Dictionary with simulation results
        """
        if not self.is_designed:
            self.design_pid_controller()
        
        # Build closed-loop system
        if self.controller_params['type'] == 'pid':
            kp = self.controller_params['kp']
            ki = self.controller_params['ki']
            kd = self.controller_params['kd']
            
            # PID transfer function
            pid_num = [kd, kp, ki]
            pid_den = [1, 0]
            pid_tf = ct.TransferFunction(pid_num, pid_den)
            
            # Default plant
            plant_num = [1]
            plant_den = [1, 2*0.1*10, 10**2]
            plant_tf = ct.TransferFunction(plant_num, plant_den)
            
            # Closed-loop transfer function
            loop_tf = ct.series(pid_tf, plant_tf)
            cl_tf = ct.feedback(loop_tf, 1)
            
        elif self.controller_params['type'] == 'lqg':
            # Use state-space controller
            controller_ss = self.controller_params['controller_ss']
            
            # Default plant (would be provided in practice)
            A_plant = np.array([[0, 1], [-100, -20]])
            B_plant = np.array([[0], [1]]) 
            C_plant = np.array([[1, 0]])
            D_plant = np.array([[0]])
            plant_ss = ct.StateSpace(A_plant, B_plant, C_plant, D_plant)
            
            # Closed-loop system
            cl_tf = ct.feedback(ct.series(controller_ss, plant_ss), 1)
        
        # Simulate response
        try:
            t, y = ct.forced_response(cl_tf, time_vector, reference_signal)
            
            # Calculate performance metrics
            settling_time = self._calculate_settling_time(t, y, reference_signal[-1])
            overshoot = self._calculate_overshoot(y, reference_signal[-1])
            steady_state_error = abs(y[-1] - reference_signal[-1])
            
            results = {
                'time': t,
                'output': y,
                'reference': reference_signal,
                'error': reference_signal - y,
                'settling_time': settling_time,
                'overshoot_percent': overshoot,
                'steady_state_error': steady_state_error
            }
            
            self.logger.info(f"Simulation complete: settling={settling_time:.3f}s, overshoot={overshoot:.1f}%")
            return results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return {}
    
    def _calculate_settling_time(self, time: np.ndarray, response: np.ndarray, 
                               final_value: float, tolerance: float = 0.02) -> float:
        """Calculate settling time (2% criterion)."""
        try:
            settling_band = tolerance * abs(final_value)
            settling_indices = np.where(abs(response - final_value) <= settling_band)[0]
            
            if len(settling_indices) > 0:
                return time[settling_indices[0]]
            else:
                return time[-1]  # Never settled
        except:
            return float('inf')
    
    def _calculate_overshoot(self, response: np.ndarray, final_value: float) -> float:
        """Calculate percentage overshoot."""
        try:
            if final_value != 0:
                max_value = np.max(response)
                overshoot = (max_value - final_value) / abs(final_value) * 100
                return max(0, overshoot)  # Only positive overshoot
            else:
                return 0
        except:
            return 0
    
    def frequency_response_analysis(self, freq_range: Tuple[float, float] = (0.1, 1000)) -> Dict[str, np.ndarray]:
        """
        Analyze frequency response of control system.
        
        Args:
            freq_range: Frequency range (Hz) as (min, max)
            
        Returns:
            Dictionary with frequency response data
        """
        if not self.is_designed:
            self.design_pid_controller()
        
        # Build control system
        if self.controller_params['type'] == 'pid':
            kp = self.controller_params['kp']
            ki = self.controller_params['ki']
            kd = self.controller_params['kd']
            
            pid_num = [kd, kp, ki]
            pid_den = [1, 0]
            controller_tf = ct.TransferFunction(pid_num, pid_den)
        
        # Default plant
        plant_num = [1]
        plant_den = [1, 2*0.1*10, 10**2]
        plant_tf = ct.TransferFunction(plant_num, plant_den)
        
        # Open-loop transfer function
        loop_tf = ct.series(controller_tf, plant_tf)
        
        # Frequency response
        omega = np.logspace(np.log10(2*PI*freq_range[0]), np.log10(2*PI*freq_range[1]), 1000)
        mag, phase, freq = ct.bode(loop_tf, omega, plot=False)
        
        # Convert to dB and degrees
        mag_db = 20 * np.log10(mag)
        phase_deg = phase * 180 / PI
        freq_hz = freq / (2 * PI)
        
        # Calculate margins
        gm, pm, wg, wp = ct.margin(loop_tf)
        gm_db = 20 * np.log10(gm) if gm > 0 else -100
        pm_deg = pm * 180 / PI if pm > 0 else 0
        
        results = {
            'frequency_hz': freq_hz,
            'magnitude_db': mag_db,
            'phase_deg': phase_deg,
            'gain_margin_db': gm_db,
            'phase_margin_deg': pm_deg,
            'gain_crossover_hz': wp / (2 * PI) if wp > 0 else 0,
            'phase_crossover_hz': wg / (2 * PI) if wg > 0 else 0
        }
        
        self.logger.info(f"Frequency analysis: GM={gm_db:.1f}dB, PM={pm_deg:.1f}°")
        return results
    
    def real_time_control_update(self, error_signal: float, dt: float) -> float:
        """
        Real-time control signal calculation.
        
        Args:
            error_signal: Current error signal
            dt: Time step (s)
            
        Returns:
            Control signal
        """
        if not self.is_designed or self.controller_params['type'] != 'pid':
            self.design_pid_controller()
        
        kp = self.controller_params['kp']
        ki = self.controller_params['ki']
        kd = self.controller_params['kd']
        
        # Initialize integrator and differentiator states if needed
        if not hasattr(self, '_integral_state'):
            self._integral_state = 0.0
            self._previous_error = 0.0
        
        # PID calculation
        proportional = kp * error_signal
        
        self._integral_state += error_signal * dt
        integral = ki * self._integral_state
        
        derivative = kd * (error_signal - self._previous_error) / dt if dt > 0 else 0
        self._previous_error = error_signal
        
        control_signal = proportional + integral + derivative
        
        self.logger.debug(f"Control update: P={proportional:.3f}, I={integral:.3f}, D={derivative:.3f}")
        return control_signal


if __name__ == "__main__":
    """Example usage and validation of interferometric control system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example optical properties
    optical_props = OpticalProperties(
        wavelength=632.8e-9,     # He-Ne laser
        refractive_index=1.5,    # Typical glass
        electro_optic_coeff=30e-12,  # m/V
        path_length=0.01         # 1 cm
    )
    
    # Controller specifications
    controller_specs = ControllerSpecs(
        gain_margin_db=19.24,
        phase_margin_deg=91.7,
        bandwidth_hz=1000
    )
    
    print("=== ADVANCED INTERFEROMETRIC CONTROL SYSTEM ===")
    
    # Create control system
    control_system = AdvancedInterferometricControl(optical_props, controller_specs)
    
    # Test phase shift calculations
    print(f"\n=== PHASE SHIFT CALCULATIONS ===")
    E_field = 1e6  # V/m
    delta_n = control_system.calculate_refractive_index_change(E_field)
    phase_shift = control_system.calculate_phase_shift(delta_n)
    
    print(f"Electric field: {E_field:.2e} V/m")
    print(f"Refractive index change: {delta_n:.6e}")
    print(f"Phase shift: {phase_shift:.4f} rad ({phase_shift*180/PI:.2f}°)")
    
    # Design PID controller
    print(f"\n=== CONTROLLER DESIGN ===")
    pid_params = control_system.design_pid_controller()
    
    print(f"PID Parameters:")
    print(f"  Kp = {pid_params['kp']:.3f}")
    print(f"  Ki = {pid_params['ki']:.3f}")
    print(f"  Kd = {pid_params['kd']:.4f}")
    print(f"Performance:")
    print(f"  Gain Margin: {pid_params['gain_margin_db']:.1f} dB")
    print(f"  Phase Margin: {pid_params['phase_margin_deg']:.1f}°")
    
    # Frequency response analysis
    print(f"\n=== FREQUENCY RESPONSE ANALYSIS ===")
    freq_response = control_system.frequency_response_analysis()
    
    print(f"Final Margins:")
    print(f"  Gain Margin: {freq_response['gain_margin_db']:.1f} dB")
    print(f"  Phase Margin: {freq_response['phase_margin_deg']:.1f}°")
    print(f"  Gain Crossover: {freq_response['gain_crossover_hz']:.1f} Hz")
    
    # Step response simulation
    print(f"\n=== STEP RESPONSE SIMULATION ===")
    t = np.linspace(0, 0.1, 1000)  # 100 ms simulation
    reference = np.ones_like(t)    # Unit step
    
    step_response = control_system.simulate_closed_loop_response(reference, t)
    
    if step_response:
        print(f"Step Response Performance:")
        print(f"  Settling Time: {step_response['settling_time']*1000:.1f} ms")
        print(f"  Overshoot: {step_response['overshoot_percent']:.1f}%")
        print(f"  Steady-State Error: {step_response['steady_state_error']:.4f}")
    
    # Real-time control example
    print(f"\n=== REAL-TIME CONTROL EXAMPLE ===")
    dt = 0.001  # 1 ms time step
    errors = [1.0, 0.8, 0.5, 0.2, 0.1]  # Decreasing error
    
    print(f"Real-time control signals:")
    for i, error in enumerate(errors):
        control_signal = control_system.real_time_control_update(error, dt)
        print(f"  Step {i+1}: error={error:.1f}, control={control_signal:.3f}")
