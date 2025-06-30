"""
Enhanced Translational Drift Control System
===========================================

This module implements advanced multi-material thermal compensation with
enhanced PID control for achieving ≤0.1 nm/hour translational drift.

Mathematical Formulation:
δ_drift(t) = ∑ᵢ₌₁ᴹ [Lᵢ × αᵢ × ΔTᵢ(t) × fᵢ_compensation(t)] ≤ 1×10⁻¹⁰ m/hr

Enhanced compensation function:
fᵢ_compensation(t) = 1 + [Kp_thermal × eᵢ(t) + Ki_thermal × ∫eᵢ(t)dt]

PID gains optimized for thermal dynamics:
Kp_thermal = (2ζωn τ - 1)/K_thermal
Ki_thermal = ωn²τ/K_thermal  
Kd_thermal = τ/K_thermal

where: ζ = 0.7, ωn = 0.21 rad/s, τ = 100s
"""

import numpy as np
from scipy.integrate import odeint, quad
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable, NamedTuple
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json

# Physical constants and requirements
DRIFT_LIMIT_M_PER_HOUR = 1e-10  # 0.1 nm/hour
SECONDS_PER_HOUR = 3600
DRIFT_RATE_LIMIT = DRIFT_LIMIT_M_PER_HOUR / SECONDS_PER_HOUR  # m/s

class MaterialType(Enum):
    """Enhanced material types with precise thermal coefficients."""
    ZERODUR = "zerodur"
    SILICON = "silicon"
    ALUMINUM = "aluminum"
    INVAR = "invar"
    STEEL = "steel"

@dataclass
class EnhancedThermalProperties:
    """Enhanced thermal properties with uncertainty quantification."""
    
    material: MaterialType
    # Enhanced thermal expansion coefficients from workspace analysis
    alpha_linear: float         # Primary expansion coefficient (1/K)
    alpha_uncertainty: float    # Uncertainty in expansion coefficient
    alpha_nonlinear: float = 0.0  # Second-order coefficient (1/K²)
    
    # Thermal dynamics parameters
    thermal_time_constant: float = 100.0  # s
    thermal_conductivity: float = 1.0     # W/(m·K)
    specific_heat: float = 500.0          # J/(kg·K)
    density: float = 2500.0               # kg/m³
    
    @classmethod
    def get_enhanced_materials(cls) -> Dict[MaterialType, 'EnhancedThermalProperties']:
        """
        Get enhanced material properties with precise coefficients from workspace analysis.
        
        Enhanced coefficients based on:
        - α_zerodur = 5×10⁻⁸ ± 2×10⁻⁹ K⁻¹  [best choice: 5 nm @ 20 mK]
        - α_silicon = 2.6×10⁻⁶ K⁻¹
        - α_aluminum = 2.3×10⁻⁵ K⁻¹
        """
        return {
            MaterialType.ZERODUR: cls(
                material=MaterialType.ZERODUR,
                alpha_linear=5e-8,        # Enhanced coefficient from workspace
                alpha_uncertainty=2e-9,   # Precise uncertainty
                alpha_nonlinear=1e-12,
                thermal_time_constant=150.0,
                thermal_conductivity=1.46,
                specific_heat=821,
                density=2530
            ),
            MaterialType.SILICON: cls(
                material=MaterialType.SILICON,
                alpha_linear=2.6e-6,
                alpha_uncertainty=0.1e-6,
                alpha_nonlinear=5e-9,
                thermal_time_constant=80.0,
                thermal_conductivity=148,
                specific_heat=712,
                density=2330
            ),
            MaterialType.ALUMINUM: cls(
                material=MaterialType.ALUMINUM,
                alpha_linear=2.3e-5,
                alpha_uncertainty=0.2e-5,
                alpha_nonlinear=1e-8,
                thermal_time_constant=50.0,
                thermal_conductivity=167,
                specific_heat=896,
                density=2700
            ),
            MaterialType.INVAR: cls(
                material=MaterialType.INVAR,
                alpha_linear=1.2e-6,
                alpha_uncertainty=0.1e-6,
                alpha_nonlinear=2e-9,
                thermal_time_constant=120.0,
                thermal_conductivity=13.8,
                specific_heat=515,
                density=8100
            ),
            MaterialType.STEEL: cls(
                material=MaterialType.STEEL,
                alpha_linear=1.2e-5,
                alpha_uncertainty=0.1e-5,
                alpha_nonlinear=8e-9,
                thermal_time_constant=90.0,
                thermal_conductivity=16.2,
                specific_heat=500,
                density=7900
            )
        }

@dataclass
class ThermalPIDParams:
    """Enhanced PID parameters for thermal drift compensation."""
    
    # Thermal control design parameters
    zeta: float = 0.7           # Damping ratio
    omega_n: float = 0.21       # Natural frequency (rad/s)
    tau: float = 100.0          # Time constant (s)
    
    # Calculated PID gains (will be computed)
    Kp_thermal: float = 0.0
    Ki_thermal: float = 0.0
    Kd_thermal: float = 0.0
    
    # Thermal system gain
    K_thermal: float = 1.0
    
    def calculate_pid_gains(self):
        """
        Calculate PID gains using enhanced formulation:
        Kp_thermal = (2ζωn τ - 1)/K_thermal
        Ki_thermal = ωn²τ/K_thermal  
        Kd_thermal = τ/K_thermal
        """
        self.Kp_thermal = (2 * self.zeta * self.omega_n * self.tau - 1) / self.K_thermal
        self.Ki_thermal = (self.omega_n**2 * self.tau) / self.K_thermal
        self.Kd_thermal = self.tau / self.K_thermal
        
        # Ensure positive gains
        self.Kp_thermal = max(0.01, self.Kp_thermal)
        self.Ki_thermal = max(0.001, self.Ki_thermal)
        self.Kd_thermal = max(0.0001, self.Kd_thermal)

class DriftComponent(NamedTuple):
    """Individual drift component analysis."""
    thermal_drift: float
    mechanical_drift: float
    electronic_drift: float
    total_drift: float

class EnhancedTranslationalDriftControl:
    """
    Enhanced translational drift control system for achieving ≤0.1 nm/hour stability.
    
    LaTeX Formulations Implemented:
    
    1. Multi-Material Drift Model:
    δ_drift(t) = ∑ᵢ₌₁ᴹ [Lᵢ × αᵢ × ΔTᵢ(t) × fᵢ_compensation(t)]
    
    2. Enhanced Compensation Function:
    fᵢ_compensation(t) = 1 + [Kp_thermal × eᵢ(t) + Ki_thermal × ∫eᵢ(t)dt]
    
    3. Optimized PID Gains:
    Kp_thermal = (2ζωn τ - 1)/K_thermal
    Ki_thermal = ωn²τ/K_thermal  
    Kd_thermal = τ/K_thermal
    
    4. Drift Rate Constraint:
    drift_rate = d(position)/dt ≤ 2.78×10⁻¹⁴ m/s
    
    5. Predictive Drift Model:
    x_predicted(t) = x₀ + v₀×t + ½a₀×t² + ∑ᵢ Aᵢ×exp(-t/τᵢ)×cos(ωᵢt + φᵢ)
    """
    
    def __init__(self, material_configs: List[Dict], 
                 pid_params: Optional[ThermalPIDParams] = None):
        """
        Initialize enhanced translational drift control system.
        
        Args:
            material_configs: List of material configuration dictionaries
            pid_params: PID parameters, uses defaults if None
        """
        self.material_configs = material_configs
        self.materials = EnhancedThermalProperties.get_enhanced_materials()
        
        # Initialize PID parameters
        self.pid_params = pid_params or ThermalPIDParams()
        self.pid_params.calculate_pid_gains()
        
        self.logger = logging.getLogger(__name__)
        
        # Control system state
        self.thermal_integrators = {}
        self.previous_errors = {}
        self.drift_history = []
        self.compensation_history = []
        
        # Predictive model state
        self.model_coefficients = {}
        self.is_model_trained = False
        
        # Performance monitoring
        self.drift_rate_history = []
        self.constraint_violations = 0
        
        # Initialize integrator states for each material
        for i, config in enumerate(material_configs):
            material = config['material']
            self.thermal_integrators[f'material_{i}_{material.value}'] = 0.0
            self.previous_errors[f'material_{i}_{material.value}'] = 0.0
        
        self.logger.info(f"Enhanced drift control initialized for {len(material_configs)} materials")
    
    def calculate_multi_material_drift(self, temperature_deltas: List[float],
                                     timestamp: float = None) -> DriftComponent:
        """
        Calculate multi-material thermal drift with enhanced compensation.
        
        LaTeX: δ_drift(t) = ∑ᵢ₌₁ᴹ [Lᵢ × αᵢ × ΔTᵢ(t) × fᵢ_compensation(t)]
        
        Args:
            temperature_deltas: Temperature changes for each material (K)
            timestamp: Current timestamp for history tracking
            
        Returns:
            DriftComponent with all drift contributions
        """
        if len(temperature_deltas) != len(self.material_configs):
            raise ValueError("Temperature deltas must match material configurations")
        
        total_thermal_drift = 0.0
        material_drifts = {}
        
        for i, (delta_T, config) in enumerate(zip(temperature_deltas, self.material_configs)):
            material = config['material']
            length = config['length']
            contribution_factor = config.get('contribution_factor', 1.0)
            
            if material not in self.materials:
                raise ValueError(f"Unknown material: {material}")
            
            props = self.materials[material]
            
            # Basic thermal expansion
            basic_expansion = props.alpha_linear * length * delta_T
            
            # Add nonlinear terms
            nonlinear_expansion = props.alpha_nonlinear * length * (delta_T**2)
            
            # Calculate compensation
            material_key = f'material_{i}_{material.value}'
            compensation_factor = self._calculate_compensation_factor(
                delta_T, material_key, props
            )
            
            # Total material drift
            material_drift = (basic_expansion + nonlinear_expansion) * compensation_factor * contribution_factor
            total_thermal_drift += material_drift
            
            material_drifts[material_key] = {
                'basic_expansion': basic_expansion,
                'nonlinear_expansion': nonlinear_expansion,
                'compensation_factor': compensation_factor,
                'total_drift': material_drift,
                'temperature_delta': delta_T
            }
        
        # Add other drift sources
        mechanical_drift = self._calculate_mechanical_drift(timestamp)
        electronic_drift = self._calculate_electronic_drift(timestamp)
        
        total_drift = total_thermal_drift + mechanical_drift + electronic_drift
        
        drift_component = DriftComponent(
            thermal_drift=total_thermal_drift,
            mechanical_drift=mechanical_drift,
            electronic_drift=electronic_drift,
            total_drift=total_drift
        )
        
        # Store in history
        if timestamp is not None:
            self.drift_history.append({
                'timestamp': timestamp,
                'drift_component': drift_component,
                'material_drifts': material_drifts,
                'temperature_deltas': temperature_deltas
            })
        
        self.logger.debug(f"Multi-material drift: thermal={total_thermal_drift*1e9:.3f}nm, "
                         f"total={total_drift*1e9:.3f}nm")
        
        return drift_component
    
    def _calculate_compensation_factor(self, delta_T: float, material_key: str,
                                     props: EnhancedThermalProperties) -> float:
        """
        Calculate enhanced compensation factor using PID control.
        
        LaTeX: fᵢ_compensation(t) = 1 + [Kp_thermal × eᵢ(t) + Ki_thermal × ∫eᵢ(t)dt]
        
        Args:
            delta_T: Temperature change (K)
            material_key: Unique material identifier
            props: Material thermal properties
            
        Returns:
            Compensation factor
        """
        # Error signal (deviation from reference temperature)
        error = delta_T  # Assuming reference is 0
        
        # PID compensation calculation
        # Proportional term
        proportional = self.pid_params.Kp_thermal * error
        
        # Integral term
        if material_key not in self.thermal_integrators:
            self.thermal_integrators[material_key] = 0.0
        
        dt = 1.0  # Time step assumption (would be actual dt in real system)
        self.thermal_integrators[material_key] += error * dt
        
        # Anti-windup for integrator
        max_integral = 0.1  # Maximum integral contribution
        integral_limit = max_integral / self.pid_params.Ki_thermal if self.pid_params.Ki_thermal > 0 else 1e6
        self.thermal_integrators[material_key] = np.clip(
            self.thermal_integrators[material_key], 
            -integral_limit, 
            integral_limit
        )
        
        integral = self.pid_params.Ki_thermal * self.thermal_integrators[material_key]
        
        # Derivative term
        if material_key not in self.previous_errors:
            self.previous_errors[material_key] = error
        
        derivative = self.pid_params.Kd_thermal * (error - self.previous_errors[material_key]) / dt
        self.previous_errors[material_key] = error
        
        # Total PID compensation
        pid_compensation = proportional + integral + derivative
        
        # Compensation factor (additive correction)
        compensation_factor = 1.0 + pid_compensation
        
        # Store compensation history
        self.compensation_history.append({
            'material_key': material_key,
            'error': error,
            'proportional': proportional,
            'integral': integral,
            'derivative': derivative,
            'compensation_factor': compensation_factor
        })
        
        return compensation_factor
    
    def _calculate_mechanical_drift(self, timestamp: Optional[float]) -> float:
        """
        Calculate mechanical drift contribution (vibration, aging, stress relaxation).
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Mechanical drift (m)
        """
        if timestamp is None:
            return 0.0
        
        # Model mechanical drift as combination of aging and vibration
        # Aging drift: exponential approach to equilibrium
        aging_drift = 1e-12 * (1 - np.exp(-timestamp / 86400))  # 1 pm/day aging
        
        # Vibrational drift: random walk with time correlation
        vibration_drift = 5e-13 * np.sqrt(timestamp) * np.random.randn()  # Random component
        
        mechanical_drift = aging_drift + vibration_drift
        
        return mechanical_drift
    
    def _calculate_electronic_drift(self, timestamp: Optional[float]) -> float:
        """
        Calculate electronic drift contribution (sensor drift, amplifier drift).
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Electronic drift (m)
        """
        if timestamp is None:
            return 0.0
        
        # Model electronic drift as 1/f noise plus thermal effects
        # 1/f noise component
        f_noise_drift = 2e-13 * np.log(1 + timestamp / 3600)  # 0.2 pm/hour log drift
        
        # Thermal electronic drift
        thermal_electronic = 1e-13 * timestamp / 3600  # 0.1 pm/hour linear drift
        
        electronic_drift = f_noise_drift + thermal_electronic
        
        return electronic_drift
    
    def predictive_drift_model(self, time_vector: np.ndarray,
                             initial_conditions: Dict[str, float] = None) -> np.ndarray:
        """
        Implement predictive drift model for feedforward compensation.
        
        LaTeX: x_predicted(t) = x₀ + v₀×t + ½a₀×t² + ∑ᵢ Aᵢ×exp(-t/τᵢ)×cos(ωᵢt + φᵢ)
        
        Args:
            time_vector: Time points for prediction (s)
            initial_conditions: Initial state dictionary
            
        Returns:
            Predicted drift values (m)
        """
        if initial_conditions is None:
            initial_conditions = {
                'x0': 0.0,           # Initial position
                'v0': 1e-14,         # Initial velocity (m/s)
                'a0': 1e-18,         # Initial acceleration (m/s²)
                'amplitudes': [1e-12, 5e-13, 2e-13],  # Exponential amplitudes
                'time_constants': [3600, 600, 60],     # Time constants (s)
                'frequencies': [1e-4, 1e-3, 1e-2],    # Angular frequencies (rad/s)
                'phases': [0, np.pi/4, np.pi/2]        # Phase angles (rad)
            }
        
        x0 = initial_conditions['x0']
        v0 = initial_conditions['v0']
        a0 = initial_conditions['a0']
        A = initial_conditions['amplitudes']
        tau = initial_conditions['time_constants']
        omega = initial_conditions['frequencies']
        phi = initial_conditions['phases']
        
        # Polynomial terms
        polynomial_drift = x0 + v0 * time_vector + 0.5 * a0 * time_vector**2
        
        # Exponential decay terms with oscillation
        exponential_drift = np.zeros_like(time_vector)
        for i in range(len(A)):
            exponential_drift += A[i] * np.exp(-time_vector / tau[i]) * \
                               np.cos(omega[i] * time_vector + phi[i])
        
        predicted_drift = polynomial_drift + exponential_drift
        
        return predicted_drift
    
    def train_predictive_model(self, historical_data: List[Dict]) -> Dict[str, float]:
        """
        Train predictive model using historical drift data.
        
        Args:
            historical_data: List of historical drift measurements
            
        Returns:
            Trained model coefficients
        """
        if len(historical_data) < 10:
            self.logger.warning("Insufficient data for model training")
            return {}
        
        # Extract time and drift data
        times = np.array([entry['timestamp'] for entry in historical_data])
        drifts = np.array([entry['drift_component'].total_drift for entry in historical_data])
        
        # Normalize time to start from 0
        times = times - times[0]
        
        def model_func(params, t):
            """Predictive model function for fitting."""
            x0, v0, a0 = params[:3]
            A = params[3:6]
            tau = params[6:9]
            omega = params[9:12]
            phi = params[12:15]
            
            polynomial = x0 + v0 * t + 0.5 * a0 * t**2
            exponential = sum(A[i] * np.exp(-t / tau[i]) * np.cos(omega[i] * t + phi[i]) 
                            for i in range(3))
            
            return polynomial + exponential
        
        def objective(params):
            """Objective function for parameter fitting."""
            try:
                predicted = np.array([model_func(params, t) for t in times])
                return np.sum((predicted - drifts)**2)
            except:
                return 1e12
        
        # Initial parameter guess
        initial_params = [
            0.0,           # x0
            np.mean(np.diff(drifts) / np.diff(times)),  # v0 (estimated velocity)
            0.0,           # a0
            1e-12, 5e-13, 2e-13,  # A (amplitudes)
            3600, 600, 60,        # tau (time constants)
            1e-4, 1e-3, 1e-2,     # omega (frequencies)
            0, np.pi/4, np.pi/2   # phi (phases)
        ]
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, initial_params, method='Nelder-Mead')
            
            if result.success:
                # Store trained coefficients
                self.model_coefficients = {
                    'x0': result.x[0],
                    'v0': result.x[1],
                    'a0': result.x[2],
                    'amplitudes': result.x[3:6].tolist(),
                    'time_constants': result.x[6:9].tolist(),
                    'frequencies': result.x[9:12].tolist(),
                    'phases': result.x[12:15].tolist()
                }
                
                self.is_model_trained = True
                
                # Calculate model accuracy
                predicted = np.array([model_func(result.x, t) for t in times])
                rms_error = np.sqrt(np.mean((predicted - drifts)**2))
                
                self.logger.info(f"Predictive model trained successfully: RMS error = {rms_error*1e9:.3f} nm")
                
                return self.model_coefficients
            else:
                self.logger.warning("Predictive model training failed")
                return {}
                
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return {}
    
    def feedforward_compensation(self, prediction_horizon: float = 3600.0) -> np.ndarray:
        """
        Calculate feedforward compensation based on predictive model.
        
        Args:
            prediction_horizon: Time horizon for prediction (s)
            
        Returns:
            Feedforward compensation signal
        """
        if not self.is_model_trained:
            self.logger.warning("Predictive model not trained, using zero feedforward")
            return np.array([0.0])
        
        # Generate prediction time vector
        dt = 60.0  # 1 minute resolution
        time_vector = np.arange(0, prediction_horizon, dt)
        
        # Predict drift
        predicted_drift = self.predictive_drift_model(time_vector, self.model_coefficients)
        
        # Feedforward compensation (negative of predicted drift)
        feedforward_signal = -predicted_drift
        
        return feedforward_signal
    
    def check_drift_constraint(self, drift_rate: float, 
                             measurement_window: float = 3600.0) -> Dict[str, bool]:
        """
        Check if translational drift constraint is satisfied.
        
        Args:
            drift_rate: Current drift rate (m/s)
            measurement_window: Time window for constraint checking (s)
            
        Returns:
            Dictionary with constraint satisfaction results
        """
        # Convert drift rate to nm/hour for reporting
        drift_rate_nm_per_hour = drift_rate * 1e9 * 3600
        
        # Check constraint
        constraint_satisfied = abs(drift_rate) <= DRIFT_RATE_LIMIT
        
        # Calculate projected drift over measurement window
        projected_drift = drift_rate * measurement_window
        projected_drift_nm = projected_drift * 1e9
        
        # Store in history
        self.drift_rate_history.append({
            'timestamp': time.time(),
            'drift_rate_m_per_s': drift_rate,
            'drift_rate_nm_per_hour': drift_rate_nm_per_hour,
            'constraint_satisfied': constraint_satisfied,
            'projected_drift_nm': projected_drift_nm
        })
        
        if not constraint_satisfied:
            self.constraint_violations += 1
        
        results = {
            'constraint_satisfied': constraint_satisfied,
            'drift_rate_nm_per_hour': drift_rate_nm_per_hour,
            'requirement_nm_per_hour': DRIFT_LIMIT_M_PER_HOUR * 1e9,
            'projected_drift_nm': projected_drift_nm,
            'margin_factor': DRIFT_RATE_LIMIT / abs(drift_rate) if drift_rate != 0 else float('inf'),
            'total_violations': self.constraint_violations
        }
        
        return results
    
    def optimize_pid_parameters(self, target_settling_time: float = 600.0) -> ThermalPIDParams:
        """
        Optimize PID parameters for target settling time and stability.
        
        Args:
            target_settling_time: Target settling time (s)
            
        Returns:
            Optimized PID parameters
        """
        def objective(params):
            """Optimization objective for PID tuning."""
            zeta, omega_n, tau = params
            
            # Calculate PID gains
            test_pid = ThermalPIDParams(zeta=zeta, omega_n=omega_n, tau=tau)
            test_pid.calculate_pid_gains()
            
            # Settling time for second-order system
            if zeta > 1:
                # Overdamped
                settling_time = 4 * tau
            elif zeta == 1:
                # Critically damped
                settling_time = 4 / omega_n
            else:
                # Underdamped
                settling_time = 4 / (zeta * omega_n)
            
            # Objective: minimize deviation from target settling time
            settling_error = (settling_time - target_settling_time)**2 / target_settling_time**2
            
            # Penalty for poor damping
            damping_penalty = (zeta - 0.7)**2 if zeta < 0.5 or zeta > 1.5 else 0
            
            return settling_error + 0.1 * damping_penalty
        
        # Optimization bounds
        bounds = [
            (0.3, 1.2),   # zeta
            (0.01, 0.5),  # omega_n
            (50, 200)     # tau
        ]
        
        # Initial guess
        initial_params = [self.pid_params.zeta, self.pid_params.omega_n, self.pid_params.tau]
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimized_pid = ThermalPIDParams(
                    zeta=result.x[0],
                    omega_n=result.x[1],
                    tau=result.x[2]
                )
                optimized_pid.calculate_pid_gains()
                
                self.logger.info(f"PID optimization successful: ζ={optimized_pid.zeta:.3f}, "
                               f"ωn={optimized_pid.omega_n:.3f}, Kp={optimized_pid.Kp_thermal:.3f}")
                
                return optimized_pid
            else:
                self.logger.warning("PID optimization failed, using current parameters")
                return self.pid_params
                
        except Exception as e:
            self.logger.error(f"PID optimization error: {e}")
            return self.pid_params
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary for drift control.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.drift_history:
            return {'status': 'no_data'}
        
        # Extract recent performance data
        recent_drifts = [entry['drift_component'] for entry in self.drift_history[-100:]]
        recent_rates = [entry for entry in self.drift_rate_history[-100:]]
        
        if not recent_drifts:
            return {'status': 'insufficient_data'}
        
        # Calculate statistics
        thermal_drifts = [d.thermal_drift for d in recent_drifts]
        total_drifts = [d.total_drift for d in recent_drifts]
        
        if recent_rates:
            constraint_satisfaction = [r['constraint_satisfied'] for r in recent_rates]
            drift_rates_nm_hr = [r['drift_rate_nm_per_hour'] for r in recent_rates]
        else:
            constraint_satisfaction = []
            drift_rates_nm_hr = []
        
        performance = {
            'drift_constraint_satisfaction': {
                'success_rate_percent': np.mean(constraint_satisfaction) * 100 if constraint_satisfaction else 0,
                'current_requirement_nm_per_hour': DRIFT_LIMIT_M_PER_HOUR * 1e9,
                'total_violations': self.constraint_violations
            },
            'drift_statistics': {
                'rms_thermal_drift_nm': np.sqrt(np.mean(np.array(thermal_drifts)**2)) * 1e9,
                'rms_total_drift_nm': np.sqrt(np.mean(np.array(total_drifts)**2)) * 1e9,
                'max_thermal_drift_nm': np.max(np.abs(thermal_drifts)) * 1e9,
                'max_total_drift_nm': np.max(np.abs(total_drifts)) * 1e9
            },
            'control_system_performance': {
                'pid_parameters': {
                    'Kp': self.pid_params.Kp_thermal,
                    'Ki': self.pid_params.Ki_thermal,
                    'Kd': self.pid_params.Kd_thermal
                },
                'model_trained': self.is_model_trained,
                'compensation_entries': len(self.compensation_history),
                'materials_count': len(self.material_configs)
            }
        }
        
        if drift_rates_nm_hr:
            performance['drift_rate_statistics'] = {
                'current_rate_nm_per_hour': drift_rates_nm_hr[-1] if drift_rates_nm_hr else 0,
                'average_rate_nm_per_hour': np.mean(drift_rates_nm_hr),
                'max_rate_nm_per_hour': np.max(np.abs(drift_rates_nm_hr))
            }
        
        return performance


if __name__ == "__main__":
    """Example usage of enhanced translational drift control."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== ENHANCED TRANSLATIONAL DRIFT CONTROL ===")
    print("Target: ≤0.1 nm/hour translational drift")
    
    # Define material configurations
    material_configs = [
        {
            'material': MaterialType.ZERODUR,
            'length': 3e-3,
            'contribution_factor': 0.7
        },
        {
            'material': MaterialType.SILICON,
            'length': 1.5e-3,
            'contribution_factor': 0.3
        }
    ]
    
    # Initialize controller
    controller = EnhancedTranslationalDriftControl(material_configs)
    
    print(f"\nPID Parameters:")
    print(f"  Kp_thermal: {controller.pid_params.Kp_thermal:.4f}")
    print(f"  Ki_thermal: {controller.pid_params.Ki_thermal:.4f}")
    print(f"  Kd_thermal: {controller.pid_params.Kd_thermal:.4f}")
    
    # Simulate temperature changes
    temperature_deltas = [0.5, -0.3]  # K
    
    # Calculate drift
    drift_component = controller.calculate_multi_material_drift(
        temperature_deltas, timestamp=time.time()
    )
    
    print(f"\nDrift Components:")
    print(f"  Thermal drift: {drift_component.thermal_drift*1e9:.3f} nm")
    print(f"  Mechanical drift: {drift_component.mechanical_drift*1e9:.3f} nm")
    print(f"  Electronic drift: {drift_component.electronic_drift*1e9:.3f} nm")
    print(f"  Total drift: {drift_component.total_drift*1e9:.3f} nm")
    
    # Check constraint
    drift_rate = drift_component.total_drift / 3600  # Convert to rate
    constraint_results = controller.check_drift_constraint(drift_rate)
    
    print(f"\nConstraint Satisfaction:")
    print(f"  Constraint satisfied: {'✓' if constraint_results['constraint_satisfied'] else '✗'}")
    print(f"  Drift rate: {constraint_results['drift_rate_nm_per_hour']:.3f} nm/hour")
    print(f"  Requirement: {constraint_results['requirement_nm_per_hour']:.1f} nm/hour")
    print(f"  Margin: {constraint_results['margin_factor']:.2f}x")
    
    # Predictive modeling
    print(f"\nPredictive Modeling:")
    time_vector = np.linspace(0, 3600, 61)  # 1 hour prediction
    predicted_drift = controller.predictive_drift_model(time_vector)
    
    print(f"  Predicted drift range: {np.min(predicted_drift)*1e9:.3f} to {np.max(predicted_drift)*1e9:.3f} nm")
    print(f"  Predicted 1-hour drift: {predicted_drift[-1]*1e9:.3f} nm")
    
    # PID optimization
    print(f"\nOptimizing PID parameters...")
    optimized_pid = controller.optimize_pid_parameters(target_settling_time=300)
    
    print(f"Optimized PID Parameters:")
    print(f"  Kp_thermal: {optimized_pid.Kp_thermal:.4f}")
    print(f"  Ki_thermal: {optimized_pid.Ki_thermal:.4f}")
    print(f"  Kd_thermal: {optimized_pid.Kd_thermal:.4f}")
    
    # Performance summary
    performance = controller.get_performance_summary()
    if 'drift_constraint_satisfaction' in performance:
        print(f"\nPerformance Summary:")
        dcs = performance['drift_constraint_satisfaction']
        print(f"  Success rate: {dcs['success_rate_percent']:.1f}%")
        print(f"  Requirement: {dcs['current_requirement_nm_per_hour']:.1f} nm/hour")
        
        ds = performance['drift_statistics']
        print(f"  RMS thermal drift: {ds['rms_thermal_drift_nm']:.3f} nm")
        print(f"  RMS total drift: {ds['rms_total_drift_nm']:.3f} nm")
