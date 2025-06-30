"""
Digital Twin Integration Module
==============================

This module integrates all digital twin components into a unified system,
providing seamless coordination between multi-physics modeling, Bayesian
state estimation, uncertainty propagation, predictive control, and validation.

System Architecture:

┌─────────────────────────────────────────────────────────────────────┐
│                    Digital Twin Integration                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  Multi-Physics  │  │   Bayesian      │  │  Uncertainty    │    │
│  │  Digital Twin   │←→│ State Estimation│←→│  Propagation    │    │
│  │     Core        │  │                 │  │                 │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│           ↕                     ↕                     ↕            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   Predictive    │  │   Validation    │  │   Enhanced      │    │
│  │    Control      │  │   Framework     │  │   Control       │    │
│  │                 │  │                 │  │  Architecture   │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘

Integration Targets:
- Real-time performance: ≤ 1 ms latency
- System fidelity: R² ≥ 0.99
- Uncertainty coverage: ≥ 95%
- Control performance: ≤ 1 µrad parallelism
"""

import numpy as np
from scipy import linalg, optimize
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import warnings

# Import all digital twin components
try:
    from .multi_physics_digital_twin import (
        MultiPhysicsDigitalTwin, MultiPhysicsState, 
        DigitalTwinParameters, SynchronizationResult
    )
    from .bayesian_state_estimation import (
        BayesianStateEstimationSystem, EstimationParameters,
        EstimationType, MeasurementType
    )
    from .uncertainty_propagation import (
        UncertaintyPropagationSystem, UncertainVariable,
        UncertaintyParameters, DistributionType, UncertaintyMethod
    )
    from .predictive_control import (
        PredictiveControlSystem, MPCParameters, MPCType,
        FailurePredictionSystem
    )
    from .validation_framework import (
        DigitalTwinValidationFramework, ValidationParameters,
        ValidationType, ModelSelectionCriterion
    )
except ImportError:
    # Handle relative imports for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from multi_physics_digital_twin import (
        MultiPhysicsDigitalTwin, MultiPhysicsState, 
        DigitalTwinParameters, SynchronizationResult
    )
    from bayesian_state_estimation import (
        BayesianStateEstimationSystem, EstimationParameters,
        EstimationType, MeasurementType
    )
    from uncertainty_propagation import (
        UncertaintyPropagationSystem, UncertainVariable,
        UncertaintyParameters, DistributionType, UncertaintyMethod
    )
    from predictive_control import (
        PredictiveControlSystem, MPCParameters, MPCType,
        FailurePredictionSystem
    )
    from validation_framework import (
        DigitalTwinValidationFramework, ValidationParameters,
        ValidationType, ModelSelectionCriterion
    )

# Import control system
try:
    from ..control.enhanced_angular_parallelism_control import (
        EnhancedAngularParallelismControl, ParallelismControllerParams
    )
except ImportError:
    # Fallback for standalone execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'control'))
    from enhanced_angular_parallelism_control import (
        EnhancedAngularParallelismControl, ParallelismControllerParams
    )

# Performance targets for integrated system
INTEGRATION_LATENCY_TARGET = 1e-3  # 1 ms
SYSTEM_FIDELITY_TARGET = 0.99  # R²
UNCERTAINTY_COVERAGE_TARGET = 0.95  # 95%
CONTROL_PERFORMANCE_TARGET = 1e-6  # 1 µrad

class IntegrationMode(Enum):
    """Digital twin integration modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    HYBRID = "hybrid"
    REAL_TIME = "real_time"

class ComponentStatus(Enum):
    """Component status indicators."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    SYNCHRONIZED = "synchronized"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class IntegrationParameters:
    """Parameters for digital twin integration."""
    
    # Integration mode
    mode: IntegrationMode = IntegrationMode.HYBRID
    
    # Timing parameters
    update_frequency_hz: float = 1000.0  # 1 kHz
    sync_timeout_s: float = 0.1
    max_latency_s: float = INTEGRATION_LATENCY_TARGET
    
    # Component parameters
    digital_twin_params: DigitalTwinParameters = field(default_factory=DigitalTwinParameters)
    estimation_params: EstimationParameters = field(default_factory=EstimationParameters)
    uncertainty_params: UncertaintyParameters = field(default_factory=UncertaintyParameters)
    mpc_params: MPCParameters = field(default_factory=MPCParameters)
    validation_params: ValidationParameters = field(default_factory=ValidationParameters)
    control_params: ParallelismControllerParams = field(default_factory=ParallelismControllerParams)
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_window_size: int = 1000
    
    # Error handling
    enable_error_recovery: bool = True
    max_retries: int = 3
    
    # Parallel processing
    use_parallel_processing: bool = True
    max_workers: int = 4

class IntegratedDigitalTwin:
    """
    Integrated digital twin system combining all components.
    
    Features:
    1. Multi-physics digital twin core
    2. Bayesian state estimation
    3. Uncertainty propagation
    4. Predictive control
    5. Validation framework
    6. Enhanced control architecture
    7. Real-time performance monitoring
    8. Automatic error recovery
    """
    
    def __init__(self, params: Optional[IntegrationParameters] = None):
        """
        Initialize integrated digital twin system.
        
        Args:
            params: Integration parameters
        """
        self.params = params or IntegrationParameters()
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.component_status = {}
        
        # Initialize components
        self._initialize_components()
        
        # Performance monitoring
        self.performance_metrics = deque(maxlen=self.params.monitoring_window_size)
        self.integration_history = deque(maxlen=1000)
        
        # Threading for real-time operation
        self.executor = ThreadPoolExecutor(max_workers=self.params.max_workers)
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Current system state
        self.current_state = MultiPhysicsState()
        self.current_control = np.zeros(3)
        self.current_measurements = {}
        
        self.logger.info("Integrated digital twin system initialized")
    
    def _initialize_components(self):
        """Initialize all digital twin components."""
        try:
            # Multi-physics digital twin core
            self.digital_twin = MultiPhysicsDigitalTwin(self.params.digital_twin_params)
            self.component_status['digital_twin'] = ComponentStatus.INITIALIZED
            
            # Bayesian state estimation
            self.state_estimator = BayesianStateEstimationSystem(
                state_size=18,  # 6 mechanical + 6 thermal + 3 electromagnetic + 3 quantum
                measurement_size=6,  # Position and velocity measurements
                estimation_params=self.params.estimation_params
            )
            self.component_status['state_estimator'] = ComponentStatus.INITIALIZED
            
            # Uncertainty propagation
            # Define uncertain variables for Casimir system
            uncertain_variables = [
                UncertainVariable("casimir_coefficient", DistributionType.NORMAL,
                                {'mean': 1.0, 'std': 0.05}),
                UncertainVariable("gap_distance", DistributionType.NORMAL,
                                {'mean': 100e-9, 'std': 5e-9}),
                UncertainVariable("temperature", DistributionType.UNIFORM,
                                {'low': 295.0, 'high': 305.0}),
                UncertainVariable("material_property", DistributionType.BETA,
                                {'alpha': 2.0, 'beta': 5.0})
            ]
            
            self.uncertainty_propagator = UncertaintyPropagationSystem(
                uncertain_variables, self.params.uncertainty_params
            )
            self.component_status['uncertainty_propagator'] = ComponentStatus.INITIALIZED
            
            # Predictive control
            self.predictive_controller = PredictiveControlSystem(
                state_size=18, control_size=3, params=self.params.mpc_params
            )
            self.component_status['predictive_controller'] = ComponentStatus.INITIALIZED
            
            # Enhanced angular parallelism control
            self.angular_controller = EnhancedAngularParallelismControl(
                params=self.params.control_params, n_actuators=5
            )
            self.component_status['angular_controller'] = ComponentStatus.INITIALIZED
            
            # Validation framework
            self.validation_framework = DigitalTwinValidationFramework(
                self.params.validation_params
            )
            self.component_status['validation_framework'] = ComponentStatus.INITIALIZED
            
            # Initialize system matrices for linear MPC
            self._setup_system_matrices()
            
            self.is_initialized = True
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def _setup_system_matrices(self):
        """Setup system matrices for linear MPC controller."""
        # Simplified linear system matrices for Casimir nanopositioning
        dt = self.params.mpc_params.sample_time
        
        # State: [x, y, z, vx, vy, vz, T_mech, T_thermal, E_field, ...]
        A = np.eye(18)
        
        # Position integration
        A[0, 3] = dt  # x += vx * dt
        A[1, 4] = dt  # y += vy * dt
        A[2, 5] = dt  # z += vz * dt
        
        # Simple thermal dynamics
        A[6, 6] = 0.99  # Mechanical temperature decay
        A[7, 7] = 0.999  # Thermal temperature decay
        
        # Electromagnetic field decay
        A[8, 8] = 0.98
        
        # Control input matrix
        B = np.zeros((18, 3))
        B[3, 0] = dt / 1e-12  # Force to acceleration (assuming 1 pg mass)
        B[4, 1] = dt / 1e-12
        B[5, 2] = dt / 1e-12
        
        # Add linear controller to predictive control system
        self.predictive_controller.add_linear_controller(A, B)
        
        self.logger.info("System matrices configured for linear MPC")
    
    def start_real_time_operation(self):
        """Start real-time digital twin operation."""
        if self.is_running:
            self.logger.warning("Digital twin already running")
            return
        
        if not self.is_initialized:
            raise RuntimeError("Digital twin not initialized")
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._real_time_update_loop,
            name="DigitalTwinUpdateLoop"
        )
        self.update_thread.start()
        
        # Update component status
        for component in self.component_status:
            if self.component_status[component] == ComponentStatus.INITIALIZED:
                self.component_status[component] = ComponentStatus.RUNNING
        
        self.logger.info(f"Real-time operation started at {self.params.update_frequency_hz} Hz")
    
    def stop_real_time_operation(self):
        """Stop real-time digital twin operation."""
        if not self.is_running:
            return
        
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        self.is_running = False
        
        # Update component status
        for component in self.component_status:
            if self.component_status[component] == ComponentStatus.RUNNING:
                self.component_status[component] = ComponentStatus.INITIALIZED
        
        self.logger.info("Real-time operation stopped")
    
    def _real_time_update_loop(self):
        """Real-time update loop for digital twin."""
        update_period = 1.0 / self.params.update_frequency_hz
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            try:
                # Perform integrated update
                self._integrated_update_step()
                
                # Calculate timing
                update_time = time.time() - start_time
                
                # Sleep for remaining time
                remaining_time = update_period - update_time
                if remaining_time > 0:
                    time.sleep(remaining_time)
                elif update_time > self.params.max_latency_s:
                    self.logger.warning(f"Update exceeded latency target: {update_time:.4f}s")
                
            except Exception as e:
                self.logger.error(f"Real-time update error: {e}")
                if not self.params.enable_error_recovery:
                    break
    
    def _integrated_update_step(self):
        """Perform one integrated update step."""
        start_time = time.time()
        
        # 1. State estimation update
        if self.current_measurements:
            estimation_result = self._update_state_estimation()
        
        # 2. Digital twin synchronization
        sync_result = self._synchronize_digital_twin()
        
        # 3. Uncertainty propagation
        uncertainty_result = self._propagate_uncertainty()
        
        # 4. Predictive control
        control_result = self._update_predictive_control()
        
        # 5. Angular parallelism control
        angular_control_result = self._update_angular_control()
        
        # 6. Performance monitoring
        self._update_performance_monitoring(start_time)
        
        # Store integration result
        integration_result = {
            'timestamp': time.time(),
            'state_estimation': estimation_result if 'estimation_result' in locals() else None,
            'digital_twin_sync': sync_result,
            'uncertainty_propagation': uncertainty_result,
            'predictive_control': control_result,
            'angular_control': angular_control_result,
            'update_time_s': time.time() - start_time
        }
        
        self.integration_history.append(integration_result)
    
    def _update_state_estimation(self):
        """Update Bayesian state estimation."""
        try:
            # Convert measurements to appropriate format
            measurement_vector = self._measurements_to_vector()
            
            # Define measurement function
            def measurement_function(state):
                # Extract position and velocity from full state
                return state[:6]  # [x, y, z, vx, vy, vz]
            
            # Define dynamics function
            def dynamics_function(state, control, dt):
                return self.digital_twin.f_coupled(state, control, dt)
            
            # Perform estimation
            result = self.state_estimator.estimate(
                measurement_vector, measurement_function,
                dynamics_function, self.current_control,
                self.params.mpc_params.sample_time
            )
            
            # Update current state
            self.current_state = self._vector_to_state(result.state_estimate)
            
            return result
            
        except Exception as e:
            self.logger.error(f"State estimation update failed: {e}")
            return None
    
    def _synchronize_digital_twin(self):
        """Synchronize digital twin with estimated state."""
        try:
            # Real-time measurements (simulated)
            real_measurements = {
                'position': self.current_state.mechanical.position,
                'velocity': self.current_state.mechanical.velocity,
                'temperature': self.current_state.thermal.temperature_distribution[0],
                'electromagnetic_field': self.current_state.electromagnetic.field_strength[0]
            }
            
            # Synchronize digital twin
            sync_result = self.digital_twin.synchronize_with_real_system(
                real_measurements, adaptive_correction=True
            )
            
            return sync_result
            
        except Exception as e:
            self.logger.error(f"Digital twin synchronization failed: {e}")
            return None
    
    def _propagate_uncertainty(self):
        """Propagate uncertainty through the system."""
        try:
            # Define model function for uncertainty propagation
            def casimir_model(inputs):
                """Simplified Casimir force model for uncertainty propagation."""
                coeff, gap, temp, material = inputs
                
                # Physical constants
                hbar_c = 1.97e-25  # ħc in J⋅m
                A = 1e-6  # Area in m²
                
                # Temperature factor
                temp_factor = 1.0 + 0.01 * (temp - 300)
                
                # Casimir force
                force = coeff * (hbar_c * np.pi**2 / 240) * (A / gap**4) * material * temp_factor
                
                return force
            
            # Propagate uncertainty
            result = self.uncertainty_propagator.propagate_uncertainty(
                casimir_model, UncertaintyMethod.MONTE_CARLO
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Uncertainty propagation failed: {e}")
            return None
    
    def _update_predictive_control(self):
        """Update predictive control."""
        try:
            # Current state vector
            current_state_vector = self._state_to_vector(self.current_state)
            
            # Reference (target state)
            reference = np.zeros(18)  # Target is zero for all states
            
            # Perform predictive control step
            control_result = self.predictive_controller.control_step(
                current_state_vector, reference
            )
            
            # Update current control
            self.current_control = control_result['control_signal']
            
            return control_result
            
        except Exception as e:
            self.logger.error(f"Predictive control update failed: {e}")
            return None
    
    def _update_angular_control(self):
        """Update angular parallelism control."""
        try:
            # Simulate actuator forces (in practice, these would be measured)
            actuator_forces = np.array([1e-9, 1.1e-9, 0.9e-9, 1.05e-9, 0.95e-9])
            target_force = 1e-9
            actuator_positions = np.linspace(-50e-6, 50e-6, 5)
            
            # Calculate angular errors
            angular_errors = self.angular_controller.calculate_angular_error(
                actuator_forces, target_force, actuator_positions
            )
            
            # Multi-rate control update
            control_signals = self.angular_controller.multi_rate_control_update(
                angular_errors
            )
            
            # Check constraints
            constraint_results = self.angular_controller.check_parallelism_constraint(
                angular_errors
            )
            
            return {
                'angular_errors': angular_errors,
                'control_signals': control_signals,
                'constraint_results': constraint_results
            }
            
        except Exception as e:
            self.logger.error(f"Angular control update failed: {e}")
            return None
    
    def _update_performance_monitoring(self, start_time: float):
        """Update performance monitoring metrics."""
        update_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = {
            'timestamp': time.time(),
            'update_time_s': update_time,
            'latency_target_met': update_time <= self.params.max_latency_s,
            'frequency_hz': 1.0 / update_time if update_time > 0 else 0,
            'component_status': dict(self.component_status),
            'system_health': self._calculate_system_health()
        }
        
        self.performance_metrics.append(metrics)
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        health_factors = []
        
        # Component status health
        running_components = sum(1 for status in self.component_status.values() 
                               if status in [ComponentStatus.RUNNING, ComponentStatus.SYNCHRONIZED])
        total_components = len(self.component_status)
        component_health = running_components / total_components if total_components > 0 else 0
        health_factors.append(component_health)
        
        # Timing health
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-10:]
            timing_health = np.mean([m['latency_target_met'] for m in recent_metrics])
            health_factors.append(timing_health)
        
        # Integration health (successful updates)
        if self.integration_history:
            recent_integrations = list(self.integration_history)[-10:]
            integration_health = np.mean([1 if i['digital_twin_sync'] is not None else 0 
                                        for i in recent_integrations])
            health_factors.append(integration_health)
        
        return np.mean(health_factors) if health_factors else 0.0
    
    def _measurements_to_vector(self) -> np.ndarray:
        """Convert measurement dictionary to vector."""
        # Default measurements if none available
        if not self.current_measurements:
            return np.zeros(6)  # [x, y, z, vx, vy, vz]
        
        # Extract measurements
        position = self.current_measurements.get('position', np.zeros(3))
        velocity = self.current_measurements.get('velocity', np.zeros(3))
        
        return np.concatenate([position, velocity])
    
    def _state_to_vector(self, state: MultiPhysicsState) -> np.ndarray:
        """Convert MultiPhysicsState to vector."""
        vector = np.concatenate([
            state.mechanical.position,
            state.mechanical.velocity,
            state.mechanical.acceleration,
            [state.thermal.temperature_distribution[0] if len(state.thermal.temperature_distribution) > 0 else 300.0],
            [state.thermal.heat_flux[0] if len(state.thermal.heat_flux) > 0 else 0.0],
            state.thermal.thermal_stress[:4],  # Take first 4 components
            [state.electromagnetic.field_strength[0] if len(state.electromagnetic.field_strength) > 0 else 0.0],
            [state.electromagnetic.current_density[0] if len(state.electromagnetic.current_density) > 0 else 0.0],
            [state.electromagnetic.magnetic_field[0] if len(state.electromagnetic.magnetic_field) > 0 else 0.0],
            state.quantum.wave_function_amplitude[:3]  # Take first 3 components
        ])
        
        # Ensure vector is exactly 18 elements
        if len(vector) < 18:
            vector = np.pad(vector, (0, 18 - len(vector)), 'constant')
        elif len(vector) > 18:
            vector = vector[:18]
        
        return vector
    
    def _vector_to_state(self, vector: np.ndarray) -> MultiPhysicsState:
        """Convert vector to MultiPhysicsState."""
        state = MultiPhysicsState()
        
        # Mechanical state
        state.mechanical.position = vector[:3]
        state.mechanical.velocity = vector[3:6]
        state.mechanical.acceleration = vector[6:9]
        
        # Thermal state
        state.thermal.temperature_distribution = np.array([vector[9]])
        state.thermal.heat_flux = np.array([vector[10]])
        state.thermal.thermal_stress = vector[11:15]
        
        # Electromagnetic state
        state.electromagnetic.field_strength = np.array([vector[15]])
        state.electromagnetic.current_density = np.array([vector[16]])
        state.electromagnetic.magnetic_field = np.array([vector[17]])
        
        # Quantum state (initialize with defaults)
        state.quantum.wave_function_amplitude = np.array([1.0, 0.0, 0.0])
        
        return state
    
    def add_measurement(self, measurement_type: str, value: np.ndarray, 
                       timestamp: Optional[float] = None):
        """
        Add measurement to the system.
        
        Args:
            measurement_type: Type of measurement ('position', 'velocity', etc.)
            value: Measurement value
            timestamp: Measurement timestamp (if None, use current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.current_measurements[measurement_type] = {
            'value': value,
            'timestamp': timestamp
        }
        
        self.logger.debug(f"Added measurement: {measurement_type} = {value}")
    
    def get_current_state(self) -> MultiPhysicsState:
        """Get current system state."""
        return self.current_state
    
    def get_predicted_trajectory(self, horizon_steps: int = 20) -> np.ndarray:
        """
        Get predicted system trajectory.
        
        Args:
            horizon_steps: Number of prediction steps
            
        Returns:
            Predicted trajectory matrix
        """
        try:
            # Get latest control result
            if self.integration_history:
                latest_integration = self.integration_history[-1]
                if latest_integration['predictive_control']:
                    return latest_integration['predictive_control'].get('predicted_trajectory', 
                                                                       np.zeros((horizon_steps, 18)))
            
            # Fallback: predict using current model
            current_state_vector = self._state_to_vector(self.current_state)
            trajectory = np.zeros((horizon_steps, 18))
            state = current_state_vector.copy()
            
            for k in range(horizon_steps):
                trajectory[k] = state
                # Simple prediction (would use actual dynamics)
                state = state + 0.001 * self.current_control[:3].tolist() + [0] * 15
            
            return trajectory
            
        except Exception as e:
            self.logger.error(f"Trajectory prediction failed: {e}")
            return np.zeros((horizon_steps, 18))
    
    def validate_system(self, validation_data: Optional[Dict] = None) -> Dict:
        """
        Validate the integrated digital twin system.
        
        Args:
            validation_data: Optional validation data
            
        Returns:
            Validation results
        """
        try:
            # Generate validation data if not provided
            if validation_data is None:
                validation_data = self._generate_validation_data()
            
            # Define integrated model for validation
            def integrated_model(X):
                """Integrated digital twin model for validation."""
                outputs = []
                for x in X:
                    # Simulate state evolution
                    state = self._vector_to_state(x)
                    state_vector = self._state_to_vector(state)
                    
                    # Use digital twin dynamics
                    next_state = self.digital_twin.f_coupled(
                        state_vector, self.current_control, 
                        self.params.mpc_params.sample_time
                    )
                    
                    # Return position as output
                    outputs.append(next_state[:3])
                
                return np.array(outputs)
            
            # Validate using framework
            validation_result = self.validation_framework.validate_model(
                integrated_model, validation_data
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return {'error': str(e)}
    
    def _generate_validation_data(self) -> Dict:
        """Generate synthetic validation data."""
        n_samples = 100
        n_features = 18
        
        # Generate random state vectors
        X = np.random.randn(n_samples, n_features) * 1e-9  # nm scale
        
        # Generate corresponding outputs (simplified)
        y = X[:, :3] + 0.01 * np.random.randn(n_samples, 3)  # Position with noise
        
        return {
            'X': X,
            'y': y,
            'parameters': {
                'casimir_coefficient': 1.0,
                'gap_distance': 100e-9,
                'temperature': 300.0
            }
        }
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary."""
        # Component status summary
        component_summary = {}
        for component, status in self.component_status.items():
            component_summary[component] = {
                'status': status.value,
                'healthy': status in [ComponentStatus.RUNNING, ComponentStatus.SYNCHRONIZED]
            }
        
        # Performance summary
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-50:]
            avg_update_time = np.mean([m['update_time_s'] for m in recent_metrics])
            avg_frequency = np.mean([m['frequency_hz'] for m in recent_metrics])
            latency_satisfaction = np.mean([m['latency_target_met'] for m in recent_metrics])
            avg_health = np.mean([m['system_health'] for m in recent_metrics])
        else:
            avg_update_time = 0.0
            avg_frequency = 0.0
            latency_satisfaction = 0.0
            avg_health = 0.0
        
        # Integration summary
        if self.integration_history:
            recent_integrations = list(self.integration_history)[-20:]
            successful_integrations = sum(1 for i in recent_integrations 
                                        if i['digital_twin_sync'] is not None)
            integration_success_rate = successful_integrations / len(recent_integrations)
        else:
            integration_success_rate = 0.0
        
        summary = {
            'system_status': {
                'initialized': self.is_initialized,
                'running': self.is_running,
                'components_total': len(self.component_status),
                'components_healthy': sum(1 for s in component_summary.values() if s['healthy']),
                'overall_health': avg_health
            },
            'performance_metrics': {
                'avg_update_time_s': avg_update_time,
                'avg_frequency_hz': avg_frequency,
                'latency_target_satisfied': latency_satisfaction >= 0.95,
                'latency_target': self.params.max_latency_s,
                'integration_success_rate': integration_success_rate
            },
            'component_status': component_summary,
            'current_state_summary': {
                'position': self.current_state.mechanical.position.tolist(),
                'velocity': self.current_state.mechanical.velocity.tolist(),
                'temperature': self.current_state.thermal.temperature_distribution[0] 
                              if len(self.current_state.thermal.temperature_distribution) > 0 else 0.0
            },
            'targets_met': {
                'latency': avg_update_time <= INTEGRATION_LATENCY_TARGET,
                'health': avg_health >= 0.8,
                'integration_success': integration_success_rate >= 0.95
            }
        }
        
        return summary
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_real_time_operation()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


if __name__ == "__main__":
    """Example usage of integrated digital twin system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== INTEGRATED DIGITAL TWIN SYSTEM ===")
    print("Comprehensive integration of all components")
    
    # Initialize integration parameters
    params = IntegrationParameters(
        mode=IntegrationMode.HYBRID,
        update_frequency_hz=100.0,  # 100 Hz for demonstration
        use_parallel_processing=True
    )
    
    print(f"\nInitialization parameters:")
    print(f"  Mode: {params.mode.value}")
    print(f"  Update frequency: {params.update_frequency_hz} Hz")
    print(f"  Max latency: {params.max_latency_s*1000:.1f} ms")
    print(f"  Parallel processing: {params.use_parallel_processing}")
    
    # Initialize integrated system
    with IntegratedDigitalTwin(params) as integrated_system:
        
        print(f"\nIntegrated digital twin initialized:")
        print(f"  Components initialized: {integrated_system.is_initialized}")
        print(f"  Component count: {len(integrated_system.component_status)}")
        
        # Display component status
        print(f"\nComponent Status:")
        for component, status in integrated_system.component_status.items():
            print(f"  {component}: {status.value}")
        
        # Add some measurements
        print(f"\nAdding measurements...")
        integrated_system.add_measurement('position', np.array([1e-9, 0.5e-9, 0]))
        integrated_system.add_measurement('velocity', np.array([0, 0, 0]))
        
        # Start real-time operation
        print(f"\nStarting real-time operation...")
        integrated_system.start_real_time_operation()
        
        # Run for a short time
        time.sleep(2.0)  # Run for 2 seconds
        
        # Get system summary
        summary = integrated_system.get_system_summary()
        
        print(f"\nSystem Performance Summary:")
        
        system_status = summary.get('system_status', {})
        print(f"  System running: {'✓' if system_status.get('running', False) else '✗'}")
        print(f"  Components healthy: {system_status.get('components_healthy', 0)}/{system_status.get('components_total', 0)}")
        print(f"  Overall health: {system_status.get('overall_health', 0):.3f}")
        
        performance = summary.get('performance_metrics', {})
        print(f"  Average update time: {performance.get('avg_update_time_s', 0)*1000:.2f} ms")
        print(f"  Average frequency: {performance.get('avg_frequency_hz', 0):.1f} Hz")
        print(f"  Latency target satisfied: {'✓' if performance.get('latency_target_satisfied', False) else '✗'}")
        print(f"  Integration success rate: {performance.get('integration_success_rate', 0)*100:.1f}%")
        
        targets_met = summary.get('targets_met', {})
        print(f"Integration Targets:")
        print(f"  Latency: {'✓' if targets_met.get('latency', False) else '✗'}")
        print(f"  Health: {'✓' if targets_met.get('health', False) else '✗'}")
        print(f"  Integration: {'✓' if targets_met.get('integration_success', False) else '✗'}")
        
        # Current state
        current_state = summary.get('current_state_summary', {})
        print(f"Current State:")
        print(f"  Position: {current_state.get('position', [0,0,0])}")
        print(f"  Velocity: {current_state.get('velocity', [0,0,0])}")
        print(f"  Temperature: {current_state.get('temperature', 0):.1f} K")
        
        # Get predicted trajectory
        print(f"\nPredicted Trajectory:")
        trajectory = integrated_system.get_predicted_trajectory(horizon_steps=10)
        print(f"  Trajectory shape: {trajectory.shape}")
        print(f"  Final position: {trajectory[-1, :3]}")
        
        # System validation
        print(f"\nPerforming system validation...")
        validation_result = integrated_system.validate_system()
        
        if 'error' not in validation_result:
            print(f"  Validation passed: {'✓' if validation_result.get('overall_passed', False) else '✗'}")
            perf_summary = validation_result.get('performance_summary', {})
            print(f"  Validation score: {perf_summary.get('overall_validation_score', 0):.3f}")
            print(f"  Validation time: {perf_summary.get('total_validation_time_s', 0):.3f} s")
        else:
            print(f"  Validation error: {validation_result['error']}")
        
        # Stop real-time operation
        print(f"\nStopping real-time operation...")
        integrated_system.stop_real_time_operation()
        
        print(f"Final system health: {integrated_system._calculate_system_health():.3f}")
    
    print(f"\nIntegrated digital twin demonstration complete!")
