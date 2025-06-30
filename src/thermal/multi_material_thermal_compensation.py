"""
Multi-Material Thermal Compensation System
=========================================

This module implements comprehensive thermal drift compensation including:
- Multi-material thermal analysis with precise coefficients
- Advanced thermal modeling with material-specific behaviors
- Real-time thermal compensation algorithms
- Integrated temperature sensing and control

Based on formulations found in workspace survey from:
- warp-bubble-optimizer/UQ-TODO.ndjson
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import json

class MaterialType(Enum):
    """Supported materials for thermal analysis."""
    ZERODUR = "zerodur"
    INVAR = "invar"
    SILICON = "silicon"
    ALUMINUM = "aluminum"
    STEEL = "steel"
    TITANIUM = "titanium"

@dataclass
class ThermalProperties:
    """Comprehensive thermal properties for materials."""
    
    material: MaterialType
    expansion_coeff: float          # Linear expansion coefficient (1/K)
    expansion_coeff_uncertainty: float  # Uncertainty in expansion coefficient
    temp_range: Tuple[float, float] # Operating temperature range (K)
    thermal_conductivity: float     # W/(m·K)
    specific_heat: float           # J/(kg·K)
    density: float                 # kg/m³
    
    # Advanced properties
    nonlinear_coeff: float = 0.0   # Second-order expansion coefficient (1/K²)
    anisotropy_ratio: float = 1.0  # Expansion anisotropy ratio
    
    @classmethod
    def get_standard_materials(cls) -> Dict[MaterialType, 'ThermalProperties']:
        """
        Get thermal properties for standard materials with precise coefficients.
        
        LaTeX formulations:
        f_thermal(T, material) = {
            Zerodur: 1 + 5×10⁻⁹ ΔT
            Invar: 1 + 1.2×10⁻⁶ ΔT  
            Silicon: 1 + 2.6×10⁻⁶ ΔT
            Aluminum: 1 + 2.3×10⁻⁵ ΔT
        }
        """
        return {
            MaterialType.ZERODUR: cls(
                material=MaterialType.ZERODUR,
                expansion_coeff=5e-9,        # Ultra-low expansion
                expansion_coeff_uncertainty=0.5e-9,
                temp_range=(253, 323),       # -20°C to 50°C
                thermal_conductivity=1.46,
                specific_heat=821,
                density=2530,
                nonlinear_coeff=1e-12,
                anisotropy_ratio=1.0
            ),
            MaterialType.INVAR: cls(
                material=MaterialType.INVAR,
                expansion_coeff=1.2e-6,      # Low expansion alloy
                expansion_coeff_uncertainty=0.1e-6,
                temp_range=(200, 400),
                thermal_conductivity=13.8,
                specific_heat=515,
                density=8100,
                nonlinear_coeff=2e-9,
                anisotropy_ratio=1.0
            ),
            MaterialType.SILICON: cls(
                material=MaterialType.SILICON,
                expansion_coeff=2.6e-6,      # Single crystal silicon
                expansion_coeff_uncertainty=0.1e-6,
                temp_range=(200, 500),
                thermal_conductivity=148,
                specific_heat=712,
                density=2330,
                nonlinear_coeff=5e-9,
                anisotropy_ratio=1.0
            ),
            MaterialType.ALUMINUM: cls(
                material=MaterialType.ALUMINUM,
                expansion_coeff=2.3e-5,      # Aluminum 6061
                expansion_coeff_uncertainty=0.2e-5,
                temp_range=(200, 600),
                thermal_conductivity=167,
                specific_heat=896,
                density=2700,
                nonlinear_coeff=1e-8,
                anisotropy_ratio=1.0
            ),
            MaterialType.STEEL: cls(
                material=MaterialType.STEEL,
                expansion_coeff=1.2e-5,      # Stainless steel 304
                expansion_coeff_uncertainty=0.1e-5,
                temp_range=(200, 800),
                thermal_conductivity=16.2,
                specific_heat=500,
                density=7900,
                nonlinear_coeff=8e-9,
                anisotropy_ratio=1.0
            ),
            MaterialType.TITANIUM: cls(
                material=MaterialType.TITANIUM,
                expansion_coeff=8.6e-6,      # Titanium Grade 2
                expansion_coeff_uncertainty=0.2e-6,
                temp_range=(200, 700),
                thermal_conductivity=16.4,
                specific_heat=523,
                density=4510,
                nonlinear_coeff=3e-9,
                anisotropy_ratio=1.0
            )
        }

@dataclass
class ThermalSensorConfig:
    """Configuration for thermal sensors."""
    sensor_type: str               # Type of temperature sensor
    accuracy: float               # Sensor accuracy (K)
    resolution: float             # Sensor resolution (K)
    response_time: float          # Response time (s)
    locations: List[Tuple[float, float, float]]  # Sensor positions (x, y, z)

class MultiMaterialThermalCompensation:
    """
    Multi-material thermal compensation system.
    
    LaTeX Formulations Implemented:
    
    1. Basic Thermal Correction:
    f_thermal(T, material) = 1 + α_material × ΔT
    
    2. Nonlinear Thermal Correction:
    f_thermal(T, material) = 1 + α₁ × ΔT + α₂ × (ΔT)²
    
    3. Multi-dimensional Thermal Expansion:
    L(T) = L₀ × [1 + α₁ΔT + α₂(ΔT)² + α₃(ΔT)³]
    
    4. Anisotropic Expansion:
    Lₓ(T) = L₀,ₓ × [1 + αₓΔT]
    Lᵧ(T) = L₀,ᵧ × [1 + αᵧΔT] 
    Lᵧ(T) = L₀,ᵧ × [1 + αᵧΔT]
    """
    
    def __init__(self, thermal_sensor_config: Optional[ThermalSensorConfig] = None):
        """
        Initialize thermal compensation system.
        
        Args:
            thermal_sensor_config: Configuration for thermal sensors
        """
        self.materials = ThermalProperties.get_standard_materials()
        self.sensor_config = thermal_sensor_config
        self.logger = logging.getLogger(__name__)
        
        # Thermal compensation state
        self.reference_temperature = 293.15  # 20°C reference
        self.active_materials = []
        self.compensation_history = []
        
        # Calibration data
        self.calibration_data = {}
        
    def calculate_thermal_correction_factor(self, material: MaterialType, 
                                          delta_T: float,
                                          include_nonlinear: bool = True) -> float:
        """
        Calculate thermal correction factor for given material and temperature change.
        
        LaTeX: f_thermal = 1 + α₁ΔT + α₂(ΔT)² (with nonlinear terms)
        
        Args:
            material: Material type
            delta_T: Temperature change from reference (K)
            include_nonlinear: Include nonlinear expansion terms
            
        Returns:
            Thermal correction factor
        """
        if material not in self.materials:
            raise ValueError(f"Unknown material: {material}")
        
        props = self.materials[material]
        
        # Linear term
        correction = 1.0 + props.expansion_coeff * delta_T
        
        # Nonlinear term
        if include_nonlinear and props.nonlinear_coeff != 0:
            correction += props.nonlinear_coeff * (delta_T**2)
        
        self.logger.debug(f"Thermal correction for {material.value}: f = {correction:.8f}")
        return correction
    
    def calculate_thermal_drift(self, material: MaterialType,
                              baseline_length: float,
                              delta_T: float,
                              include_nonlinear: bool = True) -> float:
        """
        Calculate absolute thermal drift.
        
        Args:
            material: Material type
            baseline_length: Reference length (m)
            delta_T: Temperature change (K)
            include_nonlinear: Include nonlinear terms
            
        Returns:
            Thermal drift (m)
        """
        correction_factor = self.calculate_thermal_correction_factor(
            material, delta_T, include_nonlinear
        )
        
        drift = baseline_length * (correction_factor - 1.0)
        
        self.logger.debug(f"Thermal drift: {drift:.2e} m for {material.value}")
        return drift
    
    def calculate_anisotropic_expansion(self, material: MaterialType,
                                      dimensions: Tuple[float, float, float],
                                      delta_T: float) -> Tuple[float, float, float]:
        """
        Calculate anisotropic thermal expansion in 3D.
        
        Args:
            material: Material type
            dimensions: Initial dimensions (x, y, z) in meters
            delta_T: Temperature change (K)
            
        Returns:
            Expanded dimensions (x, y, z) in meters
        """
        props = self.materials[material]
        
        # For most materials, expansion is isotropic
        # Real implementation would include crystal orientation effects
        correction = self.calculate_thermal_correction_factor(material, delta_T)
        
        expanded_dimensions = (
            dimensions[0] * correction,
            dimensions[1] * correction,
            dimensions[2] * correction
        )
        
        self.logger.debug(f"Anisotropic expansion: {dimensions} -> {expanded_dimensions}")
        return expanded_dimensions
    
    def multi_material_system_drift(self, material_configs: List[Dict],
                                   temperature_profile: List[float]) -> Dict[str, np.ndarray]:
        """
        Calculate thermal drift for multi-material system.
        
        Args:
            material_configs: List of material configurations
                Each dict contains: 'material', 'length', 'position'
            temperature_profile: Temperature at each time point (K)
            
        Returns:
            Dictionary with drift analysis for each material and total system
        """
        temperature_array = np.array(temperature_profile)
        delta_T_array = temperature_array - self.reference_temperature
        
        results = {}
        total_drift = np.zeros_like(delta_T_array)
        
        for i, config in enumerate(material_configs):
            material = config['material']
            length = config['length']
            position = config.get('position', 0.0)
            
            # Calculate drift for this material
            material_drift = np.array([
                self.calculate_thermal_drift(material, length, dt)
                for dt in delta_T_array
            ])
            
            results[f'material_{i}_{material.value}'] = {
                'drift': material_drift,
                'length': length,
                'position': position,
                'max_drift': np.max(np.abs(material_drift)),
                'rms_drift': np.sqrt(np.mean(material_drift**2))
            }
            
            # Add to total system drift (considering position)
            total_drift += material_drift * config.get('contribution_factor', 1.0)
        
        results['total_system'] = {
            'drift': total_drift,
            'max_drift': np.max(np.abs(total_drift)),
            'rms_drift': np.sqrt(np.mean(total_drift**2)),
            'temperature_profile': temperature_array,
            'delta_T_profile': delta_T_array
        }
        
        self.logger.info(f"Multi-material analysis: max system drift = {results['total_system']['max_drift']:.2e} m")
        return results
    
    def optimize_material_selection(self, target_drift: float,
                                  system_length: float,
                                  temperature_range: Tuple[float, float]) -> Dict[MaterialType, float]:
        """
        Optimize material selection to minimize thermal drift.
        
        Args:
            target_drift: Maximum allowable drift (m)
            system_length: Total system length (m)
            temperature_range: Operating temperature range (K)
            
        Returns:
            Dictionary with material suitability scores
        """
        delta_T_max = max(
            abs(temperature_range[0] - self.reference_temperature),
            abs(temperature_range[1] - self.reference_temperature)
        )
        
        material_scores = {}
        
        for material, props in self.materials.items():
            # Check if material can operate in temperature range
            if (temperature_range[0] < props.temp_range[0] or 
                temperature_range[1] > props.temp_range[1]):
                material_scores[material] = 0.0  # Not suitable
                continue
            
            # Calculate worst-case drift
            max_drift = abs(self.calculate_thermal_drift(material, system_length, delta_T_max))
            
            # Calculate suitability score
            if max_drift <= target_drift:
                # Score based on margin and other factors
                margin = target_drift / max_drift
                thermal_mass_factor = props.density * props.specific_heat
                stability_factor = 1.0 / props.expansion_coeff_uncertainty
                
                score = margin * np.log10(stability_factor) / np.log10(thermal_mass_factor)
                material_scores[material] = min(score, 10.0)  # Cap at 10
            else:
                material_scores[material] = 0.0  # Exceeds drift requirement
        
        # Sort by score
        sorted_materials = dict(sorted(material_scores.items(), 
                                     key=lambda x: x[1], reverse=True))
        
        self.logger.info(f"Material optimization complete. Best: {list(sorted_materials.keys())[0]}")
        return sorted_materials
    
    def design_thermal_compensation_algorithm(self, material_configs: List[Dict],
                                            compensation_bandwidth: float = 0.1) -> Dict[str, float]:
        """
        Design active thermal compensation algorithm.
        
        Args:
            material_configs: Material configuration list
            compensation_bandwidth: Compensation bandwidth (Hz)
            
        Returns:
            Compensation algorithm parameters
        """
        # Calculate system thermal time constants
        thermal_time_constants = []
        
        for config in material_configs:
            material = config['material']
            props = self.materials[material]
            length = config['length']
            
            # Estimate thermal time constant
            # τ = ρ c L² / k (simplified 1D model)
            tau = (props.density * props.specific_heat * length**2) / props.thermal_conductivity
            thermal_time_constants.append(tau)
        
        # Design compensator based on slowest time constant
        dominant_time_constant = max(thermal_time_constants)
        
        # Compensation filter design (low-pass to avoid noise amplification)
        cutoff_freq = min(compensation_bandwidth, 1.0 / (2 * PI * dominant_time_constant))
        
        # PID-like compensation parameters
        compensation_params = {
            'proportional_gain': 1.0,
            'integral_gain': 2 * PI * cutoff_freq,
            'derivative_gain': 1.0 / (2 * PI * cutoff_freq),
            'cutoff_frequency': cutoff_freq,
            'dominant_time_constant': dominant_time_constant,
            'compensation_bandwidth': compensation_bandwidth
        }
        
        self.logger.info(f"Compensation algorithm designed: τ_dom = {dominant_time_constant:.3f}s")
        return compensation_params
    
    def real_time_thermal_compensation(self, current_temperatures: List[float],
                                     material_configs: List[Dict],
                                     compensation_params: Dict[str, float]) -> Dict[str, float]:
        """
        Real-time thermal compensation calculation.
        
        Args:
            current_temperatures: Current temperature readings (K)
            material_configs: Material configuration list
            compensation_params: Compensation algorithm parameters
            
        Returns:
            Dictionary with compensation signals
        """
        if len(current_temperatures) != len(material_configs):
            raise ValueError("Temperature readings must match material configurations")
        
        compensation_signals = {}
        total_compensation = 0.0
        
        for i, (temp, config) in enumerate(zip(current_temperatures, material_configs)):
            material = config['material']
            length = config['length']
            
            # Calculate required compensation
            delta_T = temp - self.reference_temperature
            thermal_drift = self.calculate_thermal_drift(material, length, delta_T)
            
            # Apply compensation algorithm
            kp = compensation_params['proportional_gain']
            compensation_signal = -kp * thermal_drift  # Negative feedback
            
            compensation_signals[f'material_{i}'] = {
                'temperature': temp,
                'delta_T': delta_T,
                'thermal_drift': thermal_drift,
                'compensation_signal': compensation_signal
            }
            
            total_compensation += compensation_signal * config.get('contribution_factor', 1.0)
        
        compensation_signals['total_compensation'] = total_compensation
        
        # Log for monitoring
        self.compensation_history.append({
            'timestamp': len(self.compensation_history),
            'temperatures': current_temperatures,
            'total_compensation': total_compensation
        })
        
        self.logger.debug(f"Real-time compensation: {total_compensation:.2e} m")
        return compensation_signals
    
    def thermal_calibration_procedure(self, calibration_temperatures: List[float],
                                    reference_measurements: List[float],
                                    material: MaterialType) -> Dict[str, float]:
        """
        Perform thermal calibration to refine material parameters.
        
        Args:
            calibration_temperatures: Calibration temperature points (K)
            reference_measurements: Reference length measurements (m)
            material: Material being calibrated
            
        Returns:
            Calibrated thermal parameters
        """
        temps = np.array(calibration_temperatures)
        measurements = np.array(reference_measurements)
        
        # Reference measurement (typically at room temperature)
        ref_temp = temps[np.argmin(np.abs(temps - self.reference_temperature))]
        ref_measurement = measurements[np.argmin(np.abs(temps - self.reference_temperature))]
        
        # Calculate relative changes
        delta_T = temps - ref_temp
        relative_change = (measurements - ref_measurement) / ref_measurement
        
        # Fit thermal expansion model
        def thermal_model(delta_T, alpha1, alpha2):
            return alpha1 * delta_T + alpha2 * delta_T**2
        
        try:
            # Fit parameters
            popt, pcov = curve_fit(thermal_model, delta_T, relative_change)
            
            calibrated_alpha1 = popt[0]
            calibrated_alpha2 = popt[1] if len(popt) > 1 else 0.0
            
            # Calculate uncertainties
            param_uncertainties = np.sqrt(np.diag(pcov))
            alpha1_uncertainty = param_uncertainties[0]
            alpha2_uncertainty = param_uncertainties[1] if len(param_uncertainties) > 1 else 0.0
            
            # Update material properties
            if material in self.materials:
                self.materials[material].expansion_coeff = calibrated_alpha1
                self.materials[material].nonlinear_coeff = calibrated_alpha2
                self.materials[material].expansion_coeff_uncertainty = alpha1_uncertainty
            
            calibration_results = {
                'material': material.value,
                'calibrated_alpha1': calibrated_alpha1,
                'calibrated_alpha2': calibrated_alpha2,
                'alpha1_uncertainty': alpha1_uncertainty,
                'alpha2_uncertainty': alpha2_uncertainty,
                'fit_quality': np.corrcoef(relative_change, thermal_model(delta_T, *popt))[0, 1]**2
            }
            
            # Store calibration data
            self.calibration_data[material] = calibration_results
            
            self.logger.info(f"Thermal calibration complete for {material.value}: α₁={calibrated_alpha1:.2e}")
            return calibration_results
            
        except Exception as e:
            self.logger.error(f"Thermal calibration failed: {e}")
            return {}
    
    def export_thermal_analysis_report(self, filename: str):
        """
        Export comprehensive thermal analysis report.
        
        Args:
            filename: Output filename
        """
        report = {
            'materials': {
                material.value: {
                    'expansion_coeff': props.expansion_coeff,
                    'expansion_coeff_uncertainty': props.expansion_coeff_uncertainty,
                    'temp_range': props.temp_range,
                    'thermal_conductivity': props.thermal_conductivity,
                    'nonlinear_coeff': props.nonlinear_coeff
                }
                for material, props in self.materials.items()
            },
            'calibration_data': {
                material.value: data for material, data in self.calibration_data.items()
            },
            'compensation_history': self.compensation_history[-100:],  # Last 100 entries
            'reference_temperature': self.reference_temperature
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Thermal analysis report exported to {filename}")


if __name__ == "__main__":
    """Example usage and validation of thermal compensation system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== MULTI-MATERIAL THERMAL COMPENSATION SYSTEM ===")
    
    # Create thermal compensation system
    thermal_system = MultiMaterialThermalCompensation()
    
    # Display material properties
    print(f"\n=== MATERIAL THERMAL PROPERTIES ===")
    for material, props in thermal_system.materials.items():
        print(f"\n{material.value.upper()}:")
        print(f"  Expansion coeff: {props.expansion_coeff:.2e} ± {props.expansion_coeff_uncertainty:.2e} 1/K")
        print(f"  Temperature range: {props.temp_range[0]:.0f} - {props.temp_range[1]:.0f} K")
        print(f"  Thermal conductivity: {props.thermal_conductivity:.1f} W/(m·K)")
    
    # Thermal correction example
    print(f"\n=== THERMAL CORRECTION CALCULATIONS ===")
    delta_T_test = 10.0  # 10 K temperature change
    
    for material in [MaterialType.ZERODUR, MaterialType.SILICON, MaterialType.ALUMINUM]:
        correction = thermal_system.calculate_thermal_correction_factor(material, delta_T_test)
        drift = thermal_system.calculate_thermal_drift(material, 1e-3, delta_T_test)  # 1 mm baseline
        
        print(f"{material.value}: correction={correction:.8f}, drift={drift*1e9:.2f} nm")
    
    # Multi-material system analysis
    print(f"\n=== MULTI-MATERIAL SYSTEM ANALYSIS ===")
    
    # Define a mixed-material system
    material_configs = [
        {'material': MaterialType.ZERODUR, 'length': 2e-3, 'contribution_factor': 0.5},
        {'material': MaterialType.SILICON, 'length': 1e-3, 'contribution_factor': 0.3},
        {'material': MaterialType.ALUMINUM, 'length': 0.5e-3, 'contribution_factor': 0.2}
    ]
    
    # Temperature profile (20°C to 30°C)
    temp_profile = np.linspace(293, 303, 11).tolist()
    
    multi_material_results = thermal_system.multi_material_system_drift(
        material_configs, temp_profile
    )
    
    print(f"Multi-material system:")
    print(f"  Total max drift: {multi_material_results['total_system']['max_drift']*1e9:.2f} nm")
    print(f"  Total RMS drift: {multi_material_results['total_system']['rms_drift']*1e9:.2f} nm")
    
    # Material optimization
    print(f"\n=== MATERIAL OPTIMIZATION ===")
    target_drift = 1e-9  # 1 nm target
    system_length = 5e-3  # 5 mm system
    temp_range = (288, 308)  # 15°C to 35°C
    
    material_scores = thermal_system.optimize_material_selection(
        target_drift, system_length, temp_range
    )
    
    print(f"Material suitability scores (target: {target_drift*1e9:.0f} nm):")
    for material, score in list(material_scores.items())[:3]:  # Top 3
        print(f"  {material.value}: {score:.2f}")
    
    # Compensation algorithm design
    print(f"\n=== COMPENSATION ALGORITHM ===")
    compensation_params = thermal_system.design_thermal_compensation_algorithm(material_configs)
    
    print(f"Compensation parameters:")
    print(f"  Dominant time constant: {compensation_params['dominant_time_constant']:.3f} s")
    print(f"  Cutoff frequency: {compensation_params['cutoff_frequency']:.3f} Hz")
    
    # Real-time compensation example
    print(f"\n=== REAL-TIME COMPENSATION EXAMPLE ===")
    current_temps = [295.0, 296.5, 294.8]  # Current temperature readings
    
    compensation_signals = thermal_system.real_time_thermal_compensation(
        current_temps, material_configs, compensation_params
    )
    
    print(f"Real-time compensation:")
    print(f"  Total compensation signal: {compensation_signals['total_compensation']*1e9:.2f} nm")
    
    # Export report
    thermal_system.export_thermal_analysis_report("thermal_compensation_report.json")
    print(f"\nThermal analysis report exported to thermal_compensation_report.json")
