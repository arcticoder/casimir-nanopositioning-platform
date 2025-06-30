"""
Enhanced Casimir Force Calculations with Quantum Corrections
============================================================

This module implements advanced Casimir force formulations incorporating:
- Polymer quantization effects
- Metamaterial enhancements  
- Material dispersion corrections
- Quantum field theory corrections

Based on formulations found in workspace survey from:
- negative-energy-generator/src/hardware/polymer_coupling.py
- unified-lqg/unified_LQG_QFT_key_discoveries.txt
- unified-lqg-qft/src/drude_model.py
"""

import numpy as np
import scipy.integrate as integrate
from scipy.special import spherical_jn, spherical_yn
from typing import Dict, Tuple, Optional, Callable
import logging

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
C = 299792458.0         # m/s
PI = np.pi
KB = 1.380649e-23       # J/K

class EnhancedCasimirForce:
    """
    Enhanced Casimir force calculator with quantum corrections and material effects.
    
    LaTeX Formulations Implemented:
    
    1. Basic Casimir Force:
    F_Casimir = -π²ℏc/(240d⁴) × A × η_material
    
    2. Polymer-Modified Casimir Force:
    F_Casimir^poly = -π²ℏc/(240d⁴) A η_material × sin²(μ_g√(k²+m_g²))/(k²+m_g²)
    
    3. Metamaterial-Enhanced Casimir Force:
    F_Casimir^meta = -π²ℏc/(240d⁴) A × 1/√|ε_eff| × F(ω)
    
    4. Material Dispersion-Corrected Force:
    F_Casimir^disp = -π²ℏc/(240d⁴) A ∫₀^∞ dω/(2π) Re[ε(ω)] g(ωd/c)
    """
    
    def __init__(self, material_properties: Dict[str, float]):
        """
        Initialize enhanced Casimir force calculator.
        
        Args:
            material_properties: Dict containing material parameters
                - eta_material: Material correction factor
                - epsilon_eff: Effective permittivity for metamaterials
                - mu_g: Polymer quantization parameter
                - m_g: Polymer mass parameter
                - plasma_freq: Plasma frequency for dispersion
                - damping_freq: Damping frequency for dispersion
        """
        self.material_props = material_properties
        self.logger = logging.getLogger(__name__)
        
        # Validate required parameters
        required_params = ['eta_material']
        for param in required_params:
            if param not in material_properties:
                raise ValueError(f"Required parameter '{param}' missing from material_properties")
    
    def basic_casimir_force(self, d: float, A: float) -> float:
        """
        Calculate basic Casimir force between parallel plates.
        
        LaTeX: F_Casimir = -π²ℏc/(240d⁴) × A × η_material
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            
        Returns:
            Casimir force (N) - negative indicates attractive
        """
        if d <= 0:
            raise ValueError("Plate separation must be positive")
        
        force = -(PI**2 * HBAR * C * A * self.material_props['eta_material']) / (240 * d**4)
        
        self.logger.debug(f"Basic Casimir force: d={d:.2e}m, A={A:.2e}m², F={force:.2e}N")
        return force
    
    def polymer_modified_casimir_force(self, d: float, A: float, k: float) -> float:
        """
        Calculate polymer-modified Casimir force with quantum corrections.
        
        LaTeX: F_Casimir^poly = -π²ℏc/(240d⁴) A η_material × sin²(μ_g√(k²+m_g²))/(k²+m_g²)
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            k: Wave vector parameter (m⁻¹)
            
        Returns:
            Polymer-modified Casimir force (N)
        """
        if 'mu_g' not in self.material_props or 'm_g' not in self.material_props:
            raise ValueError("Polymer parameters mu_g and m_g required")
        
        mu_g = self.material_props['mu_g']
        m_g = self.material_props['m_g']
        
        # Basic force
        F_basic = self.basic_casimir_force(d, A)
        
        # Polymer correction factor
        k_eff = np.sqrt(k**2 + m_g**2)
        correction = (np.sin(mu_g * k_eff))**2 / (k**2 + m_g**2)
        
        force = F_basic * correction
        
        self.logger.debug(f"Polymer-modified force: correction={correction:.4f}, F={force:.2e}N")
        return force
    
    def metamaterial_enhanced_casimir_force(self, d: float, A: float, omega: float) -> float:
        """
        Calculate metamaterial-enhanced Casimir force.
        
        LaTeX: F_Casimir^meta = -π²ℏc/(240d⁴) A × 1/√|ε_eff| × F(ω)
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            omega: Angular frequency (rad/s)
            
        Returns:
            Metamaterial-enhanced Casimir force (N)
        """
        if 'epsilon_eff' not in self.material_props:
            raise ValueError("Effective permittivity epsilon_eff required for metamaterial calculation")
        
        epsilon_eff = self.material_props['epsilon_eff']
        
        # Basic force
        F_basic = self.basic_casimir_force(d, A)
        
        # Metamaterial enhancement factor
        enhancement = 1.0 / np.sqrt(abs(epsilon_eff))
        
        # Frequency-dependent function F(ω) - simplified model
        F_omega = 1.0 + 0.1 * np.sin(omega * d / C)  # Simple frequency dependence
        
        force = F_basic * enhancement * F_omega
        
        self.logger.debug(f"Metamaterial-enhanced force: enhancement={enhancement:.4f}, F_ω={F_omega:.4f}, F={force:.2e}N")
        return force
    
    def _drude_permittivity(self, omega: float) -> complex:
        """
        Calculate frequency-dependent permittivity using Drude model.
        
        Args:
            omega: Angular frequency (rad/s)
            
        Returns:
            Complex permittivity
        """
        if 'plasma_freq' not in self.material_props or 'damping_freq' not in self.material_props:
            # Default values for metallic materials
            omega_p = 1.0e16  # Plasma frequency (rad/s)
            gamma = 1.0e14    # Damping frequency (rad/s)
        else:
            omega_p = self.material_props['plasma_freq']
            gamma = self.material_props['damping_freq']
        
        epsilon = 1.0 - omega_p**2 / (omega**2 + 1j * gamma * omega)
        return epsilon
    
    def _dispersion_kernel(self, x: float) -> float:
        """
        Dispersion kernel function g(x) for integration.
        
        Args:
            x: Dimensionless frequency parameter ωd/c
            
        Returns:
            Kernel value
        """
        if x < 1e-10:
            return 1.0  # Limiting behavior
        
        return x**3 / (np.exp(x) - 1.0)
    
    def material_dispersion_corrected_force(self, d: float, A: float, 
                                          omega_max: Optional[float] = None) -> float:
        """
        Calculate material dispersion-corrected Casimir force.
        
        LaTeX: F_Casimir^disp = -π²ℏc/(240d⁴) A ∫₀^∞ dω/(2π) Re[ε(ω)] g(ωd/c)
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            omega_max: Maximum frequency for integration (rad/s), defaults to plasma frequency
            
        Returns:
            Dispersion-corrected Casimir force (N)
        """
        if omega_max is None:
            omega_max = self.material_props.get('plasma_freq', 1.0e16)
        
        def integrand(omega):
            """Integration kernel for dispersion correction."""
            epsilon = self._drude_permittivity(omega)
            x = omega * d / C
            kernel = self._dispersion_kernel(x)
            return np.real(epsilon) * kernel
        
        # Perform numerical integration
        integral, error = integrate.quad(integrand, 0, omega_max, limit=100)
        
        # Apply normalization and constants
        prefactor = -(PI**2 * HBAR * C * A) / (240 * d**4 * 2 * PI)
        force = prefactor * integral
        
        self.logger.debug(f"Dispersion-corrected force: integral={integral:.4e}, error={error:.2e}, F={force:.2e}N")
        return force
    
    def complete_enhanced_force(self, d: float, A: float, k: float, omega: float,
                              include_polymer: bool = True,
                              include_metamaterial: bool = True,
                              include_dispersion: bool = True) -> Dict[str, float]:
        """
        Calculate complete enhanced Casimir force with all corrections.
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            k: Wave vector parameter (m⁻¹)
            omega: Angular frequency (rad/s)
            include_polymer: Include polymer quantization effects
            include_metamaterial: Include metamaterial enhancements
            include_dispersion: Include material dispersion corrections
            
        Returns:
            Dictionary with all force components and total enhanced force
        """
        results = {}
        
        # Basic force
        F_basic = self.basic_casimir_force(d, A)
        results['basic'] = F_basic
        
        # Enhanced forces
        if include_polymer:
            try:
                F_polymer = self.polymer_modified_casimir_force(d, A, k)
                results['polymer'] = F_polymer
            except ValueError as e:
                self.logger.warning(f"Polymer calculation skipped: {e}")
                results['polymer'] = F_basic
        
        if include_metamaterial:
            try:
                F_meta = self.metamaterial_enhanced_casimir_force(d, A, omega)
                results['metamaterial'] = F_meta
            except ValueError as e:
                self.logger.warning(f"Metamaterial calculation skipped: {e}")
                results['metamaterial'] = F_basic
        
        if include_dispersion:
            F_disp = self.material_dispersion_corrected_force(d, A)
            results['dispersion'] = F_disp
        
        # Combined enhancement (multiplicative approach for small corrections)
        enhancement_factor = 1.0
        if include_polymer and 'polymer' in results:
            enhancement_factor *= (results['polymer'] / F_basic)
        if include_metamaterial and 'metamaterial' in results:
            enhancement_factor *= (results['metamaterial'] / F_basic)
        if include_dispersion and 'dispersion' in results:
            enhancement_factor *= (results['dispersion'] / F_basic)
        
        results['total_enhanced'] = F_basic * enhancement_factor
        results['enhancement_factor'] = enhancement_factor
        
        self.logger.info(f"Complete enhanced force calculation: enhancement={enhancement_factor:.4f}")
        return results
    
    def force_gradient(self, d: float, A: float) -> float:
        """
        Calculate Casimir force gradient for stability analysis.
        
        LaTeX: ∂F_Casimir/∂z = 4F_Casimir/z
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            
        Returns:
            Force gradient (N/m)
        """
        F_casimir = self.basic_casimir_force(d, A)
        gradient = 4 * F_casimir / d
        
        self.logger.debug(f"Force gradient: ∂F/∂z = {gradient:.2e} N/m")
        return gradient


def create_example_materials() -> Dict[str, Dict[str, float]]:
    """
    Create example material property dictionaries for testing.
    
    Returns:
        Dictionary of material property sets
    """
    materials = {
        'silicon': {
            'eta_material': 1.0,
            'epsilon_eff': 11.7,
            'mu_g': 0.1,
            'm_g': 1e6,
            'plasma_freq': 2.0e15,
            'damping_freq': 1.0e13
        },
        'gold': {
            'eta_material': 0.85,
            'epsilon_eff': -10.0 + 1.5j,
            'mu_g': 0.05,
            'm_g': 5e5,
            'plasma_freq': 1.37e16,
            'damping_freq': 4.08e13
        },
        'metamaterial': {
            'eta_material': 1.2,
            'epsilon_eff': -2.0 + 0.1j,
            'mu_g': 0.15,
            'm_g': 2e6,
            'plasma_freq': 5.0e15,
            'damping_freq': 2.0e14
        }
    }
    return materials


if __name__ == "__main__":
    """Example usage and validation of enhanced Casimir force calculations."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example parameters
    d = 100e-9  # 100 nm separation
    A = 1e-6    # 1 mm² area
    k = 1e6     # Wave vector
    omega = 1e15 # Angular frequency
    
    # Test with different materials
    materials = create_example_materials()
    
    for material_name, props in materials.items():
        print(f"\n=== {material_name.upper()} MATERIAL ===")
        
        calculator = EnhancedCasimirForce(props)
        
        # Calculate all force components
        results = calculator.complete_enhanced_force(d, A, k, omega)
        
        print(f"Basic Casimir force: {results['basic']:.3e} N")
        if 'polymer' in results:
            print(f"Polymer-modified force: {results['polymer']:.3e} N")
        if 'metamaterial' in results:
            print(f"Metamaterial-enhanced force: {results['metamaterial']:.3e} N")
        if 'dispersion' in results:
            print(f"Dispersion-corrected force: {results['dispersion']:.3e} N")
        
        print(f"Total enhanced force: {results['total_enhanced']:.3e} N")
        print(f"Enhancement factor: {results['enhancement_factor']:.4f}")
        
        # Force gradient for stability
        gradient = calculator.force_gradient(d, A)
        print(f"Force gradient: {gradient:.3e} N/m")
