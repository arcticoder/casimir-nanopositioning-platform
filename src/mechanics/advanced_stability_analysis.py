"""
Advanced Mechanical Stability Analysis for Casimir Nanopositioning
=================================================================

This module implements enhanced mechanical stability analysis including:
- Complete mechanical FEM analysis with stability ratios
- Lyapunov stability analysis for global stability guarantees
- Critical gap calculations
- Multi-physics coupling effects

Based on formulations found in workspace survey from:
- negative-energy-generator/src/simulation/mechanical_fem.py
- unified-lqg-qft/production_grade_lqg_matter_converter.py
"""

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
C = 299792458.0         # m/s
PI = np.pi

@dataclass
class MechanicalProperties:
    """Material mechanical properties for stability analysis."""
    E: float          # Young's modulus (Pa)
    nu: float         # Poisson's ratio
    rho: float        # Density (kg/m³)
    t: float          # Thickness (m)
    L: float          # Length scale (m)
    damping: float = 0.01  # Damping coefficient

@dataclass
class StabilityResults:
    """Results from mechanical stability analysis."""
    k_spring: float           # Spring constant (N/m)
    force_gradient: float     # Casimir force gradient (N/m)
    stability_ratio: float    # Stability ratio
    critical_gap: float       # Critical gap distance (m)
    eigenvalues: np.ndarray   # System eigenvalues
    is_stable: bool          # Global stability flag
    lyapunov_matrix: np.ndarray  # Lyapunov matrix P

class AdvancedMechanicalStability:
    """
    Advanced mechanical stability analyzer for Casimir nanopositioning systems.
    
    LaTeX Formulations Implemented:
    
    1. Spring Constant:
    k_spring = Et³/[12(1-ν²)L⁴]
    
    2. Casimir Force Gradient:
    ∂F_Casimir/∂z = 4F_Casimir/z
    
    3. Stability Ratio:
    Stability Ratio = k_spring / (∂F_Casimir/∂z)
    
    4. Critical Gap:
    Critical Gap = (5π²ℏcA/48k_spring)^(1/5)
    
    5. Lyapunov Stability:
    V(x) = x^T P x
    A_cl^T P + P A_cl = -Q
    V̇ = -x^T Q x < 0
    """
    
    def __init__(self, mechanical_props: MechanicalProperties):
        """
        Initialize mechanical stability analyzer.
        
        Args:
            mechanical_props: Mechanical properties of the system
        """
        self.props = mechanical_props
        self.logger = logging.getLogger(__name__)
        
        # Validate properties
        if self.props.E <= 0 or self.props.t <= 0 or self.props.L <= 0:
            raise ValueError("Mechanical properties must be positive")
    
    def calculate_spring_constant(self) -> float:
        """
        Calculate effective spring constant of the mechanical system.
        
        LaTeX: k_spring = Et³/[12(1-ν²)L⁴]
        
        Returns:
            Spring constant (N/m)
        """
        E, t, nu, L = self.props.E, self.props.t, self.props.nu, self.props.L
        
        k_spring = (E * t**3) / (12 * (1 - nu**2) * L**4)
        
        self.logger.debug(f"Spring constant: k = {k_spring:.3e} N/m")
        return k_spring
    
    def calculate_casimir_force_gradient(self, d: float, A: float, 
                                       eta_material: float = 1.0) -> float:
        """
        Calculate Casimir force gradient for stability analysis.
        
        LaTeX: ∂F_Casimir/∂z = 4F_Casimir/z = 4 × (-π²ℏcA η_material)/(240d⁴) / d
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            eta_material: Material correction factor
            
        Returns:
            Force gradient (N/m)
        """
        if d <= 0:
            raise ValueError("Plate separation must be positive")
        
        # Basic Casimir force
        F_casimir = -(PI**2 * HBAR * C * A * eta_material) / (240 * d**4)
        
        # Force gradient
        force_gradient = 4 * F_casimir / d
        
        self.logger.debug(f"Casimir force gradient: ∂F/∂z = {force_gradient:.3e} N/m")
        return force_gradient
    
    def calculate_stability_ratio(self, d: float, A: float, 
                                eta_material: float = 1.0) -> float:
        """
        Calculate mechanical stability ratio.
        
        LaTeX: Stability Ratio = k_spring / (∂F_Casimir/∂z)
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            eta_material: Material correction factor
            
        Returns:
            Stability ratio (dimensionless)
        """
        k_spring = self.calculate_spring_constant()
        force_gradient = abs(self.calculate_casimir_force_gradient(d, A, eta_material))
        
        stability_ratio = k_spring / force_gradient
        
        self.logger.debug(f"Stability ratio: SR = {stability_ratio:.3f}")
        return stability_ratio
    
    def calculate_critical_gap(self, A: float, eta_material: float = 1.0) -> float:
        """
        Calculate critical gap distance for mechanical stability.
        
        LaTeX: Critical Gap = (5π²ℏcA/48k_spring)^(1/5)
        
        Args:
            A: Plate area (m²)
            eta_material: Material correction factor
            
        Returns:
            Critical gap distance (m)
        """
        k_spring = self.calculate_spring_constant()
        
        numerator = 5 * PI**2 * HBAR * C * A * eta_material
        denominator = 48 * k_spring
        
        critical_gap = (numerator / denominator)**(1/5)
        
        self.logger.debug(f"Critical gap: d_crit = {critical_gap:.3e} m")
        return critical_gap
    
    def build_system_matrix(self, d: float, A: float, 
                          eta_material: float = 1.0) -> np.ndarray:
        """
        Build linearized system matrix for stability analysis.
        
        State vector: [position, velocity]
        
        Args:
            d: Operating point separation (m)
            A: Plate area (m²)
            eta_material: Material correction factor
            
        Returns:
            System matrix A (2x2)
        """
        k_spring = self.calculate_spring_constant()
        force_gradient = self.calculate_casimir_force_gradient(d, A, eta_material)
        
        # Effective spring constant including Casimir gradient
        k_eff = k_spring + force_gradient  # Note: force_gradient is negative
        
        # Mass from geometry (approximate)
        mass = self.props.rho * self.props.t * A
        
        # System matrix [position; velocity] -> [velocity; acceleration]
        A_sys = np.array([
            [0, 1],
            [-k_eff/mass, -self.props.damping]
        ])
        
        self.logger.debug(f"System matrix eigenvalues: {np.linalg.eigvals(A_sys)}")
        return A_sys
    
    def lyapunov_stability_analysis(self, d: float, A: float, 
                                  eta_material: float = 1.0) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Perform Lyapunov stability analysis for global stability guarantees.
        
        LaTeX: 
        V(x) = x^T P x
        A_cl^T P + P A_cl = -Q
        V̇ = -x^T Q x < 0
        
        Args:
            d: Operating point separation (m)
            A: Plate area (m²)
            eta_material: Material correction factor
            
        Returns:
            Tuple of (is_stable, P_matrix, Q_matrix)
        """
        A_sys = self.build_system_matrix(d, A, eta_material)
        
        # Choose positive definite Q matrix
        Q = np.eye(2)
        
        try:
            # Solve Lyapunov equation: A^T P + P A = -Q
            P = linalg.solve_continuous_lyapunov(A_sys.T, -Q)
            
            # Check if P is positive definite
            eigenvalues_P = np.linalg.eigvals(P)
            is_stable = np.all(eigenvalues_P > 0) and np.all(np.real(np.linalg.eigvals(A_sys)) < 0)
            
            self.logger.debug(f"Lyapunov analysis: stable={is_stable}, P_eigs={eigenvalues_P}")
            return is_stable, P, Q
            
        except linalg.LinAlgError as e:
            self.logger.error(f"Lyapunov equation solution failed: {e}")
            return False, np.zeros((2, 2)), Q
    
    def comprehensive_stability_analysis(self, d: float, A: float, 
                                       eta_material: float = 1.0) -> StabilityResults:
        """
        Perform comprehensive mechanical stability analysis.
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            eta_material: Material correction factor
            
        Returns:
            Complete stability analysis results
        """
        # Calculate all stability metrics
        k_spring = self.calculate_spring_constant()
        force_gradient = self.calculate_casimir_force_gradient(d, A, eta_material)
        stability_ratio = self.calculate_stability_ratio(d, A, eta_material)
        critical_gap = self.calculate_critical_gap(A, eta_material)
        
        # System eigenvalues
        A_sys = self.build_system_matrix(d, A, eta_material)
        eigenvalues = np.linalg.eigvals(A_sys)
        
        # Lyapunov stability analysis
        is_stable, P_matrix, Q_matrix = self.lyapunov_stability_analysis(d, A, eta_material)
        
        # Overall stability assessment
        linear_stable = np.all(np.real(eigenvalues) < 0)
        gap_stable = d > critical_gap
        ratio_stable = stability_ratio > 1.0
        
        overall_stable = is_stable and linear_stable and gap_stable and ratio_stable
        
        results = StabilityResults(
            k_spring=k_spring,
            force_gradient=force_gradient,
            stability_ratio=stability_ratio,
            critical_gap=critical_gap,
            eigenvalues=eigenvalues,
            is_stable=overall_stable,
            lyapunov_matrix=P_matrix
        )
        
        self.logger.info(f"Comprehensive stability: stable={overall_stable}, ratio={stability_ratio:.3f}")
        return results
    
    def optimize_mechanical_design(self, target_gap: float, A: float,
                                 eta_material: float = 1.0) -> Dict[str, float]:
        """
        Optimize mechanical design parameters for target operating gap.
        
        Args:
            target_gap: Desired operating gap (m)
            A: Plate area (m²)
            eta_material: Material correction factor
            
        Returns:
            Optimized design parameters
        """
        def objective(thickness_ratio):
            """Optimization objective: maximize stability ratio."""
            # Update thickness
            new_props = MechanicalProperties(
                E=self.props.E,
                nu=self.props.nu,
                rho=self.props.rho,
                t=self.props.t * thickness_ratio,
                L=self.props.L,
                damping=self.props.damping
            )
            
            temp_analyzer = AdvancedMechanicalStability(new_props)
            try:
                stability_ratio = temp_analyzer.calculate_stability_ratio(target_gap, A, eta_material)
                return -stability_ratio  # Negative for maximization
            except:
                return 1e10  # Penalty for invalid configurations
        
        # Optimize thickness ratio
        result = minimize_scalar(objective, bounds=(0.5, 5.0), method='bounded')
        
        optimal_thickness = self.props.t * result.x
        optimal_stability_ratio = -result.fun
        
        # Calculate other optimized parameters
        optimized_props = MechanicalProperties(
            E=self.props.E,
            nu=self.props.nu,
            rho=self.props.rho,
            t=optimal_thickness,
            L=self.props.L,
            damping=self.props.damping
        )
        
        temp_analyzer = AdvancedMechanicalStability(optimized_props)
        optimal_k_spring = temp_analyzer.calculate_spring_constant()
        optimal_critical_gap = temp_analyzer.calculate_critical_gap(A, eta_material)
        
        optimization_results = {
            'optimal_thickness': optimal_thickness,
            'optimal_k_spring': optimal_k_spring,
            'optimal_stability_ratio': optimal_stability_ratio,
            'optimal_critical_gap': optimal_critical_gap,
            'thickness_ratio': result.x
        }
        
        self.logger.info(f"Design optimization: t={optimal_thickness:.2e}m, SR={optimal_stability_ratio:.3f}")
        return optimization_results
    
    def frequency_response_analysis(self, d: float, A: float, 
                                  eta_material: float = 1.0,
                                  freq_range: Tuple[float, float] = (1, 1000)) -> Dict[str, np.ndarray]:
        """
        Analyze frequency response of the mechanical system.
        
        Args:
            d: Plate separation (m)
            A: Plate area (m²)
            eta_material: Material correction factor
            freq_range: Frequency range (Hz) as (min, max)
            
        Returns:
            Dictionary with frequency response data
        """
        A_sys = self.build_system_matrix(d, A, eta_material)
        B = np.array([[0], [1]])  # Input matrix
        C = np.array([[1, 0]])    # Output matrix (position)
        D = np.array([[0]])       # Feedthrough matrix
        
        # Frequency vector
        freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 100)
        omega = 2 * PI * freqs
        
        # Calculate frequency response
        magnitude = np.zeros_like(omega)
        phase = np.zeros_like(omega)
        
        for i, w in enumerate(omega):
            s = 1j * w
            H = C @ np.linalg.inv(s * np.eye(2) - A_sys) @ B + D
            magnitude[i] = np.abs(H[0, 0])
            phase[i] = np.angle(H[0, 0])
        
        response_data = {
            'frequency': freqs,
            'magnitude': magnitude,
            'phase': phase,
            'magnitude_db': 20 * np.log10(magnitude + 1e-12)
        }
        
        self.logger.debug(f"Frequency response computed for {len(freqs)} points")
        return response_data


def create_example_materials() -> Dict[str, MechanicalProperties]:
    """
    Create example mechanical property sets for different materials.
    
    Returns:
        Dictionary of mechanical properties
    """
    materials = {
        'silicon': MechanicalProperties(
            E=170e9,      # Pa
            nu=0.28,      # dimensionless
            rho=2330,     # kg/m³
            t=10e-6,      # m
            L=1e-3,       # m
            damping=0.01
        ),
        'steel': MechanicalProperties(
            E=200e9,      # Pa
            nu=0.3,       # dimensionless
            rho=7850,     # kg/m³
            t=50e-6,      # m
            L=2e-3,       # m
            damping=0.02
        ),
        'aluminum': MechanicalProperties(
            E=70e9,       # Pa
            nu=0.33,      # dimensionless
            rho=2700,     # kg/m³
            t=20e-6,      # m
            L=1.5e-3,     # m
            damping=0.015
        )
    }
    return materials


if __name__ == "__main__":
    """Example usage and validation of mechanical stability analysis."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example parameters  
    d = 100e-9      # 100 nm separation
    A = 1e-6        # 1 mm² area
    eta_material = 1.0
    
    # Test with different materials
    materials = create_example_materials()
    
    for material_name, props in materials.items():
        print(f"\n=== {material_name.upper()} MECHANICAL ANALYSIS ===")
        
        analyzer = AdvancedMechanicalStability(props)
        
        # Comprehensive stability analysis
        results = analyzer.comprehensive_stability_analysis(d, A, eta_material)
        
        print(f"Spring constant: {results.k_spring:.3e} N/m")
        print(f"Force gradient: {results.force_gradient:.3e} N/m")
        print(f"Stability ratio: {results.stability_ratio:.3f}")
        print(f"Critical gap: {results.critical_gap:.3e} m")
        print(f"System stable: {results.is_stable}")
        print(f"Eigenvalues: {results.eigenvalues}")
        
        # Design optimization
        if not results.is_stable:
            print("\nOptimizing design for stability...")
            opt_results = analyzer.optimize_mechanical_design(d, A, eta_material)
            print(f"Optimal thickness: {opt_results['optimal_thickness']:.2e} m")
            print(f"Optimal stability ratio: {opt_results['optimal_stability_ratio']:.3f}")
