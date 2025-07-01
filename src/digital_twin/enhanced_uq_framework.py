"""
Enhanced Uncertainty Quantification Framework
Digital Twin Advancement for Casimir Nanopositioning Platform

Implements cross-domain correlation modeling and enhanced UQ with:
1. Multi-physics covariance matrix (Σ_enhanced)
2. Cross-domain correlation coefficients (ρ_ij)
3. Uncertainty propagation across mechanical, thermal, electromagnetic, and quantum domains
4. Real-time correlation matrix updates

Mathematical Foundation:
Σ_enhanced = [Σ_mech    Σ_mt      Σ_me      Σ_mq    ]
             [Σ_mt     Σ_therm   Σ_te      Σ_tq    ]
             [Σ_me     Σ_te      Σ_em      Σ_eq    ]
             [Σ_mq     Σ_tq      Σ_eq      Σ_quantum]

Cross-Domain Correlation: ρ_ij = Cov(X_i, X_j) / √(Var(X_i) × Var(X_j))

Author: Digital Twin Enhancement Team
Version: 1.0.0 (Enhanced UQ Framework)
"""

import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

@dataclass
class UQDomainParams:
    """Parameters for each uncertainty quantification domain."""
    # Mechanical domain
    mechanical_vars: List[str] = field(default_factory=lambda: 
        ['displacement', 'velocity', 'acceleration', 'force', 'stiffness'])
    mechanical_uncertainties: np.ndarray = field(default_factory=lambda: 
        np.array([1e-9, 1e-6, 1e-3, 1e-12, 1e3]))  # Base uncertainties
    
    # Thermal domain
    thermal_vars: List[str] = field(default_factory=lambda: 
        ['temperature', 'heat_flux', 'thermal_expansion', 'conductivity'])
    thermal_uncertainties: np.ndarray = field(default_factory=lambda: 
        np.array([0.1, 1e-3, 1e-8, 0.01]))
    
    # Electromagnetic domain
    electromagnetic_vars: List[str] = field(default_factory=lambda: 
        ['electric_field', 'magnetic_field', 'permittivity', 'permeability', 'voltage'])
    electromagnetic_uncertainties: np.ndarray = field(default_factory=lambda: 
        np.array([1e3, 1e-6, 0.1, 0.01, 1.0]))
    
    # Quantum domain
    quantum_vars: List[str] = field(default_factory=lambda: 
        ['coherence_time', 'squeezing_parameter', 'entanglement', 'decoherence_rate'])
    quantum_uncertainties: np.ndarray = field(default_factory=lambda: 
        np.array([1e-9, 0.05, 0.02, 1e6]))
    
    # Cross-domain correlation coefficients
    correlation_coefficients: Dict[str, float] = field(default_factory=lambda: {
        'mechanical_thermal': 0.75,      # ρ_mt: strong thermal-mechanical coupling
        'mechanical_electromagnetic': 0.45,  # ρ_me: moderate electromechanical coupling
        'mechanical_quantum': 0.25,      # ρ_mq: weak quantum-mechanical coupling
        'thermal_electromagnetic': 0.60, # ρ_te: strong thermomagnetic coupling
        'thermal_quantum': 0.15,         # ρ_tq: weak thermal-quantum coupling
        'electromagnetic_quantum': 0.35  # ρ_eq: moderate electromagnetic-quantum coupling
    })

@dataclass
class UQState:
    """Current state of uncertainty quantification system."""
    timestamp: float
    enhanced_covariance_matrix: np.ndarray
    correlation_matrix: np.ndarray
    domain_variances: Dict[str, np.ndarray]
    cross_correlations: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    effective_degrees_freedom: float

class EnhancedUQFramework:
    """Enhanced uncertainty quantification framework with cross-domain correlations."""
    
    def __init__(self, params: UQDomainParams):
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Initialize domain dimensions
        self.n_mechanical = len(params.mechanical_vars)
        self.n_thermal = len(params.thermal_vars)
        self.n_electromagnetic = len(params.electromagnetic_vars)
        self.n_quantum = len(params.quantum_vars)
        self.total_dimensions = (self.n_mechanical + self.n_thermal + 
                                self.n_electromagnetic + self.n_quantum)
        
        # Initialize enhanced covariance matrix
        self.enhanced_covariance = self._initialize_enhanced_covariance()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.computation_times = []
        
    def _initialize_enhanced_covariance(self) -> np.ndarray:
        """
        Initialize the enhanced covariance matrix Σ_enhanced.
        
        Returns:
            Enhanced covariance matrix with cross-domain correlations
        """
        try:
            # Create block diagonal structure
            Sigma = np.zeros((self.total_dimensions, self.total_dimensions))
            
            # Domain variance blocks (diagonal)
            start_idx = 0
            
            # Mechanical domain block
            end_idx = start_idx + self.n_mechanical
            Sigma_mech = np.diag(self.params.mechanical_uncertainties**2)
            Sigma[start_idx:end_idx, start_idx:end_idx] = Sigma_mech
            start_idx = end_idx
            
            # Thermal domain block
            end_idx = start_idx + self.n_thermal
            Sigma_therm = np.diag(self.params.thermal_uncertainties**2)
            Sigma[start_idx:end_idx, start_idx:end_idx] = Sigma_therm
            start_idx = end_idx
            
            # Electromagnetic domain block
            end_idx = start_idx + self.n_electromagnetic
            Sigma_em = np.diag(self.params.electromagnetic_uncertainties**2)
            Sigma[start_idx:end_idx, start_idx:end_idx] = Sigma_em
            start_idx = end_idx
            
            # Quantum domain block
            end_idx = start_idx + self.n_quantum
            Sigma_quantum = np.diag(self.params.quantum_uncertainties**2)
            Sigma[start_idx:end_idx, start_idx:end_idx] = Sigma_quantum
            
            # Add cross-domain correlations (off-diagonal blocks)
            self._add_cross_domain_correlations(Sigma)
            
            # Ensure positive definiteness
            Sigma = self._ensure_positive_definite(Sigma)
            
            self.logger.info(f"Enhanced covariance matrix initialized: {Sigma.shape}")
            return Sigma
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced covariance: {e}")
            return np.eye(self.total_dimensions) * 1e-6
    
    def _add_cross_domain_correlations(self, Sigma: np.ndarray) -> None:
        """Add cross-domain correlation terms to covariance matrix."""
        try:
            corr_coeffs = self.params.correlation_coefficients
            
            # Define domain index ranges
            mech_range = (0, self.n_mechanical)
            therm_range = (self.n_mechanical, self.n_mechanical + self.n_thermal)
            em_range = (self.n_mechanical + self.n_thermal, 
                       self.n_mechanical + self.n_thermal + self.n_electromagnetic)
            quantum_range = (self.n_mechanical + self.n_thermal + self.n_electromagnetic,
                           self.total_dimensions)
            
            # Mechanical-Thermal coupling
            rho_mt = corr_coeffs['mechanical_thermal']
            for i in range(mech_range[0], mech_range[1]):
                for j in range(therm_range[0], therm_range[1]):
                    if i < j:  # Only upper triangle, will be symmetrized
                        cov_ij = rho_mt * np.sqrt(Sigma[i,i] * Sigma[j,j])
                        Sigma[i,j] = cov_ij
                        Sigma[j,i] = cov_ij
            
            # Mechanical-Electromagnetic coupling
            rho_me = corr_coeffs['mechanical_electromagnetic']
            for i in range(mech_range[0], mech_range[1]):
                for j in range(em_range[0], em_range[1]):
                    cov_ij = rho_me * np.sqrt(Sigma[i,i] * Sigma[j,j])
                    Sigma[i,j] = cov_ij
                    Sigma[j,i] = cov_ij
            
            # Mechanical-Quantum coupling
            rho_mq = corr_coeffs['mechanical_quantum']
            for i in range(mech_range[0], mech_range[1]):
                for j in range(quantum_range[0], quantum_range[1]):
                    cov_ij = rho_mq * np.sqrt(Sigma[i,i] * Sigma[j,j])
                    Sigma[i,j] = cov_ij
                    Sigma[j,i] = cov_ij
            
            # Thermal-Electromagnetic coupling
            rho_te = corr_coeffs['thermal_electromagnetic']
            for i in range(therm_range[0], therm_range[1]):
                for j in range(em_range[0], em_range[1]):
                    cov_ij = rho_te * np.sqrt(Sigma[i,i] * Sigma[j,j])
                    Sigma[i,j] = cov_ij
                    Sigma[j,i] = cov_ij
            
            # Thermal-Quantum coupling
            rho_tq = corr_coeffs['thermal_quantum']
            for i in range(therm_range[0], therm_range[1]):
                for j in range(quantum_range[0], quantum_range[1]):
                    cov_ij = rho_tq * np.sqrt(Sigma[i,i] * Sigma[j,j])
                    Sigma[i,j] = cov_ij
                    Sigma[j,i] = cov_ij
            
            # Electromagnetic-Quantum coupling
            rho_eq = corr_coeffs['electromagnetic_quantum']
            for i in range(em_range[0], em_range[1]):
                for j in range(quantum_range[0], quantum_range[1]):
                    cov_ij = rho_eq * np.sqrt(Sigma[i,i] * Sigma[j,j])
                    Sigma[i,j] = cov_ij
                    Sigma[j,i] = cov_ij
                    
        except Exception as e:
            self.logger.warning(f"Failed to add cross-domain correlations: {e}")
    
    def _ensure_positive_definite(self, Sigma: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive definite."""
        try:
            # Eigenvalue decomposition
            eigenvals, eigenvecs = la.eigh(Sigma)
            
            # Clip negative eigenvalues
            min_eigenval = 1e-12
            eigenvals_clipped = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            Sigma_pd = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
            
            # Verify positive definiteness
            if np.all(eigenvals_clipped > 0):
                self.logger.debug("Covariance matrix is positive definite")
                return Sigma_pd
            else:
                self.logger.warning("Had to clip negative eigenvalues")
                return Sigma_pd
                
        except Exception as e:
            self.logger.error(f"Failed to ensure positive definiteness: {e}")
            return np.eye(Sigma.shape[0]) * 1e-6
    
    def compute_cross_domain_correlations(self, measurements: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute cross-domain correlation coefficients from measurements.
        
        ρ_ij = Cov(X_i, X_j) / √(Var(X_i) × Var(X_j))
        """
        start_time = time.time()
        
        try:
            correlations = {}
            
            # Extract domain measurements
            mech_data = measurements.get('mechanical', np.array([]))
            therm_data = measurements.get('thermal', np.array([]))
            em_data = measurements.get('electromagnetic', np.array([]))
            quantum_data = measurements.get('quantum', np.array([]))
            
            # Compute correlations between domain pairs
            if len(mech_data) > 0 and len(therm_data) > 0:
                correlations['mechanical_thermal'] = self._compute_correlation_coefficient(
                    mech_data, therm_data)
            
            if len(mech_data) > 0 and len(em_data) > 0:
                correlations['mechanical_electromagnetic'] = self._compute_correlation_coefficient(
                    mech_data, em_data)
            
            if len(mech_data) > 0 and len(quantum_data) > 0:
                correlations['mechanical_quantum'] = self._compute_correlation_coefficient(
                    mech_data, quantum_data)
            
            if len(therm_data) > 0 and len(em_data) > 0:
                correlations['thermal_electromagnetic'] = self._compute_correlation_coefficient(
                    therm_data, em_data)
            
            if len(therm_data) > 0 and len(quantum_data) > 0:
                correlations['thermal_quantum'] = self._compute_correlation_coefficient(
                    therm_data, quantum_data)
            
            if len(em_data) > 0 and len(quantum_data) > 0:
                correlations['electromagnetic_quantum'] = self._compute_correlation_coefficient(
                    em_data, quantum_data)
            
            computation_time = time.time() - start_time
            self.computation_times.append(computation_time)
            
            self.logger.info(f"Cross-domain correlations computed in {computation_time:.3f}s")
            return correlations
            
        except Exception as e:
            self.logger.error(f"Failed to compute cross-domain correlations: {e}")
            return {}
    
    def _compute_correlation_coefficient(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Compute correlation coefficient between two datasets."""
        try:
            # Ensure same length
            min_len = min(len(data1), len(data2))
            data1 = data1[:min_len]
            data2 = data2[:min_len]
            
            # Compute correlation
            if min_len > 1:
                correlation_matrix = np.corrcoef(data1, data2)
                return float(correlation_matrix[0, 1])
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Correlation computation failed: {e}")
            return 0.0
    
    def update_enhanced_covariance(self, new_measurements: Dict[str, np.ndarray],
                                  adaptation_rate: float = 0.1) -> np.ndarray:
        """
        Update enhanced covariance matrix with new measurements.
        
        Args:
            new_measurements: New measurement data for each domain
            adaptation_rate: Rate of adaptation to new data
            
        Returns:
            Updated enhanced covariance matrix
        """
        with self._lock:
            try:
                # Compute new correlations
                new_correlations = self.compute_cross_domain_correlations(new_measurements)
                
                # Update correlation coefficients with exponential moving average
                for key, new_corr in new_correlations.items():
                    if key in self.params.correlation_coefficients:
                        old_corr = self.params.correlation_coefficients[key]
                        updated_corr = (1 - adaptation_rate) * old_corr + adaptation_rate * new_corr
                        self.params.correlation_coefficients[key] = updated_corr
                
                # Rebuild covariance matrix with updated correlations
                self.enhanced_covariance = self._initialize_enhanced_covariance()
                
                self.logger.info("Enhanced covariance matrix updated")
                return self.enhanced_covariance
                
            except Exception as e:
                self.logger.error(f"Failed to update enhanced covariance: {e}")
                return self.enhanced_covariance
    
    def propagate_uncertainty(self, state_mean: np.ndarray, 
                            jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate uncertainty through nonlinear transformation.
        
        Args:
            state_mean: Mean state vector
            jacobian: Jacobian matrix of transformation
            
        Returns:
            Propagated mean and covariance
        """
        try:
            # Linear uncertainty propagation: Σ_out = J × Σ_in × J^T
            propagated_covariance = jacobian @ self.enhanced_covariance @ jacobian.T
            
            # For nonlinear case, could add higher-order terms
            # Second-order correction: + 0.5 × tr(H_i × Σ_in) for each Hessian H_i
            
            return state_mean, propagated_covariance
            
        except Exception as e:
            self.logger.error(f"Uncertainty propagation failed: {e}")
            return state_mean, np.eye(len(state_mean)) * 1e-6
    
    def generate_monte_carlo_samples(self, n_samples: int, 
                                   mean_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate Monte Carlo samples from enhanced uncertainty distribution.
        
        Args:
            n_samples: Number of samples to generate
            mean_state: Mean state (zero if not provided)
            
        Returns:
            Monte Carlo samples [n_samples × n_dimensions]
        """
        try:
            if mean_state is None:
                mean_state = np.zeros(self.total_dimensions)
            
            # Generate samples from multivariate normal distribution
            samples = np.random.multivariate_normal(
                mean_state, self.enhanced_covariance, size=n_samples)
            
            self.logger.debug(f"Generated {n_samples} Monte Carlo samples")
            return samples
            
        except Exception as e:
            self.logger.error(f"Monte Carlo sampling failed: {e}")
            return np.zeros((n_samples, self.total_dimensions))
    
    def compute_confidence_intervals(self, samples: np.ndarray, 
                                   confidence_level: float = 0.95) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute confidence intervals for each domain.
        
        Args:
            samples: Monte Carlo samples
            confidence_level: Confidence level (0.95 for 95%)
            
        Returns:
            Confidence intervals for each domain
        """
        try:
            alpha = 1 - confidence_level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            intervals = {}
            
            # Mechanical domain
            mech_samples = samples[:, :self.n_mechanical]
            intervals['mechanical'] = (
                np.percentile(mech_samples, lower_percentile, axis=0),
                np.percentile(mech_samples, upper_percentile, axis=0)
            )
            
            # Thermal domain
            start_idx = self.n_mechanical
            end_idx = start_idx + self.n_thermal
            therm_samples = samples[:, start_idx:end_idx]
            intervals['thermal'] = (
                np.percentile(therm_samples, lower_percentile, axis=0),
                np.percentile(therm_samples, upper_percentile, axis=0)
            )
            
            # Electromagnetic domain
            start_idx = end_idx
            end_idx = start_idx + self.n_electromagnetic
            em_samples = samples[:, start_idx:end_idx]
            intervals['electromagnetic'] = (
                np.percentile(em_samples, lower_percentile, axis=0),
                np.percentile(em_samples, upper_percentile, axis=0)
            )
            
            # Quantum domain
            start_idx = end_idx
            quantum_samples = samples[:, start_idx:]
            intervals['quantum'] = (
                np.percentile(quantum_samples, lower_percentile, axis=0),
                np.percentile(quantum_samples, upper_percentile, axis=0)
            )
            
            return intervals
            
        except Exception as e:
            self.logger.error(f"Confidence interval computation failed: {e}")
            return {}
    
    def get_current_uq_state(self) -> UQState:
        """Get current uncertainty quantification state."""
        try:
            # Compute correlation matrix from covariance
            correlation_matrix = self._covariance_to_correlation(self.enhanced_covariance)
            
            # Extract domain variances
            domain_variances = {
                'mechanical': np.diag(self.enhanced_covariance)[:self.n_mechanical],
                'thermal': np.diag(self.enhanced_covariance)[
                    self.n_mechanical:self.n_mechanical + self.n_thermal],
                'electromagnetic': np.diag(self.enhanced_covariance)[
                    self.n_mechanical + self.n_thermal:
                    self.n_mechanical + self.n_thermal + self.n_electromagnetic],
                'quantum': np.diag(self.enhanced_covariance)[
                    self.n_mechanical + self.n_thermal + self.n_electromagnetic:]
            }
            
            # Compute effective degrees of freedom
            effective_dof = np.trace(self.enhanced_covariance)**2 / np.trace(
                self.enhanced_covariance @ self.enhanced_covariance)
            
            return UQState(
                timestamp=time.time(),
                enhanced_covariance_matrix=self.enhanced_covariance.copy(),
                correlation_matrix=correlation_matrix,
                domain_variances=domain_variances,
                cross_correlations=self.params.correlation_coefficients.copy(),
                confidence_intervals={},  # Would be computed with specific samples
                effective_degrees_freedom=effective_dof
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get UQ state: {e}")
            return UQState(
                timestamp=time.time(),
                enhanced_covariance_matrix=np.eye(self.total_dimensions),
                correlation_matrix=np.eye(self.total_dimensions),
                domain_variances={},
                cross_correlations={},
                confidence_intervals={},
                effective_degrees_freedom=self.total_dimensions
            )
    
    def _covariance_to_correlation(self, covariance: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        try:
            D_inv = np.diag(1.0 / np.sqrt(np.diag(covariance)))
            correlation = D_inv @ covariance @ D_inv
            return correlation
        except Exception:
            return np.eye(covariance.shape[0])

def main():
    """Demonstration of enhanced UQ framework."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Enhanced UQ Framework Demonstration")
    
    # Initialize framework
    params = UQDomainParams()
    uq_framework = EnhancedUQFramework(params)
    
    # Get initial state
    initial_state = uq_framework.get_current_uq_state()
    
    print(f"\n📊 ENHANCED UQ FRAMEWORK STATUS:")
    print(f"   Total Dimensions:        {uq_framework.total_dimensions}")
    print(f"   Mechanical Variables:    {uq_framework.n_mechanical}")
    print(f"   Thermal Variables:       {uq_framework.n_thermal}")
    print(f"   Electromagnetic Variables: {uq_framework.n_electromagnetic}")
    print(f"   Quantum Variables:       {uq_framework.n_quantum}")
    print(f"   Effective DOF:           {initial_state.effective_degrees_freedom:.1f}")
    
    print(f"\n🔗 CROSS-DOMAIN CORRELATIONS:")
    for key, value in initial_state.cross_correlations.items():
        print(f"   {key:25s}: {value:.3f}")
    
    # Generate samples and compute confidence intervals
    n_samples = 10000
    samples = uq_framework.generate_monte_carlo_samples(n_samples)
    confidence_intervals = uq_framework.compute_confidence_intervals(samples)
    
    print(f"\n📈 CONFIDENCE INTERVALS (95%):")
    for domain, (lower, upper) in confidence_intervals.items():
        print(f"   {domain.capitalize()} Domain:")
        for i, (l, u) in enumerate(zip(lower, upper)):
            var_name = getattr(params, f"{domain}_vars")[i]
            print(f"     {var_name:20s}: [{l:8.2e}, {u:8.2e}]")
    
    # Simulate measurement update
    print(f"\n🔄 TESTING ADAPTIVE UPDATE:")
    fake_measurements = {
        'mechanical': np.random.randn(100) * 0.1,
        'thermal': np.random.randn(100) * 0.05,
        'electromagnetic': np.random.randn(100) * 0.2,
        'quantum': np.random.randn(100) * 0.03
    }
    
    updated_covariance = uq_framework.update_enhanced_covariance(fake_measurements)
    updated_state = uq_framework.get_current_uq_state()
    
    print(f"   Updated Correlations:")
    for key, value in updated_state.cross_correlations.items():
        print(f"     {key:25s}: {value:.3f}")
    
    print(f"\n✅ Enhanced UQ Framework Successfully Demonstrated")

if __name__ == "__main__":
    main()
