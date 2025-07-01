"""
Enhanced Numerical Stability Framework for UQ Operations
Comprehensive protection against numerical issues in uncertainty quantification

This module provides robust numerical stability protection for all UQ operations
including matrix conditioning, overflow/underflow detection, and automatic
regularization techniques.

CRITICAL RESOLUTION: CRITICAL-001
- Addresses missing numerical stability protection in covariance operations
- Implements comprehensive stability monitoring and regularization
- Provides fallback mechanisms for singular matrix conditions

Author: UQ Numerical Stability Team  
Version: 1.0.0 (Critical Resolution)
"""

import numpy as np
import scipy.linalg as la
from scipy.linalg import LinAlgError, LinAlgWarning
import logging
import warnings
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import sys

@dataclass
class NumericalStabilityConfig:
    """Configuration for numerical stability checks."""
    max_condition_number: float = 1e12
    min_eigenvalue: float = 1e-15
    regularization_factor: float = 1e-10
    overflow_threshold: float = 1e100
    underflow_threshold: float = 1e-100
    nan_check_enabled: bool = True
    inf_check_enabled: bool = True

class NumericalStabilityManager:
    """Manager for numerical stability in UQ operations."""
    
    def __init__(self, config: NumericalStabilityConfig = None):
        self.config = config or NumericalStabilityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup warning filters
        warnings.filterwarnings('error', category=LinAlgWarning)
    
    def check_matrix_stability(self, matrix: np.ndarray, 
                              matrix_name: str = "matrix") -> Tuple[bool, str]:
        """
        Comprehensive matrix stability check.
        
        Args:
            matrix: Matrix to check
            matrix_name: Name for logging
            
        Returns:
            (is_stable, error_message)
        """
        try:
            # Check for NaN or Inf values
            if self.config.nan_check_enabled and np.any(np.isnan(matrix)):
                return False, f"{matrix_name} contains NaN values"
            
            if self.config.inf_check_enabled and np.any(np.isinf(matrix)):
                return False, f"{matrix_name} contains infinite values"
            
            # Check for overflow/underflow
            max_val = np.max(np.abs(matrix))
            if max_val > self.config.overflow_threshold:
                return False, f"{matrix_name} has overflow: max value {max_val:.2e}"
            
            if max_val < self.config.underflow_threshold and max_val > 0:
                return False, f"{matrix_name} has underflow: max value {max_val:.2e}"
            
            # Check condition number for square matrices
            if matrix.shape[0] == matrix.shape[1]:
                try:
                    cond_num = np.linalg.cond(matrix)
                    if cond_num > self.config.max_condition_number:
                        return False, f"{matrix_name} is ill-conditioned: cond={cond_num:.2e}"
                except LinAlgError:
                    return False, f"{matrix_name} condition number calculation failed"
            
            # Check positive semi-definite for covariance matrices
            if self._is_covariance_matrix(matrix):
                eigenvals = np.linalg.eigvals(matrix)
                min_eigenval = np.min(eigenvals)
                if min_eigenval < -self.config.min_eigenvalue:
                    return False, f"{matrix_name} not positive semi-definite: min eigenvalue {min_eigenval:.2e}"
            
            return True, "Matrix is numerically stable"
            
        except Exception as e:
            return False, f"Stability check failed for {matrix_name}: {str(e)}"
    
    def regularize_covariance_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Regularize covariance matrix to ensure numerical stability.
        
        Args:
            cov_matrix: Input covariance matrix
            
        Returns:
            Regularized covariance matrix
        """
        try:
            # Check if regularization is needed
            is_stable, _ = self.check_matrix_stability(cov_matrix, "covariance")
            
            if is_stable:
                return cov_matrix
            
            self.logger.warning("Regularizing unstable covariance matrix")
            
            # Method 1: Diagonal regularization
            regularized = cov_matrix.copy()
            eigenvals, eigenvecs = np.linalg.eigh(regularized)
            
            # Clip negative eigenvalues
            eigenvals = np.maximum(eigenvals, self.config.min_eigenvalue)
            
            # Reconstruct matrix
            regularized = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Method 2: Add diagonal regularization if still unstable
            is_stable, _ = self.check_matrix_stability(regularized, "regularized_covariance")
            if not is_stable:
                regularized += np.eye(regularized.shape[0]) * self.config.regularization_factor
                self.logger.warning(f"Applied diagonal regularization: {self.config.regularization_factor}")
            
            # Final stability check
            is_stable, error_msg = self.check_matrix_stability(regularized, "final_regularized")
            if not is_stable:
                self.logger.error(f"Regularization failed: {error_msg}")
                # Fallback: Return identity matrix scaled appropriately
                trace_val = np.trace(cov_matrix)
                fallback_matrix = np.eye(cov_matrix.shape[0]) * (trace_val / cov_matrix.shape[0])
                self.logger.warning("Using fallback identity-based covariance matrix")
                return fallback_matrix
            
            return regularized
            
        except Exception as e:
            self.logger.error(f"Covariance regularization failed: {e}")
            # Emergency fallback
            return np.eye(cov_matrix.shape[0]) * 1e-6
    
    def safe_matrix_operations(self, operation: str, *args, **kwargs) -> Union[np.ndarray, Tuple]:
        """
        Perform matrix operations with stability checking.
        
        Args:
            operation: Operation name ('inv', 'chol', 'solve', 'eig')
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result or fallback
        """
        try:
            if operation == 'inv':
                matrix = args[0]
                is_stable, error_msg = self.check_matrix_stability(matrix, "inverse_input")
                if not is_stable:
                    self.logger.warning(f"Unstable matrix for inversion: {error_msg}")
                    matrix = self.regularize_covariance_matrix(matrix)
                
                return np.linalg.inv(matrix)
                
            elif operation == 'chol':
                matrix = args[0]
                is_stable, error_msg = self.check_matrix_stability(matrix, "cholesky_input")
                if not is_stable:
                    self.logger.warning(f"Unstable matrix for Cholesky: {error_msg}")
                    matrix = self.regularize_covariance_matrix(matrix)
                
                return np.linalg.cholesky(matrix)
                
            elif operation == 'solve':
                A, b = args[0], args[1]
                is_stable, error_msg = self.check_matrix_stability(A, "solve_matrix")
                if not is_stable:
                    self.logger.warning(f"Unstable matrix for solve: {error_msg}")
                    A = self.regularize_covariance_matrix(A)
                
                return np.linalg.solve(A, b)
                
            elif operation == 'eig':
                matrix = args[0]
                is_stable, error_msg = self.check_matrix_stability(matrix, "eigenvalue_input")
                if not is_stable:
                    self.logger.warning(f"Unstable matrix for eigenvalues: {error_msg}")
                    matrix = self.regularize_covariance_matrix(matrix)
                
                return np.linalg.eigh(matrix)
                
            else:
                raise ValueError(f"Unsupported operation: {operation}")
                
        except (LinAlgError, LinAlgWarning) as e:
            self.logger.error(f"Matrix operation '{operation}' failed: {e}")
            return self._fallback_result(operation, *args)
        
        except Exception as e:
            self.logger.error(f"Unexpected error in '{operation}': {e}")
            return self._fallback_result(operation, *args)
    
    def validate_numerical_result(self, result: Union[np.ndarray, float], 
                                 result_name: str = "result") -> Union[np.ndarray, float]:
        """
        Validate and clean numerical results.
        
        Args:
            result: Numerical result to validate
            result_name: Name for logging
            
        Returns:
            Validated result
        """
        try:
            if isinstance(result, np.ndarray):
                # Check for NaN/Inf
                if np.any(np.isnan(result)):
                    self.logger.error(f"{result_name} contains NaN values")
                    return np.zeros_like(result)
                
                if np.any(np.isinf(result)):
                    self.logger.error(f"{result_name} contains infinite values")
                    # Replace infinities with large finite values
                    result = np.where(np.isinf(result), 
                                    np.sign(result) * self.config.overflow_threshold, 
                                    result)
                
                # Check for overflow/underflow
                max_val = np.max(np.abs(result))
                if max_val > self.config.overflow_threshold:
                    self.logger.warning(f"{result_name} has potential overflow")
                    result = result / max_val * (self.config.overflow_threshold * 0.1)
                
            elif isinstance(result, (float, int)):
                if np.isnan(result):
                    self.logger.error(f"{result_name} is NaN")
                    return 0.0
                
                if np.isinf(result):
                    self.logger.error(f"{result_name} is infinite")
                    return np.sign(result) * self.config.overflow_threshold
            
            return result
            
        except Exception as e:
            self.logger.error(f"Result validation failed for {result_name}: {e}")
            if isinstance(result, np.ndarray):
                return np.zeros_like(result)
            else:
                return 0.0
    
    def _is_covariance_matrix(self, matrix: np.ndarray) -> bool:
        """Check if matrix is likely a covariance matrix."""
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # Check if symmetric (within tolerance)
        if not np.allclose(matrix, matrix.T, rtol=1e-10):
            return False
        
        # Check if diagonal elements are non-negative
        diag_elements = np.diag(matrix)
        if np.any(diag_elements < 0):
            return False
        
        return True
    
    def _fallback_result(self, operation: str, *args):
        """Provide fallback results for failed operations."""
        if operation == 'inv':
            matrix = args[0]
            return np.eye(matrix.shape[0]) / np.trace(matrix) * matrix.shape[0]
        
        elif operation == 'chol':
            matrix = args[0]
            # Return identity matrix scaled by sqrt of diagonal mean
            diag_mean = np.mean(np.diag(matrix))
            return np.eye(matrix.shape[0]) * np.sqrt(np.abs(diag_mean))
        
        elif operation == 'solve':
            A, b = args[0], args[1]
            # Return least squares solution
            try:
                return np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                return np.zeros_like(b)
        
        elif operation == 'eig':
            matrix = args[0]
            # Return diagonal elements as eigenvalues and identity as eigenvectors
            eigenvals = np.diag(matrix)
            eigenvecs = np.eye(matrix.shape[0])
            return eigenvals, eigenvecs
        
        else:
            return None

# Global stability manager instance
_stability_manager = None

def get_stability_manager() -> NumericalStabilityManager:
    """Get global numerical stability manager."""
    global _stability_manager
    if _stability_manager is None:
        _stability_manager = NumericalStabilityManager()
    return _stability_manager

def safe_matrix_op(operation: str, *args, **kwargs):
    """Convenient wrapper for safe matrix operations."""
    manager = get_stability_manager()
    return manager.safe_matrix_operations(operation, *args, **kwargs)

def validate_result(result, name: str = "result"):
    """Convenient wrapper for result validation."""
    manager = get_stability_manager()
    return manager.validate_numerical_result(result, name)

def main():
    """Demonstrate numerical stability framework."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 Numerical Stability Framework Demonstration")
    
    # Test with unstable matrix
    unstable_matrix = np.array([[1e10, 1e10-1], [1e10-1, 1e10]])
    
    manager = NumericalStabilityManager()
    
    # Check stability
    is_stable, error_msg = manager.check_matrix_stability(unstable_matrix, "test_matrix")
    print(f"Matrix stability: {is_stable}")
    print(f"Error message: {error_msg}")
    
    # Regularize
    regularized = manager.regularize_covariance_matrix(unstable_matrix)
    print(f"Regularized matrix condition number: {np.linalg.cond(regularized):.2e}")
    
    # Safe operations
    try:
        inverse = safe_matrix_op('inv', regularized)
        print(f"Safe inverse computed successfully")
        print(f"Inverse condition number: {np.linalg.cond(inverse):.2e}")
    except Exception as e:
        print(f"Safe inverse failed: {e}")
    
    print("✅ Numerical stability framework operational")

if __name__ == "__main__":
    main()
