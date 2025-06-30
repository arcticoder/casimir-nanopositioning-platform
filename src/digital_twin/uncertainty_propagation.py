"""
Uncertainty Propagation Module
=============================

This module implements advanced uncertainty propagation techniques for the
digital twin, including Monte Carlo methods, polynomial chaos expansion,
and sensitivity analysis.

Mathematical Formulation:

Monte Carlo Propagation:
Y = f(X₁, X₂, ..., Xₙ)
E[Y] ≈ (1/N) × Σᵢ f(x₁ⁱ, x₂ⁱ, ..., xₙⁱ)
Var[Y] ≈ (1/(N-1)) × Σᵢ [f(x₁ⁱ, x₂ⁱ, ..., xₙⁱ) - E[Y]]²

Polynomial Chaos Expansion:
Y ≈ Σₖ αₖ × Ψₖ(ξ)
where Ψₖ are orthogonal polynomials and ξ are standard random variables

Sensitivity Analysis:
Sᵢ = Var[E[Y|Xᵢ]] / Var[Y]  (First-order sensitivity)
STᵢ = E[Var[Y|X₋ᵢ]] / Var[Y]  (Total-order sensitivity)

Gaussian Process Surrogate:
Y(x) ~ GP(μ(x), k(x,x'))
"""

import numpy as np
from scipy import stats, optimize, linalg
from scipy.special import factorial, binom, hermite, legendre
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from typing import Dict, List, Tuple, Optional, Callable, Union, NamedTuple
import logging
from dataclasses import dataclass
import time
from collections import deque
import warnings
from abc import ABC, abstractmethod
from enum import Enum
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# Performance targets for uncertainty propagation
UQ_ACCURACY_TARGET = 0.99  # R² coefficient
UQ_EFFICIENCY_TARGET = 1e-3  # seconds per evaluation
UQ_COVERAGE_TARGET = 0.95  # Coverage probability

class UncertaintyMethod(Enum):
    """Types of uncertainty propagation methods."""
    MONTE_CARLO = "monte_carlo"
    LATIN_HYPERCUBE = "latin_hypercube"
    POLYNOMIAL_CHAOS = "polynomial_chaos"
    GAUSSIAN_PROCESS = "gaussian_process"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    MOMENT_MATCHING = "moment_matching"

class DistributionType(Enum):
    """Types of probability distributions."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    BETA = "beta"
    GAMMA = "gamma"
    LOGNORMAL = "lognormal"
    TRUNCATED_NORMAL = "truncated_normal"

@dataclass
class UncertaintyParameters:
    """Parameters for uncertainty propagation."""
    
    # Monte Carlo parameters - CRITICAL: Increased for nanopositioning precision
    n_samples: int = 50000  # Increased from 10,000 for CRITICAL precision requirements
    random_seed: Optional[int] = 42
    confidence_level: float = 0.95
    # CRITICAL: Auto-convergence parameters
    auto_convergence: bool = True
    convergence_check_interval: int = 5000  # Check every 5k samples
    min_samples: int = 10000  # Minimum before convergence check
    
    # Polynomial Chaos parameters
    pce_order: int = 3
    pce_basis: str = "hermite"  # hermite, legendre, laguerre
    
    # Gaussian Process parameters
    gp_kernel: str = "rbf"  # rbf, matern
    gp_alpha: float = 1e-10
    gp_n_restarts: int = 10
    
    # Sensitivity analysis parameters
    sobol_n_samples: int = 8192
    morris_n_levels: int = 10
    morris_grid_jump: int = 5
    
    # Convergence criteria - CRITICAL: Tightened for precision applications
    convergence_tolerance: float = 1e-8  # Tightened from 1e-6 for CRITICAL precision
    max_evaluations: int = 500000  # Increased limit for convergence
    # CRITICAL: Numerical stability parameters
    numerical_stability_check: bool = True
    overflow_threshold: float = 1e100
    underflow_threshold: float = 1e-100
    
    # Parallel processing
    n_jobs: int = -1  # Use all available cores
    use_parallel: bool = True

class UncertainVariable:
    """Representation of an uncertain variable."""
    
    def __init__(self, name: str, distribution: DistributionType, 
                 parameters: Dict):
        """
        Initialize uncertain variable.
        
        Args:
            name: Variable name
            distribution: Distribution type
            parameters: Distribution parameters
        """
        self.name = name
        self.distribution = distribution
        self.parameters = parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate distribution parameters."""
        if self.distribution == DistributionType.NORMAL:
            required = ['mean', 'std']
        elif self.distribution == DistributionType.UNIFORM:
            required = ['low', 'high']
        elif self.distribution == DistributionType.BETA:
            required = ['alpha', 'beta']
        elif self.distribution == DistributionType.GAMMA:
            required = ['shape', 'scale']
        elif self.distribution == DistributionType.LOGNORMAL:
            required = ['mean', 'sigma']
        elif self.distribution == DistributionType.TRUNCATED_NORMAL:
            required = ['mean', 'std', 'low', 'high']
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution}")
            
        for param in required:
            if param not in self.parameters:
                raise ValueError(f"Missing parameter '{param}' for {self.distribution}")
    
    def sample(self, n_samples: int, random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Generate samples from the distribution."""
        if random_state is None:
            random_state = np.random.RandomState()
            
        if self.distribution == DistributionType.NORMAL:
            return random_state.normal(self.parameters['mean'], 
                                     self.parameters['std'], n_samples)
        elif self.distribution == DistributionType.UNIFORM:
            return random_state.uniform(self.parameters['low'], 
                                      self.parameters['high'], n_samples)
        elif self.distribution == DistributionType.BETA:
            return random_state.beta(self.parameters['alpha'], 
                                   self.parameters['beta'], n_samples)
        elif self.distribution == DistributionType.GAMMA:
            return random_state.gamma(self.parameters['shape'], 
                                    self.parameters['scale'], n_samples)
        elif self.distribution == DistributionType.LOGNORMAL:
            return random_state.lognormal(self.parameters['mean'], 
                                        self.parameters['sigma'], n_samples)
        elif self.distribution == DistributionType.TRUNCATED_NORMAL:
            samples = random_state.normal(self.parameters['mean'], 
                                        self.parameters['std'], n_samples * 2)
            mask = (samples >= self.parameters['low']) & (samples <= self.parameters['high'])
            truncated = samples[mask][:n_samples]
            
            # If not enough samples, pad with boundary values
            if len(truncated) < n_samples:
                n_pad = n_samples - len(truncated)
                pad_values = random_state.uniform(self.parameters['low'], 
                                                self.parameters['high'], n_pad)
                truncated = np.concatenate([truncated, pad_values])
            
            return truncated
        else:
            raise ValueError(f"Sampling not implemented for {self.distribution}")
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density function."""
        if self.distribution == DistributionType.NORMAL:
            return stats.norm.pdf(x, self.parameters['mean'], self.parameters['std'])
        elif self.distribution == DistributionType.UNIFORM:
            return stats.uniform.pdf(x, self.parameters['low'], 
                                   self.parameters['high'] - self.parameters['low'])
        elif self.distribution == DistributionType.BETA:
            return stats.beta.pdf(x, self.parameters['alpha'], self.parameters['beta'])
        elif self.distribution == DistributionType.GAMMA:
            return stats.gamma.pdf(x, self.parameters['shape'], scale=self.parameters['scale'])
        elif self.distribution == DistributionType.LOGNORMAL:
            return stats.lognorm.pdf(x, self.parameters['sigma'], scale=np.exp(self.parameters['mean']))
        else:
            raise ValueError(f"PDF not implemented for {self.distribution}")
    
    def mean(self) -> float:
        """Compute theoretical mean."""
        if self.distribution == DistributionType.NORMAL:
            return self.parameters['mean']
        elif self.distribution == DistributionType.UNIFORM:
            return (self.parameters['low'] + self.parameters['high']) / 2
        elif self.distribution == DistributionType.BETA:
            a, b = self.parameters['alpha'], self.parameters['beta']
            return a / (a + b)
        elif self.distribution == DistributionType.GAMMA:
            return self.parameters['shape'] * self.parameters['scale']
        elif self.distribution == DistributionType.LOGNORMAL:
            return np.exp(self.parameters['mean'] + self.parameters['sigma']**2 / 2)
        else:
            raise ValueError(f"Mean not implemented for {self.distribution}")
    
    def variance(self) -> float:
        """Compute theoretical variance."""
        if self.distribution == DistributionType.NORMAL:
            return self.parameters['std']**2
        elif self.distribution == DistributionType.UNIFORM:
            diff = self.parameters['high'] - self.parameters['low']
            return diff**2 / 12
        elif self.distribution == DistributionType.BETA:
            a, b = self.parameters['alpha'], self.parameters['beta']
            return (a * b) / ((a + b)**2 * (a + b + 1))
        elif self.distribution == DistributionType.GAMMA:
            return self.parameters['shape'] * self.parameters['scale']**2
        elif self.distribution == DistributionType.LOGNORMAL:
            mu, sigma = self.parameters['mean'], self.parameters['sigma']
            return (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        else:
            raise ValueError(f"Variance not implemented for {self.distribution}")

class UncertaintyPropagator(ABC):
    """Abstract base class for uncertainty propagation methods."""
    
    def __init__(self, uncertain_variables: List[UncertainVariable], 
                 params: UncertaintyParameters):
        self.uncertain_variables = uncertain_variables
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.evaluation_times = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=100)
        
    @abstractmethod
    def propagate(self, model_function: Callable) -> Dict:
        """Propagate uncertainty through model."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get propagation performance metrics."""
        if not self.evaluation_times:
            return {'status': 'no_evaluations'}
        
        avg_time = np.mean(list(self.evaluation_times))
        max_time = np.max(list(self.evaluation_times))
        
        # Accuracy (if available)
        avg_accuracy = 0.0
        if self.accuracy_history:
            avg_accuracy = np.mean(list(self.accuracy_history))
        
        return {
            'avg_evaluation_time_s': avg_time,
            'max_evaluation_time_s': max_time,
            'avg_accuracy': avg_accuracy,
            'efficiency_satisfied': avg_time <= UQ_EFFICIENCY_TARGET,
            'accuracy_satisfied': avg_accuracy >= UQ_ACCURACY_TARGET,
            'n_evaluations': len(self.evaluation_times)
        }

class MonteCarloUncertaintyPropagator(UncertaintyPropagator):
    """
    Monte Carlo uncertainty propagation.
    
    Mathematical Implementation:
    1. Generate N samples from input distributions
    2. Evaluate model at each sample point
    3. Compute statistics of output distribution
    
    Statistical Moments:
    μ = E[Y] = (1/N) × Σᵢ f(xᵢ)
    σ² = Var[Y] = (1/(N-1)) × Σᵢ [f(xᵢ) - μ]²
    """
    
    def __init__(self, uncertain_variables: List[UncertainVariable], 
                 params: UncertaintyParameters):
        super().__init__(uncertain_variables, params)
        self.random_state = np.random.RandomState(params.random_seed)
        
    def propagate(self, model_function: Callable) -> Dict:
        """
        CRITICAL: Monte Carlo uncertainty propagation with convergence validation.
        
        Enhanced implementation with automatic convergence checking and numerical
        stability monitoring for precision nanopositioning applications.
        """
        start_time = time.time()
        
        # CRITICAL: Initialize convergence tracking
        converged = False
        convergence_history = []
        n_current = self.params.min_samples
        
        # Initial sample generation
        samples = self._generate_samples(n_current)
        outputs = []
        
        # CRITICAL: Iterative sampling with convergence checking
        while not converged and n_current <= self.params.max_evaluations:
            # Evaluate model for current batch
            if self.params.use_parallel and self.params.n_jobs != 1:
                batch_outputs = self._parallel_evaluation(model_function, samples)
            else:
                batch_outputs = self._sequential_evaluation(model_function, samples)
            
            outputs.extend(batch_outputs)
            
            # CRITICAL: Numerical stability check
            if self.params.numerical_stability_check:
                stability_issues = self._check_numerical_stability(outputs)
                if stability_issues['has_issues']:
                    self.logger.warning(f"Numerical stability issues detected: {stability_issues}")
            
            # CRITICAL: Convergence check
            if self.params.auto_convergence and n_current >= self.params.min_samples:
                convergence_result = self._check_convergence(outputs)
                convergence_history.append(convergence_result)
                
                if convergence_result['converged']:
                    converged = True
                    self.logger.info(f"Monte Carlo converged at {n_current} samples")
                elif n_current < self.params.max_evaluations:
                    # Generate additional samples
                    additional_samples = self.params.convergence_check_interval
                    new_samples = self._generate_samples(additional_samples)
                    samples = np.vstack([samples, new_samples])
                    n_current += additional_samples
                else:
                    self.logger.warning(f"Maximum evaluations reached without convergence")
                    break
            else:
                converged = True  # Skip convergence check if disabled
        
        # Convert to numpy array for processing
        outputs = np.array(outputs)
        
        # CRITICAL: Final validation of results
        final_validation = self._validate_final_results(outputs)
        if not final_validation['valid']:
            self.logger.error(f"Final validation failed: {final_validation['reason']}")
        
        # Compute statistics
        statistics = self._compute_statistics(outputs)
        
        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            outputs, self.params.confidence_level
        )
        
        # Performance tracking
        evaluation_time = time.time() - start_time
        self.evaluation_times.append(evaluation_time)
        
        # Convergence analysis
        convergence_analysis = self._analyze_convergence(outputs)
        
        return {
            'method': 'monte_carlo',
            'n_samples': len(outputs),
            'converged': converged,
            'convergence_history': convergence_history,
            'numerical_stability': self._check_numerical_stability(outputs),
            'final_validation': final_validation,
            'statistics': statistics,
            'confidence_intervals': confidence_intervals,
            'convergence': convergence_analysis,
            'samples': samples[:len(outputs)],  # Trim to actual evaluations
            'outputs': outputs,
            'evaluation_time_s': evaluation_time,
            'samples_per_second': len(outputs) / evaluation_time if evaluation_time > 0 else 0
        }
    
    def _generate_samples(self) -> np.ndarray:
        """Generate Monte Carlo samples."""
        n_vars = len(self.uncertain_variables)
        samples = np.zeros((self.params.n_samples, n_vars))
        
        for i, var in enumerate(self.uncertain_variables):
            samples[:, i] = var.sample(self.params.n_samples, self.random_state)
        
        return samples
    
    def _sequential_evaluation(self, model_function: Callable, 
                             samples: np.ndarray) -> np.ndarray:
        """Sequential model evaluation."""
        outputs = []
        
        for i, sample in enumerate(samples):
            try:
                output = model_function(sample)
                if np.isscalar(output):
                    outputs.append(output)
                else:
                    outputs.append(np.asarray(output))
            except Exception as e:
                self.logger.warning(f"Model evaluation failed for sample {i}: {e}")
                outputs.append(np.nan)
        
        return np.array(outputs)
    
    def _parallel_evaluation(self, model_function: Callable, 
                           samples: np.ndarray) -> np.ndarray:
        """Parallel model evaluation."""
        outputs = [None] * len(samples)
        
        with ThreadPoolExecutor(max_workers=self.params.n_jobs) as executor:
            # Submit tasks
            future_to_index = {
                executor.submit(model_function, sample): i 
                for i, sample in enumerate(samples)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if np.isscalar(result):
                        outputs[index] = result
                    else:
                        outputs[index] = np.asarray(result)
                except Exception as e:
                    self.logger.warning(f"Parallel evaluation failed for sample {index}: {e}")
                    outputs[index] = np.nan
        
        return np.array(outputs)
    
    def _compute_statistics(self, outputs: np.ndarray) -> Dict:
        """Compute output statistics."""
        # Remove NaN values
        valid_outputs = outputs[~np.isnan(outputs)]
        
        if len(valid_outputs) == 0:
            return {'error': 'no_valid_outputs'}
        
        # Basic statistics
        statistics = {
            'mean': np.mean(valid_outputs),
            'std': np.std(valid_outputs, ddof=1),
            'variance': np.var(valid_outputs, ddof=1),
            'min': np.min(valid_outputs),
            'max': np.max(valid_outputs),
            'median': np.median(valid_outputs),
            'q25': np.percentile(valid_outputs, 25),
            'q75': np.percentile(valid_outputs, 75),
            'skewness': stats.skew(valid_outputs),
            'kurtosis': stats.kurtosis(valid_outputs),
            'n_valid': len(valid_outputs),
            'n_invalid': len(outputs) - len(valid_outputs)
        }
        
        # Higher-order moments
        if len(valid_outputs) > 3:
            statistics.update({
                'moment_3': stats.moment(valid_outputs, moment=3),
                'moment_4': stats.moment(valid_outputs, moment=4)
            })
        
        return statistics
    
    def _compute_confidence_intervals(self, outputs: np.ndarray, 
                                    confidence_level: float) -> Dict:
        """Compute confidence intervals."""
        valid_outputs = outputs[~np.isnan(outputs)]
        
        if len(valid_outputs) < 2:
            return {'error': 'insufficient_data'}
        
        alpha = 1 - confidence_level
        
        # Percentile-based intervals
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        intervals = {
            'confidence_level': confidence_level,
            'percentile_lower': np.percentile(valid_outputs, lower_percentile),
            'percentile_upper': np.percentile(valid_outputs, upper_percentile),
        }
        
        # Bootstrap confidence intervals for mean
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = self.random_state.choice(
                valid_outputs, size=len(valid_outputs), replace=True
            )
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        intervals.update({
            'bootstrap_mean_lower': np.percentile(bootstrap_means, lower_percentile),
            'bootstrap_mean_upper': np.percentile(bootstrap_means, upper_percentile)
        })
        
        return intervals
    
    def _analyze_convergence(self, outputs: np.ndarray) -> Dict:
        """Analyze Monte Carlo convergence."""
        valid_outputs = outputs[~np.isnan(outputs)]
        n_valid = len(valid_outputs)
        
        if n_valid < 10:
            return {'error': 'insufficient_data'}
        
        # Running statistics
        running_means = np.cumsum(valid_outputs) / np.arange(1, n_valid + 1)
        running_vars = []
        
        for i in range(2, n_valid + 1):
            running_vars.append(np.var(valid_outputs[:i], ddof=1))
        running_vars = np.array(running_vars)
        
        # Convergence criteria
        final_mean = running_means[-1]
        final_var = running_vars[-1]
        
        # Check convergence of mean (last 10% of samples)
        n_check = max(10, n_valid // 10)
        recent_means = running_means[-n_check:]
        mean_convergence = np.std(recent_means) / np.abs(final_mean) if final_mean != 0 else np.std(recent_means)
        
        # Check convergence of variance
        recent_vars = running_vars[-n_check:]
        var_convergence = np.std(recent_vars) / final_var if final_var != 0 else np.std(recent_vars)
        
        return {
            'mean_convergence_ratio': mean_convergence,
            'variance_convergence_ratio': var_convergence,
            'running_means': running_means.tolist(),
            'running_variances': running_vars.tolist(),
            'is_converged': (mean_convergence < self.params.convergence_tolerance and 
                           var_convergence < self.params.convergence_tolerance),
            'suggested_n_samples': self._estimate_required_samples(valid_outputs)
        }
    
    def _estimate_required_samples(self, outputs: np.ndarray) -> int:
        """Estimate required sample size for convergence."""
        if len(outputs) < 100:
            return self.params.n_samples * 2
        
        # Use central limit theorem approximation
        sample_std = np.std(outputs, ddof=1)
        sample_mean = np.mean(outputs)
        
        # Target relative error
        target_relative_error = self.params.convergence_tolerance
        
        if sample_mean != 0:
            # For relative error
            z_score = stats.norm.ppf(1 - 0.05/2)  # 95% confidence
            required_n = (z_score * sample_std / (target_relative_error * np.abs(sample_mean)))**2
        else:
            # For absolute error
            target_absolute_error = target_relative_error
            required_n = (z_score * sample_std / target_absolute_error)**2
        
        return max(int(required_n), len(outputs))
    
    def _check_convergence(self, outputs: np.ndarray) -> Dict:
        """
        CRITICAL: Check Monte Carlo convergence using multiple criteria.
        
        Implements rigorous convergence assessment for precision nanopositioning
        applications requiring validated uncertainty bounds.
        """
        if len(outputs) < self.params.min_samples:
            return {'converged': False, 'reason': 'insufficient_samples'}
        
        # Statistical convergence criteria
        n = len(outputs)
        mean_estimate = np.mean(outputs)
        std_estimate = np.std(outputs, ddof=1)
        
        # 1. Standard error convergence
        standard_error = std_estimate / np.sqrt(n)
        relative_error = standard_error / np.abs(mean_estimate) if mean_estimate != 0 else standard_error
        
        se_converged = relative_error < self.params.convergence_tolerance
        
        # 2. Running average convergence (last 20% vs total)
        split_point = int(0.8 * n)
        if split_point < n // 2:
            running_converged = False
        else:
            early_mean = np.mean(outputs[:split_point])
            late_mean = np.mean(outputs[split_point:])
            running_diff = np.abs(late_mean - early_mean) / np.abs(early_mean) if early_mean != 0 else np.abs(late_mean)
            running_converged = running_diff < self.params.convergence_tolerance
        
        # 3. Batch convergence (compare last 1000 samples with total)
        if n > 2000:
            batch_mean = np.mean(outputs[-1000:])
            total_mean = np.mean(outputs)
            batch_diff = np.abs(batch_mean - total_mean) / np.abs(total_mean) if total_mean != 0 else np.abs(batch_mean)
            batch_converged = batch_diff < self.params.convergence_tolerance
        else:
            batch_converged = True  # Not enough samples for batch test
        
        # Overall convergence
        converged = se_converged and running_converged and batch_converged
        
        return {
            'converged': converged,
            'n_samples': n,
            'standard_error_converged': se_converged,
            'running_average_converged': running_converged,
            'batch_converged': batch_converged,
            'relative_error': relative_error,
            'standard_error': standard_error,
            'confidence_bound': 1.96 * standard_error  # 95% confidence
        }
    
    def _check_numerical_stability(self, outputs: np.ndarray) -> Dict:
        """
        CRITICAL: Check for numerical stability issues.
        
        Identifies overflow, underflow, and other numerical problems that
        could compromise uncertainty quantification accuracy.
        """
        outputs_array = np.array(outputs)
        
        # Check for numerical issues
        has_inf = np.any(np.isinf(outputs_array))
        has_nan = np.any(np.isnan(outputs_array))
        has_overflow = np.any(np.abs(outputs_array) > self.params.overflow_threshold)
        has_underflow = np.any((outputs_array != 0) & (np.abs(outputs_array) < self.params.underflow_threshold))
        
        # Dynamic range check
        valid_outputs = outputs_array[np.isfinite(outputs_array)]
        if len(valid_outputs) > 0:
            max_val = np.max(np.abs(valid_outputs))
            min_val = np.min(np.abs(valid_outputs[valid_outputs != 0])) if np.any(valid_outputs != 0) else 1.0
            dynamic_range = max_val / min_val if min_val > 0 else np.inf
        else:
            dynamic_range = 0
        
        # Condition number estimate (if sufficient samples)
        condition_estimate = np.inf
        if len(valid_outputs) > 10:
            # Simple condition estimate based on sample statistics
            sample_std = np.std(valid_outputs)
            sample_mean = np.mean(valid_outputs)
            if sample_std > 0 and sample_mean != 0:
                condition_estimate = np.abs(sample_mean) / sample_std
        
        has_issues = has_inf or has_nan or has_overflow or has_underflow or dynamic_range > 1e12
        
        return {
            'has_issues': has_issues,
            'has_inf': has_inf,
            'has_nan': has_nan,
            'has_overflow': has_overflow,
            'has_underflow': has_underflow,
            'dynamic_range': dynamic_range,
            'condition_estimate': condition_estimate,
            'valid_fraction': len(valid_outputs) / len(outputs_array) if len(outputs_array) > 0 else 0,
            'recommendations': self._get_stability_recommendations(has_inf, has_nan, has_overflow, has_underflow, dynamic_range)
        }
    
    def _get_stability_recommendations(self, has_inf: bool, has_nan: bool, 
                                     has_overflow: bool, has_underflow: bool, 
                                     dynamic_range: float) -> List[str]:
        """Get recommendations for numerical stability issues."""
        recommendations = []
        
        if has_inf:
            recommendations.append("Use log-space calculations to prevent infinite values")
        if has_nan:
            recommendations.append("Add input validation and handle edge cases")
        if has_overflow:
            recommendations.append("Scale down model inputs or use higher precision arithmetic")
        if has_underflow:
            recommendations.append("Increase numerical precision or adjust model scaling")
        if dynamic_range > 1e10:
            recommendations.append("Consider log-normal or scaled input distributions")
        
        return recommendations
    
    def _validate_final_results(self, outputs: np.ndarray) -> Dict:
        """
        CRITICAL: Final validation of Monte Carlo results.
        
        Ensures that the uncertainty quantification results meet precision
        requirements for nanopositioning applications.
        """
        valid_outputs = outputs[np.isfinite(outputs)]
        
        # Basic validity checks
        if len(valid_outputs) == 0:
            return {'valid': False, 'reason': 'no_valid_outputs'}
        
        if len(valid_outputs) < 0.9 * len(outputs):
            return {'valid': False, 'reason': 'excessive_invalid_outputs', 
                   'valid_fraction': len(valid_outputs) / len(outputs)}
        
        # Statistical validity
        mean_val = np.mean(valid_outputs)
        std_val = np.std(valid_outputs, ddof=1)
        
        # Coefficient of variation check
        if mean_val != 0:
            cv = std_val / np.abs(mean_val)
            if cv > 10:  # Excessive variability
                return {'valid': False, 'reason': 'excessive_variability', 'cv': cv}
        
        # Sample size adequacy
        if len(valid_outputs) < self.params.min_samples:
            return {'valid': False, 'reason': 'insufficient_valid_samples', 
                   'valid_samples': len(valid_outputs)}
        
        # Distribution shape checks (normality test for large samples)
        from scipy import stats
        if len(valid_outputs) > 5000:
            # Shapiro-Wilk test on subsample (max 5000 samples)
            test_sample = np.random.choice(valid_outputs, size=min(5000, len(valid_outputs)), replace=False)
            _, p_value = stats.shapiro(test_sample)
            
            # If p < 0.01, likely non-normal - check for reasonableness
            if p_value < 0.01:
                # Check for extreme skewness or kurtosis
                skewness = stats.skew(valid_outputs)
                kurtosis = stats.kurtosis(valid_outputs)
                
                if np.abs(skewness) > 5 or np.abs(kurtosis) > 10:
                    return {'valid': False, 'reason': 'extreme_distribution_shape',
                           'skewness': skewness, 'kurtosis': kurtosis}
        
        return {
            'valid': True,
            'valid_fraction': len(valid_outputs) / len(outputs),
            'mean': mean_val,
            'std': std_val,
            'cv': std_val / np.abs(mean_val) if mean_val != 0 else np.inf
        }
    
    def _generate_samples(self, n_samples: Optional[int] = None) -> np.ndarray:
        """Generate samples from uncertain variables with numerical stability."""
        if n_samples is None:
            n_samples = self.params.n_samples
            
        n_vars = len(self.uncertain_variables)
        samples = np.zeros((n_samples, n_vars))
        
        for i, variable in enumerate(self.uncertain_variables):
            try:
                var_samples = variable.sample(n_samples, random_state=self.random_state)
                
                # CRITICAL: Numerical stability check for samples
                if not np.all(np.isfinite(var_samples)):
                    self.logger.warning(f"Non-finite samples detected for variable {variable.name}")
                    # Replace non-finite values with distribution mean
                    finite_mask = np.isfinite(var_samples)
                    if np.any(finite_mask):
                        replacement_value = np.mean(var_samples[finite_mask])
                    else:
                        replacement_value = variable.parameters.get('mean', 0)
                    var_samples[~finite_mask] = replacement_value
                
                samples[:, i] = var_samples
                
            except Exception as e:
                self.logger.error(f"Error sampling variable {variable.name}: {e}")
                # Fallback to normal distribution
                samples[:, i] = self.random_state.normal(0, 1, n_samples)
        
        return samples

class LatinHypercubeUncertaintyPropagator(UncertaintyPropagator):
    """
    Latin Hypercube Sampling for more efficient space-filling design.
    
    Mathematical Implementation:
    1. Divide each input dimension into N equal intervals
    2. Sample once from each interval for each dimension
    3. Randomly permute the samples to avoid correlation
    """
    
    def __init__(self, uncertain_variables: List[UncertainVariable], 
                 params: UncertaintyParameters):
        super().__init__(uncertain_variables, params)
        self.random_state = np.random.RandomState(params.random_seed)
    
    def propagate(self, model_function: Callable) -> Dict:
        """Latin Hypercube uncertainty propagation."""
        start_time = time.time()
        
        # Generate LHS samples
        samples = self._generate_lhs_samples()
        
        # Evaluate model
        if self.params.use_parallel and self.params.n_jobs != 1:
            outputs = self._parallel_evaluation(model_function, samples)
        else:
            outputs = self._sequential_evaluation(model_function, samples)
        
        # Compute statistics (reuse Monte Carlo methods)
        mc_propagator = MonteCarloUncertaintyPropagator(self.uncertain_variables, self.params)
        statistics = mc_propagator._compute_statistics(outputs)
        confidence_intervals = mc_propagator._compute_confidence_intervals(
            outputs, self.params.confidence_level
        )
        
        # Performance tracking
        evaluation_time = time.time() - start_time
        self.evaluation_times.append(evaluation_time)
        
        return {
            'method': 'latin_hypercube',
            'n_samples': len(outputs),
            'statistics': statistics,
            'confidence_intervals': confidence_intervals,
            'samples': samples,
            'outputs': outputs,
            'evaluation_time_s': evaluation_time,
            'space_filling_quality': self._assess_space_filling(samples)
        }
    
    def _generate_lhs_samples(self) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        n_vars = len(self.uncertain_variables)
        n_samples = self.params.n_samples
        
        # Generate LHS design in [0,1]^d
        lhs_design = np.zeros((n_samples, n_vars))
        
        for j in range(n_vars):
            # Divide interval into n_samples segments
            segments = np.arange(n_samples, dtype=float) / n_samples
            
            # Add random offset within each segment
            offsets = self.random_state.random(n_samples) / n_samples
            uniform_samples = segments + offsets
            
            # Random permutation to decorrelate dimensions
            lhs_design[:, j] = self.random_state.permutation(uniform_samples)
        
        # Transform to actual distributions
        samples = np.zeros_like(lhs_design)
        for j, var in enumerate(self.uncertain_variables):
            # Use inverse CDF transformation
            uniform_values = lhs_design[:, j]
            samples[:, j] = self._inverse_cdf_transform(var, uniform_values)
        
        return samples
    
    def _inverse_cdf_transform(self, variable: UncertainVariable, 
                              uniform_values: np.ndarray) -> np.ndarray:
        """Transform uniform samples to variable distribution."""
        if variable.distribution == DistributionType.NORMAL:
            return stats.norm.ppf(uniform_values, 
                                variable.parameters['mean'], 
                                variable.parameters['std'])
        elif variable.distribution == DistributionType.UNIFORM:
            low, high = variable.parameters['low'], variable.parameters['high']
            return low + uniform_values * (high - low)
        elif variable.distribution == DistributionType.BETA:
            return stats.beta.ppf(uniform_values,
                                variable.parameters['alpha'],
                                variable.parameters['beta'])
        elif variable.distribution == DistributionType.GAMMA:
            return stats.gamma.ppf(uniform_values,
                                 variable.parameters['shape'],
                                 scale=variable.parameters['scale'])
        elif variable.distribution == DistributionType.LOGNORMAL:
            return stats.lognorm.ppf(uniform_values,
                                   variable.parameters['sigma'],
                                   scale=np.exp(variable.parameters['mean']))
        else:
            # Fallback: use numerical inversion
            return self._numerical_inverse_cdf(variable, uniform_values)
    
    def _numerical_inverse_cdf(self, variable: UncertainVariable,
                              uniform_values: np.ndarray) -> np.ndarray:
        """Numerical inverse CDF for unsupported distributions."""
        # This is a simplified implementation
        # In practice, you might want to use more sophisticated methods
        samples = []
        for u in uniform_values:
            # Use root finding to solve F(x) = u
            def cdf_equation(x):
                # Approximate CDF using samples
                test_samples = variable.sample(1000, self.random_state)
                return np.mean(test_samples <= x) - u
            
            try:
                # Find root
                result = optimize.brentq(cdf_equation, -10, 10)
                samples.append(result)
            except:
                # Fallback to direct sampling
                samples.append(variable.sample(1, self.random_state)[0])
        
        return np.array(samples)
    
    def _sequential_evaluation(self, model_function: Callable, 
                             samples: np.ndarray) -> np.ndarray:
        """Sequential model evaluation (same as Monte Carlo)."""
        outputs = []
        
        for i, sample in enumerate(samples):
            try:
                output = model_function(sample)
                if np.isscalar(output):
                    outputs.append(output)
                else:
                    outputs.append(np.asarray(output))
            except Exception as e:
                self.logger.warning(f"Model evaluation failed for sample {i}: {e}")
                outputs.append(np.nan)
        
        return np.array(outputs)
    
    def _parallel_evaluation(self, model_function: Callable, 
                           samples: np.ndarray) -> np.ndarray:
        """Parallel model evaluation (same as Monte Carlo)."""
        outputs = [None] * len(samples)
        
        with ThreadPoolExecutor(max_workers=self.params.n_jobs) as executor:
            future_to_index = {
                executor.submit(model_function, sample): i 
                for i, sample in enumerate(samples)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if np.isscalar(result):
                        outputs[index] = result
                    else:
                        outputs[index] = np.asarray(result)
                except Exception as e:
                    self.logger.warning(f"Parallel evaluation failed for sample {index}: {e}")
                    outputs[index] = np.nan
        
        return np.array(outputs)
    
    def _assess_space_filling(self, samples: np.ndarray) -> Dict:
        """Assess space-filling quality of LHS design."""
        n_samples, n_dims = samples.shape
        
        # Minimum distance criterion
        distances = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(samples[i] - samples[j])
                distances.append(dist)
        
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        
        # Coverage uniformity (simplified)
        # Divide each dimension into bins and check occupancy
        n_bins = int(np.ceil(n_samples**(1/n_dims)))
        uniformity_scores = []
        
        for dim in range(n_dims):
            hist, _ = np.histogram(samples[:, dim], bins=n_bins)
            expected_count = n_samples / n_bins
            uniformity = 1.0 / (1.0 + np.std(hist) / expected_count) if expected_count > 0 else 0.0
            uniformity_scores.append(uniformity)
        
        return {
            'min_distance': min_distance,
            'avg_distance': avg_distance,
            'distance_ratio': min_distance / avg_distance,
            'dimension_uniformity': uniformity_scores,
            'avg_uniformity': np.mean(uniformity_scores),
            'space_filling_quality': np.mean(uniformity_scores) * (min_distance / avg_distance)
        }

class MultiDomainUncertaintyPropagator:
    """
    HIGH SEVERITY RESOLUTION: Multi-domain uncertainty propagation with correlation modeling.
    
    This class addresses the critical need for cross-domain uncertainty correlation
    in multi-physics systems where mechanical, thermal, electromagnetic, and quantum
    domains are coupled.
    """
    
    def __init__(self, domain_propagators: Dict[str, UncertaintyPropagator]):
        """
        Initialize multi-domain uncertainty propagator.
        
        Args:
            domain_propagators: Dictionary of domain-specific uncertainty propagators
        """
        self.domain_propagators = domain_propagators
        self.correlation_matrix = None
        self.joint_samples = None
        self.logger = logging.getLogger(__name__)
    
    def estimate_correlation_matrix(self, measurement_history: List[Dict]) -> np.ndarray:
        """
        HIGH SEVERITY: Estimate cross-domain correlation matrix from measurement data.
        
        This addresses the critical gap in correlation modeling between physics domains.
        """
        if len(measurement_history) < 20:
            self.logger.warning("Insufficient data for correlation estimation")
            return np.eye(len(self.domain_propagators))
        
        # Extract domain measurements
        domain_names = list(self.domain_propagators.keys())
        domain_data = {name: [] for name in domain_names}
        
        for entry in measurement_history[-100:]:  # Use last 100 measurements
            measurements = entry.get('measurements', {})
            
            # Convert PhysicsDomain keys to strings if needed
            for domain_key, measurement in measurements.items():
                domain_str = domain_key.value if hasattr(domain_key, 'value') else str(domain_key)
                if domain_str in domain_data:
                    if np.isscalar(measurement):
                        domain_data[domain_str].append(measurement)
                    else:
                        # Use first component if vector
                        domain_data[domain_str].append(np.asarray(measurement).flatten()[0])
        
        # Ensure equal length data
        min_length = min(len(data) for data in domain_data.values() if data)
        if min_length < 10:
            self.logger.warning(f"Insufficient data for correlation estimation: {min_length} samples")
            return np.eye(len(domain_names))
        
        # Create data matrix
        data_matrix = []
        valid_domains = []
        
        for domain_name in domain_names:
            if len(domain_data[domain_name]) >= min_length:
                domain_array = np.array(domain_data[domain_name][:min_length])
                
                # Standardize data
                if np.std(domain_array) > 0:
                    domain_array = (domain_array - np.mean(domain_array)) / np.std(domain_array)
                
                data_matrix.append(domain_array)
                valid_domains.append(domain_name)
        
        if len(data_matrix) < 2:
            return np.eye(len(domain_names))
        
        data_matrix = np.array(data_matrix)
        
        # Calculate correlation matrix
        try:
            correlation_matrix = np.corrcoef(data_matrix)
            
            # Ensure positive definite (regularize if needed)
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval < 1e-6:
                # Regularize by adding small diagonal term
                regularization = 1e-6 - min_eigenval + 1e-8
                correlation_matrix += regularization * np.eye(correlation_matrix.shape[0])
                self.logger.info(f"Regularized correlation matrix with λ={regularization:.2e}")
            
            # Expand to full size if some domains were missing
            if len(valid_domains) < len(domain_names):
                full_correlation = np.eye(len(domain_names))
                valid_indices = [domain_names.index(name) for name in valid_domains]
                
                for i, idx_i in enumerate(valid_indices):
                    for j, idx_j in enumerate(valid_indices):
                        full_correlation[idx_i, idx_j] = correlation_matrix[i, j]
                
                correlation_matrix = full_correlation
            
            self.correlation_matrix = correlation_matrix
            
            self.logger.info(f"Estimated correlation matrix from {min_length} samples")
            self.logger.debug(f"Max off-diagonal correlation: {np.max(np.abs(correlation_matrix - np.eye(len(domain_names)))):.3f}")
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Correlation estimation failed: {e}")
            return np.eye(len(domain_names))
    
    def generate_correlated_samples(self, n_samples: int, 
                                  correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        HIGH SEVERITY: Generate correlated samples across domains.
        
        Uses Cholesky decomposition to generate samples that preserve
        cross-domain correlation structure.
        """
        if correlation_matrix is None:
            correlation_matrix = self.correlation_matrix
            
        if correlation_matrix is None:
            correlation_matrix = np.eye(len(self.domain_propagators))
        
        domain_names = list(self.domain_propagators.keys())
        n_domains = len(domain_names)
        
        try:
            # Cholesky decomposition for correlated sampling
            L = np.linalg.cholesky(correlation_matrix)
            
            # Generate independent standard normal samples
            independent_samples = np.random.standard_normal((n_samples, n_domains))
            
            # Transform to correlated samples
            correlated_samples = independent_samples @ L.T
            
            # Transform to domain-specific distributions
            domain_samples = {}
            
            for i, domain_name in enumerate(domain_names):
                propagator = self.domain_propagators[domain_name]
                
                # Get standard normal samples for this domain
                std_normal_samples = correlated_samples[:, i]
                
                # Transform to domain distribution (simplified - assumes normal)
                if hasattr(propagator, 'uncertain_variables') and propagator.uncertain_variables:
                    # Use first uncertain variable as representative
                    var = propagator.uncertain_variables[0]
                    
                    if var.distribution == DistributionType.NORMAL:
                        mean = var.parameters.get('mean', 0)
                        std = var.parameters.get('std', 1)
                        domain_samples[domain_name] = mean + std * std_normal_samples
                    elif var.distribution == DistributionType.UNIFORM:
                        low = var.parameters.get('low', -1)
                        high = var.parameters.get('high', 1)
                        # Transform normal to uniform using CDF
                        from scipy.stats import norm
                        uniform_samples = norm.cdf(std_normal_samples)
                        domain_samples[domain_name] = low + (high - low) * uniform_samples
                    else:
                        # Fallback to normal
                        domain_samples[domain_name] = std_normal_samples
                else:
                    # No variables defined, use standard normal
                    domain_samples[domain_name] = std_normal_samples
                    
            self.joint_samples = domain_samples
            
            return domain_samples
            
        except np.linalg.LinAlgError as e:
            self.logger.error(f"Cholesky decomposition failed: {e}. Using independent samples.")
            
            # Fallback to independent samples
            domain_samples = {}
            for domain_name in domain_names:
                domain_samples[domain_name] = np.random.standard_normal(n_samples)
            
            return domain_samples
    
    def propagate_joint_uncertainty(self, coupled_model_function: Callable, 
                                  n_samples: int = 10000,
                                  measurement_history: Optional[List[Dict]] = None) -> Dict:
        """
        HIGH SEVERITY: Propagate uncertainty accounting for cross-domain correlations.
        
        This method addresses the critical need for system-level uncertainty
        quantification in multi-physics systems.
        """
        start_time = time.time()
        
        # Estimate correlations if measurement history is provided
        if measurement_history is not None:
            correlation_matrix = self.estimate_correlation_matrix(measurement_history)
        else:
            correlation_matrix = self.correlation_matrix or np.eye(len(self.domain_propagators))
        
        # Generate correlated samples
        correlated_samples = self.generate_correlated_samples(n_samples, correlation_matrix)
        
        # Evaluate coupled model
        outputs = []
        valid_evaluations = 0
        
        for i in range(n_samples):
            try:
                # Create input vector from correlated samples
                input_vector = []
                for domain_name in self.domain_propagators.keys():
                    input_vector.append(correlated_samples[domain_name][i])
                
                # Evaluate coupled model
                output = coupled_model_function(np.array(input_vector))
                
                if np.isfinite(output):
                    outputs.append(output)
                    valid_evaluations += 1
                else:
                    outputs.append(np.nan)
                    
            except Exception as e:
                outputs.append(np.nan)
                
        outputs = np.array(outputs)
        valid_outputs = outputs[np.isfinite(outputs)]
        
        if len(valid_outputs) == 0:
            return {'error': 'no_valid_outputs', 'total_samples': n_samples}
        
        # Calculate statistics
        statistics = {
            'mean': np.mean(valid_outputs),
            'std': np.std(valid_outputs, ddof=1),
            'var': np.var(valid_outputs, ddof=1),
            'min': np.min(valid_outputs),
            'max': np.max(valid_outputs),
            'median': np.median(valid_outputs),
            'q25': np.percentile(valid_outputs, 25),
            'q75': np.percentile(valid_outputs, 75),
            'skewness': float(self._calculate_skewness(valid_outputs)),
            'kurtosis': float(self._calculate_kurtosis(valid_outputs))
        }
        
        # Confidence intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        confidence_intervals = {
            f'{confidence_level*100:.0f}%': {
                'lower': np.percentile(valid_outputs, lower_percentile),
                'upper': np.percentile(valid_outputs, upper_percentile)
            }
        }
        
        # Correlation analysis
        correlation_analysis = {
            'correlation_matrix': correlation_matrix.tolist(),
            'max_correlation': float(np.max(np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0])))),
            'correlation_strength': self._assess_correlation_strength(correlation_matrix),
            'effective_dimensions': float(np.trace(correlation_matrix))  # Effective dimensionality
        }
        
        evaluation_time = time.time() - start_time
        
        return {
            'method': 'joint_multi_domain',
            'n_samples': n_samples,
            'valid_evaluations': valid_evaluations,
            'success_rate': valid_evaluations / n_samples,
            'statistics': statistics,
            'confidence_intervals': confidence_intervals,
            'correlation_analysis': correlation_analysis,
            'correlated_samples': {k: v.tolist() for k, v in correlated_samples.items()},
            'outputs': valid_outputs.tolist(),
            'evaluation_time_s': evaluation_time,
            'numerical_stability': {
                'finite_fraction': valid_evaluations / n_samples,
                'has_issues': valid_evaluations < 0.9 * n_samples
            }
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate sample skewness."""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((data - mean) / std) ** 3)
        # Bias correction
        skew_corrected = skew * np.sqrt(n * (n - 1)) / (n - 2)
        
        return skew_corrected
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate sample kurtosis (excess kurtosis)."""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        kurt = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        
        # Bias correction
        kurt_corrected = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * kurt + 6)
        
        return kurt_corrected
    
    def _assess_correlation_strength(self, correlation_matrix: np.ndarray) -> str:
        """Assess overall correlation strength in the system."""
        off_diagonal = correlation_matrix - np.eye(correlation_matrix.shape[0])
        max_corr = np.max(np.abs(off_diagonal))
        
        if max_corr < 0.1:
            return 'negligible'
        elif max_corr < 0.3:
            return 'weak'
        elif max_corr < 0.7:
            return 'moderate'
        else:
            return 'strong'

class UncertaintyPropagationSystem:
    """
    Comprehensive uncertainty propagation system with multiple methods.
    
    Features:
    1. Multiple propagation methods (Monte Carlo, LHS, PCE, GP)
    2. Automatic method selection based on problem characteristics
    3. Adaptive sampling strategies
    4. Multi-output uncertainty propagation
    5. Real-time performance monitoring
    """
    
    def __init__(self, uncertain_variables: List[UncertainVariable],
                 params: Optional[UncertaintyParameters] = None):
        """
        Initialize uncertainty propagation system.
        
        Args:
            uncertain_variables: List of uncertain input variables
            params: Propagation parameters
        """
        self.uncertain_variables = uncertain_variables
        self.params = params or UncertaintyParameters()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize propagators
        self.propagators = {
            UncertaintyMethod.MONTE_CARLO: MonteCarloUncertaintyPropagator(
                uncertain_variables, self.params
            ),
            UncertaintyMethod.LATIN_HYPERCUBE: LatinHypercubeUncertaintyPropagator(
                uncertain_variables, self.params
            )
        }
        
        # Active method
        self.active_method = UncertaintyMethod.MONTE_CARLO
        
        # Results history
        self.propagation_results = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        
        self.logger.info(f"Uncertainty propagation system initialized with {len(uncertain_variables)} variables")
    
    def propagate_uncertainty(self, model_function: Callable,
                            method: Optional[UncertaintyMethod] = None) -> Dict:
        """
        Propagate uncertainty through model.
        
        Args:
            model_function: Model function f(x) to evaluate
            method: Propagation method (if None, use active method)
            
        Returns:
            Uncertainty propagation results
        """
        if method is None:
            method = self.active_method
        
        if method not in self.propagators:
            raise ValueError(f"Method {method} not available")
        
        # Select propagator
        propagator = self.propagators[method]
        
        # Propagate uncertainty
        start_time = time.time()
        results = propagator.propagate(model_function)
        total_time = time.time() - start_time
        
        # Add system-level information
        results.update({
            'system_info': {
                'method_used': method.value,
                'n_variables': len(self.uncertain_variables),
                'variable_names': [var.name for var in self.uncertain_variables],
                'total_evaluation_time_s': total_time
            }
        })
        
        # Store results
        self.propagation_results.append(results)
        
        # Update performance tracking
        self._update_performance_tracking(propagator, results)
        
        return results
    
    def sensitivity_analysis(self, model_function: Callable) -> Dict:
        """
        Perform global sensitivity analysis using simplified Sobol indices.
        
        Note: This is a simplified implementation. For production use,
        install SALib: pip install SALib
        
        Args:
            model_function: Model function to analyze
            
        Returns:
            Sensitivity analysis results
        """
        try:
            # Try to import SALib if available
            from SALib.sample import saltelli
            from SALib.analyze import sobol
            use_salib = True
        except ImportError:
            self.logger.warning("SALib not available. Using simplified sensitivity analysis.")
            use_salib = False
        
        if use_salib:
            return self._salib_sensitivity_analysis(model_function)
        else:
            return self._simplified_sensitivity_analysis(model_function)
    
    def _salib_sensitivity_analysis(self, model_function: Callable) -> Dict:
        """Sensitivity analysis using SALib."""
        from SALib.sample import saltelli
        from SALib.analyze import sobol
        
        # Define problem for SALib
        problem = {
            'num_vars': len(self.uncertain_variables),
            'names': [var.name for var in self.uncertain_variables],
            'bounds': []
        }
        
        # Convert distributions to bounds
        for var in self.uncertain_variables:
            if var.distribution == DistributionType.NORMAL:
                # Use 3-sigma bounds for normal distribution
                mean, std = var.parameters['mean'], var.parameters['std']
                bounds = [mean - 3*std, mean + 3*std]
            elif var.distribution == DistributionType.UNIFORM:
                bounds = [var.parameters['low'], var.parameters['high']]
            else:
                # Fallback: use percentile-based bounds
                samples = var.sample(10000)
                bounds = [np.percentile(samples, 1), np.percentile(samples, 99)]
            
            problem['bounds'].append(bounds)
        
        # Generate samples
        n_samples = self.params.sobol_n_samples
        samples = saltelli.sample(problem, n_samples)
        
        # Evaluate model
        outputs = []
        for sample in samples:
            try:
                output = model_function(sample)
                if np.isscalar(output):
                    outputs.append(output)
                else:
                    outputs.append(np.asarray(output).item())
            except:
                outputs.append(np.nan)
        
        outputs = np.array(outputs)
        
        # Remove NaN values
        valid_mask = ~np.isnan(outputs)
        if not np.any(valid_mask):
            return {'error': 'no_valid_outputs'}
        
        samples_clean = samples[valid_mask]
        outputs_clean = outputs[valid_mask]
        
        # Perform Sobol analysis
        try:
            sobol_indices = sobol.analyze(problem, outputs_clean)
            
            sensitivity_results = {
                'method': 'sobol',
                'first_order': dict(zip(problem['names'], sobol_indices['S1'])),
                'total_order': dict(zip(problem['names'], sobol_indices['ST'])),
                'second_order': sobol_indices['S2'],
                'first_order_conf': dict(zip(problem['names'], sobol_indices['S1_conf'])),
                'total_order_conf': dict(zip(problem['names'], sobol_indices['ST_conf'])),
                'n_samples': len(outputs_clean),
                'variance_explained': np.sum(sobol_indices['S1'])
            }
            
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"Sobol analysis failed: {e}")
            return {'error': f'sobol_analysis_failed: {e}'}
    
    def _simplified_sensitivity_analysis(self, model_function: Callable) -> Dict:
        """
        CRITICAL: Simplified sensitivity analysis for cases where SALib is not available.
        
        Uses variance-based sensitivity analysis with elementary effects.
        """
        n_vars = len(self.uncertain_variables)
        n_samples = min(self.params.sobol_n_samples, 10000)  # Limit for simplified method
        
        # Generate base samples
        base_samples = self._generate_samples(n_samples)
        
        # Evaluate base model outputs
        base_outputs = []
        for sample in base_samples:
            try:
                output = model_function(sample)
                if np.isscalar(output):
                    base_outputs.append(output)
                else:
                    base_outputs.append(np.asarray(output).item())
            except:
                base_outputs.append(np.nan)
        
        base_outputs = np.array(base_outputs)
        valid_mask = ~np.isnan(base_outputs)
        
        if not np.any(valid_mask):
            return {'error': 'no_valid_base_outputs'}
        
        base_outputs_clean = base_outputs[valid_mask]
        base_samples_clean = base_samples[valid_mask]
        base_variance = np.var(base_outputs_clean, ddof=1)
        
        if base_variance == 0:
            return {'error': 'zero_variance'}
        
        # Elementary effects sensitivity analysis
        first_order_indices = {}
        total_order_indices = {}
        
        for i, var in enumerate(self.uncertain_variables):
            # Perturb variable i while keeping others fixed
            perturbed_samples = base_samples_clean.copy()
            
            # Resample variable i
            var_samples = var.sample(len(base_samples_clean))
            perturbed_samples[:, i] = var_samples
            
            # Evaluate perturbed outputs
            perturbed_outputs = []
            for sample in perturbed_samples:
                try:
                    output = model_function(sample)
                    if np.isscalar(output):
                        perturbed_outputs.append(output)
                    else:
                        perturbed_outputs.append(np.asarray(output).item())
                except:
                    perturbed_outputs.append(np.nan)
            
            perturbed_outputs = np.array(perturbed_outputs)
            perturbed_valid = ~np.isnan(perturbed_outputs)
            
            if np.any(perturbed_valid):
                # First-order sensitivity (main effect)
                diff = perturbed_outputs[perturbed_valid] - base_outputs_clean[perturbed_valid]
                first_order_var = np.var(diff, ddof=1) / 2  # Variance of differences / 2
                first_order_indices[var.name] = first_order_var / base_variance
                
                # Total sensitivity (approximate)
                total_var = np.var(perturbed_outputs[perturbed_valid], ddof=1)
                total_order_indices[var.name] = 1 - (base_variance - first_order_var) / base_variance
            else:
                first_order_indices[var.name] = 0.0
                total_order_indices[var.name] = 0.0
        
        # Normalize indices to ensure they sum to <= 1
        total_first_order = sum(first_order_indices.values())
        if total_first_order > 1:
            for var_name in first_order_indices:
                first_order_indices[var_name] /= total_first_order
        
        return {
            'method': 'simplified_sobol',
            'first_order': first_order_indices,
            'total_order': total_order_indices,
            'n_samples': len(base_outputs_clean),
            'variance_explained': sum(first_order_indices.values()),
            'note': 'Simplified implementation. Install SALib for full Sobol analysis.'
        }
    
    def compare_methods(self, model_function: Callable, 
                       methods: Optional[List[UncertaintyMethod]] = None) -> Dict:
        """
        Compare different uncertainty propagation methods.
        
        Args:
            model_function: Model function to evaluate
            methods: Methods to compare (if None, compare all available)
            
        Returns:
            Method comparison results
        """
        if methods is None:
            methods = list(self.propagators.keys())
        
        comparison_results = {}
        
        for method in methods:
            if method not in self.propagators:
                continue
            
            self.logger.info(f"Running comparison for method: {method.value}")
            
            try:
                # Run propagation
                results = self.propagate_uncertainty(model_function, method)
                
                # Extract key metrics
                if 'statistics' in results:
                    stats = results['statistics']
                    comparison_results[method.value] = {
                        'mean': stats.get('mean', np.nan),
                        'std': stats.get('std', np.nan),
                        'evaluation_time_s': results.get('evaluation_time_s', np.nan),
                        'n_samples': results.get('n_samples', 0),
                        'efficiency': results.get('samples_per_second', 0),
                        'convergence': results.get('convergence', {})
                    }
                else:
                    comparison_results[method.value] = {'error': 'no_statistics'}
                    
            except Exception as e:
                self.logger.error(f"Method {method.value} failed: {e}")
                comparison_results[method.value] = {'error': str(e)}
        
        # Analyze comparison
        analysis = self._analyze_method_comparison(comparison_results)
        
        return {
            'method_results': comparison_results,
            'analysis': analysis,
            'recommendation': self._recommend_method(analysis)
        }
    
    def _update_performance_tracking(self, propagator: UncertaintyPropagator, 
                                   results: Dict):
        """Update performance tracking."""
        performance_metrics = propagator.get_performance_metrics()
        performance_metrics.update({
            'timestamp': time.time(),
            'method': results.get('method', 'unknown'),
            'n_samples': results.get('n_samples', 0)
        })
        
        self.performance_history.append(performance_metrics)
    
    def _analyze_method_comparison(self, comparison_results: Dict) -> Dict:
        """Analyze method comparison results."""
        analysis = {
            'fastest_method': None,
            'most_accurate_method': None,  # Based on sample size and convergence
            'most_efficient_method': None,  # Based on samples per second
            'agreement_analysis': {}
        }
        
        # Find fastest method
        min_time = float('inf')
        for method, results in comparison_results.items():
            if 'error' not in results:
                eval_time = results.get('evaluation_time_s', float('inf'))
                if eval_time < min_time:
                    min_time = eval_time
                    analysis['fastest_method'] = method
        
        # Find most efficient method
        max_efficiency = 0
        for method, results in comparison_results.items():
            if 'error' not in results:
                efficiency = results.get('efficiency', 0)
                if efficiency > max_efficiency:
                    max_efficiency = efficiency
                    analysis['most_efficient_method'] = method
        
        # Agreement analysis (compare means and stds)
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        
        if len(valid_results) > 1:
            means = [results['mean'] for results in valid_results.values()]
            stds = [results['std'] for results in valid_results.values()]
            
            analysis['agreement_analysis'] = {
                'mean_agreement': {
                    'coefficient_of_variation': np.std(means) / np.abs(np.mean(means)) if np.mean(means) != 0 else np.std(means),
                    'range': np.max(means) - np.min(means),
                    'methods': list(valid_results.keys())
                },
                'std_agreement': {
                    'coefficient_of_variation': np.std(stds) / np.mean(stds) if np.mean(stds) != 0 else np.std(stds),
                    'range': np.max(stds) - np.min(stds)
                }
            }
        
        return analysis
    
    def _recommend_method(self, analysis: Dict) -> Dict:
        """Recommend best method based on analysis."""
        recommendation = {
            'primary_recommendation': None,
            'reasoning': '',
            'alternatives': []
        }
        
        # Simple recommendation logic
        if analysis['fastest_method'] == analysis['most_efficient_method']:
            recommendation['primary_recommendation'] = analysis['fastest_method']
            recommendation['reasoning'] = 'Best balance of speed and efficiency'
        elif analysis['most_efficient_method']:
            recommendation['primary_recommendation'] = analysis['most_efficient_method']
            recommendation['reasoning'] = 'Most samples per second'
        elif analysis['fastest_method']:
            recommendation['primary_recommendation'] = analysis['fastest_method']
            recommendation['reasoning'] = 'Fastest execution time'
        else:
            recommendation['primary_recommendation'] = 'monte_carlo'
            recommendation['reasoning'] = 'Default robust method'
        
        # Add alternatives
        methods = set()
        if analysis['fastest_method']:
            methods.add(analysis['fastest_method'])
        if analysis['most_efficient_method']:
            methods.add(analysis['most_efficient_method'])
        
        recommendation['alternatives'] = list(methods - {recommendation['primary_recommendation']})
        
        return recommendation
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary."""
        # Recent performance
        if self.performance_history:
            recent_performance = list(self.performance_history)[-20:]
            avg_efficiency = np.mean([p.get('efficiency_satisfied', False) for p in recent_performance])
            avg_accuracy = np.mean([p.get('accuracy_satisfied', False) for p in recent_performance])
        else:
            avg_efficiency = 0.0
            avg_accuracy = 0.0
        
        # Variable summary
        variable_summary = []
        for var in self.uncertain_variables:
            var_info = {
                'name': var.name,
                'distribution': var.distribution.value,
                'parameters': var.parameters,
                'theoretical_mean': var.mean(),
                'theoretical_variance': var.variance()
            }
            variable_summary.append(var_info)
        
        summary = {
            'system_status': {
                'n_variables': len(self.uncertain_variables),
                'active_method': self.active_method.value,
                'available_methods': [method.value for method in self.propagators.keys()],
                'n_propagation_results': len(self.propagation_results)
            },
            'performance_summary': {
                'avg_efficiency_satisfied': avg_efficiency,
                'avg_accuracy_satisfied': avg_accuracy,
                'efficiency_target': UQ_EFFICIENCY_TARGET,
                'accuracy_target': UQ_ACCURACY_TARGET
            },
            'variable_summary': variable_summary,
            'recent_results_count': len(self.propagation_results)
        }
        
        return summary


if __name__ == "__main__":
    """Example usage of uncertainty propagation system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== UNCERTAINTY PROPAGATION SYSTEM ===")
    print("Advanced uncertainty quantification with multiple methods")
    
    # Define uncertain variables
    uncertain_variables = [
        UncertainVariable(
            name="casimir_force_coeff",
            distribution=DistributionType.NORMAL,
            parameters={'mean': 1.0, 'std': 0.05}  # 5% uncertainty
        ),
        UncertainVariable(
            name="gap_distance",
            distribution=DistributionType.NORMAL,
            parameters={'mean': 100e-9, 'std': 5e-9}  # 100 nm ± 5 nm
        ),
        UncertainVariable(
            name="temperature",
            distribution=DistributionType.UNIFORM,
            parameters={'low': 295.0, 'high': 305.0}  # 295-305 K
        ),
        UncertainVariable(
            name="material_property",
            distribution=DistributionType.BETA,
            parameters={'alpha': 2.0, 'beta': 5.0}  # Skewed distribution
        )
    ]
    
    print(f"\nDefined {len(uncertain_variables)} uncertain variables:")
    for var in uncertain_variables:
        print(f"  {var.name}: {var.distribution.value} with μ={var.mean():.3e}, σ²={var.variance():.3e}")
    
    # Initialize propagation system
    params = UncertaintyParameters(
        n_samples=5000,
        confidence_level=0.95,
        use_parallel=True
    )
    
    uq_system = UncertaintyPropagationSystem(uncertain_variables, params)
    print(f"\nInitialized UQ system with {len(uq_system.propagators)} methods")
    
    # Define model function (simplified Casimir force model)
    def casimir_model(inputs):
        """
        Simplified Casimir force model.
        F = -coeff * (ħc π²/240) * (A/d⁴) * material_factor * temp_factor
        """
        coeff, gap, temp, material = inputs
        
        # Physical constants (simplified)
        hbar_c = 1.97e-25  # ħc in J⋅m
        A = 1e-6  # Area in m²
        
        # Temperature factor (simplified)
        temp_factor = 1.0 + 0.01 * (temp - 300)  # Linear approximation
        
        # Casimir force magnitude
        force = coeff * (hbar_c * np.pi**2 / 240) * (A / gap**4) * material * temp_factor
        
        return force
    
    print(f"\nDefined Casimir force model with 4 uncertain inputs")
    
    # Test individual methods
    print(f"\n=== MONTE CARLO PROPAGATION ===")
    mc_results = uq_system.propagate_uncertainty(casimir_model, UncertaintyMethod.MONTE_CARLO)
    
    if 'statistics' in mc_results:
        stats = mc_results['statistics']
        print(f"Statistics:")
        print(f"  Mean: {stats['mean']:.3e} N")
        print(f"  Std:  {stats['std']:.3e} N")
        print(f"  CV:   {stats['std']/stats['mean']*100:.1f}%")
        print(f"  95% CI: [{mc_results['confidence_intervals']['percentile_lower']:.3e}, "
              f"{mc_results['confidence_intervals']['percentile_upper']:.3e}] N")
    
    print(f"Evaluation time: {mc_results['evaluation_time_s']:.3f} s")
    print(f"Samples per second: {mc_results.get('samples_per_second', 0):.1f}")
    
    if 'convergence' in mc_results and mc_results['convergence'].get('is_converged', False):
        print(f"✓ Monte Carlo converged")
    else:
        print(f"⚠ Monte Carlo may need more samples")
    
    print(f"\n=== LATIN HYPERCUBE PROPAGATION ===")
    lhs_results = uq_system.propagate_uncertainty(casimir_model, UncertaintyMethod.LATIN_HYPERCUBE)
    
    if 'statistics' in lhs_results:
        stats = lhs_results['statistics']
        print(f"Statistics:")
        print(f"  Mean: {stats['mean']:.3e} N")
        print(f"  Std:  {stats['std']:.3e} N")
        print(f"  CV:   {stats['std']/stats['mean']*100:.1f}%")
    
    print(f"Evaluation time: {lhs_results['evaluation_time_s']:.3f} s")
    print(f"Space-filling quality: {lhs_results.get('space_filling_quality', 0):.3f}")
    
    # Method comparison
    print(f"\n=== METHOD COMPARISON ===")
    comparison = uq_system.compare_methods(casimir_model)
    
    if 'analysis' in comparison:
        analysis = comparison['analysis']
        print(f"Fastest method: {analysis.get('fastest_method', 'N/A')}")
        print(f"Most efficient: {analysis.get('most_efficient_method', 'N/A')}")
        
        if 'agreement_analysis' in analysis:
            mean_agreement = analysis['agreement_analysis'].get('mean_agreement', {})
            cv = mean_agreement.get('coefficient_of_variation', 0)
            print(f"Mean agreement (CV): {cv:.3f}")
    
    if 'recommendation' in comparison:
        rec = comparison['recommendation']
        print(f"Recommended method: {rec.get('primary_recommendation', 'N/A')}")
        print(f"Reasoning: {rec.get('reasoning', 'N/A')}")
    
    # Sensitivity analysis
    print(f"\n=== SENSITIVITY ANALYSIS ===")
    try:
        sensitivity = uq_system.sensitivity_analysis(casimir_model)
        
        if 'first_order' in sensitivity:
            print(f"First-order Sobol indices:")
            for var_name, index in sensitivity['first_order'].items():
                print(f"  {var_name}: {index:.3f}")
            
            print(f"Total variance explained: {sensitivity.get('variance_explained', 0):.3f}")
            
            # Find most influential variable
            first_order = sensitivity['first_order']
            most_influential = max(first_order.items(), key=lambda x: x[1])
            print(f"Most influential variable: {most_influential[0]} (S₁ = {most_influential[1]:.3f})")
        else:
            print(f"Sensitivity analysis failed: {sensitivity.get('error', 'unknown')}")
    except ImportError:
        print("SALib not available for sensitivity analysis")
    except Exception as e:
        print(f"Sensitivity analysis error: {e}")
    
    # System summary
    print(f"\n=== SYSTEM SUMMARY ===")
    summary = uq_system.get_system_summary()
    
    system_status = summary.get('system_status', {})
    print(f"System status:")
    print(f"  Variables: {system_status.get('n_variables', 0)}")
    print(f"  Active method: {system_status.get('active_method', 'N/A')}")
    print(f"  Results generated: {system_status.get('n_propagation_results', 0)}")
    
    performance = summary.get('performance_summary', {})
    print(f"Performance:")
    print(f"  Efficiency target met: {performance.get('avg_efficiency_satisfied', 0)*100:.1f}%")
    print(f"  Accuracy target met: {performance.get('avg_accuracy_satisfied', 0)*100:.1f}%")
    
    print(f"\nUncertainty propagation demonstration complete!")
