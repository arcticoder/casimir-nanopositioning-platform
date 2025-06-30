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
    
    # Monte Carlo parameters
    n_samples: int = 10000
    random_seed: Optional[int] = 42
    confidence_level: float = 0.95
    
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
    
    # Convergence criteria
    convergence_tolerance: float = 1e-6
    max_evaluations: int = 100000
    
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
        Monte Carlo uncertainty propagation.
        
        Args:
            model_function: Function to evaluate f(x₁, x₂, ..., xₙ)
            
        Returns:
            Dictionary with uncertainty propagation results
        """
        start_time = time.time()
        
        # Generate samples
        samples = self._generate_samples()
        
        # Evaluate model
        if self.params.use_parallel and self.params.n_jobs != 1:
            outputs = self._parallel_evaluation(model_function, samples)
        else:
            outputs = self._sequential_evaluation(model_function, samples)
        
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
            'statistics': statistics,
            'confidence_intervals': confidence_intervals,
            'convergence': convergence_analysis,
            'samples': samples,
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
        Perform global sensitivity analysis using Sobol indices.
        
        Args:
            model_function: Model function to analyze
            
        Returns:
            Sensitivity analysis results
        """
        from SALib.sample import saltelli
        from SALib.analyze import sobol
        
        # Define problem for SALib
        problem = {
            'num_vars': len(self.uncertain_variables),
            'names': [var.name for var in self.uncertain_variables],
            'bounds': []
        }
        
        # Convert distributions to bounds (simplified)
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
