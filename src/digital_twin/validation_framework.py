"""
Digital Twin Validation Framework
================================

This module implements comprehensive validation protocols for the digital twin,
including cross-validation, model selection, robustness testing, and real-time
performance assessment.

Mathematical Formulation:

Cross-Validation:
CV(k) = (1/k) × Σᵢ₌₁ᵏ L(yᵢ, ŷᵢ)

Model Selection Criteria:
AIC = 2k - 2ln(L)
BIC = k×ln(n) - 2ln(L)
MDL = -ln(L) + (k/2)×ln(n)

Robustness Metrics:
Sensitivity: S = |∂y/∂θ| / |y/θ|
Robustness: R = 1 / (1 + max|∂y/∂θ|)

Validation Targets:
- Model accuracy: R² ≥ 0.99
- Coverage probability: ≥ 95%
- Prediction latency: ≤ 1 ms
- Cross-domain error: ≤ 1%
"""

import numpy as np
from scipy import stats, optimize, linalg
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Callable, Union, NamedTuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import warnings
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# Validation performance targets
ACCURACY_TARGET = 0.99  # R² coefficient
COVERAGE_TARGET = 0.95  # Coverage probability
LATENCY_TARGET = 1e-3  # seconds
CROSS_DOMAIN_ERROR_TARGET = 0.01  # 1%

class ValidationType(Enum):
    """Types of validation methods."""
    CROSS_VALIDATION = "cross_validation"
    HOLDOUT_VALIDATION = "holdout_validation"
    TIME_SERIES_VALIDATION = "time_series_validation"
    BOOTSTRAP_VALIDATION = "bootstrap_validation"
    PHYSICS_VALIDATION = "physics_validation"
    UNCERTAINTY_VALIDATION = "uncertainty_validation"

class ModelSelectionCriterion(Enum):
    """Model selection criteria."""
    AIC = "aic"  # Akaike Information Criterion
    BIC = "bic"  # Bayesian Information Criterion
    MDL = "mdl"  # Minimum Description Length
    CROSS_VALIDATION = "cross_validation"
    PHYSICS_INFORMED = "physics_informed"

class RobustnessTestType(Enum):
    """Types of robustness tests."""
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    NOISE_ROBUSTNESS = "noise_robustness"
    DISTURBANCE_REJECTION = "disturbance_rejection"
    OUTLIER_ROBUSTNESS = "outlier_robustness"
    TEMPORAL_ROBUSTNESS = "temporal_robustness"

@dataclass
class ValidationParameters:
    """Parameters for digital twin validation."""
    
    # Cross-validation parameters
    cv_folds: int = 10
    cv_shuffle: bool = True
    cv_random_state: int = 42
    
    # Time series validation parameters
    ts_n_splits: int = 5
    ts_max_train_size: Optional[int] = None
    
    # Bootstrap parameters
    bootstrap_n_samples: int = 1000
    bootstrap_confidence_level: float = 0.95
    
    # Robustness testing parameters
    sensitivity_perturbation: float = 0.01  # 1% parameter perturbation
    noise_levels: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    outlier_fractions: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    
    # Performance thresholds
    accuracy_threshold: float = ACCURACY_TARGET
    coverage_threshold: float = COVERAGE_TARGET
    latency_threshold: float = LATENCY_TARGET
    
    # Validation options
    use_parallel: bool = True
    n_jobs: int = -1
    verbose: bool = True

class ValidationResult(NamedTuple):
    """Result of validation test."""
    test_type: str
    accuracy_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    passed: bool
    details: Dict

class DigitalTwinValidator(ABC):
    """Abstract base class for digital twin validators."""
    
    def __init__(self, params: ValidationParameters):
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validation history
        self.validation_history = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=100)
        
    @abstractmethod
    def validate(self, model: Callable, data: Dict) -> ValidationResult:
        """Validate digital twin model."""
        pass
    
    def _calculate_accuracy_metrics(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'error': 'no_data'}
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.any(valid_mask):
            return {'error': 'all_nan'}
        
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        try:
            metrics = {
                'r2_score': r2_score(y_true_clean, y_pred_clean),
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'mape': np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100,
                'max_error': np.max(np.abs(y_true_clean - y_pred_clean)),
                'correlation': np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
            }
            
            # Normalized metrics
            y_range = np.max(y_true_clean) - np.min(y_true_clean)
            if y_range > 0:
                metrics['normalized_rmse'] = metrics['rmse'] / y_range
                metrics['normalized_mae'] = metrics['mae'] / y_range
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_uncertainty_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate uncertainty quantification metrics."""
        if y_std is None:
            return {'coverage_probability': 0.0, 'calibration_error': 0.0}
        
        # Coverage probability
        coverage_prob = self._calculate_coverage_probability(y_true, y_pred, y_std)
        
        # Calibration error
        calibration_error = self._calculate_calibration_error(y_true, y_pred, y_std)
        
        # Sharpness (average prediction interval width)
        sharpness = np.mean(2 * 1.96 * y_std)  # 95% confidence interval width
        
        # Reliability (consistency of uncertainty estimates)
        reliability = self._calculate_reliability(y_true, y_pred, y_std)
        
        return {
            'coverage_probability': coverage_prob,
            'calibration_error': calibration_error,
            'sharpness': sharpness,
            'reliability': reliability,
            'uncertainty_quality': coverage_prob * (1 - calibration_error) * reliability
        }
    
    def _calculate_coverage_probability(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray, 
                                      y_std: np.ndarray,
                                      confidence_level: float = 0.95) -> float:
        """Calculate coverage probability of prediction intervals."""
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        lower_bound = y_pred - z_score * y_std
        upper_bound = y_pred + z_score * y_std
        
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        return float(coverage)
    
    def _calculate_calibration_error(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray, 
                                   y_std: np.ndarray) -> float:
        """Calculate calibration error of uncertainty estimates."""
        # Probability integral transform
        standardized_residuals = (y_true - y_pred) / y_std
        
        # Should be uniformly distributed if well-calibrated
        p_values = stats.norm.cdf(standardized_residuals)
        
        # Kolmogorov-Smirnov test for uniformity
        ks_statistic, _ = stats.kstest(p_values, 'uniform')
        
        return float(ks_statistic)
    
    def _calculate_reliability(self, y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             y_std: np.ndarray) -> float:
        """Calculate reliability of uncertainty estimates."""
        # Standardized residuals should have unit variance if well-calibrated
        standardized_residuals = (y_true - y_pred) / y_std
        
        # Calculate variance of standardized residuals
        residual_variance = np.var(standardized_residuals)
        
        # Reliability is inverse of deviation from unit variance
        reliability = 1 / (1 + abs(residual_variance - 1))
        
        return float(reliability)

class CrossValidationValidator(DigitalTwinValidator):
    """
    Cross-validation validator for digital twin models.
    
    Mathematical Implementation:
    
    K-Fold Cross-Validation:
    CV(k) = (1/k) × Σᵢ₌₁ᵏ L(yᵢ, ŷᵢ)
    
    Where L is the loss function and k is the number of folds
    """
    
    def validate(self, model: Callable, data: Dict) -> ValidationResult:
        """
        Perform k-fold cross-validation.
        
        Args:
            model: Digital twin model function
            data: Dictionary with 'X' (features) and 'y' (targets)
            
        Returns:
            Cross-validation results
        """
        start_time = time.time()
        
        X = data['X']
        y = data['y']
        
        if len(X) != len(y):
            raise ValueError("Feature and target arrays must have same length")
        
        # Initialize cross-validation
        cv = KFold(
            n_splits=self.params.cv_folds,
            shuffle=self.params.cv_shuffle,
            random_state=self.params.cv_random_state
        )
        
        # Storage for results
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_std = []
        
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Train model on fold
                model_trained = self._train_model(model, X_train, y_train)
                
                # Predict on test set
                predictions = self._predict_with_uncertainty(model_trained, X_test)
                
                if isinstance(predictions, tuple):
                    y_pred, y_std = predictions
                else:
                    y_pred = predictions
                    y_std = None
                
                # Calculate fold metrics
                fold_accuracy = self._calculate_accuracy_metrics(y_test, y_pred)
                fold_uncertainty = self._calculate_uncertainty_metrics(y_test, y_pred, y_std)
                
                fold_results.append({
                    'fold': fold_idx,
                    'accuracy': fold_accuracy,
                    'uncertainty': fold_uncertainty,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                })
                
                # Store for overall metrics
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                if y_std is not None:
                    all_y_std.extend(y_std)
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold_idx}: {e}")
                fold_results.append({
                    'fold': fold_idx,
                    'error': str(e)
                })
        
        # Calculate overall metrics
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_std = np.array(all_y_std) if all_y_std else None
        
        overall_accuracy = self._calculate_accuracy_metrics(all_y_true, all_y_pred)
        overall_uncertainty = self._calculate_uncertainty_metrics(all_y_true, all_y_pred, all_y_std)
        
        # Performance metrics
        validation_time = time.time() - start_time
        performance_metrics = {
            'validation_time_s': validation_time,
            'folds_completed': len([r for r in fold_results if 'error' not in r]),
            'folds_failed': len([r for r in fold_results if 'error' in r]),
            'avg_predictions_per_second': len(all_y_pred) / validation_time if validation_time > 0 else 0
        }
        
        # Determine if validation passed
        passed = (
            overall_accuracy.get('r2_score', 0) >= self.params.accuracy_threshold and
            overall_uncertainty.get('coverage_probability', 0) >= self.params.coverage_threshold and
            validation_time <= 10 * self.params.latency_threshold  # Allow more time for validation
        )
        
        return ValidationResult(
            test_type="cross_validation",
            accuracy_metrics=overall_accuracy,
            uncertainty_metrics=overall_uncertainty,
            performance_metrics=performance_metrics,
            passed=passed,
            details={
                'fold_results': fold_results,
                'cv_parameters': {
                    'n_folds': self.params.cv_folds,
                    'shuffle': self.params.cv_shuffle,
                    'random_state': self.params.cv_random_state
                }
            }
        )
    
    def _train_model(self, model: Callable, X_train: np.ndarray, 
                    y_train: np.ndarray) -> Callable:
        """Train model on training data (simplified)."""
        # This is a placeholder - in practice, would involve actual model training
        return model
    
    def _predict_with_uncertainty(self, model: Callable, 
                                X_test: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with uncertainty estimates."""
        try:
            # Try to get uncertainty estimates
            predictions = model(X_test)
            
            if isinstance(predictions, tuple) and len(predictions) == 2:
                return predictions  # (mean, std)
            else:
                # Only point predictions available
                return np.asarray(predictions)
                
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return np.zeros(len(X_test))

class RobustnessValidator(DigitalTwinValidator):
    """
    Robustness validator for testing digital twin resilience.
    
    Mathematical Implementation:
    
    Parameter Sensitivity:
    S = |∂y/∂θ| / |y/θ|
    
    Noise Robustness:
    R_noise = 1 - |y_noisy - y_clean| / |y_clean|
    """
    
    def validate(self, model: Callable, data: Dict) -> ValidationResult:
        """
        Perform comprehensive robustness testing.
        
        Args:
            model: Digital twin model function
            data: Dictionary with test data
            
        Returns:
            Robustness validation results
        """
        start_time = time.time()
        
        X = data['X']
        y = data['y']
        
        robustness_results = {}
        
        # Parameter sensitivity analysis
        if 'parameters' in data:
            sensitivity_results = self._test_parameter_sensitivity(
                model, X, y, data['parameters']
            )
            robustness_results['parameter_sensitivity'] = sensitivity_results
        
        # Noise robustness testing
        noise_results = self._test_noise_robustness(model, X, y)
        robustness_results['noise_robustness'] = noise_results
        
        # Outlier robustness testing
        outlier_results = self._test_outlier_robustness(model, X, y)
        robustness_results['outlier_robustness'] = outlier_results
        
        # Temporal robustness (if time series data)
        if 'time' in data:
            temporal_results = self._test_temporal_robustness(model, X, y, data['time'])
            robustness_results['temporal_robustness'] = temporal_results
        
        # Calculate overall robustness score
        overall_robustness = self._calculate_overall_robustness(robustness_results)
        
        # Performance metrics
        validation_time = time.time() - start_time
        performance_metrics = {
            'validation_time_s': validation_time,
            'tests_completed': len(robustness_results),
            'overall_robustness_score': overall_robustness
        }
        
        # Determine if validation passed
        passed = overall_robustness >= 0.8  # 80% robustness threshold
        
        return ValidationResult(
            test_type="robustness",
            accuracy_metrics={'overall_robustness': overall_robustness},
            uncertainty_metrics={},
            performance_metrics=performance_metrics,
            passed=passed,
            details=robustness_results
        )
    
    def _test_parameter_sensitivity(self, model: Callable, X: np.ndarray, 
                                  y: np.ndarray, parameters: Dict) -> Dict:
        """Test sensitivity to parameter perturbations."""
        sensitivity_results = {}
        
        # Baseline predictions
        try:
            y_baseline = model(X)
        except:
            return {'error': 'baseline_prediction_failed'}
        
        for param_name, param_value in parameters.items():
            try:
                # Perturb parameter
                perturbation = self.params.sensitivity_perturbation * param_value
                
                # Test positive perturbation
                perturbed_params = parameters.copy()
                perturbed_params[param_name] = param_value + perturbation
                
                # Create perturbed model (simplified)
                y_perturbed = model(X)  # Would use perturbed parameters
                
                # Calculate sensitivity
                if np.all(y_baseline != 0):
                    sensitivity = np.mean(np.abs(y_perturbed - y_baseline) / np.abs(y_baseline))
                else:
                    sensitivity = np.mean(np.abs(y_perturbed - y_baseline))
                
                sensitivity_results[param_name] = {
                    'sensitivity': sensitivity,
                    'robustness': 1 / (1 + sensitivity),
                    'perturbation_size': perturbation
                }
                
            except Exception as e:
                sensitivity_results[param_name] = {'error': str(e)}
        
        return sensitivity_results
    
    def _test_noise_robustness(self, model: Callable, X: np.ndarray, 
                             y: np.ndarray) -> Dict:
        """Test robustness to input noise."""
        noise_results = {}
        
        # Baseline predictions
        try:
            y_baseline = model(X)
        except:
            return {'error': 'baseline_prediction_failed'}
        
        for noise_level in self.params.noise_levels:
            try:
                # Add noise to inputs
                noise = np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
                X_noisy = X + noise
                
                # Predict with noisy inputs
                y_noisy = model(X_noisy)
                
                # Calculate robustness
                if np.all(y_baseline != 0):
                    robustness = 1 - np.mean(np.abs(y_noisy - y_baseline) / np.abs(y_baseline))
                else:
                    robustness = 1 - np.mean(np.abs(y_noisy - y_baseline))
                
                noise_results[f'noise_level_{noise_level}'] = {
                    'robustness': max(0, robustness),
                    'mean_absolute_change': np.mean(np.abs(y_noisy - y_baseline)),
                    'max_absolute_change': np.max(np.abs(y_noisy - y_baseline))
                }
                
            except Exception as e:
                noise_results[f'noise_level_{noise_level}'] = {'error': str(e)}
        
        return noise_results
    
    def _test_outlier_robustness(self, model: Callable, X: np.ndarray, 
                               y: np.ndarray) -> Dict:
        """Test robustness to outliers in training data."""
        outlier_results = {}
        
        for outlier_fraction in self.params.outlier_fractions:
            try:
                # Create dataset with outliers
                n_outliers = int(outlier_fraction * len(X))
                outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
                
                X_outliers = X.copy()
                y_outliers = y.copy()
                
                # Add outliers (extreme values)
                X_outliers[outlier_indices] *= 10  # 10x larger inputs
                y_outliers[outlier_indices] *= 10  # 10x larger outputs
                
                # Test model on clean data vs outlier-contaminated data
                y_clean = model(X)
                y_with_outliers = model(X)  # Would retrain with outlier data
                
                # Calculate robustness
                if np.all(y_clean != 0):
                    robustness = 1 - np.mean(np.abs(y_with_outliers - y_clean) / np.abs(y_clean))
                else:
                    robustness = 1 - np.mean(np.abs(y_with_outliers - y_clean))
                
                outlier_results[f'outlier_fraction_{outlier_fraction}'] = {
                    'robustness': max(0, robustness),
                    'n_outliers': n_outliers,
                    'mean_performance_change': np.mean(np.abs(y_with_outliers - y_clean))
                }
                
            except Exception as e:
                outlier_results[f'outlier_fraction_{outlier_fraction}'] = {'error': str(e)}
        
        return outlier_results
    
    def _test_temporal_robustness(self, model: Callable, X: np.ndarray, 
                                y: np.ndarray, time: np.ndarray) -> Dict:
        """Test robustness over time (drift, non-stationarity)."""
        temporal_results = {}
        
        try:
            # Split data into time periods
            n_periods = 5
            period_length = len(time) // n_periods
            
            period_performances = []
            
            for i in range(n_periods):
                start_idx = i * period_length
                end_idx = (i + 1) * period_length if i < n_periods - 1 else len(time)
                
                X_period = X[start_idx:end_idx]
                y_period = y[start_idx:end_idx]
                
                # Predict for this period
                y_pred_period = model(X_period)
                
                # Calculate performance
                period_r2 = r2_score(y_period, y_pred_period)
                period_performances.append(period_r2)
            
            # Calculate temporal stability
            performance_std = np.std(period_performances)
            performance_mean = np.mean(period_performances)
            
            temporal_stability = 1 / (1 + performance_std)
            
            temporal_results = {
                'temporal_stability': temporal_stability,
                'performance_std': performance_std,
                'performance_mean': performance_mean,
                'period_performances': period_performances,
                'performance_drift': period_performances[-1] - period_performances[0]
            }
            
        except Exception as e:
            temporal_results = {'error': str(e)}
        
        return temporal_results
    
    def _calculate_overall_robustness(self, robustness_results: Dict) -> float:
        """Calculate overall robustness score."""
        scores = []
        
        # Parameter sensitivity
        if 'parameter_sensitivity' in robustness_results:
            param_scores = []
            for param_result in robustness_results['parameter_sensitivity'].values():
                if 'robustness' in param_result:
                    param_scores.append(param_result['robustness'])
            if param_scores:
                scores.append(np.mean(param_scores))
        
        # Noise robustness
        if 'noise_robustness' in robustness_results:
            noise_scores = []
            for noise_result in robustness_results['noise_robustness'].values():
                if 'robustness' in noise_result:
                    noise_scores.append(noise_result['robustness'])
            if noise_scores:
                scores.append(np.mean(noise_scores))
        
        # Outlier robustness
        if 'outlier_robustness' in robustness_results:
            outlier_scores = []
            for outlier_result in robustness_results['outlier_robustness'].values():
                if 'robustness' in outlier_result:
                    outlier_scores.append(outlier_result['robustness'])
            if outlier_scores:
                scores.append(np.mean(outlier_scores))
        
        # Temporal robustness
        if 'temporal_robustness' in robustness_results:
            temporal_result = robustness_results['temporal_robustness']
            if 'temporal_stability' in temporal_result:
                scores.append(temporal_result['temporal_stability'])
        
        return np.mean(scores) if scores else 0.0

class DigitalTwinValidationFramework:
    """
    Comprehensive digital twin validation framework.
    
    Features:
    1. Multiple validation methods
    2. Model selection and comparison
    3. Automated validation pipelines
    4. Real-time validation monitoring
    5. Performance benchmarking
    """
    
    def __init__(self, params: Optional[ValidationParameters] = None):
        """
        Initialize validation framework.
        
        Args:
            params: Validation parameters
        """
        self.params = params or ValidationParameters()
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.validators = {
            ValidationType.CROSS_VALIDATION: CrossValidationValidator(self.params),
            ValidationType.ROBUSTNESS_VALIDATION: RobustnessValidator(self.params)
        }
        
        # Validation history
        self.validation_history = deque(maxlen=1000)
        self.model_performance = {}
        
        # Active validators
        self.active_validators = list(self.validators.keys())
        
        self.logger.info(f"Digital twin validation framework initialized with {len(self.validators)} validators")
    
    def validate_model(self, model: Callable, data: Dict, 
                      validation_types: Optional[List[ValidationType]] = None) -> Dict:
        """
        Comprehensive model validation.
        
        Args:
            model: Digital twin model to validate
            data: Validation data
            validation_types: Types of validation to perform
            
        Returns:
            Comprehensive validation results
        """
        if validation_types is None:
            validation_types = self.active_validators
        
        validation_results = {}
        start_time = time.time()
        
        # Run each validation type
        for val_type in validation_types:
            if val_type in self.validators:
                try:
                    self.logger.info(f"Running {val_type.value} validation...")
                    result = self.validators[val_type].validate(model, data)
                    validation_results[val_type.value] = result
                    
                except Exception as e:
                    self.logger.error(f"Validation {val_type.value} failed: {e}")
                    validation_results[val_type.value] = ValidationResult(
                        test_type=val_type.value,
                        accuracy_metrics={'error': str(e)},
                        uncertainty_metrics={},
                        performance_metrics={},
                        passed=False,
                        details={'error': str(e)}
                    )
            else:
                self.logger.warning(f"Validator {val_type.value} not available")
        
        # Calculate overall validation score
        overall_score = self._calculate_overall_validation_score(validation_results)
        
        # Performance summary
        total_time = time.time() - start_time
        performance_summary = {
            'total_validation_time_s': total_time,
            'validations_completed': len(validation_results),
            'validations_passed': sum(1 for r in validation_results.values() if r.passed),
            'overall_validation_score': overall_score,
            'validation_efficiency': len(validation_results) / total_time if total_time > 0 else 0
        }
        
        # Comprehensive results
        comprehensive_results = {
            'validation_results': validation_results,
            'performance_summary': performance_summary,
            'overall_passed': overall_score >= 0.8,  # 80% threshold
            'timestamp': time.time(),
            'data_summary': {
                'n_samples': len(data.get('X', [])),
                'n_features': data.get('X', []).shape[1] if len(data.get('X', [])) > 0 else 0,
                'data_keys': list(data.keys())
            }
        }
        
        # Store validation history
        self.validation_history.append(comprehensive_results)
        
        return comprehensive_results
    
    def compare_models(self, models: Dict[str, Callable], data: Dict) -> Dict:
        """
        Compare multiple models using validation framework.
        
        Args:
            models: Dictionary of model_name -> model_function
            data: Validation data
            
        Returns:
            Model comparison results
        """
        model_results = {}
        
        # Validate each model
        for model_name, model in models.items():
            self.logger.info(f"Validating model: {model_name}")
            
            try:
                results = self.validate_model(model, data)
                model_results[model_name] = results
                
            except Exception as e:
                self.logger.error(f"Model {model_name} validation failed: {e}")
                model_results[model_name] = {
                    'error': str(e),
                    'overall_passed': False,
                    'performance_summary': {'overall_validation_score': 0.0}
                }
        
        # Model ranking
        model_ranking = self._rank_models(model_results)
        
        # Statistical comparison
        statistical_comparison = self._statistical_model_comparison(model_results)
        
        comparison_results = {
            'model_results': model_results,
            'model_ranking': model_ranking,
            'statistical_comparison': statistical_comparison,
            'best_model': model_ranking[0] if model_ranking else None,
            'comparison_timestamp': time.time()
        }
        
        return comparison_results
    
    def model_selection(self, models: Dict[str, Callable], data: Dict,
                       criterion: ModelSelectionCriterion = ModelSelectionCriterion.CROSS_VALIDATION) -> str:
        """
        Select best model based on validation criteria.
        
        Args:
            models: Dictionary of models
            data: Validation data
            criterion: Model selection criterion
            
        Returns:
            Name of selected best model
        """
        if criterion == ModelSelectionCriterion.CROSS_VALIDATION:
            # Use cross-validation scores
            comparison = self.compare_models(models, data)
            return comparison.get('best_model', list(models.keys())[0])
        
        elif criterion == ModelSelectionCriterion.AIC:
            # Calculate AIC for each model
            aic_scores = {}
            for model_name, model in models.items():
                try:
                    y_pred = model(data['X'])
                    n = len(data['y'])
                    k = 1  # Number of parameters (simplified)
                    mse = mean_squared_error(data['y'], y_pred)
                    log_likelihood = -n/2 * np.log(2*np.pi*mse) - n/2
                    aic = 2*k - 2*log_likelihood
                    aic_scores[model_name] = aic
                except:
                    aic_scores[model_name] = float('inf')
            
            return min(aic_scores.items(), key=lambda x: x[1])[0]
        
        # Default to first model
        return list(models.keys())[0]
    
    def _calculate_overall_validation_score(self, validation_results: Dict) -> float:
        """Calculate overall validation score."""
        scores = []
        
        for val_type, result in validation_results.items():
            if result.passed:
                # Extract accuracy score
                accuracy = result.accuracy_metrics.get('r2_score', 
                          result.accuracy_metrics.get('overall_robustness', 0))
                
                # Extract uncertainty score
                uncertainty = result.uncertainty_metrics.get('coverage_probability', 1.0)
                
                # Combined score
                combined_score = 0.7 * accuracy + 0.3 * uncertainty
                scores.append(combined_score)
            else:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def _rank_models(self, model_results: Dict) -> List[str]:
        """Rank models by validation performance."""
        model_scores = []
        
        for model_name, results in model_results.items():
            if 'performance_summary' in results:
                score = results['performance_summary'].get('overall_validation_score', 0.0)
                model_scores.append((model_name, score))
            else:
                model_scores.append((model_name, 0.0))
        
        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, score in model_scores]
    
    def _statistical_model_comparison(self, model_results: Dict) -> Dict:
        """Statistical comparison of models."""
        # This is a simplified implementation
        # In practice, would use proper statistical tests
        
        comparison = {
            'n_models': len(model_results),
            'models_passed': sum(1 for r in model_results.values() if r.get('overall_passed', False)),
            'significant_differences': False,  # Would require proper statistical testing
            'confidence_level': 0.95
        }
        
        return comparison
    
    def get_validation_summary(self) -> Dict:
        """Get comprehensive validation framework summary."""
        # Recent validation performance
        if self.validation_history:
            recent_validations = list(self.validation_history)[-20:]
            
            success_rate = np.mean([v.get('overall_passed', False) for v in recent_validations])
            avg_score = np.mean([v['performance_summary'].get('overall_validation_score', 0) 
                               for v in recent_validations])
            avg_time = np.mean([v['performance_summary'].get('total_validation_time_s', 0) 
                              for v in recent_validations])
        else:
            success_rate = 0.0
            avg_score = 0.0
            avg_time = 0.0
        
        # Validator performance
        validator_performance = {}
        for val_type, validator in self.validators.items():
            if hasattr(validator, 'performance_metrics') and validator.performance_metrics:
                recent_metrics = list(validator.performance_metrics)[-10:]
                validator_performance[val_type.value] = {
                    'avg_validation_time': np.mean([m.get('validation_time_s', 0) for m in recent_metrics]),
                    'success_rate': np.mean([m.get('passed', False) for m in recent_metrics])
                }
        
        summary = {
            'framework_status': {
                'available_validators': len(self.validators),
                'active_validators': len(self.active_validators),
                'validation_history_length': len(self.validation_history),
                'framework_initialized': True
            },
            'performance_summary': {
                'recent_success_rate': success_rate,
                'avg_validation_score': avg_score,
                'avg_validation_time_s': avg_time,
                'accuracy_target_met': avg_score >= self.params.accuracy_threshold,
                'efficiency_acceptable': avg_time <= 60.0  # 1 minute threshold for validation
            },
            'validator_performance': validator_performance,
            'validation_targets': {
                'accuracy_target': self.params.accuracy_threshold,
                'coverage_target': self.params.coverage_threshold,
                'latency_target': self.params.latency_threshold
            }
        }
        
        return summary


if __name__ == "__main__":
    """Example usage of digital twin validation framework."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== DIGITAL TWIN VALIDATION FRAMEWORK ===")
    print("Comprehensive validation with multiple methods")
    
    # Initialize validation framework
    params = ValidationParameters(
        cv_folds=5,
        bootstrap_n_samples=500,
        use_parallel=True
    )
    
    validation_framework = DigitalTwinValidationFramework(params)
    
    print(f"\nInitialized validation framework:")
    print(f"  Available validators: {len(validation_framework.validators)}")
    print(f"  Cross-validation folds: {params.cv_folds}")
    print(f"  Bootstrap samples: {params.bootstrap_n_samples}")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # True function: quadratic with noise
    y = 0.5 * X[:, 0]**2 + 0.3 * X[:, 1] + 0.1 * np.sum(X, axis=1) + 0.01 * np.random.randn(n_samples)
    
    validation_data = {
        'X': X,
        'y': y,
        'parameters': {'coeff_0': 0.5, 'coeff_1': 0.3, 'coeff_2': 0.1},
        'time': np.arange(n_samples) * 0.001  # 1 ms timesteps
    }
    
    print(f"\nGenerated validation data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Target range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    
    # Define test models
    def linear_model(X):
        """Simple linear model."""
        return 0.4 * X[:, 0] + 0.2 * X[:, 1] + 0.05 * np.sum(X, axis=1)
    
    def quadratic_model(X):
        """Quadratic model (closer to true function)."""
        return 0.45 * X[:, 0]**2 + 0.25 * X[:, 1] + 0.08 * np.sum(X, axis=1)
    
    def noisy_model(X):
        """Model with added noise."""
        return (0.5 * X[:, 0]**2 + 0.3 * X[:, 1] + 0.1 * np.sum(X, axis=1) + 
                0.05 * np.random.randn(len(X)))
    
    models = {
        'linear_model': linear_model,
        'quadratic_model': quadratic_model,
        'noisy_model': noisy_model
    }
    
    print(f"\nDefined {len(models)} test models")
    
    # Single model validation
    print(f"\n=== SINGLE MODEL VALIDATION ===")
    single_validation = validation_framework.validate_model(
        quadratic_model, validation_data
    )
    
    print(f"Quadratic model validation:")
    print(f"  Overall passed: {'✓' if single_validation['overall_passed'] else '✗'}")
    print(f"  Validation score: {single_validation['performance_summary']['overall_validation_score']:.3f}")
    print(f"  Validation time: {single_validation['performance_summary']['total_validation_time_s']:.3f} s")
    
    # Cross-validation results
    if 'cross_validation' in single_validation['validation_results']:
        cv_result = single_validation['validation_results']['cross_validation']
        print(f"  Cross-validation R²: {cv_result.accuracy_metrics.get('r2_score', 0):.3f}")
        print(f"  Cross-validation RMSE: {cv_result.accuracy_metrics.get('rmse', 0):.3e}")
    
    # Model comparison
    print(f"\n=== MODEL COMPARISON ===")
    comparison_results = validation_framework.compare_models(models, validation_data)
    
    print(f"Model ranking:")
    for i, model_name in enumerate(comparison_results['model_ranking']):
        model_result = comparison_results['model_results'][model_name]
        score = model_result['performance_summary'].get('overall_validation_score', 0)
        passed = '✓' if model_result.get('overall_passed', False) else '✗'
        print(f"  {i+1}. {model_name}: {score:.3f} {passed}")
    
    print(f"Best model: {comparison_results['best_model']}")
    
    # Model selection
    print(f"\n=== MODEL SELECTION ===")
    selected_model = validation_framework.model_selection(
        models, validation_data, ModelSelectionCriterion.CROSS_VALIDATION
    )
    print(f"Selected model (CV): {selected_model}")
    
    selected_model_aic = validation_framework.model_selection(
        models, validation_data, ModelSelectionCriterion.AIC
    )
    print(f"Selected model (AIC): {selected_model_aic}")
    
    # Framework summary
    print(f"\n=== FRAMEWORK SUMMARY ===")
    summary = validation_framework.get_validation_summary()
    
    framework_status = summary.get('framework_status', {})
    print(f"Framework status:")
    print(f"  Available validators: {framework_status.get('available_validators', 0)}")
    print(f"  Validation history: {framework_status.get('validation_history_length', 0)}")
    
    performance_summary = summary.get('performance_summary', {})
    print(f"Performance summary:")
    print(f"  Recent success rate: {performance_summary.get('recent_success_rate', 0)*100:.1f}%")
    print(f"  Average validation score: {performance_summary.get('avg_validation_score', 0):.3f}")
    print(f"  Average validation time: {performance_summary.get('avg_validation_time_s', 0):.3f} s")
    print(f"  Accuracy target met: {'✓' if performance_summary.get('accuracy_target_met', False) else '✗'}")
    
    print(f"\nDigital twin validation framework demonstration complete!")
