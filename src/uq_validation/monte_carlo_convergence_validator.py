"""
Monte Carlo Convergence Validation Framework
Comprehensive convergence diagnostics for uncertainty quantification

This module provides robust Monte Carlo convergence validation using:
1. Gelman-Rubin diagnostics (R-hat statistic)
2. Effective sample size estimation
3. Adaptive sample sizing until convergence
4. Multiple chains analysis

CRITICAL RESOLUTION: CRITICAL-002
- Addresses missing Monte Carlo convergence validation
- Implements automatic convergence checking with R-hat < 1.1 criterion
- Provides adaptive sample sizing for reliable uncertainty estimates

Mathematical Foundation:
R-hat = √((n-1)/n + (B/W)(m+1)/(m*n))

Where:
B = between-chain variance
W = within-chain variance  
n = chain length
m = number of chains

Author: Monte Carlo Validation Team
Version: 1.0.0 (Critical Resolution)
"""

import numpy as np
import scipy.stats as stats
from scipy import signal
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

@dataclass
class ConvergenceConfig:
    """Configuration for Monte Carlo convergence validation."""
    target_r_hat: float = 1.1           # Gelman-Rubin threshold
    min_samples: int = 1000             # Minimum samples per chain
    max_samples: int = 100000           # Maximum samples per chain
    min_chains: int = 4                 # Minimum number of chains
    max_chains: int = 8                 # Maximum number of chains
    min_effective_samples: int = 1000   # Minimum effective sample size
    convergence_check_interval: int = 1000  # Check every N samples
    warmup_fraction: float = 0.5        # Fraction for warmup/burn-in
    autocorr_max_lag: int = 100         # Maximum lag for autocorrelation
    thinning_enabled: bool = True       # Enable sample thinning

@dataclass
class ConvergenceResults:
    """Results from convergence analysis."""
    converged: bool
    r_hat_values: Dict[str, float]
    effective_sample_sizes: Dict[str, float]
    total_samples: int
    chains_used: int
    convergence_time: float
    autocorrelation_times: Dict[str, float]
    convergence_diagnostics: Dict[str, any]

class MonteCarloConvergenceValidator:
    """Monte Carlo convergence validation using multiple diagnostic methods."""
    
    def __init__(self, config: ConvergenceConfig = None):
        self.config = config or ConvergenceConfig()
        self.logger = logging.getLogger(__name__)
        
    def validate_convergence(self, sampling_function: callable, 
                           parameter_names: List[str],
                           true_values: Optional[Dict[str, float]] = None) -> ConvergenceResults:
        """
        Validate Monte Carlo convergence using multiple chains and diagnostics.
        
        Args:
            sampling_function: Function that generates samples
            parameter_names: Names of parameters being sampled
            true_values: Optional true values for validation
            
        Returns:
            Convergence validation results
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 Starting Monte Carlo convergence validation")
        self.logger.info(f"   Parameters: {parameter_names}")
        self.logger.info(f"   Target R-hat: {self.config.target_r_hat}")
        
        # Initialize multiple chains
        chains = []
        for chain_id in range(self.config.min_chains):
            chain = self._initialize_chain(sampling_function, chain_id)
            chains.append(chain)
        
        # Progressive convergence checking
        current_samples = self.config.min_samples
        converged = False
        final_r_hat = {}
        final_ess = {}
        final_autocorr = {}
        
        while current_samples <= self.config.max_samples and not converged:
            # Extend chains to current sample size
            self._extend_chains(chains, current_samples, sampling_function)
            
            # Extract samples after warmup
            warmup_samples = int(current_samples * self.config.warmup_fraction)
            chain_samples = []
            
            for chain in chains:
                if len(chain) > warmup_samples:
                    chain_samples.append(chain[warmup_samples:])
                else:
                    chain_samples.append(chain)
            
            # Calculate diagnostics for each parameter
            r_hat_values = {}
            ess_values = {}
            autocorr_times = {}
            
            convergence_achieved = True
            
            for i, param_name in enumerate(parameter_names):
                # Extract parameter values from all chains
                param_chains = []
                for chain_data in chain_samples:
                    if len(chain_data) > 0:
                        param_values = np.array([sample[i] if isinstance(sample, (list, tuple)) 
                                               else sample for sample in chain_data])
                        param_chains.append(param_values)
                
                if len(param_chains) > 0 and all(len(chain) > 10 for chain in param_chains):
                    # Calculate R-hat
                    r_hat = self._calculate_gelman_rubin_statistic(param_chains)
                    r_hat_values[param_name] = r_hat
                    
                    # Calculate effective sample size
                    ess = self._calculate_effective_sample_size(param_chains)
                    ess_values[param_name] = ess
                    
                    # Calculate autocorrelation time
                    autocorr_time = self._calculate_autocorrelation_time(param_chains)
                    autocorr_times[param_name] = autocorr_time
                    
                    # Check convergence criteria
                    if (r_hat > self.config.target_r_hat or 
                        ess < self.config.min_effective_samples):
                        convergence_achieved = False
                else:
                    convergence_achieved = False
            
            final_r_hat = r_hat_values
            final_ess = ess_values
            final_autocorr = autocorr_times
            
            # Log progress
            if r_hat_values:
                max_r_hat = max(r_hat_values.values())
                min_ess = min(ess_values.values()) if ess_values else 0
                self.logger.info(f"   Samples: {current_samples}, Max R-hat: {max_r_hat:.4f}, Min ESS: {min_ess:.0f}")
            
            if convergence_achieved and r_hat_values:
                converged = True
                self.logger.info(f"✅ Convergence achieved at {current_samples} samples")
                break
            
            # Increase sample size
            current_samples += self.config.convergence_check_interval
        
        # Additional diagnostics
        diagnostics = self._compute_additional_diagnostics(chain_samples, parameter_names)
        
        # Validate against true values if provided
        if true_values:
            diagnostics['bias_analysis'] = self._analyze_bias(chain_samples, parameter_names, true_values)
        
        convergence_time = time.time() - start_time
        
        # Create results
        results = ConvergenceResults(
            converged=converged,
            r_hat_values=final_r_hat,
            effective_sample_sizes=final_ess,
            total_samples=current_samples * len(chains),
            chains_used=len(chains),
            convergence_time=convergence_time,
            autocorrelation_times=final_autocorr,
            convergence_diagnostics=diagnostics
        )
        
        self._log_convergence_results(results)
        
        return results
    
    def _initialize_chain(self, sampling_function: callable, chain_id: int) -> List:
        """Initialize a single Markov chain."""
        np.random.seed(chain_id * 12345)  # Different seed for each chain
        return []
    
    def _extend_chains(self, chains: List[List], target_length: int, 
                      sampling_function: callable) -> None:
        """Extend chains to target length."""
        for chain in chains:
            while len(chain) < target_length:
                try:
                    sample = sampling_function()
                    chain.append(sample)
                except Exception as e:
                    self.logger.warning(f"Sampling failed: {e}")
                    break
    
    def _calculate_gelman_rubin_statistic(self, chains: List[np.ndarray]) -> float:
        """
        Calculate Gelman-Rubin R-hat statistic.
        
        Args:
            chains: List of chains for a single parameter
            
        Returns:
            R-hat statistic
        """
        try:
            if len(chains) < 2:
                return float('inf')
            
            # Ensure all chains have same length
            min_length = min(len(chain) for chain in chains)
            if min_length < 10:
                return float('inf')
            
            chains_array = np.array([chain[:min_length] for chain in chains])
            
            m, n = chains_array.shape  # m chains, n samples per chain
            
            # Calculate chain means and overall mean
            chain_means = np.mean(chains_array, axis=1)
            overall_mean = np.mean(chain_means)
            
            # Between-chain variance (B)
            B = n * np.var(chain_means, ddof=1) if m > 1 else 0
            
            # Within-chain variance (W)
            chain_variances = np.var(chains_array, axis=1, ddof=1)
            W = np.mean(chain_variances)
            
            # Marginal posterior variance estimate
            var_plus = ((n - 1) / n) * W + (1 / n) * B
            
            # R-hat statistic
            if W > 0:
                r_hat = np.sqrt(var_plus / W)
            else:
                r_hat = float('inf')
            
            return float(r_hat)
            
        except Exception as e:
            self.logger.warning(f"R-hat calculation failed: {e}")
            return float('inf')
    
    def _calculate_effective_sample_size(self, chains: List[np.ndarray]) -> float:
        """
        Calculate effective sample size accounting for autocorrelation.
        
        Args:
            chains: List of chains for a single parameter
            
        Returns:
            Effective sample size
        """
        try:
            if not chains:
                return 0.0
            
            total_samples = sum(len(chain) for chain in chains)
            
            # Calculate autocorrelation for each chain
            autocorr_times = []
            for chain in chains:
                if len(chain) > 20:
                    autocorr_time = self._autocorrelation_time_single_chain(chain)
                    autocorr_times.append(autocorr_time)
            
            if not autocorr_times:
                return float(total_samples)
            
            # Average autocorrelation time
            avg_autocorr_time = np.mean(autocorr_times)
            
            # Effective sample size
            ess = total_samples / (2 * avg_autocorr_time + 1)
            
            return float(max(1.0, ess))
            
        except Exception as e:
            self.logger.warning(f"ESS calculation failed: {e}")
            return 0.0
    
    def _calculate_autocorrelation_time(self, chains: List[np.ndarray]) -> float:
        """Calculate autocorrelation time."""
        try:
            if not chains:
                return float('inf')
            
            autocorr_times = []
            for chain in chains:
                if len(chain) > 20:
                    autocorr_time = self._autocorrelation_time_single_chain(chain)
                    autocorr_times.append(autocorr_time)
            
            return float(np.mean(autocorr_times)) if autocorr_times else float('inf')
            
        except Exception as e:
            self.logger.warning(f"Autocorrelation calculation failed: {e}")
            return float('inf')
    
    def _autocorrelation_time_single_chain(self, chain: np.ndarray) -> float:
        """Calculate autocorrelation time for a single chain."""
        try:
            chain = np.array(chain)
            n = len(chain)
            
            if n < 20:
                return float('inf')
            
            # Calculate autocorrelation function
            max_lag = min(self.config.autocorr_max_lag, n // 4)
            autocorr = np.correlate(chain - np.mean(chain), 
                                  chain - np.mean(chain), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find integrated autocorrelation time
            autocorr_time = 1.0
            for k in range(1, min(len(autocorr), max_lag)):
                if autocorr[k] <= 0:
                    break
                autocorr_time += 2 * autocorr[k]
            
            return float(autocorr_time)
            
        except Exception as e:
            return float('inf')
    
    def _compute_additional_diagnostics(self, chain_samples: List, 
                                      parameter_names: List[str]) -> Dict:
        """Compute additional convergence diagnostics."""
        diagnostics = {}
        
        try:
            # Monte Carlo standard error
            diagnostics['mc_se'] = {}
            for i, param_name in enumerate(parameter_names):
                all_samples = []
                for chain_data in chain_samples:
                    if len(chain_data) > 0:
                        param_values = [sample[i] if isinstance(sample, (list, tuple)) 
                                      else sample for sample in chain_data]
                        all_samples.extend(param_values)
                
                if all_samples:
                    mc_se = np.std(all_samples) / np.sqrt(len(all_samples))
                    diagnostics['mc_se'][param_name] = float(mc_se)
            
            # Geweke diagnostic
            diagnostics['geweke'] = self._geweke_diagnostic(chain_samples, parameter_names)
            
            return diagnostics
            
        except Exception as e:
            self.logger.warning(f"Additional diagnostics failed: {e}")
            return diagnostics
    
    def _geweke_diagnostic(self, chain_samples: List, parameter_names: List[str]) -> Dict:
        """Compute Geweke convergence diagnostic."""
        geweke_results = {}
        
        try:
            for i, param_name in enumerate(parameter_names):
                # Combine all chains for this parameter
                all_samples = []
                for chain_data in chain_samples:
                    if len(chain_data) > 0:
                        param_values = [sample[i] if isinstance(sample, (list, tuple)) 
                                      else sample for sample in chain_data]
                        all_samples.extend(param_values)
                
                if len(all_samples) > 100:
                    # First 10% vs last 50%
                    n = len(all_samples)
                    first_part = all_samples[:n//10]
                    last_part = all_samples[n//2:]
                    
                    if len(first_part) > 10 and len(last_part) > 10:
                        # Z-score for difference in means
                        mean_diff = np.mean(last_part) - np.mean(first_part)
                        se_diff = np.sqrt(np.var(first_part)/len(first_part) + 
                                        np.var(last_part)/len(last_part))
                        
                        if se_diff > 0:
                            z_score = mean_diff / se_diff
                            geweke_results[param_name] = float(z_score)
            
            return geweke_results
            
        except Exception as e:
            self.logger.warning(f"Geweke diagnostic failed: {e}")
            return {}
    
    def _analyze_bias(self, chain_samples: List, parameter_names: List[str], 
                     true_values: Dict[str, float]) -> Dict:
        """Analyze bias relative to true values."""
        bias_analysis = {}
        
        try:
            for i, param_name in enumerate(parameter_names):
                if param_name in true_values:
                    all_samples = []
                    for chain_data in chain_samples:
                        if len(chain_data) > 0:
                            param_values = [sample[i] if isinstance(sample, (list, tuple)) 
                                          else sample for sample in chain_data]
                            all_samples.extend(param_values)
                    
                    if all_samples:
                        estimated_mean = np.mean(all_samples)
                        true_value = true_values[param_name]
                        bias = estimated_mean - true_value
                        relative_bias = bias / true_value if true_value != 0 else float('inf')
                        
                        bias_analysis[param_name] = {
                            'absolute_bias': float(bias),
                            'relative_bias': float(relative_bias),
                            'estimated_mean': float(estimated_mean),
                            'true_value': float(true_value)
                        }
            
            return bias_analysis
            
        except Exception as e:
            self.logger.warning(f"Bias analysis failed: {e}")
            return {}
    
    def _log_convergence_results(self, results: ConvergenceResults) -> None:
        """Log convergence validation results."""
        
        self.logger.info(f"🏁 Monte Carlo Convergence Results:")
        self.logger.info(f"   Converged: {'✅ YES' if results.converged else '❌ NO'}")
        self.logger.info(f"   Total Samples: {results.total_samples:,}")
        self.logger.info(f"   Chains Used: {results.chains_used}")
        self.logger.info(f"   Validation Time: {results.convergence_time:.2f}s")
        
        if results.r_hat_values:
            self.logger.info(f"   R-hat Statistics:")
            for param, r_hat in results.r_hat_values.items():
                status = "✅" if r_hat <= self.config.target_r_hat else "❌"
                self.logger.info(f"     {param}: {r_hat:.4f} {status}")
        
        if results.effective_sample_sizes:
            self.logger.info(f"   Effective Sample Sizes:")
            for param, ess in results.effective_sample_sizes.items():
                status = "✅" if ess >= self.config.min_effective_samples else "❌"
                self.logger.info(f"     {param}: {ess:.0f} {status}")

def main():
    """Demonstrate Monte Carlo convergence validation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Monte Carlo Convergence Validation Demonstration")
    
    # Example sampling function
    def sample_bivariate_normal():
        return np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
    
    # Validate convergence
    validator = MonteCarloConvergenceValidator()
    results = validator.validate_convergence(
        sampling_function=sample_bivariate_normal,
        parameter_names=['param1', 'param2'],
        true_values={'param1': 0.0, 'param2': 0.0}
    )
    
    print(f"\n📊 CONVERGENCE VALIDATION SUMMARY:")
    print(f"   Converged: {results.converged}")
    print(f"   Total Samples: {results.total_samples:,}")
    print(f"   R-hat Values: {results.r_hat_values}")
    print(f"   ESS Values: {results.effective_sample_sizes}")
    
    print("\n✅ Monte Carlo convergence validation operational")

if __name__ == "__main__":
    main()
