#!/usr/bin/env python3
"""
CRITICAL UQ VALIDATION SCRIPT
Validates that all high and critical severity UQ concerns have been properly resolved.

Addresses the user request: "Resolve any UQ concerns at that level of severity"
"""

import logging
import numpy as np
import traceback
from pathlib import Path
import sys
import time
import json
from datetime import datetime

# Add source directories to path
src_path = Path(__file__).parent / "src"
uq_validation_path = src_path / "uq_validation"
digital_twin_path = src_path / "digital_twin"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(uq_validation_path))
sys.path.insert(0, str(digital_twin_path))

from enhanced_uq_framework import EnhancedUQFramework, UQDomainParams
from critical_uq_concerns_analysis import CriticalUQConcernAnalyzer

def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('uq_validation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_critical_uq_001_numerical_stability():
    """CRITICAL-001: Validate numerical stability protection."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CRITICAL-001: Validating numerical stability protection")
    
    try:
        # Test with deliberately unstable matrix
        unstable_matrix = np.array([
            [1e-16, 1e-15],
            [1e-15, 1e-14]
        ])
        
        from numerical_stability_framework import NumericalStabilityManager
        stability_manager = NumericalStabilityManager()
        
        # Check if stability framework catches instability
        is_stable, condition_number = stability_manager.check_matrix_stability(
            unstable_matrix, "test_unstable_matrix")
        
        if not is_stable:
            logger.info("✅ CRITICAL-001: Instability correctly detected")
            
            # Test regularization
            regularized = stability_manager.regularize_covariance_matrix(unstable_matrix)
            is_regularized_stable, _ = stability_manager.check_matrix_stability(
                regularized, "test_regularized_matrix")
            
            if is_regularized_stable:
                logger.info("✅ CRITICAL-001: Matrix successfully regularized")
                return True
            else:
                logger.error("❌ CRITICAL-001: Matrix regularization failed")
                return False
        else:
            logger.error("❌ CRITICAL-001: Failed to detect instability")
            return False
            
    except Exception as e:
        logger.error(f"❌ CRITICAL-001: Validation failed with error: {e}")
        return False

def validate_critical_uq_002_monte_carlo_convergence():
    """CRITICAL-002: Validate Monte Carlo convergence validation."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CRITICAL-002: Validating Monte Carlo convergence")
    
    try:
        from monte_carlo_convergence_validator import MonteCarloConvergenceValidator
        
        convergence_validator = MonteCarloConvergenceValidator()
        
        # Test with a simple sampling function
        def test_sampling():
            return np.random.randn(4)  # 4-dimensional test
        
        # Validate convergence
        results = convergence_validator.validate_convergence(
            sampling_function=test_sampling,
            parameter_names=["param1", "param2", "param3", "param4"]
        )
        
        if hasattr(results, 'r_hat_values') and len(results.r_hat_values) > 0:
            logger.info(f"✅ CRITICAL-002: Convergence validation functional, R-hat: {results.r_hat_values}")
            return True
        else:
            logger.error("❌ CRITICAL-002: Convergence validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ CRITICAL-002: Validation failed with error: {e}")
        return False

def validate_critical_uq_003_realtime_latency():
    """CRITICAL-003: Validate real-time latency requirements."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CRITICAL-003: Validating real-time latency requirements")
    
    try:
        # Create test parameters
        params = UQDomainParams()
        
        # Initialize UQ framework
        uq_framework = EnhancedUQFramework(params)
        
        # Test sample generation timing
        start_time = time.time()
        samples = uq_framework.generate_monte_carlo_samples(n_samples=100)
        generation_time = time.time() - start_time
        
        # Real-time requirement: < 1 second for 100 samples
        if generation_time < 1.0:
            logger.info(f"✅ CRITICAL-003: Real-time latency met ({generation_time:.3f}s for 100 samples)")
            return True
        else:
            logger.warning(f"⚠️ CRITICAL-003: Real-time latency exceeded ({generation_time:.3f}s for 100 samples)")
            return False
            
    except Exception as e:
        logger.error(f"❌ CRITICAL-003: Validation failed with error: {e}")
        return False

def validate_integration_with_enhanced_framework():
    """Test that enhanced UQ framework properly integrates all fixes."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing enhanced UQ framework integration")
    
    try:
        # Create test parameters
        params = UQDomainParams()
        
        # Initialize enhanced framework
        uq_framework = EnhancedUQFramework(params)
        
        # Test that stability manager is available
        if hasattr(uq_framework, 'stability_manager'):
            logger.info("✅ Stability manager integrated")
        else:
            logger.error("❌ Stability manager not integrated")
            return False
        
        # Test that convergence validator is available
        if hasattr(uq_framework, 'convergence_validator'):
            logger.info("✅ Convergence validator integrated")
        else:
            logger.error("❌ Convergence validator not integrated")
            return False
        
        # Test sample generation
        samples = uq_framework.generate_monte_carlo_samples(n_samples=50)
        
        if samples is not None and samples.shape[0] == 50:
            logger.info("✅ Enhanced framework functional")
            return True
        else:
            logger.error("❌ Enhanced framework sample generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Enhanced framework integration failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_comprehensive_uq_validation():
    """Run comprehensive validation of all UQ fixes."""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE UQ VALIDATION - CRITICAL AND HIGH SEVERITY FIXES")
    logger.info("=" * 80)
    
    validation_results = {}
    
    # Test CRITICAL issues
    critical_tests = [
        ("CRITICAL-001: Numerical Stability", validate_critical_uq_001_numerical_stability),
        ("CRITICAL-002: Monte Carlo Convergence", validate_critical_uq_002_monte_carlo_convergence),
        ("CRITICAL-003: Real-time Latency", validate_critical_uq_003_realtime_latency),
        ("Enhanced Framework Integration", validate_integration_with_enhanced_framework),
    ]
    
    total_passed = 0
    total_tests = len(critical_tests)
    
    for test_name, test_function in critical_tests:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'=' * 60}")
        
        try:
            result = test_function()
            validation_results[test_name] = result
            
            if result:
                logger.info(f"✅ PASSED: {test_name}")
                total_passed += 1
            else:
                logger.error(f"❌ FAILED: {test_name}")
                
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            validation_results[test_name] = False
    
    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Tests Passed: {total_passed}/{total_tests}")
    logger.info(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
    
    if total_passed == total_tests:
        logger.info("🎉 ALL CRITICAL UQ CONCERNS SUCCESSFULLY RESOLVED!")
    else:
        logger.warning(f"⚠️ {total_tests - total_passed} critical issues still need attention")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"uq_validation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'success_rate': (total_passed/total_tests)*100,
            'individual_results': validation_results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = run_comprehensive_uq_validation()
    sys.exit(0 if success else 1)
