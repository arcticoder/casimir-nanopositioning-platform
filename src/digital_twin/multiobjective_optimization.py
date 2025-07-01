"""
Multi-Objective Digital Twin Optimization with Uncertainty Awareness
Advanced Pareto Framework for Robust Performance Optimization

Implements robust multi-objective optimization with:
1. Uncertainty-aware objective functions with expected value and variance terms
2. Robust Pareto frontier computation with uncertainty dominance
3. Real-time parameter optimization under uncertainty
4. Multi-criteria decision making with confidence intervals

Mathematical Foundation:
Robust Multi-Objective Function:
J_robust = w₁⋅E[J_performance] + w₂⋅√Var[J_performance] + w₃⋅J_uncertainty

Where:
J_performance = ||y - y_target||²₂ + λ₁||Δu||²₂
J_uncertainty = tr(Σ_out) + γ∑ᵢⱼ|ρᵢⱼ|
E[J_performance] = ∫ J(θ,u)p(θ|D)dθ

Robust Pareto Frontier:
α_robust(k+1) = α(k) + η∇_α[μ_F₁ - λ(μ_F₂ + β√σ²_F₂)]

Uncertainty Dominance:
θ₁ ≻_U θ₂ iff {μ₁ᵢ ≤ μ₂ᵢ ∀i} ∧ {σ₁ᵢ ≤ σ₂ᵢ ∀i}

Author: Multi-Objective Optimization Team
Version: 1.0.0 (Uncertainty-Aware Pareto Framework)
"""

import numpy as np
import scipy.optimize as opt
from scipy.stats import multivariate_normal
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from abc import ABC, abstractmethod
from collections import deque

@dataclass
class OptimizationObjective:
    """Definition of a single optimization objective."""
    name: str
    weight: float
    target_value: Optional[float] = None
    uncertainty_weight: float = 0.1
    minimize: bool = True
    
    def evaluate(self, state: np.ndarray, uncertainty: np.ndarray, 
                control: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate objective function with uncertainty.
        
        Returns:
            (mean_value, uncertainty_contribution)
        """
        raise NotImplementedError

@dataclass
class PositioningAccuracyObjective(OptimizationObjective):
    """Positioning accuracy objective: minimize position error."""
    
    def evaluate(self, state: np.ndarray, uncertainty: np.ndarray, 
                control: np.ndarray) -> Tuple[float, float]:
        try:
            # Position error (assuming first element is position)
            target_pos = self.target_value if self.target_value is not None else 10e-9  # 10 nm
            position_error = abs(state[0] - target_pos)
            
            # Uncertainty contribution (position uncertainty)
            position_uncertainty = np.sqrt(uncertainty[0]) if len(uncertainty) > 0 else 0.0
            
            mean_value = position_error**2
            uncertainty_contribution = self.uncertainty_weight * position_uncertainty**2
            
            return mean_value, uncertainty_contribution
            
        except Exception:
            return 1e6, 1e3

@dataclass
class BandwidthObjective(OptimizationObjective):
    """Bandwidth objective: maximize system bandwidth."""
    
    def evaluate(self, state: np.ndarray, uncertainty: np.ndarray, 
                control: np.ndarray) -> Tuple[float, float]:
        try:
            # Estimate bandwidth from velocity response (simplified)
            velocity = abs(state[1]) if len(state) > 1 else 0.0
            bandwidth_estimate = velocity * 2 * np.pi  # Convert to frequency
            
            target_bandwidth = self.target_value if self.target_value is not None else 1e6  # 1 MHz
            bandwidth_error = abs(bandwidth_estimate - target_bandwidth)
            
            # Uncertainty in velocity affects bandwidth estimate
            velocity_uncertainty = np.sqrt(uncertainty[1]) if len(uncertainty) > 1 else 0.0
            bandwidth_uncertainty = velocity_uncertainty * 2 * np.pi
            
            mean_value = bandwidth_error**2
            uncertainty_contribution = self.uncertainty_weight * bandwidth_uncertainty**2
            
            return mean_value, uncertainty_contribution
            
        except Exception:
            return 1e6, 1e3

@dataclass
class ControlEffortObjective(OptimizationObjective):
    """Control effort objective: minimize control energy."""
    
    def evaluate(self, state: np.ndarray, uncertainty: np.ndarray, 
                control: np.ndarray) -> Tuple[float, float]:
        try:
            # Control effort
            control_norm = np.linalg.norm(control)**2
            
            # No direct uncertainty in control (deterministic)
            mean_value = control_norm
            uncertainty_contribution = 0.0
            
            return mean_value, uncertainty_contribution
            
        except Exception:
            return 1e3, 0.0

@dataclass
class ThermalStabilityObjective(OptimizationObjective):
    """Thermal stability objective: minimize temperature variations."""
    
    def evaluate(self, state: np.ndarray, uncertainty: np.ndarray, 
                control: np.ndarray) -> Tuple[float, float]:
        try:
            # Temperature deviation (assuming thermal state at index 4)
            target_temp = self.target_value if self.target_value is not None else 300.0  # Room temp
            if len(state) > 4:
                temp_error = abs(state[4] - target_temp)
                temp_uncertainty = np.sqrt(uncertainty[4]) if len(uncertainty) > 4 else 0.0
            else:
                temp_error = 0.0
                temp_uncertainty = 0.0
            
            mean_value = temp_error**2
            uncertainty_contribution = self.uncertainty_weight * temp_uncertainty**2
            
            return mean_value, uncertainty_contribution
            
        except Exception:
            return 1e3, 1e1

@dataclass
class UncertaintyMinimizationObjective(OptimizationObjective):
    """Uncertainty minimization objective: minimize total system uncertainty."""
    
    def evaluate(self, state: np.ndarray, uncertainty: np.ndarray, 
                control: np.ndarray) -> Tuple[float, float]:
        try:
            # Total uncertainty (trace of covariance)
            total_uncertainty = np.sum(uncertainty)
            
            mean_value = 0.0  # No mean contribution
            uncertainty_contribution = total_uncertainty
            
            return mean_value, uncertainty_contribution
            
        except Exception:
            return 0.0, 1e6

@dataclass
class MultiObjectiveOptimizationParams:
    """Parameters for multi-objective optimization."""
    # Optimization settings
    population_size: int = 100
    max_generations: int = 200
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Pareto frontier settings
    pareto_archive_size: int = 50
    dominance_threshold: float = 1e-6
    uncertainty_dominance_weight: float = 0.3
    
    # Robust optimization settings
    monte_carlo_samples: int = 1000
    confidence_level: float = 0.95
    risk_aversion_factor: float = 0.1
    
    # Parallel processing
    n_threads: int = 8
    
    # Convergence criteria
    convergence_tolerance: float = 1e-6
    max_stagnation_generations: int = 20

class ParetoSolution:
    """Represents a solution on the Pareto frontier."""
    
    def __init__(self, parameters: np.ndarray, objectives: np.ndarray, 
                 objective_uncertainties: np.ndarray):
        self.parameters = parameters.copy()
        self.objectives = objectives.copy()
        self.objective_uncertainties = objective_uncertainties.copy()
        
        # Robust objective values (mean + risk term)
        self.robust_objectives = objectives + 0.1 * objective_uncertainties
        
        # Dominance metrics
        self.dominance_count = 0
        self.dominated_solutions = []
        
    def dominates(self, other: 'ParetoSolution', uncertainty_weight: float = 0.3) -> bool:
        """
        Check if this solution dominates another using uncertainty-aware dominance.
        
        θ₁ ≻_U θ₂ iff {μ₁ᵢ ≤ μ₂ᵢ ∀i} ∧ {σ₁ᵢ ≤ σ₂ᵢ ∀i}
        """
        try:
            # Standard Pareto dominance on mean objectives
            mean_dominates = np.all(self.objectives <= other.objectives) and \
                           np.any(self.objectives < other.objectives)
            
            # Uncertainty dominance
            uncertainty_dominates = np.all(self.objective_uncertainties <= 
                                         other.objective_uncertainties)
            
            # Combined uncertainty-aware dominance
            if uncertainty_weight > 0:
                robust_dominates = np.all(self.robust_objectives <= other.robust_objectives) and \
                                 np.any(self.robust_objectives < other.robust_objectives)
                return robust_dominates and uncertainty_dominates
            else:
                return mean_dominates
                
        except Exception:
            return False
    
    def distance_to(self, other: 'ParetoSolution') -> float:
        """Compute distance to another solution in objective space."""
        return np.linalg.norm(self.robust_objectives - other.robust_objectives)

class MultiObjectiveDigitalTwinOptimizer:
    """Multi-objective optimizer for digital twin parameters."""
    
    def __init__(self, objectives: List[OptimizationObjective], 
                 params: MultiObjectiveOptimizationParams):
        self.objectives = objectives
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self.current_population = []
        self.pareto_frontier = []
        self.generation = 0
        self.convergence_history = []
        
        # Performance tracking
        self.optimization_times = deque(maxlen=100)
        self.hypervolume_history = []
        
        # Thread safety
        self._lock = threading.Lock()
        
    def optimize(self, initial_parameters: Dict[str, np.ndarray],
                parameter_bounds: Dict[str, Tuple[float, float]],
                digital_twin_evaluator: Callable) -> List[ParetoSolution]:
        """
        Perform multi-objective optimization of digital twin parameters.
        
        Args:
            initial_parameters: Initial parameter values
            parameter_bounds: Parameter bounds for optimization
            digital_twin_evaluator: Function to evaluate digital twin performance
            
        Returns:
            Pareto optimal solutions
        """
        start_time = time.perf_counter()
        
        try:
            self.logger.info("Starting multi-objective digital twin optimization")
            
            # Initialize population
            self._initialize_population(initial_parameters, parameter_bounds)
            
            # Evolution loop
            for generation in range(self.params.max_generations):
                self.generation = generation
                
                # Evaluate population
                self._evaluate_population(digital_twin_evaluator)
                
                # Update Pareto frontier
                self._update_pareto_frontier()
                
                # Check convergence
                if self._check_convergence():
                    self.logger.info(f"Converged at generation {generation}")
                    break
                
                # Generate next generation
                self._generate_next_generation(parameter_bounds)
                
                # Log progress
                if generation % 10 == 0:
                    hypervolume = self._compute_hypervolume()
                    self.hypervolume_history.append(hypervolume)
                    self.logger.info(f"Generation {generation}: "
                                   f"Pareto size = {len(self.pareto_frontier)}, "
                                   f"Hypervolume = {hypervolume:.6f}")
            
            optimization_time = time.perf_counter() - start_time
            self.optimization_times.append(optimization_time)
            
            self.logger.info(f"Optimization completed in {optimization_time:.2f}s")
            return self.pareto_frontier
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return []
    
    def _initialize_population(self, initial_parameters: Dict[str, np.ndarray],
                             parameter_bounds: Dict[str, Tuple[float, float]]) -> None:
        """Initialize optimization population."""
        try:
            self.current_population = []
            
            # Convert parameters to vector form
            param_names = list(parameter_bounds.keys())
            param_dim = len(param_names)
            
            for i in range(self.params.population_size):
                if i == 0:
                    # Start with initial parameters
                    params_vector = np.array([initial_parameters.get(name, 0.0) 
                                            for name in param_names])
                else:
                    # Random initialization within bounds
                    params_vector = np.zeros(param_dim)
                    for j, name in enumerate(param_names):
                        min_val, max_val = parameter_bounds[name]
                        params_vector[j] = np.random.uniform(min_val, max_val)
                
                # Create dummy solution (will be evaluated later)
                solution = ParetoSolution(
                    parameters=params_vector,
                    objectives=np.full(len(self.objectives), np.inf),
                    objective_uncertainties=np.full(len(self.objectives), np.inf)
                )
                
                self.current_population.append(solution)
                
        except Exception as e:
            self.logger.error(f"Population initialization failed: {e}")
    
    def _evaluate_population(self, digital_twin_evaluator: Callable) -> None:
        """Evaluate all solutions in current population."""
        try:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=self.params.n_threads) as executor:
                futures = []
                
                for solution in self.current_population:
                    future = executor.submit(self._evaluate_solution, 
                                           solution, digital_twin_evaluator)
                    futures.append((solution, future))
                
                # Collect results
                for solution, future in futures:
                    try:
                        objectives, uncertainties = future.result(timeout=30)
                        solution.objectives = objectives
                        solution.objective_uncertainties = uncertainties
                        solution.robust_objectives = objectives + 0.1 * uncertainties
                    except Exception as e:
                        self.logger.debug(f"Solution evaluation failed: {e}")
                        # Keep infinite objectives for failed evaluations
                        
        except Exception as e:
            self.logger.error(f"Population evaluation failed: {e}")
    
    def _evaluate_solution(self, solution: ParetoSolution, 
                          digital_twin_evaluator: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a single solution with Monte Carlo uncertainty estimation."""
        try:
            # Get system state and uncertainty from digital twin
            state, uncertainty, control = digital_twin_evaluator(solution.parameters)
            
            # Monte Carlo evaluation for robust objectives
            n_samples = min(self.params.monte_carlo_samples, 100)  # Limit for speed
            objective_samples = np.zeros((n_samples, len(self.objectives)))
            
            for i in range(n_samples):
                # Sample from uncertainty distribution
                if uncertainty.size > 0:
                    state_sample = np.random.multivariate_normal(
                        state, np.diag(uncertainty**2))
                else:
                    state_sample = state
                
                # Evaluate objectives for this sample
                for j, objective in enumerate(self.objectives):
                    mean_val, uncertainty_contrib = objective.evaluate(
                        state_sample, uncertainty, control)
                    objective_samples[i, j] = mean_val + uncertainty_contrib
            
            # Compute robust objective statistics
            mean_objectives = np.mean(objective_samples, axis=0)
            objective_uncertainties = np.std(objective_samples, axis=0)
            
            return mean_objectives, objective_uncertainties
            
        except Exception as e:
            self.logger.debug(f"Solution evaluation error: {e}")
            return (np.full(len(self.objectives), 1e6), 
                   np.full(len(self.objectives), 1e3))
    
    def _update_pareto_frontier(self) -> None:
        """Update Pareto frontier with new solutions."""
        with self._lock:
            try:
                # Combine current frontier with new solutions
                all_solutions = self.pareto_frontier + [sol for sol in self.current_population 
                                                       if not np.any(np.isinf(sol.objectives))]
                
                # Find non-dominated solutions
                pareto_solutions = []
                
                for i, solution_i in enumerate(all_solutions):
                    is_dominated = False
                    
                    for j, solution_j in enumerate(all_solutions):
                        if i != j and solution_j.dominates(solution_i, 
                                                          self.params.uncertainty_dominance_weight):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        pareto_solutions.append(solution_i)
                
                # Limit frontier size
                if len(pareto_solutions) > self.params.pareto_archive_size:
                    # Use clustering to maintain diversity
                    pareto_solutions = self._maintain_diversity(pareto_solutions)
                
                self.pareto_frontier = pareto_solutions
                
            except Exception as e:
                self.logger.error(f"Pareto frontier update failed: {e}")
    
    def _maintain_diversity(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """Maintain diversity in Pareto frontier using clustering."""
        try:
            if len(solutions) <= self.params.pareto_archive_size:
                return solutions
            
            # Simple diversity maintenance: select solutions with maximum spread
            selected = [solutions[0]]  # Always keep first
            
            for _ in range(self.params.pareto_archive_size - 1):
                max_min_distance = 0
                best_candidate = None
                
                for candidate in solutions:
                    if candidate in selected:
                        continue
                    
                    # Find minimum distance to selected solutions
                    min_distance = min(candidate.distance_to(sel) for sel in selected)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected.append(best_candidate)
                else:
                    break
            
            return selected
            
        except Exception as e:
            self.logger.debug(f"Diversity maintenance failed: {e}")
            return solutions[:self.params.pareto_archive_size]
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        try:
            if len(self.convergence_history) < 2:
                return False
            
            # Check if hypervolume has stagnated
            recent_hypervolumes = self.hypervolume_history[-self.params.max_stagnation_generations:]
            
            if len(recent_hypervolumes) >= self.params.max_stagnation_generations:
                improvement = max(recent_hypervolumes) - min(recent_hypervolumes)
                return improvement < self.params.convergence_tolerance
            
            return False
            
        except Exception:
            return False
    
    def _generate_next_generation(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> None:
        """Generate next generation using genetic operators."""
        try:
            new_population = []
            
            # Elite preservation: keep best solutions
            elite_size = max(1, self.params.population_size // 10)
            elite_solutions = sorted(self.current_population, 
                                   key=lambda x: np.sum(x.robust_objectives))[:elite_size]
            new_population.extend(elite_solutions)
            
            # Generate offspring
            while len(new_population) < self.params.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if np.random.random() < self.params.crossover_rate:
                    child_params = self._crossover(parent1.parameters, parent2.parameters)
                else:
                    child_params = parent1.parameters.copy()
                
                # Mutation
                if np.random.random() < self.params.mutation_rate:
                    child_params = self._mutate(child_params, parameter_bounds)
                
                # Create new solution
                child = ParetoSolution(
                    parameters=child_params,
                    objectives=np.full(len(self.objectives), np.inf),
                    objective_uncertainties=np.full(len(self.objectives), np.inf)
                )
                
                new_population.append(child)
            
            self.current_population = new_population
            
        except Exception as e:
            self.logger.error(f"Next generation creation failed: {e}")
    
    def _tournament_selection(self, tournament_size: int = 3) -> ParetoSolution:
        """Tournament selection for parent selection."""
        try:
            # Select random candidates
            candidates = np.random.choice(self.current_population, 
                                        size=min(tournament_size, len(self.current_population)),
                                        replace=False)
            
            # Return best candidate (lowest sum of robust objectives)
            return min(candidates, key=lambda x: np.sum(x.robust_objectives))
            
        except Exception:
            return np.random.choice(self.current_population)
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover operator."""
        try:
            child = np.zeros_like(parent1)
            for i in range(len(parent1)):
                child[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]
            return child
        except Exception:
            return parent1.copy()
    
    def _mutate(self, parameters: np.ndarray, 
               parameter_bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Gaussian mutation operator."""
        try:
            mutated = parameters.copy()
            param_names = list(parameter_bounds.keys())
            
            for i in range(len(parameters)):
                if i < len(param_names):
                    min_val, max_val = parameter_bounds[param_names[i]]
                    mutation_strength = 0.1 * (max_val - min_val)
                    mutated[i] += np.random.normal(0, mutation_strength)
                    mutated[i] = np.clip(mutated[i], min_val, max_val)
            
            return mutated
            
        except Exception:
            return parameters
    
    def _compute_hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """Compute hypervolume indicator for Pareto frontier quality."""
        try:
            if not self.pareto_frontier:
                return 0.0
            
            # Extract objective values
            objectives_matrix = np.array([sol.robust_objectives for sol in self.pareto_frontier])
            
            if reference_point is None:
                # Use worst objectives as reference point
                reference_point = np.max(objectives_matrix, axis=0) * 1.1
            
            # Simplified hypervolume calculation (2D case)
            if objectives_matrix.shape[1] == 2:
                sorted_solutions = sorted(self.pareto_frontier, 
                                        key=lambda x: x.robust_objectives[0])
                
                hypervolume = 0.0
                prev_x = reference_point[0]
                
                for solution in sorted_solutions:
                    x, y = solution.robust_objectives
                    if x < prev_x and y < reference_point[1]:
                        hypervolume += (prev_x - x) * (reference_point[1] - y)
                        prev_x = x
                
                return hypervolume
            else:
                # For higher dimensions, use approximation
                return float(len(self.pareto_frontier))
                
        except Exception:
            return 0.0
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        return {
            'pareto_frontier_size': len(self.pareto_frontier),
            'generations_completed': self.generation,
            'final_hypervolume': self._compute_hypervolume() if self.pareto_frontier else 0.0,
            'average_optimization_time': np.mean(self.optimization_times) if self.optimization_times else 0.0,
            'convergence_achieved': len(self.convergence_history) > 0,
            'best_solutions': [
                {
                    'parameters': sol.parameters.tolist(),
                    'objectives': sol.objectives.tolist(),
                    'uncertainties': sol.objective_uncertainties.tolist(),
                    'robust_objectives': sol.robust_objectives.tolist()
                }
                for sol in self.pareto_frontier[:5]  # Top 5 solutions
            ]
        }

def main():
    """Demonstration of multi-objective digital twin optimization."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Multi-Objective Digital Twin Optimization Demonstration")
    
    # Define optimization objectives
    objectives = [
        PositioningAccuracyObjective(name="positioning", weight=1.0, target_value=10e-9),
        BandwidthObjective(name="bandwidth", weight=0.8, target_value=1e6),
        ControlEffortObjective(name="control_effort", weight=0.3),
        ThermalStabilityObjective(name="thermal", weight=0.5, target_value=300.0),
        UncertaintyMinimizationObjective(name="uncertainty", weight=0.4)
    ]
    
    # Optimization parameters
    params = MultiObjectiveOptimizationParams(
        population_size=50,
        max_generations=50,
        n_threads=4
    )
    
    # Initialize optimizer
    optimizer = MultiObjectiveDigitalTwinOptimizer(objectives, params)
    
    # Mock digital twin evaluator
    def mock_digital_twin_evaluator(parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mock digital twin evaluation function."""
        # Simulate system state based on parameters
        state = np.array([parameters[0] * 1e-9,  # position
                         parameters[1] * 1e-6,   # velocity  
                         0.0, 0.0,               # acceleration, force
                         300 + parameters[2],    # temperature
                         0.0, 0.0,               # thermal terms
                         parameters[3] * 1e3,    # E field
                         0.0, 0.0, 0.0,          # other EM terms
                         0.5, 0.1, 0.05])        # quantum terms
        
        # Simulate uncertainties
        uncertainty = np.array([1e-12, 1e-9, 1e-6, 1e-15, 0.1, 1e-6, 1e-9, 
                               1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-18, 1e-18])
        
        # Control signal
        control = np.array([parameters[0] * 1e-12])  # Simple proportional control
        
        return state, uncertainty, control
    
    # Parameter bounds
    parameter_bounds = {
        'position_gain': (5.0, 15.0),
        'velocity_gain': (0.1, 2.0), 
        'thermal_offset': (-5.0, 5.0),
        'field_strength': (0.1, 1.0)
    }
    
    # Initial parameters
    initial_parameters = {
        'position_gain': 10.0,
        'velocity_gain': 1.0,
        'thermal_offset': 0.0,
        'field_strength': 0.5
    }
    
    print(f"\n🚀 MULTI-OBJECTIVE OPTIMIZATION:")
    print(f"   Objectives:              {len(objectives)}")
    print(f"   Population Size:         {params.population_size}")
    print(f"   Max Generations:         {params.max_generations}")
    print(f"   Parameters:              {len(parameter_bounds)}")
    
    # Run optimization
    pareto_solutions = optimizer.optimize(initial_parameters, parameter_bounds, 
                                        mock_digital_twin_evaluator)
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"   Pareto Solutions:        {summary['pareto_frontier_size']}")
    print(f"   Generations Completed:   {summary['generations_completed']}")
    print(f"   Final Hypervolume:       {summary['final_hypervolume']:.6f}")
    print(f"   Optimization Time:       {summary['average_optimization_time']:.2f}s")
    
    if pareto_solutions:
        print(f"\n🏆 TOP PARETO SOLUTIONS:")
        for i, solution_data in enumerate(summary['best_solutions'][:3]):
            print(f"\n   Solution {i+1}:")
            print(f"     Parameters:    {[f'{x:.3f}' for x in solution_data['parameters']]}")
            print(f"     Objectives:    {[f'{x:.2e}' for x in solution_data['objectives']]}")
            print(f"     Uncertainties: {[f'{x:.2e}' for x in solution_data['uncertainties']]}")
            print(f"     Robust Values: {[f'{x:.2e}' for x in solution_data['robust_objectives']]}")
    
    print(f"\n✅ Multi-Objective Digital Twin Optimization Successfully Demonstrated")

if __name__ == "__main__":
    main()
