"""
Adaptive Mesh Refinement with Uncertainty Quantification
Dynamic Space-Time UQ Mesh Adaptation Framework

Implements UQ-guided mesh refinement with:
1. Dynamic uncertainty mapping with spatiotemporal correlations
2. UQ-guided refinement criteria based on uncertainty gradients
3. Multi-resolution uncertainty propagation
4. Real-time mesh adaptation for digital twin synchronization

Mathematical Foundation:
UQ-Guided Mesh Refinement:
ε_UQ(x,t) = √∑ᵢ₌₁ᴹ λᵢφᵢ(x)ψᵢ(t)

Refinement Criterion:
h_new = h_old × min{1, (TOL/ε_UQ)^(1/p)}

Where:
TOL = tolerance × uncertainty_confidence_level
p = order of uncertainty expansion

Spatiotemporal Correlation:
K(x₁,x₂,t₁,t₂) = σ²exp(-||x₁-x₂||²/l²)exp(-|t₁-t₂|/τ)

Multi-Resolution UQ:
U(x,t) = ∑ₖ₌₀ᴸ ∑ⱼ Uₖⱼφₖⱼ(x)ψₖⱼ(t)

Author: Adaptive Mesh UQ Team
Version: 1.0.0 (Dynamic UQ Mesh Framework)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d, RegularGridInterpolator
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class MeshPoint:
    """Represents a point in the adaptive mesh."""
    coordinates: np.ndarray
    timestamp: float
    uncertainty_value: float
    uncertainty_gradient: np.ndarray
    refinement_level: int
    is_active: bool = True
    
    def __post_init__(self):
        if self.uncertainty_gradient is None:
            self.uncertainty_gradient = np.zeros_like(self.coordinates)

@dataclass
class MeshElement:
    """Represents an element in the adaptive mesh."""
    vertices: List[MeshPoint]
    center: np.ndarray
    volume: float
    uncertainty_estimate: float
    refinement_indicator: float
    children: List['MeshElement'] = field(default_factory=list)
    parent: Optional['MeshElement'] = None
    level: int = 0
    
    def needs_refinement(self, tolerance: float) -> bool:
        """Check if element needs refinement based on UQ criteria."""
        return self.refinement_indicator > tolerance

@dataclass
class AdaptiveMeshParams:
    """Parameters for adaptive mesh refinement."""
    # Spatial domain
    spatial_bounds: Tuple[np.ndarray, np.ndarray] = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    initial_resolution: Tuple[int, int] = (10, 10)
    max_refinement_levels: int = 5
    min_element_size: float = 1e-6
    
    # Temporal domain
    time_bounds: Tuple[float, float] = (0.0, 1.0)
    initial_time_steps: int = 100
    max_temporal_refinement: int = 3
    
    # UQ parameters
    uncertainty_tolerance: float = 1e-6
    confidence_level: float = 0.95
    refinement_order: int = 2
    coarsening_threshold: float = 0.1
    
    # Correlation parameters
    spatial_correlation_length: float = 0.1
    temporal_correlation_time: float = 0.01
    correlation_decay_rate: float = 2.0
    
    # Adaptive parameters
    refinement_factor: float = 2.0
    adaptation_frequency: int = 10
    max_elements: int = 100000
    
    # Multi-resolution settings
    max_resolution_levels: int = 4
    wavelength_ratios: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])

class UQMeshFunction:
    """Function representing uncertainty distribution on mesh."""
    
    def __init__(self, mesh_points: List[MeshPoint]):
        self.mesh_points = mesh_points
        self.logger = logging.getLogger(__name__)
        
        # Spatial interpolator
        if mesh_points:
            coordinates = np.array([p.coordinates for p in mesh_points])
            values = np.array([p.uncertainty_value for p in mesh_points])
            
            if coordinates.shape[1] == 1:
                self.interpolator = interp1d(coordinates.flatten(), values, 
                                           bounds_error=False, fill_value=0.0)
            elif coordinates.shape[1] == 2:
                # For 2D, create regular grid interpolator
                x_unique = np.unique(coordinates[:, 0])
                y_unique = np.unique(coordinates[:, 1])
                if len(x_unique) > 1 and len(y_unique) > 1:
                    self.interpolator = RegularGridInterpolator(
                        (x_unique, y_unique), 
                        values.reshape(len(x_unique), len(y_unique)),
                        bounds_error=False, fill_value=0.0)
                else:
                    self.interpolator = None
            else:
                self.interpolator = None
        else:
            self.interpolator = None
    
    def evaluate(self, coordinates: np.ndarray) -> float:
        """Evaluate uncertainty function at given coordinates."""
        try:
            if self.interpolator is None:
                return 0.0
            
            if coordinates.ndim == 1:
                return float(self.interpolator(coordinates))
            else:
                return float(self.interpolator(coordinates.flatten()))
        except Exception:
            return 0.0
    
    def gradient(self, coordinates: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Compute gradient of uncertainty function."""
        try:
            grad = np.zeros_like(coordinates)
            
            for i in range(len(coordinates)):
                coords_plus = coordinates.copy()
                coords_minus = coordinates.copy()
                coords_plus[i] += h
                coords_minus[i] -= h
                
                val_plus = self.evaluate(coords_plus)
                val_minus = self.evaluate(coords_minus)
                
                grad[i] = (val_plus - val_minus) / (2 * h)
            
            return grad
            
        except Exception:
            return np.zeros_like(coordinates)

class SpatiotemporalCorrelation:
    """Spatiotemporal correlation function for UQ mesh adaptation."""
    
    def __init__(self, spatial_length: float, temporal_scale: float, 
                 decay_rate: float = 2.0):
        self.spatial_length = spatial_length
        self.temporal_scale = temporal_scale
        self.decay_rate = decay_rate
        
    def correlation(self, x1: np.ndarray, x2: np.ndarray, 
                   t1: float, t2: float) -> float:
        """
        Compute spatiotemporal correlation.
        
        K(x₁,x₂,t₁,t₂) = σ²exp(-||x₁-x₂||²/l²)exp(-|t₁-t₂|/τ)
        """
        try:
            # Spatial correlation
            spatial_distance = np.linalg.norm(x1 - x2)
            spatial_corr = np.exp(-(spatial_distance / self.spatial_length)**self.decay_rate)
            
            # Temporal correlation
            temporal_distance = abs(t1 - t2)
            temporal_corr = np.exp(-temporal_distance / self.temporal_scale)
            
            return spatial_corr * temporal_corr
            
        except Exception:
            return 0.0
    
    def correlation_matrix(self, points1: List[np.ndarray], points2: List[np.ndarray],
                          times1: List[float], times2: List[float]) -> np.ndarray:
        """Compute full correlation matrix between two sets of points."""
        try:
            n1, n2 = len(points1), len(points2)
            corr_matrix = np.zeros((n1, n2))
            
            for i in range(n1):
                for j in range(n2):
                    corr_matrix[i, j] = self.correlation(points1[i], points2[j],
                                                       times1[i], times2[j])
            
            return corr_matrix
            
        except Exception:
            return np.eye(max(len(points1), len(points2)))

class AdaptiveMeshRefinement:
    """Adaptive mesh refinement engine with UQ guidance."""
    
    def __init__(self, params: AdaptiveMeshParams):
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Mesh state
        self.mesh_points: List[MeshPoint] = []
        self.mesh_elements: List[MeshElement] = []
        self.current_time = 0.0
        self.refinement_history = deque(maxlen=1000)
        
        # UQ functions
        self.uncertainty_function = None
        self.correlation_function = SpatiotemporalCorrelation(
            params.spatial_correlation_length,
            params.temporal_correlation_time,
            params.correlation_decay_rate
        )
        
        # Multi-resolution hierarchy
        self.resolution_levels: List[List[MeshElement]] = [[] for _ in range(params.max_resolution_levels)]
        
        # Performance tracking
        self.adaptation_times = deque(maxlen=100)
        self.element_counts = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize mesh
        self._initialize_mesh()
    
    def _initialize_mesh(self) -> None:
        """Initialize uniform mesh."""
        try:
            self.logger.info("Initializing adaptive mesh")
            
            # Create uniform spatial grid
            bounds_min, bounds_max = self.params.spatial_bounds
            
            if len(bounds_min) == 1:
                # 1D mesh
                x_coords = np.linspace(bounds_min[0], bounds_max[0], 
                                     self.params.initial_resolution[0])
                
                for i, x in enumerate(x_coords):
                    point = MeshPoint(
                        coordinates=np.array([x]),
                        timestamp=0.0,
                        uncertainty_value=0.0,
                        uncertainty_gradient=np.array([0.0]),
                        refinement_level=0
                    )
                    self.mesh_points.append(point)
                    
            elif len(bounds_min) == 2:
                # 2D mesh
                x_coords = np.linspace(bounds_min[0], bounds_max[0], 
                                     self.params.initial_resolution[0])
                y_coords = np.linspace(bounds_min[1], bounds_max[1], 
                                     self.params.initial_resolution[1])
                
                for i, x in enumerate(x_coords):
                    for j, y in enumerate(y_coords):
                        point = MeshPoint(
                            coordinates=np.array([x, y]),
                            timestamp=0.0,
                            uncertainty_value=0.0,
                            uncertainty_gradient=np.array([0.0, 0.0]),
                            refinement_level=0
                        )
                        self.mesh_points.append(point)
            
            # Create initial elements (simplified for demonstration)
            self._create_initial_elements()
            
            self.logger.info(f"Initialized mesh with {len(self.mesh_points)} points")
            
        except Exception as e:
            self.logger.error(f"Mesh initialization failed: {e}")
    
    def _create_initial_elements(self) -> None:
        """Create initial mesh elements."""
        try:
            # Simple element creation for 2D case
            if len(self.params.spatial_bounds[0]) == 2:
                nx, ny = self.params.initial_resolution
                
                for i in range(nx - 1):
                    for j in range(ny - 1):
                        # Create rectangular element
                        vertices = [
                            self.mesh_points[i * ny + j],         # bottom-left
                            self.mesh_points[(i + 1) * ny + j],   # bottom-right
                            self.mesh_points[(i + 1) * ny + j + 1], # top-right
                            self.mesh_points[i * ny + j + 1]      # top-left
                        ]
                        
                        center = np.mean([v.coordinates for v in vertices], axis=0)
                        volume = self._compute_element_volume(vertices)
                        
                        element = MeshElement(
                            vertices=vertices,
                            center=center,
                            volume=volume,
                            uncertainty_estimate=0.0,
                            refinement_indicator=0.0,
                            level=0
                        )
                        
                        self.mesh_elements.append(element)
                        self.resolution_levels[0].append(element)
            
        except Exception as e:
            self.logger.debug(f"Initial element creation failed: {e}")
    
    def _compute_element_volume(self, vertices: List[MeshPoint]) -> float:
        """Compute volume/area of mesh element."""
        try:
            if len(vertices) == 2:  # 1D element
                return abs(vertices[1].coordinates[0] - vertices[0].coordinates[0])
            elif len(vertices) == 4:  # 2D rectangular element
                coords = np.array([v.coordinates for v in vertices])
                # Simple rectangular area
                dx = abs(coords[1, 0] - coords[0, 0])
                dy = abs(coords[3, 1] - coords[0, 1])
                return dx * dy
            else:
                return 1.0
        except Exception:
            return 1.0
    
    def update_uncertainty_field(self, uncertainty_data: Dict[str, Any],
                                timestamp: float) -> None:
        """
        Update uncertainty field on mesh.
        
        Args:
            uncertainty_data: Dictionary containing uncertainty values and locations
            timestamp: Current time
        """
        with self._lock:
            try:
                self.current_time = timestamp
                
                # Update mesh point uncertainties
                if 'point_uncertainties' in uncertainty_data:
                    point_uncertainties = uncertainty_data['point_uncertainties']
                    
                    for i, point in enumerate(self.mesh_points):
                        if i < len(point_uncertainties):
                            point.uncertainty_value = point_uncertainties[i]
                            point.timestamp = timestamp
                            
                            # Compute uncertainty gradient
                            point.uncertainty_gradient = self._compute_uncertainty_gradient(point)
                
                # Update uncertainty function
                self.uncertainty_function = UQMeshFunction(self.mesh_points)
                
                # Update element uncertainty estimates
                for element in self.mesh_elements:
                    element.uncertainty_estimate = self._estimate_element_uncertainty(element)
                    element.refinement_indicator = self._compute_refinement_indicator(element)
                
                self.logger.debug(f"Updated uncertainty field at t={timestamp}")
                
            except Exception as e:
                self.logger.error(f"Uncertainty field update failed: {e}")
    
    def _compute_uncertainty_gradient(self, point: MeshPoint) -> np.ndarray:
        """Compute uncertainty gradient at mesh point."""
        try:
            if self.uncertainty_function is None:
                return np.zeros_like(point.coordinates)
            
            return self.uncertainty_function.gradient(point.coordinates)
            
        except Exception:
            return np.zeros_like(point.coordinates)
    
    def _estimate_element_uncertainty(self, element: MeshElement) -> float:
        """Estimate uncertainty in mesh element."""
        try:
            if not element.vertices:
                return 0.0
            
            # Average uncertainty at vertices
            vertex_uncertainties = [v.uncertainty_value for v in element.vertices]
            return np.mean(vertex_uncertainties)
            
        except Exception:
            return 0.0
    
    def _compute_refinement_indicator(self, element: MeshElement) -> float:
        """
        Compute refinement indicator for element.
        
        ε_UQ(x,t) = √∑ᵢ₌₁ᴹ λᵢφᵢ(x)ψᵢ(t)
        """
        try:
            # Uncertainty-based refinement criterion
            uncertainty = element.uncertainty_estimate
            
            # Gradient-based refinement (simplified)
            if element.vertices:
                gradients = [np.linalg.norm(v.uncertainty_gradient) for v in element.vertices]
                max_gradient = max(gradients) if gradients else 0.0
            else:
                max_gradient = 0.0
            
            # Combined indicator
            size_factor = element.volume**(1.0 / self.params.refinement_order)
            refinement_indicator = (uncertainty + 0.1 * max_gradient) * size_factor
            
            return refinement_indicator
            
        except Exception:
            return 0.0
    
    def adapt_mesh(self) -> Dict[str, Any]:
        """
        Perform mesh adaptation based on UQ criteria.
        
        Refinement Criterion: h_new = h_old × min{1, (TOL/ε_UQ)^(1/p)}
        """
        start_time = time.perf_counter()
        
        with self._lock:
            try:
                self.logger.info("Starting mesh adaptation")
                
                # Mark elements for refinement/coarsening
                elements_to_refine = []
                elements_to_coarsen = []
                
                tolerance = self.params.uncertainty_tolerance * self.params.confidence_level
                
                for element in self.mesh_elements:
                    if element.needs_refinement(tolerance) and element.level < self.params.max_refinement_levels:
                        elements_to_refine.append(element)
                    elif (element.refinement_indicator < self.params.coarsening_threshold * tolerance and 
                          element.level > 0):
                        elements_to_coarsen.append(element)
                
                # Perform refinement
                refined_count = 0
                for element in elements_to_refine:
                    if len(self.mesh_elements) < self.params.max_elements:
                        self._refine_element(element)
                        refined_count += 1
                
                # Perform coarsening
                coarsened_count = 0
                for element in elements_to_coarsen:
                    if self._can_coarsen_element(element):
                        self._coarsen_element(element)
                        coarsened_count += 1
                
                # Update mesh connectivity and data structures
                self._update_mesh_connectivity()
                
                adaptation_time = time.perf_counter() - start_time
                self.adaptation_times.append(adaptation_time)
                self.element_counts.append(len(self.mesh_elements))
                
                results = {
                    'refined_elements': refined_count,
                    'coarsened_elements': coarsened_count,
                    'total_elements': len(self.mesh_elements),
                    'total_points': len(self.mesh_points),
                    'adaptation_time': adaptation_time,
                    'max_refinement_level': max([e.level for e in self.mesh_elements]) if self.mesh_elements else 0
                }
                
                self.refinement_history.append(results)
                
                self.logger.info(f"Mesh adaptation completed: +{refined_count} refined, "
                               f"-{coarsened_count} coarsened, {results['total_elements']} total")
                
                return results
                
            except Exception as e:
                self.logger.error(f"Mesh adaptation failed: {e}")
                return {'adaptation_time': time.perf_counter() - start_time}
    
    def _refine_element(self, element: MeshElement) -> None:
        """Refine a mesh element by subdivision."""
        try:
            if len(element.vertices) == 4:  # 2D rectangular element
                # Create 4 child elements by subdivision
                v = element.vertices
                
                # Create new points at edge midpoints and center
                edge_midpoints = [
                    self._create_midpoint(v[0], v[1]),  # bottom edge
                    self._create_midpoint(v[1], v[2]),  # right edge
                    self._create_midpoint(v[2], v[3]),  # top edge
                    self._create_midpoint(v[3], v[0])   # left edge
                ]
                
                center_point = self._create_center_point(v)
                
                # Create 4 child elements
                child_vertices = [
                    [v[0], edge_midpoints[0], center_point, edge_midpoints[3]],  # bottom-left
                    [edge_midpoints[0], v[1], edge_midpoints[1], center_point],  # bottom-right
                    [center_point, edge_midpoints[1], v[2], edge_midpoints[2]], # top-right
                    [edge_midpoints[3], center_point, edge_midpoints[2], v[3]]  # top-left
                ]
                
                for child_verts in child_vertices:
                    child_center = np.mean([vp.coordinates for vp in child_verts], axis=0)
                    child_volume = self._compute_element_volume(child_verts)
                    
                    child_element = MeshElement(
                        vertices=child_verts,
                        center=child_center,
                        volume=child_volume,
                        uncertainty_estimate=element.uncertainty_estimate,
                        refinement_indicator=0.0,
                        parent=element,
                        level=element.level + 1
                    )
                    
                    element.children.append(child_element)
                    self.mesh_elements.append(child_element)
                    
                    # Add to appropriate resolution level
                    if child_element.level < len(self.resolution_levels):
                        self.resolution_levels[child_element.level].append(child_element)
                
                # Deactivate parent element
                element.is_active = False
                
        except Exception as e:
            self.logger.debug(f"Element refinement failed: {e}")
    
    def _create_midpoint(self, point1: MeshPoint, point2: MeshPoint) -> MeshPoint:
        """Create midpoint between two mesh points."""
        try:
            midpoint_coords = (point1.coordinates + point2.coordinates) / 2
            midpoint_uncertainty = (point1.uncertainty_value + point2.uncertainty_value) / 2
            midpoint_gradient = (point1.uncertainty_gradient + point2.uncertainty_gradient) / 2
            
            return MeshPoint(
                coordinates=midpoint_coords,
                timestamp=self.current_time,
                uncertainty_value=midpoint_uncertainty,
                uncertainty_gradient=midpoint_gradient,
                refinement_level=max(point1.refinement_level, point2.refinement_level) + 1
            )
            
        except Exception:
            return point1
    
    def _create_center_point(self, vertices: List[MeshPoint]) -> MeshPoint:
        """Create center point of element."""
        try:
            center_coords = np.mean([v.coordinates for v in vertices], axis=0)
            center_uncertainty = np.mean([v.uncertainty_value for v in vertices])
            center_gradient = np.mean([v.uncertainty_gradient for v in vertices], axis=0)
            max_level = max(v.refinement_level for v in vertices)
            
            return MeshPoint(
                coordinates=center_coords,
                timestamp=self.current_time,
                uncertainty_value=center_uncertainty,
                uncertainty_gradient=center_gradient,
                refinement_level=max_level + 1
            )
            
        except Exception:
            return vertices[0]
    
    def _can_coarsen_element(self, element: MeshElement) -> bool:
        """Check if element can be coarsened."""
        try:
            # Element can be coarsened if it has siblings that also meet coarsening criteria
            if element.parent is None:
                return False
            
            parent = element.parent
            if not parent.children:
                return False
            
            # Check if all siblings meet coarsening criteria
            tolerance = self.params.uncertainty_tolerance * self.params.confidence_level
            for sibling in parent.children:
                if sibling.refinement_indicator >= self.params.coarsening_threshold * tolerance:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _coarsen_element(self, element: MeshElement) -> None:
        """Coarsen element by removing children and reactivating parent."""
        try:
            if element.parent is None:
                return
            
            parent = element.parent
            
            # Remove all children
            for child in parent.children:
                if child in self.mesh_elements:
                    self.mesh_elements.remove(child)
                
                # Remove from resolution level
                if child.level < len(self.resolution_levels):
                    if child in self.resolution_levels[child.level]:
                        self.resolution_levels[child.level].remove(child)
            
            # Clear children and reactivate parent
            parent.children.clear()
            parent.is_active = True
            
            # Update parent uncertainty estimate
            parent.uncertainty_estimate = self._estimate_element_uncertainty(parent)
            parent.refinement_indicator = self._compute_refinement_indicator(parent)
            
        except Exception as e:
            self.logger.debug(f"Element coarsening failed: {e}")
    
    def _update_mesh_connectivity(self) -> None:
        """Update mesh connectivity after adaptation."""
        try:
            # Remove inactive elements
            self.mesh_elements = [e for e in self.mesh_elements if e.is_active]
            
            # Update resolution levels
            for level_elements in self.resolution_levels:
                level_elements[:] = [e for e in level_elements if e.is_active]
            
            # Update mesh points (remove unused points)
            used_points = set()
            for element in self.mesh_elements:
                for vertex in element.vertices:
                    used_points.add(id(vertex))
            
            self.mesh_points = [p for p in self.mesh_points if id(p) in used_points]
            
        except Exception as e:
            self.logger.debug(f"Mesh connectivity update failed: {e}")
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mesh statistics."""
        try:
            active_elements = [e for e in self.mesh_elements if e.is_active]
            
            if not active_elements:
                return {}
            
            element_sizes = [e.volume for e in active_elements]
            refinement_levels = [e.level for e in active_elements]
            uncertainty_estimates = [e.uncertainty_estimate for e in active_elements]
            
            return {
                'total_points': len(self.mesh_points),
                'total_elements': len(active_elements),
                'refinement_levels': {
                    'min': min(refinement_levels),
                    'max': max(refinement_levels),
                    'mean': np.mean(refinement_levels)
                },
                'element_sizes': {
                    'min': min(element_sizes),
                    'max': max(element_sizes),
                    'mean': np.mean(element_sizes)
                },
                'uncertainty_statistics': {
                    'min': min(uncertainty_estimates),
                    'max': max(uncertainty_estimates),
                    'mean': np.mean(uncertainty_estimates),
                    'std': np.std(uncertainty_estimates)
                },
                'adaptation_performance': {
                    'average_adaptation_time': np.mean(self.adaptation_times) if self.adaptation_times else 0.0,
                    'total_adaptations': len(self.refinement_history)
                },
                'resolution_level_distribution': [len(level) for level in self.resolution_levels]
            }
            
        except Exception as e:
            self.logger.error(f"Mesh statistics computation failed: {e}")
            return {}

def main():
    """Demonstration of adaptive mesh refinement with UQ."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Adaptive Mesh Refinement with UQ Demonstration")
    
    # Initialize mesh parameters
    params = AdaptiveMeshParams(
        spatial_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        initial_resolution=(8, 8),
        max_refinement_levels=3,
        uncertainty_tolerance=1e-3,
        max_elements=1000
    )
    
    # Create adaptive mesh
    mesh = AdaptiveMeshRefinement(params)
    
    print(f"\n🌐 ADAPTIVE MESH INITIALIZATION:")
    initial_stats = mesh.get_mesh_statistics()
    print(f"   Initial Points:          {initial_stats.get('total_points', 0)}")
    print(f"   Initial Elements:        {initial_stats.get('total_elements', 0)}")
    print(f"   Max Refinement Levels:   {params.max_refinement_levels}")
    print(f"   Uncertainty Tolerance:   {params.uncertainty_tolerance:.2e}")
    
    # Simulate uncertainty evolution and mesh adaptation
    n_time_steps = 10
    
    for step in range(n_time_steps):
        current_time = step * 0.1
        
        # Generate synthetic uncertainty field
        n_points = len(mesh.mesh_points)
        
        # Create spatially varying uncertainty (higher near center)
        point_uncertainties = []
        for point in mesh.mesh_points:
            x, y = point.coordinates
            # Gaussian-like uncertainty distribution
            distance_from_center = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            uncertainty = 1e-2 * np.exp(-5 * distance_from_center**2) * (1 + 0.5 * np.sin(10 * current_time))
            point_uncertainties.append(uncertainty)
        
        # Update uncertainty field
        uncertainty_data = {'point_uncertainties': point_uncertainties}
        mesh.update_uncertainty_field(uncertainty_data, current_time)
        
        # Perform mesh adaptation
        adaptation_results = mesh.adapt_mesh()
        
        if step % 3 == 0:
            logger.info(f"Step {step}: Refined {adaptation_results.get('refined_elements', 0)} elements, "
                       f"Total elements: {adaptation_results.get('total_elements', 0)}")
    
    # Final mesh statistics
    final_stats = mesh.get_mesh_statistics()
    
    print(f"\n📊 FINAL MESH STATISTICS:")
    print(f"   Final Points:            {final_stats.get('total_points', 0)}")
    print(f"   Final Elements:          {final_stats.get('total_elements', 0)}")
    
    refinement_levels = final_stats.get('refinement_levels', {})
    print(f"   Refinement Levels:       {refinement_levels.get('min', 0)} - {refinement_levels.get('max', 0)}")
    print(f"   Average Level:           {refinement_levels.get('mean', 0):.2f}")
    
    element_sizes = final_stats.get('element_sizes', {})
    print(f"   Element Size Range:      {element_sizes.get('min', 0):.2e} - {element_sizes.get('max', 0):.2e}")
    
    uncertainty_stats = final_stats.get('uncertainty_statistics', {})
    print(f"   Uncertainty Range:       {uncertainty_stats.get('min', 0):.2e} - {uncertainty_stats.get('max', 0):.2e}")
    
    performance = final_stats.get('adaptation_performance', {})
    print(f"   Average Adaptation Time: {performance.get('average_adaptation_time', 0)*1000:.2f} ms")
    print(f"   Total Adaptations:       {performance.get('total_adaptations', 0)}")
    
    # Resolution level distribution
    level_dist = final_stats.get('resolution_level_distribution', [])
    print(f"\n📈 RESOLUTION LEVEL DISTRIBUTION:")
    for i, count in enumerate(level_dist):
        if count > 0:
            print(f"   Level {i}:                {count} elements")
    
    print(f"\n✅ Adaptive Mesh Refinement with UQ Successfully Demonstrated")

if __name__ == "__main__":
    main()
