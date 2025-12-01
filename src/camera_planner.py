from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import heapq

import numpy as np
from scipy.spatial import KDTree


@dataclass
class PlannerConfig:
    beats_per_shot: List[int]
    ease: str
    lateral_margin_ratio: float
    zoom_margin_ratio: float
    orbit_altitude_ratio: float
    dolly_altitude_ratio: float
    min_duration: float
    max_duration: float
    safety_margin_ratio: float
    clearance_ratio: float
    duration_scale: float = 3.0
    shot_mix: Dict[str, float] | None = None


@dataclass
class ViewPoint:
    """A validated camera position with visibility info."""
    position: np.ndarray
    visible_clusters: List[int]  # Indices of clusters visible from here
    view_score: float  # Quality score for this viewpoint
    neighbors: List[int]  # Indices of connected viewpoints


class NavigableSpace:
    """Pre-computed grid of valid camera positions with connectivity.
    
    This approach solves the camera-in-wall problem by:
    1. Pre-computing ALL valid positions before shot planning begins
    2. Building a connectivity graph so we only move between validated positions
    3. Checking obstacle clearance during grid construction, not during shot planning
    """
    
    def __init__(
        self,
        scene_info: Dict,
        grid_resolution: float = 3.0,
        min_obstacle_distance: float = 2.0,
    ):
        self.scene_info = scene_info
        self.grid_resolution = grid_resolution
        self.min_obstacle_distance = min_obstacle_distance
        
        # Scene bounds
        safe_bounds = scene_info.get("safe_bounds") or scene_info["bounds"]
        self.bounds_min = np.array(safe_bounds.get("min", scene_info["bounds"]["min"]))
        self.bounds_max = np.array(safe_bounds.get("max", scene_info["bounds"]["max"]))
        self.bounds_extent = self.bounds_max - self.bounds_min
        
        # Floor/ceiling from environment
        env_metrics = scene_info.get("environment", {}).get("metrics", {})
        self.floor_height = float(env_metrics.get("floor_height", self.bounds_min[1]))
        self.ceiling_height = float(env_metrics.get("ceiling_height", self.bounds_max[1]))
        
        # Obstacles - the occupancy grid from scene analysis
        obstacles = np.array(scene_info.get("occupancy", []))
        self.obstacle_tree = KDTree(obstacles) if obstacles.size > 0 else None
        
        # Clusters (points of interest)
        self.clusters = scene_info.get("clusters", [])
        self.cluster_centers = np.array([c["center"] for c in self.clusters]) if self.clusters else np.array([])
        
        # Environment type (indoor/outdoor)
        env_info = scene_info.get("environment", {})
        self.is_indoor = env_info.get("type") == "indoor"
        
        # Will be populated by build()
        self.viewpoints: List[ViewPoint] = []
        self.position_tree: Optional[KDTree] = None
        
    def build(self) -> None:
        """Build the navigable space grid and connectivity graph."""
        print("[NavigableSpace] Building navigable space grid...")
        
        # Generate candidate positions on a 3D grid
        candidates = self._generate_grid_candidates()
        print(f"[NavigableSpace] Generated {len(candidates)} grid candidates")
        
        # Filter to valid positions (not in obstacles, has visibility)
        valid_positions = []
        valid_scores = []
        valid_visible = []
        
        for pos in candidates:
            if self._is_in_obstacle(pos):
                continue
            
            # For indoor scenes, verify position is inside the environment (surrounded by walls)
            if self.is_indoor and not self._is_inside_environment(pos):
                continue
            
            # Check visibility to clusters
            visible = self._compute_visible_clusters(pos)
            if len(visible) == 0 and len(self.clusters) > 0:
                # Must see at least one cluster if clusters exist
                continue
            
            score = self._compute_view_score(pos, visible)
            valid_positions.append(pos)
            valid_scores.append(score)
            valid_visible.append(visible)
        
        print(f"[NavigableSpace] {len(valid_positions)} positions passed validation")
        
        if len(valid_positions) == 0:
            # Fallback: use scene center and try to find ANY valid position
            print("[NavigableSpace] No valid positions found, attempting fallback...")
            center = (self.bounds_min + self.bounds_max) / 2
            
            # Try sampling around center with reducing obstacle distance
            fallback_found = False
            for obstacle_mult in [0.5, 0.25, 0.1, 0.0]:
                test_dist = self.min_obstacle_distance * obstacle_mult
                if self.obstacle_tree is None or self._distance_to_obstacle(center) > test_dist:
                    valid_positions = [center]
                    valid_scores = [1.0]
                    valid_visible = [list(range(len(self.clusters)))]
                    fallback_found = True
                    print(f"[NavigableSpace] Fallback: using scene center with obstacle_mult={obstacle_mult}")
                    break
            
            if not fallback_found:
                # Last resort: use center regardless
                valid_positions = [center]
                valid_scores = [1.0]
                valid_visible = [list(range(len(self.clusters)))]
                print("[NavigableSpace] Fallback: using scene center (no obstacle check)")
        
        # Build KD-tree for connectivity queries
        positions_array = np.array(valid_positions)
        self.position_tree = KDTree(positions_array)
        
        # Create viewpoints with connectivity
        max_neighbor_dist = self.grid_resolution * 2.5
        
        for i, (pos, score, visible) in enumerate(zip(valid_positions, valid_scores, valid_visible)):
            # Find nearby viewpoints
            nearby_indices = self.position_tree.query_ball_point(pos, max_neighbor_dist)
            
            # Filter to connected neighbors (path must be clear)
            neighbors = []
            for j in nearby_indices:
                if i == j:
                    continue
                other_pos = valid_positions[j]
                if not self._path_blocked(pos, other_pos):
                    neighbors.append(j)
            
            self.viewpoints.append(ViewPoint(
                position=pos,
                visible_clusters=visible,
                view_score=score,
                neighbors=neighbors,
            ))
        
        print(f"[NavigableSpace] Built {len(self.viewpoints)} viewpoints with connectivity")
        
        # Report connectivity stats
        if self.viewpoints:
            avg_neighbors = sum(len(vp.neighbors) for vp in self.viewpoints) / len(self.viewpoints)
            max_neighbors = max(len(vp.neighbors) for vp in self.viewpoints)
            isolated = sum(1 for vp in self.viewpoints if len(vp.neighbors) == 0)
            print(f"[NavigableSpace] Connectivity: avg={avg_neighbors:.1f}, max={max_neighbors}, isolated={isolated}")
    
    def _generate_grid_candidates(self) -> List[np.ndarray]:
        """Generate a 3D grid of candidate positions."""
        candidates = []
        
        # Determine grid steps with margin from bounds
        margin = self.bounds_extent * 0.1
        start = self.bounds_min + margin
        end = self.bounds_max - margin
        
        # Use fewer height levels, biased toward middle of space
        vertical_span = self.ceiling_height - self.floor_height
        if vertical_span < 0.1:
            vertical_span = self.bounds_extent[1]
            self.floor_height = self.bounds_min[1]
            self.ceiling_height = self.bounds_max[1]
        
        height_levels = [
            self.floor_height + 0.25 * vertical_span,
            self.floor_height + 0.40 * vertical_span,
            self.floor_height + 0.55 * vertical_span,
            self.floor_height + 0.70 * vertical_span,
        ]
        
        x_range = end[0] - start[0]
        z_range = end[2] - start[2]
        
        x_steps = max(4, int(x_range / self.grid_resolution))
        z_steps = max(4, int(z_range / self.grid_resolution))
        
        for height in height_levels:
            if height < self.bounds_min[1] or height > self.bounds_max[1]:
                continue
            for xi in range(x_steps):
                x = start[0] + (xi / max(1, x_steps - 1)) * x_range if x_steps > 1 else (start[0] + end[0]) / 2
                for zi in range(z_steps):
                    z = start[2] + (zi / max(1, z_steps - 1)) * z_range if z_steps > 1 else (start[2] + end[2]) / 2
                    candidates.append(np.array([x, height, z]))
        
        return candidates
    
    def _distance_to_obstacle(self, pos: np.ndarray) -> float:
        """Get distance to nearest obstacle."""
        if self.obstacle_tree is None:
            return float('inf')
        dist, _ = self.obstacle_tree.query(pos)
        return float(dist)
    
    def _is_in_obstacle(self, pos: np.ndarray) -> bool:
        """Check if position is too close to obstacles."""
        return self._distance_to_obstacle(pos) < self.min_obstacle_distance
    
    def _is_inside_environment(self, pos: np.ndarray) -> bool:
        """Check if position is inside the environment using multiple robust methods.
        
        For indoor scenes, we use several stringent checks:
        1. Proximity check: Position must be close to occupied voxels (geometry)
        2. Density check: Must have sufficient geometry density in vicinity
        3. Enclosure check: Rays cast outward should hit walls in most directions
        4. Vertical check: Must have both floor below AND ceiling above
        5. Diagonal check: Additional rays at 45-degree angles for corners
        """
        if self.obstacle_tree is None:
            return True
        
        scene_diagonal = np.linalg.norm(self.bounds_extent)
        
        # Check 1: Proximity to occupied space (stricter threshold)
        # A position inside the building should be close to some geometry
        dist_to_nearest = self._distance_to_obstacle(pos)
        max_allowed_dist = scene_diagonal * 0.10  # Max 10% of scene extent (stricter)
        if dist_to_nearest > max_allowed_dist:
            return False
        
        # Check 2: Density check - count obstacles within a radius
        # Indoor positions should have geometry nearby in multiple directions
        density_radius = scene_diagonal * 0.15
        nearby_indices = self.obstacle_tree.query_ball_point(pos, density_radius)
        min_nearby_points = 50  # Need at least this many geometry points nearby
        if len(nearby_indices) < min_nearby_points:
            return False
        
        # Check 3: Must be within the tight bounds (where most geometry is)
        tight_bounds = self.scene_info.get("tight_bounds", {})
        if tight_bounds:
            tight_min = np.array(tight_bounds.get("min", self.bounds_min.tolist()))
            tight_max = np.array(tight_bounds.get("max", self.bounds_max.tolist()))
            # Stricter margin - only 5% outside tight bounds allowed
            margin = (tight_max - tight_min) * 0.05
            if np.any(pos < tight_min - margin) or np.any(pos > tight_max + margin):
                return False
        
        # Check 4: Enclosure check with 24 rays (16 horizontal + 8 diagonal)
        ray_length = scene_diagonal * 0.35
        
        # Horizontal rays (16 directions)
        horizontal_hits = 0
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
            if self._ray_hits_obstacle(pos, direction, ray_length):
                horizontal_hits += 1
        
        # Diagonal rays (8 directions at 45 degrees up and down)
        diagonal_hits = 0
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            # Upward diagonal
            dir_up = np.array([np.cos(angle) * 0.707, 0.707, np.sin(angle) * 0.707])
            if self._ray_hits_obstacle(pos, dir_up, ray_length):
                diagonal_hits += 1
            # Downward diagonal
            dir_down = np.array([np.cos(angle) * 0.707, -0.707, np.sin(angle) * 0.707])
            if self._ray_hits_obstacle(pos, dir_down, ray_length):
                diagonal_hits += 1
        
        # Check 5: Vertical rays - MUST have BOTH floor AND ceiling
        up_dir = np.array([0.0, 1.0, 0.0])
        down_dir = np.array([0.0, -1.0, 0.0])
        has_ceiling = self._ray_hits_obstacle(pos, up_dir, ray_length * 0.6)
        has_floor = self._ray_hits_obstacle(pos, down_dir, ray_length * 0.6)
        
        # Strict requirements for indoor:
        # - At least 75% horizontal hits (12 of 16)
        # - At least 50% diagonal hits (8 of 16)
        # - MUST have both floor AND ceiling
        horizontal_ok = horizontal_hits >= 12
        diagonal_ok = diagonal_hits >= 8
        vertical_ok = has_floor and has_ceiling
        
        return horizontal_ok and diagonal_ok and vertical_ok
    
    def _ray_hits_obstacle(self, start: np.ndarray, direction: np.ndarray, max_dist: float) -> bool:
        """Check if a ray from start in direction hits an obstacle within max_dist."""
        if self.obstacle_tree is None:
            return False
        
        # Sample along the ray with finer granularity
        steps = max(10, int(max_dist / (self.min_obstacle_distance * 0.5)))
        for i in range(1, steps + 1):
            t = i / steps
            point = start + direction * (max_dist * t)
            dist_to_obstacle = self._distance_to_obstacle(point)
            # Hit if we get close to any obstacle
            if dist_to_obstacle < self.min_obstacle_distance * 2.0:
                return True
        
        return False

    def _path_blocked(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if path between two points is blocked by obstacles."""
        if self.obstacle_tree is None:
            return False
        
        dist = np.linalg.norm(end - start)
        if dist < 0.1:
            return False
        
        # Sample points along the path
        steps = max(3, int(dist / (self.min_obstacle_distance * 0.5)))
        for i in range(1, steps):
            t = i / steps
            point = start * (1 - t) + end * t
            if self._is_in_obstacle(point):
                return True
        return False
    
    def _compute_visible_clusters(self, pos: np.ndarray) -> List[int]:
        """Compute which clusters are visible from this position."""
        if len(self.clusters) == 0:
            return []
        
        visible = []
        for i, cluster in enumerate(self.clusters):
            center = np.array(cluster["center"])
            if not self._path_blocked(pos, center):
                visible.append(i)
        return visible
    
    def _compute_view_score(self, pos: np.ndarray, visible_clusters: List[int]) -> float:
        """Score a viewpoint based on what it can see."""
        if len(visible_clusters) == 0:
            return 0.1  # Base score for having valid position
        
        score = 0.0
        for i in visible_clusters:
            cluster = self.clusters[i]
            center = np.array(cluster["center"])
            dist = np.linalg.norm(pos - center)
            radius = cluster.get("radius", 5.0)
            
            # Prefer positions at good viewing distance (1-3x cluster radius)
            ideal_dist = radius * 2.0
            dist_score = 1.0 / (1.0 + abs(dist - ideal_dist) / max(ideal_dist, 1.0))
            
            # Weight by cluster importance
            cluster_score = cluster.get("score", 1.0)
            
            score += dist_score * cluster_score
        
        return score
    
    def find_path(self, start_idx: int, end_idx: int) -> List[int]:
        """Find shortest path between two viewpoints using A*."""
        if start_idx == end_idx:
            return [start_idx]
        
        if start_idx >= len(self.viewpoints) or end_idx >= len(self.viewpoints):
            return []
        
        end_pos = self.viewpoints[end_idx].position
        
        # A* search
        open_set = [(0.0, start_idx)]
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {start_idx: 0.0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end_idx:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for neighbor in self.viewpoints[current].neighbors:
                tentative_g = g_score[current] + np.linalg.norm(
                    self.viewpoints[current].position - self.viewpoints[neighbor].position
                )
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = float(np.linalg.norm(self.viewpoints[neighbor].position - end_pos))
                    heapq.heappush(open_set, (tentative_g + h, neighbor))
        
        return []  # No path found
    
    def get_best_viewpoints_for_cluster(self, cluster_idx: int, count: int = 5) -> List[int]:
        """Get the best viewpoints for viewing a specific cluster."""
        if cluster_idx >= len(self.clusters):
            return []
        
        cluster_center = np.array(self.clusters[cluster_idx]["center"])
        candidates = []
        
        for i, vp in enumerate(self.viewpoints):
            if cluster_idx in vp.visible_clusters:
                dist = np.linalg.norm(vp.position - cluster_center)
                # Score: prefer moderate distance, weight by view score
                dist_penalty = abs(dist - 10.0) / 10.0
                score = vp.view_score / (1 + dist_penalty)
                candidates.append((score, i))
        
        candidates.sort(reverse=True)
        return [idx for _, idx in candidates[:count]]


class CameraPlanner:
    """Turns beats + scene cues into timed camera shots using navigable space.
    
    Key differences from previous approach:
    1. Pre-computes all valid camera positions BEFORE shot planning
    2. Uses graph-based pathfinding instead of random sampling
    3. Camera never starts in a wall because all positions are pre-validated
    4. Shot variety achieved through deterministic alternation, not random retry loops
    5. Uses beat strength/downbeats to time shot transitions on important beats
    6. Tracks spatial positions to avoid consecutive shots in similar locations
    """

    def __init__(
        self,
        scene_info: Dict,
        beat_info: Dict,
        config: PlannerConfig,
    ) -> None:
        self.scene = scene_info
        self.beats = beat_info
        self.config = config
        
        # Build navigable space with adaptive parameters
        diagonal = np.linalg.norm(np.array(scene_info["bounds"]["extent"]))
        grid_res = max(2.0, min(5.0, diagonal * 0.04))  # Adaptive grid resolution
        obstacle_dist = max(1.0, min(3.0, diagonal * 0.015))  # Adaptive obstacle distance
        
        print(f"[Planner] Scene diagonal: {diagonal:.1f}, grid_res: {grid_res:.2f}, obstacle_dist: {obstacle_dist:.2f}")
        
        self.nav_space = NavigableSpace(
            scene_info,
            grid_resolution=grid_res,
            min_obstacle_distance=obstacle_dist,
        )
        self.nav_space.build()
        
        # Scene geometry
        self.bounds_center = np.array(scene_info["bounds"]["center"])
        self.extent = np.array(scene_info["bounds"]["extent"])
        self.clusters = scene_info.get("clusters", [])
        safe_bounds = scene_info.get("safe_bounds") or scene_info["bounds"]
        self.safe_min = np.array(safe_bounds.get("min", scene_info["bounds"]["min"]))
        self.safe_max = np.array(safe_bounds.get("max", scene_info["bounds"]["max"]))
        self.safe_extent = self.safe_max - self.safe_min
        self.safe_center = (self.safe_min + self.safe_max) / 2.0
        tight_bounds = scene_info.get("tight_bounds") or {}
        tight_min = np.array(tight_bounds.get("min", self.safe_min.tolist()))
        tight_max = np.array(tight_bounds.get("max", self.safe_max.tolist()))
        env_info = scene_info.get("environment") or {}
        env_type = env_info.get("type", "ambiguous")
        env_conf = float(env_info.get("confidence", 0.0))
        env_metrics = env_info.get("metrics", {})
        # Relaxed indoor check to ensure we apply enclosure logic even for lower confidence scenes
        self.is_indoor = env_type == "indoor"
        self.is_outdoor = env_type == "outdoor" and env_conf >= 0.5
        self.floor_height = float(env_metrics.get("floor_height", self.safe_min[1]))
        self.ceiling_height = float(env_metrics.get("ceiling_height", self.safe_max[1]))
        self.vertical_span = max(self.ceiling_height - self.floor_height, 1e-3)
        clamp_min = self.safe_min.copy()
        clamp_max = self.safe_max.copy()
        if self.is_indoor:
            clamp_min = np.maximum(clamp_min, tight_min)
            clamp_max = np.minimum(clamp_max, tight_max)
            padding = 0.08 * np.maximum(clamp_max - clamp_min, 1e-4)
            clamp_min = np.minimum(clamp_max - 1e-4, clamp_min + padding)
            clamp_max = np.maximum(clamp_min + 1e-4, clamp_max - padding)
        floor_buffer = 0.05 * self.vertical_span
        ceil_buffer = (0.08 if self.is_indoor else 0.04) * self.vertical_span
        clamp_min[1] = max(clamp_min[1], self.floor_height + floor_buffer)
        clamp_max[1] = min(clamp_max[1], self.ceiling_height - ceil_buffer)
        self.flight_min = np.minimum(clamp_min, clamp_max - 1e-4)
        self.flight_max = np.maximum(clamp_max, self.flight_min + 1e-4)
        self.flight_extent = np.maximum(self.flight_max - self.flight_min, 1e-4)
        self.flight_center = (self.flight_min + self.flight_max) / 2.0
        if self.is_outdoor:
            self.motion_extent = self.safe_extent
            self.motion_center = self.safe_center
        else:
            self.motion_extent = self.flight_extent
            self.motion_center = self.flight_center
            # For indoor scenes, the bounding box center might be in a void (e.g. L-shape).
            # Prefer the largest cluster center if available and reasonably central.
            if self.clusters:
                best_cluster = np.array(self.clusters[0]["center"])
                # Only use it if it's within the flight bounds
                if np.all(best_cluster >= self.flight_min) and np.all(best_cluster <= self.flight_max):
                    self.motion_center = best_cluster

        vertical_span = max(float(self.flight_extent[1]), 1e-3)
        self.clearance_step = 0.05 * vertical_span
        self.max_clearance = self.config.clearance_ratio * vertical_span
        base_ratio = 0.42 if self.is_indoor else 0.32
        self.altitude = self._height_at_ratio(base_ratio)
        
        self.obstacles = np.array(scene_info.get("occupancy", []))
        self.obstacle_tree = KDTree(self.obstacles) if self.obstacles.size > 0 else None
        # Collision radius scales with scene size; smaller for dense indoor scenes to avoid false positives
        diag = float(np.linalg.norm(self.motion_extent))
        # Adjusted base radius for balance between safety and finding shots
        self.collision_radius = float(max(0.5, min(2.5, 0.03 * max(diag, 1.0))))
        self.used_focus_points: List[np.ndarray] = []
        self.used_camera_centers: List[np.ndarray] = []
        self.cluster_usage: Dict[int, int] = {}
        self.last_zoom_angle: float | None = None
        self.last_move_vector: np.ndarray | None = None

    def plan(self) -> Dict:
        beat_times = self.beats["beat_times"]
        if len(beat_times) < 2:
            raise ValueError("Beat tracker returned too few beats for planning")

        audio_duration = float(self.beats.get("duration", beat_times[-1]))
        target_duration = min(max(audio_duration, self.config.min_duration), self.config.max_duration)
        
        # Get downbeat indices for timing shot transitions on important beats
        downbeat_indices = self.beats.get("downbeat_indices", [])
        beat_strengths = self.beats.get("beat_strengths", [])
        
        # Build shot segments based on downbeats
        # Each shot spans from one downbeat to the next (or next-next for longer shots)
        shot_segments = self._build_shot_segments_from_downbeats(
            beat_times, downbeat_indices, beat_strengths, target_duration
        )
        
        print(f"[Planner] Created {len(shot_segments)} shot segments from downbeats")

        timeline: List[Dict] = []
        shot_counter = 1

        for start_idx, end_idx in shot_segments:
            start_time = beat_times[start_idx]
            end_time = beat_times[end_idx]
            
            self.shot_counter = shot_counter
            
            # Alternate between travel and focus shots only (no multi-keyframe orbit)
            # This ensures smooth 2-point interpolation without direction changes
            shot_type = shot_counter % 2
            if shot_type == 1:
                shot = self._build_travel_shot(
                    shot_id=f"shot_{shot_counter:02d}",
                    start_time=start_time,
                    end_time=end_time,
                    beat_range=(start_idx, end_idx),
                )
            else:
                shot = self._build_focus_shot(
                    shot_id=f"shot_{shot_counter:02d}",
                    start_time=start_time,
                    end_time=end_time,
                    beat_range=(start_idx, end_idx),
                )
            
            if shot:
                timeline.append(shot)
                shot_counter += 1
            else:
                # Try fallback
                shot = self._build_fallback_shot(
                    shot_id=f"shot_{shot_counter:02d}",
                    start_time=start_time,
                    end_time=end_time,
                    beat_range=(start_idx, end_idx),
                )
                if shot:
                    timeline.append(shot)
                    shot_counter += 1
                else:
                    print(f"[Planner] Failed to generate shot {shot_counter}")

        # Normalize times to start at 0
        offset = beat_times[0]
        for shot in timeline:
            shot["startTime"] -= offset
            shot["endTime"] -= offset
            for keyframe in shot["keyframes"]:
                keyframe["time"] -= offset

        if not timeline:
            raise ValueError("Unable to create any shots from beat grid")

        schedule = {
            "shots": timeline,
            "timeline_start": 0.0,
            "timeline_end": timeline[-1]["endTime"] if timeline else beat_times[-1] - offset,
            "target_duration": target_duration,
            "tempo": self.beats.get("tempo"),
            "beat_confidence": self.beats.get("confidence"),
            "beat_times": [bt - offset for bt in beat_times],
            "downbeat_indices": downbeat_indices,
            "beat_strengths": beat_strengths,
        }
        return schedule

    def _shot_generators(self) -> Iterable:
        mapping = {
            "wide_lateral": self._build_lateral_shot,
            "zoom_focus": self._build_zoom_shot,
            "orbit": self._build_orbit_shot,
            "elevated_dolly": self._build_dolly_shot,
            "slide_peek": self._build_slide_peek_shot,
            "diagonal_push": self._build_diagonal_push_shot,
            "pan_arc": self._build_pan_arc_shot,
        }
        ordered = self._shot_order_defaults()
        sequence = self._weighted_shot_sequence(ordered)
        return itertools.cycle((sid, mapping[sid]) for sid in sequence)

    def _shot_order_defaults(self) -> List[str]:
        return ["wide_lateral", "zoom_focus"]

    def _weighted_shot_sequence(self, defaults: List[str]) -> List[str]:
        # Force mix to only include allowed shots
        mix = {"wide_lateral": 0.6, "zoom_focus": 0.4}
        
        weights: List[Tuple[str, float]] = []
        for sid in defaults:
            weight = float(max(0.0, mix.get(sid, 0.0)))
            if weight > 0.0:
                weights.append((sid, weight))
        if not weights:
            weights = [(sid, 1.0) for sid in defaults]
        total = sum(weight for _, weight in weights)
        bucket = max(len(weights), 10)
        counts = {sid: max(1, int(round((weight / total) * bucket))) for sid, weight in weights}
        max_count = max(counts.values())
        sequence: List[str] = []
        for i in range(max_count):
            for sid, _ in weights:
                if i < counts[sid]:
                    sequence.append(sid)
        return sequence or defaults

    # Real implementations ------------------------------------------------

    def _build_lateral_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        # Try to find a fresh location with random sampling
        collision_rejections = 0
        los_rejections = 0
        path_rejections = 0
        used_rejections = 0
        for attempt in range(1000):
            # 1. Pick a potential focus point (what we are looking at)
            # Prioritize clusters that haven't been used much
            if attempt > 500 and self.clusters:
                 # Fallback: Pick from ANY cluster with less noise to ensure validity
                 c_idx = np.random.randint(0, len(self.clusters))
                 base = np.array(self.clusters[c_idx]["center"])
                 noise = (np.random.rand(3) - 0.5) * 8.0
                 candidate_focus = base + noise
            elif self.clusters and np.random.rand() < 0.70:
                # Sort clusters by usage
                cluster_indices = list(range(len(self.clusters)))
                cluster_indices.sort(key=lambda i: self.cluster_usage.get(i, 0))
                
                # Pick from the least used half, with probability weighted towards the very least used
                limit = max(1, len(cluster_indices) // 2)
                candidates = cluster_indices[:limit]
                c_idx = np.random.choice(candidates)
                
                base = np.array(self.clusters[c_idx]["center"])
                # Add noise to vary the specific part of the cluster
                noise = (np.random.rand(3) - 0.5) * 15.0
                candidate_focus = base + noise
            elif self.obstacles.size > 0:
                # Sample near actual geometry instead of random bounding box
                idx = np.random.randint(0, self.obstacles.shape[0])
                base = self.obstacles[idx]
                noise = (np.random.rand(3) - 0.5) * 5.0
                candidate_focus = base + noise
            else:
                # Random point in the flight bounds (fallback)
                # Ensure we don't pick points too close to the walls to avoid staring at them
                margin = self.flight_extent * 0.15
                safe_min = self.flight_min + margin
                safe_max = self.flight_max - margin
                # If bounds are too tight, fall back to full bounds
                if np.any(safe_min >= safe_max):
                    safe_min = self.flight_min
                    safe_max = self.flight_max
                
                r = np.random.rand(3)
                candidate_focus = safe_min + r * (safe_max - safe_min)
            
            # Ensure focus point is within safe bounds (prevents looking at floor/ceiling outliers)
            candidate_focus = np.clip(candidate_focus, self.safe_min, self.safe_max)

            # Ensure focus point is near geometry (within 4.0 units)
            if self.obstacle_tree:
                dist, _ = self.obstacle_tree.query(candidate_focus)
                if dist > 4.0:
                    continue

            # Check if we've used a similar focus point recently (check last 10 shots)
            min_focus_dist = 3.0 if attempt < 800 else 1.5
            if any(np.linalg.norm(candidate_focus - p) < min_focus_dist for p in self.used_focus_points[-10:]):
                used_rejections += 1
                continue
            
            # Check if we've used a similar camera center recently (check last 5 shots)
            # This prevents "looking at the same place from a little bit different angle"
            # We don't have the exact camera centers of previous shots easily available here, 
            # but we can approximate by checking if the focus point AND the angle are similar.
            # Or we can just rely on the focus point check above which is quite strict (5.0 units).
            
            # Let's add a check for the camera center itself if we can.
            # Since we haven't calculated cam_center yet, we can't check it here.
            # We will check it after calculating cam_center.

            # 2. Pick a viewing direction (random angle in horizontal plane)
            angle = np.random.rand() * 2 * np.pi
            view_dir = np.array([np.cos(angle), 0, np.sin(angle)])
            
            # 3. Pick a distance from the focus point
            # Use cluster radius if available to keep it relative to the object size
            # But ensure a minimum distance and CAP the maximum distance to avoid going outside
            
            # Cap the base distance based on scene size to avoid going through walls
            max_scene_dim = min(self.flight_extent[0], self.flight_extent[2])
            base_dist_cap = max(4.0, 0.4 * max_scene_dim)
            
            if self.clusters:
                # Find nearest cluster to focus to get a sense of scale
                dists = [np.linalg.norm(candidate_focus - np.array(c["center"])) for c in self.clusters]
                nearest_idx = np.argmin(dists)
                nearest_r = self.clusters[nearest_idx]["radius"]
                # Cap the base distance.
                base_dist = min(max(nearest_r * 0.6, 4.0), base_dist_cap)
            else:
                base_dist = min(0.15 * np.linalg.norm(self.motion_extent), base_dist_cap)
            
            dist = base_dist + np.random.rand() * 1.5
            
            # 4. Determine Camera Center
            cam_center = candidate_focus - view_dir * dist
            
            # 5. Determine Height
            # Vary height between 30% and 70% of vertical span
            h_ratio = 0.3 + np.random.rand() * 0.4
            cam_center[1] = self._height_at_ratio(h_ratio)
            
            # Check if this center is valid (in bounds and not in collision)
            cam_center = self._clamp_point(cam_center)

            # Extra check for indoor scenes to avoid clipping floor/ceiling
            if self.is_indoor:
                if cam_center[1] < self.floor_height + 0.2 or cam_center[1] > self.ceiling_height - 0.2:
                    los_rejections += 1
                    continue

            # Check if we've used a similar camera center recently (check last 10 shots)
            # Relax this check if we are struggling
            min_dist = 3.5
            if attempt > 800:
                min_dist = 1.0
            elif attempt > 500:
                min_dist = 2.0
                
            if any(np.linalg.norm(cam_center - p) < min_dist for p in self.used_camera_centers[-10:]):
                used_rejections += 1
                continue

            # Desperation mode for collision/enclosure
            col_mult = 1.0
            enc_thresh = 7
            los_mult = 1.3
            path_mult = 1.2
            
            if attempt > 800:
                col_mult = 0.6
                enc_thresh = 4
                los_mult = 1.0
                path_mult = 1.0
            
            if self._is_in_collision(cam_center, radius_multiplier=col_mult):
                collision_rejections += 1
                continue
            
            # Check enclosure for indoor scenes (prevents being in the void)
            if self.is_indoor and not self._check_enclosure(cam_center, threshold=enc_thresh):
                los_rejections += 1
                continue
            
            # Ensure camera is near geometry (within 5.0 units) to avoid being in the void
            if self.obstacle_tree:
                dist_cam, _ = self.obstacle_tree.query(cam_center)
                if dist_cam > 5.0:
                    los_rejections += 1
                    continue

            # CRITICAL: Check line of sight to focus. 
            if self._path_intersects_obstacle(cam_center, candidate_focus, radius_multiplier=los_mult):
                los_rejections += 1
                continue
            
            # Check spatial distance from recent shot positions
            if self._is_too_close_to_recent_shots(vp.position):
                continue
            
            score = vp.view_score
            
            # Strong penalty for already being used (exponential decay)
            score *= (0.4 ** use_count)
            
            # 7. Define Travel Path
            scene_diag = np.linalg.norm(self.motion_extent)
            base_travel = 0.05 * scene_diag + np.random.rand() * 0.1 * scene_diag
            
            # Reduce travel distance in desperation mode to find a valid path
            if attempt > 800:
                base_travel = 2.0 + np.random.rand() * 2.0
                
            travel_dist = base_travel
            start = cam_center - motion_axis * travel_dist * 0.5
            end = cam_center + motion_axis * travel_dist * 0.5
            
            # Ensure start/end height matches center
            start[1] = end[1] = cam_center[1]
            
            if self._is_direction_similar(start, end):
                start, end = end, start

            # Check path collision with slightly stricter radius
            if self._path_intersects_obstacle(start, end, radius_multiplier=path_mult):
                path_rejections += 1
                continue
                
            # If we are here, we found a valid shot!
            self.used_focus_points.append(candidate_focus)
            self.used_camera_centers.append(cam_center)
            if self.clusters:
                 # Update usage for the cluster we picked (or nearest one)
                 dists = [np.linalg.norm(candidate_focus - np.array(c["center"])) for c in self.clusters]
                 nearest_idx = np.argmin(dists)
                 self.cluster_usage[nearest_idx] = self.cluster_usage.get(nearest_idx, 0) + 1
            
            # 8. Define Targets for Minimal Rotation
            # To have no rotation, the view vector must be constant.
            # View vector = candidate_focus - cam_center
            # Target(t) = Position(t) + ViewVector
            fixed_view_vec = candidate_focus - cam_center
            
            # Add small random angle change (randomly selected)
            # We perturb the end view vector slightly to introduce a subtle rotation
            view_dist = np.linalg.norm(fixed_view_vec)
            # Random offset ~10% of distance => ~5-6 degrees rotation
            angle_offset = (np.random.rand(3) - 0.5) * 0.2 * view_dist
            end_view_vec = fixed_view_vec + angle_offset

            print(f"[Planner] Shot {shot_id} (wide_lateral): Focus={candidate_focus.round(2)}, Dist={dist:.2f}, Height={cam_center[1]:.2f}, Travel={travel_dist:.2f}")

            keyframes = [
                {"time": start_time, "position": start.tolist(), "target": (start + fixed_view_vec).tolist()},
                {"time": end_time, "position": end.tolist(), "target": (end + end_view_vec).tolist()},
            ]
            
            return self._package_shot(
                shot_id,
                "wide_lateral",
                start_time,
                end_time,
                beat_range,
                self._finalize_keyframes(keyframes, fix_orientation=True),
                description="Lateral tracking shot with subtle rotation",
            )

        print(f"[Planner] Shot {shot_id} (wide_lateral): Failed after {attempt+1} attempts. Rejects: used={used_rejections}, coll={collision_rejections}, los={los_rejections}, path={path_rejections}")
        return None

    def _build_zoom_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        shot_num = self._shot_number(shot_id)
        
        # Choose focus cluster avoiding recent repetitions
        focus = self.motion_center
        if self.clusters:
            # Get all available indices
            all_indices = list(range(len(self.clusters)))
            # Filter out those whose centers are close to recently used focus points
            valid_indices = []
            for idx in all_indices:
                center = np.array(self.clusters[idx]["center"])
                if not any(np.linalg.norm(center - p) < 2.0 for p in self.used_focus_points[-10:]):
                    valid_indices.append(idx)
            
            if not valid_indices:
                # If all used recently, pick the one used least recently (or just random)
                valid_indices = all_indices
            
            # Pick one randomly from valid
            cluster_idx = np.random.choice(valid_indices)
            focus = self._clamp_point(np.array(self._choose_focus_cluster(cluster_idx)))
            if self.clusters:
                 self.cluster_usage[cluster_idx] = self.cluster_usage.get(cluster_idx, 0) + 1
        
        # Base direction from motion center to focus
        base_dir = focus - self.motion_center
        if np.linalg.norm(base_dir) < 1e-3:
            base_dir = self._horizontal_axis(self.axes[1])

        base_dir = base_dir / np.linalg.norm(base_dir)

        planar_span = float(max(1e-3, min(self.motion_extent[0], self.motion_extent[2])))
        diag = np.linalg.norm(self.motion_extent)
        out_distance_base = 0.32 * planar_span + 0.25 * self.config.zoom_margin_ratio * diag
        if self.is_outdoor:
            out_distance_base = max(out_distance_base, 0.55 * np.linalg.norm(self.extent))
        cluster_radius = 0.08 * diag
        if self.clusters:
            cluster_radius = max(cluster_radius, float(self.clusters[0]["radius"]))
        in_distance_base = max(cluster_radius * 1.1, 0.12 * planar_span)
        if in_distance_base >= out_distance_base:
            in_distance_base = 0.55 * out_distance_base

        # Try multiple approach angles and small height adjustments to find a clear path.
        angle_candidates = list(np.linspace(0, 2 * np.pi, 12, endpoint=False))
        np.random.shuffle(angle_candidates)
        chosen = None
        chosen_angle = None
        chosen_start = None
        chosen_end = None
        chosen_out = None
        chosen_in = None

        for angle in angle_candidates:
            # Enforce angle jump limit relative to previous zoom angle (if set)
            deg = np.degrees(angle)
            if self.last_zoom_angle is not None:
                diff = abs(((deg - self.last_zoom_angle + 180) % 360) - 180)
                if diff > 120:
                    continue

            c, s = np.cos(angle), np.sin(angle)
            x, z = base_dir[0], base_dir[2]
            new_x = x * c - z * s
            new_z = x * s + z * c
            direction = np.array([new_x, base_dir[1], new_z])
            direction = direction / np.linalg.norm(direction)

            # Avoid vertical singularity
            if abs(direction[1]) > 0.75:
                direction[1] = 0.75 * np.sign(direction[1])
                direction = direction / np.linalg.norm(direction)

            # allow small variation in distances
            out_distance = out_distance_base + (np.random.rand() - 0.5) * 0.2 * out_distance_base
            in_distance = in_distance_base + (np.random.rand() - 0.5) * 0.2 * in_distance_base

            start_pos = focus + direction * out_distance
            end_pos = focus + direction * in_distance

            if self._is_direction_similar(start_pos, end_pos):
                continue

            # set heights
            start_pos[1] = self._height_at_ratio(0.48)
            min_focus_height = focus[1] + 0.1 * self.vertical_span
            end_pos[1] = self._height_at_ratio(0.35, minimum=min_focus_height)

            # clamp and check collisions / line of sight
            start_pos = self._clamp_point(start_pos)
            end_pos = self._clamp_point(end_pos)
            # Relaxed collision check for zoom (0.8 radius)
            if self._is_in_collision(start_pos, radius_multiplier=0.8) or self._is_in_collision(end_pos, radius_multiplier=0.8):
                continue
            # Relaxed LOS check (1.0 radius)
            if self._path_intersects_obstacle(start_pos, focus, radius_multiplier=1.0) or self._path_intersects_obstacle(start_pos, end_pos, radius_multiplier=1.0):
                # try raising the start height slightly to clear geometry
                for h_try in [0.1, 0.25, 0.5]:
                    s_try = start_pos.copy()
                    s_try[1] = min(self.flight_max[1], s_try[1] + h_try * self.vertical_span)
                    if not (self._is_in_collision(s_try) or self._path_intersects_obstacle(s_try, focus, radius_multiplier=1.2)):
                        start_pos = s_try
                        break
                else:
                    continue

            # passes checks
            chosen = True
            chosen_angle = angle
            chosen_start = start_pos
            chosen_end = end_pos
            chosen_out = out_distance
            chosen_in = in_distance
            break

        if not chosen:
            print(f"[Planner] Shot {shot_id} (zoom_focus): no clear angle found after relax, skipping")
            return None

        # remember angle
        self.last_zoom_angle = float(np.degrees(chosen_angle))
        self.used_focus_points.append(focus)
        self.used_camera_centers.append(chosen_start)

        keyframes = [
            {"time": start_time, "position": start_vp.position.tolist(), "target": target.tolist()},
            {"time": end_time, "position": clamped_end_pos.tolist(), "target": target.tolist()},
        ]
        
        print(f"[Planner] Shot {shot_id} (focus): Cluster {target_cluster_idx}, VP {start_vp_idx} -> {end_vp_idx}, speed={speed:.1f}/s")
        
        return self._package_shot(
            shot_id, "focus", start_time, end_time, beat_range, keyframes,
            description=f"Focus shot on cluster {target_cluster_idx}",
        )

    def _build_orbit_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        if self.is_indoor:
            return None
        cluster_idx = self._cluster_index_for_shot(shot_id, default=1)
        focus = self._clamp_point(np.array(self._choose_focus_cluster(index=cluster_idx)))
        axis_a = self._horizontal_axis(self.axes[0])
        axis_b = self._horizontal_axis(self.axes[2])
        radius = 0.45 * min(self.motion_extent[0], self.motion_extent[2])
        if self.is_outdoor:
            radius = max(radius, 0.6 * max(self.extent[0], self.extent[2]))
        times = np.linspace(start_time, end_time, num=5)
        keyframes = []
        for i, t in enumerate(times):
            angle = 2 * np.pi * (i / (len(times) - 1))
            pos = focus + axis_a * (radius * np.cos(angle)) + axis_b * (radius * np.sin(angle))
            min_height = focus[1] + 0.08 * self.vertical_span
            desired_ratio = 0.55 if self.is_outdoor else 0.5
            pos[1] = self._height_at_ratio(desired_ratio, minimum=min_height)
            keyframes.append({"time": float(t), "position": pos.tolist(), "target": focus.tolist()})
        return self._package_shot(
            shot_id,
            "orbit",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Orbit move to reveal parallax",
        )

    def _build_dolly_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        axis = self._horizontal_axis(self.axes[1])
        center = self.motion_center.copy()
        travel = 0.15 * np.linalg.norm(self.motion_extent)
        if self.is_indoor:
            travel *= 0.6
        elif self.is_outdoor:
            travel = max(travel, 0.4 * np.linalg.norm(self.extent))
        near = center - axis * travel
        far = center + axis * travel
        base_height = 0.46 if self.is_indoor else 0.35
        near[1] = self._height_at_ratio(base_height)
        far[1] = self._height_at_ratio(base_height + 0.06, minimum=near[1] + 0.05 * self.vertical_span)
        
        if self._is_direction_similar(near, far):
            near, far = far, near

        keyframes = [
            {"time": start_time, "position": near.tolist(), "target": center.tolist()},
            {"time": end_time, "position": far.tolist(), "target": center.tolist()},
        ]
        return self._package_shot(
            shot_id,
            "elevated_dolly",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Floating dolly for connective tissue",
        )

    def _build_slide_peek_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        cluster_idx = self._cluster_index_for_shot(shot_id)
        focus = self._clamp_point(np.array(self._choose_focus_cluster(cluster_idx)))
        axis = self._horizontal_axis(self.axes[2])
        planar_span = max(1e-3, min(self.motion_extent[0], self.motion_extent[2]))
        offset = 0.2 * planar_span
        side = -1.0 if self._shot_number(shot_id) % 2 == 0 else 1.0
        start = focus + axis * side * offset
        end = focus - axis * 0.4 * side * offset
        min_height = focus[1] + 0.05 * self.vertical_span
        height = self._height_at_ratio(0.4, minimum=min_height)
        start[1] = end[1] = height
        
        if self._is_direction_similar(start, end):
            start, end = end, start

        mid = (start + end) / 2.0
        keyframes = [
            {"time": start_time, "position": start.tolist(), "target": focus.tolist()},
            {"time": (start_time + end_time) / 2.0, "position": mid.tolist(), "target": focus.tolist()},
            {"time": end_time, "position": end.tolist(), "target": focus.tolist()},
        ]
        return self._package_shot(
            shot_id,
            "slide_peek",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Lateral slide that peeks around an interior feature",
        )

    def _build_diagonal_push_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        corners = [
            np.array([self.flight_min[0], self.flight_min[1], self.flight_min[2]]),
            np.array([self.flight_max[0], self.flight_min[1], self.flight_min[2]]),
            np.array([self.flight_min[0], self.flight_min[1], self.flight_max[2]]),
            np.array([self.flight_max[0], self.flight_min[1], self.flight_max[2]]),
        ]
        corner = corners[self._shot_number(shot_id) % len(corners)].copy()
        center = self.motion_center.copy()
        corner[1] = self._height_at_ratio(0.48)
        center[1] = self._height_at_ratio(0.5)
        offset_axis = self._horizontal_axis(self.axes[0] + self.axes[2])
        offset = offset_axis * 0.07 * np.linalg.norm(self.motion_extent)
        
        if self._is_direction_similar(corner, center):
             corner = corners[(self._shot_number(shot_id) + 1) % len(corners)].copy()
             corner[1] = self._height_at_ratio(0.48)

        keyframes = [
            {"time": start_time, "position": corner.tolist(), "target": center.tolist()},
            {"time": (start_time + end_time) / 2.0, "position": (corner + offset).tolist(), "target": center.tolist()},
            {"time": end_time, "position": center.tolist(), "target": center.tolist()},
        ]
        return self._package_shot(
            shot_id,
            "diagonal_push",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Diagonal push from a corner toward the focal volume",
        )

    def _build_pan_arc_shot(self, shot_id: str, start_time: float, end_time: float, beat_range) -> Dict:
        if not self.is_outdoor:
            return None
        focus = self.motion_center.copy()
        radius = max(0.6 * max(self.extent[0], self.extent[2]), 0.5 * np.linalg.norm(self.motion_extent))
        axis_a = self._horizontal_axis(self.axes[0])
        axis_b = self._horizontal_axis(self.axes[2])
        shot_num = self._shot_number(shot_id)
        base_angle = (shot_num % 4) * (np.pi / 2.0)
        sweep = np.pi / 3.0
        angles = np.linspace(base_angle, base_angle + sweep, num=4)
        altitude = self._height_at_ratio(0.55, minimum=focus[1] + 0.1 * self.vertical_span)
        
        start_pos = focus + axis_a * (radius * np.cos(angles[0])) + axis_b * (radius * np.sin(angles[0]))
        end_pos = focus + axis_a * (radius * np.cos(angles[-1])) + axis_b * (radius * np.sin(angles[-1]))
        if self._is_direction_similar(start_pos, end_pos):
            angles = angles[::-1]

        keyframes = []
        for t, angle in zip(np.linspace(start_time, end_time, num=len(angles)), angles):
            pos = focus + axis_a * (radius * np.cos(angle)) + axis_b * (radius * np.sin(angle))
            pos[1] = altitude
            keyframes.append({"time": float(t), "position": pos.tolist(), "target": focus.tolist()})
        return self._package_shot(
            shot_id,
            "pan_arc",
            start_time,
            end_time,
            beat_range,
            self._finalize_keyframes(keyframes),
            description="Slow panoramic arc for outdoor vistas",
        )

    def _choose_focus_cluster(self, index: int = 0) -> List[float]:
        if not self.clusters:
            return self.motion_center.tolist()
        clamped = min(max(index, 0), len(self.clusters) - 1)
        return self.clusters[clamped]["center"]

    def _cluster_index_for_shot(self, shot_id: str, default: int = 0) -> int:
        if not self.clusters:
            return default
        return self._shot_number(shot_id) % len(self.clusters)

    def _horizontal_axis(self, axis: np.ndarray) -> np.ndarray:
        horizontal = axis.copy()
        horizontal[1] = 0.0
        if np.linalg.norm(horizontal) < 1e-3:
            horizontal = np.array([1.0, 0.0, 0.0])
        return horizontal / np.linalg.norm(horizontal)

    def _clamp_point(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self.flight_min, self.flight_max)

    def _line_exits_bounds(self, origin: np.ndarray, target: np.ndarray) -> bool:
        samples = np.linspace(0.0, 1.0, num=10)[1:-1]
        for t in samples:
            point = origin * (1 - t) + target * t
            if np.any(point < self.flight_min) or np.any(point > self.flight_max):
                return True
        return False

    def _raise_for_clearance(self, position: np.ndarray, target: np.ndarray) -> np.ndarray:
        new_pos = position.copy()
        max_height = self.flight_max[1]
        added = 0.0
        while self._line_exits_bounds(new_pos, target) and added < self.max_clearance:
            new_pos[1] = min(new_pos[1] + self.clearance_step, max_height)
            added += self.clearance_step
            if new_pos[1] >= max_height:
                break
        return new_pos

    def _finalize_keyframes(self, keyframes: List[Dict], fix_orientation: bool = False) -> List[Dict]:
        # First pass: resolve collisions for individual points
        points = []
        targets = []
        for frame in keyframes:
            original_pos = np.array(frame["position"])
            pos = self._clamp_point(original_pos)
            # Do NOT clamp the target. The target is a look-at point and can be outside the bounds.
            tgt = np.array(frame["target"])
            
            pos = self._resolve_collision(pos)
            pos = self._clamp_point(pos)
            
            if self._line_exits_bounds(pos, tgt):
                pos = self._raise_for_clearance(pos, tgt)
                
            if self._is_in_collision(pos):
                 pos = self._resolve_collision(pos)
            
            # If we need to fix orientation, apply the position delta to the target
            if fix_orientation:
                delta = pos - original_pos
                tgt = tgt + delta
            
            # Ensure we don't get too close to the target (prevents flip)
            # If distance to target is very small, push back
            to_target = tgt - pos
            dist = np.linalg.norm(to_target)
            if dist < 2.0:
                # Push back along the view vector
                push_dir = -to_target / dist
                pos = pos + push_dir * (2.0 - dist)

            points.append(pos)
            targets.append(tgt)

        # Second pass: ensure paths between keyframes are clear
        # ... (rest of the function)
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            
            if self._path_intersects_obstacle(p1, p2):
                # Determine a safe pull target for this segment
                midpoint = (p1 + p2) / 2
                pull_target = self.motion_center
                if self.clusters:
                    dists = [np.linalg.norm(midpoint - np.array(c["center"])) for c in self.clusters]
                    nearest_idx = np.argmin(dists)
                    pull_target = np.array(self.clusters[nearest_idx]["center"])

                # ... (collision resolution logic)
                original_p1 = points[i].copy()
                original_p2 = points[i+1].copy()
                
                # Iteratively pull p1 and p2 towards pull_target
                for _ in range(15): # Increased iterations
                    if not self._path_intersects_obstacle(p1, p2, radius_multiplier=1.1): # Check with margin
                        break
                    p1 = p1 * 0.75 + pull_target * 0.25 # Stronger pull
                    p2 = p2 * 0.75 + pull_target * 0.25
                
                points[i] = p1
                points[i+1] = p2
                
                if fix_orientation:
                    targets[i] += (p1 - original_p1)
                    targets[i+1] += (p2 - original_p2)

        adjusted: List[Dict] = []
        for i, frame in enumerate(keyframes):
            adjusted.append({
                "time": frame["time"], 
                "position": points[i].tolist(), 
                "target": targets[i].tolist()
            })
        return adjusted

    def _package_shot(
        self,
        shot_id: str,
        style: str,
        start_time: float,
        end_time: float,
        beat_range,
        keyframes: List[Dict],
        description: str,
    ) -> Dict:
        if keyframes:
            start_pos = np.array(keyframes[0]["position"])
            end_pos = np.array(keyframes[-1]["position"])
            self.last_move_vector = end_pos - start_pos

        return {
            "id": shot_id,
            "style": style,
            "startTime": start_time,
            "endTime": end_time,
            "beatRange": beat_range,
            "keyframes": keyframes,
            "focus": keyframes[-1]["target"],
            "ease": self.config.ease,
            "description": description,
        }

    def _shot_number(self, shot_id: str) -> int:
        digits = "".join(ch for ch in shot_id if ch.isdigit())
        return int(digits) if digits else 0

    def _height_at_ratio(self, ratio: float, minimum: float | None = None) -> float:
        value = self.floor_height + float(np.clip(ratio, 0.0, 1.0)) * self.vertical_span
        if minimum is not None:
            value = max(value, minimum)
        return float(np.clip(value, self.flight_min[1], self.flight_max[1]))

    def _is_in_collision(self, point: np.ndarray, radius_multiplier: float = 1.0) -> bool:
        if self.obstacle_tree is None:
            return False
        # Query nearest neighbor
        dist, _ = self.obstacle_tree.query(point)
        return dist < (self.collision_radius * radius_multiplier)

    def _resolve_collision(self, point: np.ndarray) -> np.ndarray:
        if not self._is_in_collision(point):
            return point
        
        # Determine safe target: nearest cluster center or motion center
        target = self.motion_center
        if self.clusters:
            # Find nearest cluster center
            dists = [np.linalg.norm(point - np.array(c["center"])) for c in self.clusters]
            nearest_idx = np.argmin(dists)
            target = np.array(self.clusters[nearest_idx]["center"])

        # Try to move towards the safe target
        direction = target - point
        dist_to_target = np.linalg.norm(direction)
        if dist_to_target < 1e-3:
            return point
            
        direction /= dist_to_target
        
        # Iteratively step towards target
        current_pos = point.copy()
        step_size = 0.5
        max_steps = 20
        
        for _ in range(max_steps):
            current_pos += direction * step_size
            if not self._is_in_collision(current_pos):
                return current_pos
                
        # If simple push fails, try moving vertically towards target height
        current_pos = point.copy()
        vertical_dir = np.array([0.0, 1.0, 0.0])
        if target[1] < point[1]:
            vertical_dir *= -1
            
        for _ in range(max_steps):
            current_pos += vertical_dir * step_size
            if not self._is_in_collision(current_pos):
                return current_pos

        # If deterministic resolution fails, try random perturbations
        for _ in range(20):
            random_dir = np.random.randn(3)
            random_dir /= np.linalg.norm(random_dir)
            # Try varying distances
            for dist in [0.5, 1.0, 2.0, 3.0]:
                candidate = point + random_dir * dist
                candidate = self._clamp_point(candidate)
                if not self._is_in_collision(candidate):
                    return candidate

        return point

    def _check_enclosure(self, point: np.ndarray, threshold: int = 7) -> bool:
        """
        Checks if the point is 'enclosed' by obstacles, indicating it's inside a room.
        Casts rays in multiple directions (cardinal + diagonal).
        Returns True if a significant portion of rays hit obstacles.
        """
        if not self.is_indoor or self.obstacle_tree is None:
            return True # Assume valid if not indoor or no obstacles

        # Use scene diagonal as max ray distance
        max_dist = np.linalg.norm(self.motion_extent)
        
        # Define directions: 4 cardinal + 4 diagonal horizontal + up + down
        directions = [
            np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
            np.array([0.707, 0.0, 0.707]), np.array([0.707, 0.0, -0.707]),
            np.array([-0.707, 0.0, 0.707]), np.array([-0.707, 0.0, -0.707]),
            np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
        ]
        
        hits = 0
        for direction in directions:
            if self._ray_hit(point, direction, max_dist):
                hits += 1
        
        # We expect to hit something in most directions if we are inside.
        # 10 rays total.
        # If we are outside, we might hit the house in 1-3 directions, and ground in 1.
        # If we are inside, we should hit walls in almost all horizontal directions + floor/ceiling.
        return hits >= threshold

    def _ray_hit(self, start: np.ndarray, direction: np.ndarray, max_dist: float) -> bool:
        end = start + direction * max_dist
        # Use a larger radius multiplier to catch sparse point clouds
        return self._path_intersects_obstacle(start, end, radius_multiplier=1.5)

    def _path_intersects_obstacle(self, start: np.ndarray, end: np.ndarray, radius_multiplier: float = 1.0) -> bool:
        if self.obstacle_tree is None:
            return False
        
        dist = np.linalg.norm(end - start)
        if dist < 1e-3:
            return self._is_in_collision(start, radius_multiplier)
            
        # Use the effective radius for step calculation to ensure we don't skip over obstacles
        effective_radius = self.collision_radius * radius_multiplier
        steps = max(5, int(dist / (effective_radius * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            point = start * (1 - t) + end * t
            if self._is_in_collision(point, radius_multiplier):
                return True
        return False

    def _is_direction_similar(self, start: np.ndarray, end: np.ndarray, threshold: float = 0.85) -> bool:
        if self.last_move_vector is None:
            return False
        
        current_vec = end - start
        curr_len = np.linalg.norm(current_vec)
        if curr_len < 1e-3:
            return False
            
        last_len = np.linalg.norm(self.last_move_vector)
        if last_len < 1e-3:
            return False
            
        # Normalize
        curr_dir = current_vec / curr_len
        last_dir = self.last_move_vector / last_len
        
        dot_product = np.dot(curr_dir, last_dir)
        return dot_product > threshold
